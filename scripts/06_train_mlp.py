#!/usr/bin/env python3
"""
Step 7: Train the deployable MLP seizure detector (PyTorch, 16-bit-bound).

Architecture (locked with the user): 10 -> 16 -> 8 -> 1, ReLU, tiny dropout.
    - 10 inputs = the selected_features.json contract (k=10, chosen by the val
      sweep in 05). We slice those columns out of dataset.npz by INDEX, never by
      re-typing feature names -- and we assert the indices still name the same
      features, so a rebuilt dataset with shifted columns fails loudly instead
      of training on the wrong data.
    - (16, 8) hidden = the hardware sweet spot: ~320 params, 3 crossbars. The
      output layer emits a raw LOGIT (no sigmoid): sigmoid is monotone, so at
      inference we threshold the logit directly and the deployed net drops a
      transcendental op.

Class imbalance (~1% positives, and 66% of train positives are time-shift
augmentation copies)
    We train on ALL negatives (no subsampling, unlike the 05 sweep) so the
    false-alarm-per-hour number the model is optimised against reflects the real
    background rate (see evaluation-strategy in memory). The imbalance is handled
    in the LOSS instead: BCEWithLogitsLoss(pos_weight=w) multiplies the positive
    term by w, making the net "more sensitive to positives" exactly as asked.

    What is the right w? train neg/pos ~= 99.6 (incl. augmentation), so w=100 is
    ~full re-balancing. Rather than guess, we sweep w in {10, 25, 50, 100} and
    pick by VAL PR-AUC -- the same rank-based, threshold-free criterion used to
    pick k. The decision threshold is tuned at inference (0.3/0.5/0.7/0.9), not
    by moving w, so w only sets training sensitivity.

Overfitting control (the "optimization avoids overfitting" requirement)
    - L2 weight decay in AdamW.
    - Dropout 0.1 (kept small -- the net is tiny, heavy dropout would underfit).
    - Early stopping on VAL PR-AUC, not val loss: pos_weight distorts the loss
      scale, so val PR-AUC is the honest "are we still improving on the real
      objective" signal. Keep the best-PR-AUC snapshot, restore it at the end.

Data hygiene (see feature-selection-strategy / evaluation-strategy)
    - StandardScaler fit on TRAIN only; its mean/scale are saved so hardware can
      fold the affine normalisation into the first layer's W1/b1.
    - VAL and TEST are scored on REAL windows only (augmentation serves train).
    - TEST is untouched during training and pos_weight selection. It is scored
      once, only when --eval-test is passed, after the model is locked.

Outputs (results/mlp/)
    mlp_model.pt          state_dict + arch + chosen pos_weight
    scaler.json           feature mean/scale (for first-layer folding)
    train_report.json     pos_weight sweep, val metrics, (optional) test metrics
    training_curve.csv     per-epoch train loss / val PR-AUC for the final run

Usage
    # sweep pos_weight, train final model on best w, write artifacts:
    python scripts/06_train_mlp.py
    # override the pos_weight candidates:
    python scripts/06_train_mlp.py --pos-weights 25 50 75 100
    # after the model is locked, score the held-out test split ONCE:
    python scripts/06_train_mlp.py --eval-test
"""

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
def build_mlp(n_in, hidden=(16, 8), p_drop=0.1):
    """10 -> 16 -> 8 -> 1 logit. ReLU + dropout between hidden layers."""
    import torch.nn as nn

    layers, prev = [], n_in
    for h in hidden:
        layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(p_drop)]
        prev = h
    layers += [nn.Linear(prev, 1)]        # raw logit, no sigmoid
    return nn.Sequential(*layers)


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def load_selected_columns(dataset_path, contract_path, with_test=False):
    """Load dataset.npz and slice the selected columns BY INDEX.

    The contract (selected_features.json) carries feature_indices -- the actual
    column numbers into dataset.npz['features']. We slice by those, and assert
    that the columns still carry the expected names, so a rebuilt/reordered
    dataset fails loudly instead of silently training on wrong columns.

    TEST is only sliced out when with_test=True (i.e. --eval-test). A plain
    training/selection run never materialises the held-out split, so the
    "test untouched" claim is enforced by the code path, not just by discipline.
    """
    contract = json.loads(Path(contract_path).read_text(encoding="utf-8"))
    cols = list(contract["feature_indices"])
    want_names = list(contract["feature_names"])

    d = np.load(dataset_path, allow_pickle=True)
    feat_names = [str(n) for n in d["feature_names"]]
    got_names = [feat_names[i] for i in cols]
    if got_names != want_names:
        sys.exit(
            "selected_features.json is out of sync with dataset.npz.\n"
            f"  indices {cols}\n"
            f"  expected {want_names}\n"
            f"  found    {got_names}\n"
            "Re-run scripts/05_validate_feature_count.py --select k.")

    X = d["features"][:, cols].astype(np.float32)
    y = d["labels"].astype(np.int64)
    split = d["split"]
    is_aug = d["is_augmented"]

    tr = split == "train"
    va = (split == "val") & ~is_aug        # real windows only
    out = {
        "cols": cols, "names": want_names,
        "Xtr": X[tr], "ytr": y[tr],
        "Xva": X[va], "yva": y[va],
    }
    if with_test:
        # TEST is loaded ONLY when explicitly requested (--eval-test), so a plain
        # training run never so much as slices the held-out split into memory.
        te = (split == "test") & ~is_aug
        out["Xte"], out["yte"] = X[te], y[te]
    return out


def standardize(Xtr, *others):
    """Fit mean/std on train, apply to train + others. Returns (scaled..., mean, scale)."""
    mean = Xtr.mean(axis=0)
    scale = Xtr.std(axis=0)
    scale[scale == 0] = 1.0                # guard constant columns
    out = [((X - mean) / scale).astype(np.float32) for X in (Xtr, *others)]
    return (*out, mean.astype(np.float64), scale.astype(np.float64))


# --------------------------------------------------------------------------- #
# Train / eval one configuration
# --------------------------------------------------------------------------- #
def train_one(Xtr, ytr, Xva, yva, pos_weight, *, seed, epochs, batch,
              lr, weight_decay, patience, p_drop, device, verbose=False,
              curve_out=None):
    """Train the MLP with a given pos_weight; early-stop on val PR-AUC.

    Returns (best_val_ap, best_state_dict, best_val_roc).
    """
    import torch
    import torch.nn as nn
    from sklearn.metrics import average_precision_score, roc_auc_score

    torch.manual_seed(seed)
    np.random.seed(seed)

    model = build_mlp(Xtr.shape[1], p_drop=p_drop).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight], dtype=torch.float32, device=device))

    Xtr_t = torch.from_numpy(Xtr).to(device)
    ytr_t = torch.from_numpy(ytr.astype(np.float32)).to(device)
    Xva_t = torch.from_numpy(Xva).to(device)

    n = Xtr_t.shape[0]
    best_ap, best_roc, best_state, bad = -1.0, 0.0, None, 0
    curve = []

    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        ep_loss = 0.0
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            xb, yb = Xtr_t[idx], ytr_t[idx]
            opt.zero_grad()
            logit = model(xb).squeeze(1)
            loss = loss_fn(logit, yb)
            loss.backward()
            opt.step()
            ep_loss += loss.item() * len(idx)
        ep_loss /= n

        # ---- val PR-AUC on real windows ----
        model.eval()
        with torch.no_grad():
            va_logit = model(Xva_t).squeeze(1).cpu().numpy()
        va_score = 1.0 / (1.0 + np.exp(-va_logit))     # sigmoid for scoring only
        ap = average_precision_score(yva, va_score)
        roc = roc_auc_score(yva, va_score)
        curve.append((ep, ep_loss, ap, roc))
        if verbose:
            print(f"    ep{ep:>3d}  loss={ep_loss:.4f}  val_PR-AUC={ap:.4f}  "
                  f"val_ROC={roc:.4f}")

        if ap > best_ap:
            best_ap, best_roc = ap, roc
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                if verbose:
                    print(f"    early stop at ep{ep} (no val PR-AUC gain "
                          f"for {patience})")
                break

    if curve_out is not None:
        with open(curve_out, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["epoch", "train_loss", "val_pr_auc", "val_roc_auc"])
            w.writerows(curve)

    return best_ap, best_state, best_roc


def eval_split(state, n_in, X, y, thresholds, p_drop, device):
    """Load a state_dict, score a split, return PR-AUC/ROC + threshold metrics."""
    import torch
    from sklearn.metrics import (average_precision_score, roc_auc_score,
                                 precision_recall_fscore_support)
    model = build_mlp(n_in, p_drop=p_drop).to(device)
    model.load_state_dict(state)
    model.eval()
    with torch.no_grad():
        logit = model(torch.from_numpy(X).to(device)).squeeze(1).cpu().numpy()
    score = 1.0 / (1.0 + np.exp(-logit))
    out = {
        "pr_auc": float(average_precision_score(y, score)),
        "roc_auc": float(roc_auc_score(y, score)),
        "thresholds": {},
    }
    for t in thresholds:
        pred = (score >= t).astype(int)
        p, r, f, _ = precision_recall_fscore_support(
            y, pred, average="binary", zero_division=0)
        out["thresholds"][str(t)] = {"recall": float(r), "precision": float(p),
                                     "f1": float(f)}
    return out


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="data/processed/dataset.npz")
    ap.add_argument("--contract",
                    default="results/feature_selection/selected_features.json")
    ap.add_argument("--outdir", default="results/mlp")
    ap.add_argument("--pos-weights", type=float, nargs="+",
                    default=[10, 25, 50, 100],
                    help="pos_weight candidates; picked by val PR-AUC")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=8192)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-sweep-seeds", type=int, default=3,
                    help="Seeds per pos_weight in the sweep; pick by mean val "
                         "PR-AUC over seeds so the choice is not a single-draw "
                         "fluke (small MLP + extreme imbalance is high-variance).")
    ap.add_argument("--thresholds", type=float, nargs="+",
                    default=[0.3, 0.5, 0.7, 0.9])
    ap.add_argument("--eval-test", action="store_true",
                    help="Score the held-out TEST split once (after locking).")
    args = ap.parse_args()

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # TEST is loaded ONLY with --eval-test, so a plain training/selection run
    # never materialises the held-out split -- the "test untouched" claim is
    # enforced by the code path, not just by discipline.
    data = load_selected_columns(project_root / args.dataset,
                                 project_root / args.contract,
                                 with_test=args.eval_test)
    Xtr, ytr = data["Xtr"], data["ytr"]
    Xva, yva = data["Xva"], data["yva"]
    if args.eval_test:
        Xtr, Xva, Xte, mean, scale = standardize(Xtr, Xva, data["Xte"])
        yte = data["yte"]
    else:
        Xtr, Xva, mean, scale = standardize(Xtr, Xva)
        Xte = yte = None
    n_in = Xtr.shape[1]

    print("=" * 70)
    print("MLP training (10 -> 16 -> 8 -> 1, PyTorch)")
    print("=" * 70)
    print(f"device: {device}")
    print(f"features ({n_in}): {data['names']}")
    print(f"train : {len(ytr):,}  pos={int((ytr==1).sum()):,}  "
          f"neg/pos={(ytr==0).sum()/max((ytr==1).sum(),1):.1f}")
    print(f"val   : {len(yva):,}  pos={int((yva==1).sum()):,} (real)")
    if args.eval_test:
        print(f"test  : {len(yte):,}  pos={int((yte==1).sum()):,} (real, held out)")
    else:
        print("test  : not loaded (pass --eval-test to score it)")
    print()

    outdir = project_root / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    def common_for(seed):
        return dict(seed=seed, epochs=args.epochs, batch=args.batch,
                    lr=args.lr, weight_decay=args.weight_decay,
                    patience=args.patience, p_drop=args.dropout, device=device)

    # ---- pos_weight sweep on val PR-AUC, averaged over seeds ----------------
    # Each pos_weight is trained over several seeds (both the batch shuffle and
    # the weight init move with the seed), so the choice reflects mean val PR-AUC
    # rather than one lucky/unlucky run. We pick by mean; ties within one std are
    # reported so the choice is auditable.
    sweep_seeds = [args.seed + s for s in range(args.n_sweep_seeds)]
    print(f"pos_weight sweep (val PR-AUC, real val windows; "
          f"{args.n_sweep_seeds} seeds -> mean +/- std):")
    sweep = {}
    for w in args.pos_weights:
        aps, rocs = [], []
        for sd in sweep_seeds:
            ap_val, _state, roc_val = train_one(
                Xtr, ytr, Xva, yva, w, **common_for(sd))
            aps.append(ap_val)
            rocs.append(roc_val)
        sweep[w] = {
            "val_pr_auc_mean": float(np.mean(aps)),
            "val_pr_auc_std": float(np.std(aps)),
            "val_roc_auc_mean": float(np.mean(rocs)),
            "val_pr_auc_per_seed": [float(a) for a in aps],
        }
        print(f"  pos_weight={w:>6.1f}  val_PR-AUC={sweep[w]['val_pr_auc_mean']:.4f}"
              f" +/-{sweep[w]['val_pr_auc_std']:.4f}  "
              f"val_ROC={sweep[w]['val_roc_auc_mean']:.4f}")

    best_w = max(sweep, key=lambda w: sweep[w]["val_pr_auc_mean"])
    best_mean = sweep[best_w]["val_pr_auc_mean"]
    best_std = sweep[best_w]["val_pr_auc_std"]
    ties = [w for w in args.pos_weights
            if w != best_w and best_mean - sweep[w]["val_pr_auc_mean"] <= best_std]
    print(f"\nbest pos_weight = {best_w} "
          f"(val PR-AUC={best_mean:.4f} +/-{best_std:.4f})")
    if ties:
        print(f"  within one std of best: {ties} "
              f"(choice of {best_w} is not statistically separated from these)")

    # ---- retrain the chosen w at the base seed, log the curve, then evaluate -
    curve_path = outdir / "training_curve.csv"
    best_ap, best_state, best_roc = train_one(
        Xtr, ytr, Xva, yva, best_w, verbose=True,
        curve_out=curve_path, **common_for(args.seed))

    val_metrics = eval_split(best_state, n_in, Xva, yva, args.thresholds,
                             args.dropout, device)
    print("\nVAL metrics (locked model):")
    print(f"  PR-AUC={val_metrics['pr_auc']:.4f}  ROC-AUC={val_metrics['roc_auc']:.4f}")
    for t in args.thresholds:
        m = val_metrics["thresholds"][str(t)]
        print(f"  thr={t:.1f}  recall={m['recall']:.3f}  "
              f"precision={m['precision']:.3f}  F1={m['f1']:.3f}")

    test_metrics = None
    if args.eval_test:
        test_metrics = eval_split(best_state, n_in, Xte, yte, args.thresholds,
                                  args.dropout, device)
        print("\nTEST metrics (held-out, scored once):")
        print(f"  PR-AUC={test_metrics['pr_auc']:.4f}  "
              f"ROC-AUC={test_metrics['roc_auc']:.4f}")
        for t in args.thresholds:
            m = test_metrics["thresholds"][str(t)]
            print(f"  thr={t:.1f}  recall={m['recall']:.3f}  "
                  f"precision={m['precision']:.3f}  F1={m['f1']:.3f}")

    # ---- artifacts ----
    torch.save({
        "state_dict": best_state,
        "arch": {"n_in": n_in, "hidden": [16, 8], "dropout": args.dropout,
                 "output": "logit (apply sigmoid or threshold logit directly)"},
        "pos_weight": best_w,
        "feature_indices": data["cols"],
        "feature_names": data["names"],
    }, outdir / "mlp_model.pt")

    (outdir / "scaler.json").write_text(json.dumps({
        "note": "z = (x - mean) / scale; foldable into first Linear layer",
        "feature_names": data["names"],
        "feature_indices": data["cols"],
        "mean": mean.tolist(),
        "scale": scale.tolist(),
    }, indent=2), encoding="utf-8")

    (outdir / "train_report.json").write_text(json.dumps({
        "created": datetime.now().isoformat(timespec="seconds"),
        "device": device,
        "arch": "10 -> 16 -> 8 -> 1, ReLU, dropout=%.2f, logit output" % args.dropout,
        "contract": str(Path(args.contract).as_posix()),
        "pos_weight_sweep": {str(w): sweep[w] for w in args.pos_weights},
        "pos_weight_chosen": best_w,
        "pos_weight_selection": {
            "criterion": "max mean val PR-AUC over seeds",
            "n_sweep_seeds": args.n_sweep_seeds,
            "within_one_std_of_best": ties,
        },
        "hyperparams": {"epochs": args.epochs, "batch": args.batch, "lr": args.lr,
                        "weight_decay": args.weight_decay,
                        "patience": args.patience, "dropout": args.dropout,
                        "seed": args.seed, "n_sweep_seeds": args.n_sweep_seeds},
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,      # null unless --eval-test
    }, indent=2), encoding="utf-8")

    print(f"\nWrote {outdir/'mlp_model.pt'}, scaler.json, train_report.json, "
          f"training_curve.csv")
    if test_metrics is None:
        print("TEST not scored (pass --eval-test once the model is final).")


if __name__ == "__main__":
    main()
