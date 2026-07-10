#!/usr/bin/env python3
"""
Step 6b: Quick validation sweep to choose the final feature count k.

Feature *ranking* (scripts/04_feature_selection.py) already produced a
de-correlated shortlist ordered by permutation importance and wrote it to
``results/feature_selection/feature_ranking.csv``. This script answers the
remaining question -- how many of those features to keep -- by training a
lightweight MLP on the TRAIN split for each candidate k and scoring it on the
VALIDATION split. The test split is not selected on or scored here.

Where the shortlist comes from (no hard-coding)
    We read feature_ranking.csv and keep the cluster representatives
    (is_cluster_representative == 1) that are *stable*: perm_mean > perm_std > 0.
    That is the same "worth selecting" set 04 prints. Reading it from the CSV
    means the two scripts can never drift out of sync -- if 04 is re-run and the
    ranking changes, this sweep follows automatically. Order is by permutation
    importance (the CSV's perm_rank), so "top k" is well defined.

Why validation, not test?
    Choosing k from test performance is data leakage (see feature-selection-
    strategy in project memory). Validation is exactly the knob-tuning set.

Why PR-AUC (average precision) as the headline metric?
    With ~1% positives, ROC-AUC is optimistic and accuracy is meaningless.
    Average precision (area under the precision-recall curve) is the honest
    threshold-independent summary for this imbalance. ROC-AUC is reported too
    for reference, plus recall/precision/F1 at a few decision thresholds.

Why a small MLP and not the RF?
    The downstream model is an MLP, so the trend across k should be measured
    with the same model family we will actually deploy. Kept tiny + early-
    stopped so the sweep runs in seconds.

Fairness notes:
    - StandardScaler is fit on the TRAIN subsample only, applied to val.
    - Negatives are subsampled for training speed/balance; val is scored on the
      FULL real distribution (no subsampling), so the metrics reflect reality.
    - The MLP training subsample keeps augmented positives (that mirrors how the
      real model will be trained); val is real windows only.
    - Same seeds / same scoring for every k, so differences are about k alone.

Output artifact:
    When --select k is given, writes results/feature_selection/
    selected_features.json with the chosen feature names, their column indices
    into dataset.npz, k, the val metrics, and all run parameters -- so the MLP
    step reads a machine-readable contract instead of a hand-copied list.

Usage:
    # sweep only (decide k by eye):
    python scripts/05_validate_feature_count.py
    python scripts/05_validate_feature_count.py --ks 8 10 11 12 13 15
    # sweep and lock in a choice (writes selected_features.json):
    python scripts/05_validate_feature_count.py --select 11
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


def load_stable_shortlist(csv_path):
    """Read feature_ranking.csv -> ordered list of stable cluster reps.

    Keep rows where is_cluster_representative == 1 (redundancy already removed
    by 04) AND perm_importance_mean > perm_importance_std > 0 (positive and
    larger than its own run-to-run noise -- i.e. statistically worth selecting;
    this is exactly the [OK] set 04 prints). Sort by perm_rank so index 0 is the
    strongest feature and "top k" means the k strongest stable reps.

    Returns a list of (feature_name, perm_mean, perm_std, gini, cluster).
    """
    rows = []
    with open(csv_path, newline="") as fh:
        for r in csv.DictReader(fh):
            if int(r["is_cluster_representative"]) != 1:
                continue
            pm = float(r["perm_importance_mean"])
            ps = float(r["perm_importance_std"])
            if not (pm > ps > 0):          # stable-and-positive gate
                continue
            rows.append({
                "rank": int(r["perm_rank"]),
                "name": r["feature"],
                "perm_mean": pm,
                "perm_std": ps,
                "gini": float(r["gini_importance"]),
                "cluster": int(r["cluster"]),
            })
    rows.sort(key=lambda d: d["rank"])
    return rows


def subsample_negatives(X, y, neg_per_pos, seed):
    """Keep all positives, randomly keep neg_per_pos negatives per positive."""
    rng = np.random.default_rng(seed)
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    n_keep = min(len(neg), len(pos) * neg_per_pos)
    neg_keep = rng.choice(neg, size=n_keep, replace=False)
    idx = np.concatenate([pos, neg_keep])
    rng.shuffle(idx)
    return X[idx], y[idx]


def eval_at_thresholds(y_true, scores, thresholds):
    """Recall / precision / F1 at each decision threshold."""
    from sklearn.metrics import precision_recall_fscore_support
    rows = []
    for t in thresholds:
        pred = (scores >= t).astype(int)
        p, r, f, _ = precision_recall_fscore_support(
            y_true, pred, average="binary", zero_division=0)
        rows.append((t, r, p, f))
    return rows


def run_one_k(cols, Xtr_full, ytr_full, Xva, yva, neg_per_pos, seeds,
              thresholds):
    """Train the small MLP over several seeds; return averaged metrics."""
    from sklearn.preprocessing import StandardScaler
    from sklearn.neural_network import MLPClassifier
    from sklearn.metrics import average_precision_score, roc_auc_score

    ap_runs, roc_runs = [], []
    thr_acc = {t: {"r": [], "p": [], "f": []} for t in thresholds}

    for seed in seeds:
        # Vary BOTH the train subsample and the MLP init with the seed so the
        # spread reflects real run-to-run variance, not just one draw.
        Xtr_k, ytr_k = subsample_negatives(
            Xtr_full[:, cols], ytr_full, neg_per_pos, seed)

        scaler = StandardScaler().fit(Xtr_k)          # fit on train only
        Xtr_s = scaler.transform(Xtr_k)
        Xva_s = scaler.transform(Xva[:, cols])

        clf = MLPClassifier(
            hidden_layer_sizes=(32, 16),
            activation="relu",
            alpha=1e-3,
            max_iter=200,
            early_stopping=True,
            n_iter_no_change=10,
            random_state=seed,
        )
        clf.fit(Xtr_s, ytr_k)
        scores = clf.predict_proba(Xva_s)[:, 1]

        ap_runs.append(average_precision_score(yva, scores))
        roc_runs.append(roc_auc_score(yva, scores))
        for t, r, p, f in eval_at_thresholds(yva, scores, thresholds):
            thr_acc[t]["r"].append(r)
            thr_acc[t]["p"].append(p)
            thr_acc[t]["f"].append(f)

    return {
        "ap_mean": float(np.mean(ap_runs)),
        "ap_std": float(np.std(ap_runs)),
        "roc_mean": float(np.mean(roc_runs)),
        "thr": {t: {"recall": float(np.mean(thr_acc[t]["r"])),
                    "precision": float(np.mean(thr_acc[t]["p"])),
                    "f1": float(np.mean(thr_acc[t]["f"]))}
                for t in thresholds},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="data/processed/dataset.npz")
    ap.add_argument("--ranking",
                    default="results/feature_selection/feature_ranking.csv")
    ap.add_argument("--outdir", default="results/feature_selection")
    ap.add_argument("--ks", type=int, nargs="+", default=None,
                    help="Candidate feature counts. Default: a spread across "
                         "the stable shortlist length.")
    ap.add_argument("--select", type=int, default=None,
                    help="If set, lock this k and write selected_features.json.")
    ap.add_argument("--neg-per-pos", type=int, default=15,
                    help="Negatives per positive in the TRAIN subsample")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-seeds", type=int, default=5,
                    help="MLP re-inits per k; report mean +/- std so that "
                         "k differences smaller than seed noise are visible")
    ap.add_argument("--thresholds", type=float, nargs="+",
                    default=[0.3, 0.5, 0.7, 0.9])
    args = ap.parse_args()

    ranking_path = project_root / args.ranking
    if not ranking_path.exists():
        sys.exit(f"ranking CSV not found: {ranking_path}\n"
                 f"Run scripts/04_feature_selection.py first.")

    shortlist = load_stable_shortlist(ranking_path)
    names = [d["name"] for d in shortlist]
    n_stable = len(names)
    if n_stable == 0:
        sys.exit("No stable representatives in the ranking CSV.")

    # Default candidate ks: a spread from a small floor up to the full stable
    # shortlist, so we see where the val curve flattens. Always include the
    # user's earlier interest point (11) if it is in range.
    if args.ks:
        ks = sorted({k for k in args.ks if 1 <= k <= n_stable})
    else:
        cand = {8, 10, 11, 12, 13, n_stable}
        ks = sorted({k for k in cand if 1 <= k <= n_stable})

    d = np.load(project_root / args.dataset, allow_pickle=True)
    feat_names = [str(n) for n in d["feature_names"]]
    name_to_idx = {n: i for i, n in enumerate(feat_names)}
    missing = [n for n in names if n not in name_to_idx]
    if missing:
        sys.exit(f"Shortlist features absent from dataset feature_names: {missing}")

    X_all = d["features"]
    y_all = d["labels"].astype(int)
    split = d["split"]
    is_aug = d["is_augmented"]

    tr = split == "train"
    # Val scored on REAL windows only -- augmented copies were already dropped
    # from val in 03_build_dataset, but filter defensively.
    va = (split == "val") & ~is_aug
    Xtr_full, ytr_full = X_all[tr], y_all[tr]
    Xva, yva = X_all[va], y_all[va]

    seeds = [args.seed + s for s in range(args.n_seeds)]

    print("=" * 70)
    print("Feature-count validation sweep (train -> val)")
    print("=" * 70)
    print(f"stable shortlist: {n_stable} features (from {ranking_path.name})")
    print(f"candidate ks: {ks}")
    print(f"train windows: {len(ytr_full):,}  (pos={int((ytr_full==1).sum()):,})")
    print(f"val   windows: {len(yva):,}  (pos={int((yva==1).sum()):,}, "
          f"prevalence={100*yva.mean():.2f}%)")
    print(f"neg_per_pos: {args.neg_per_pos}   n_seeds: {args.n_seeds}\n")

    # Show the ordered shortlist so the k->features mapping is explicit.
    print("Shortlist order (index : feature : perm_mean):")
    for i, dct in enumerate(shortlist, 1):
        print(f"  {i:>2}. {dct['name']:32s} {dct['perm_mean']:.5f}")
    print()

    results = {}
    for k in ks:
        cols = [name_to_idx[names[i]] for i in range(k)]
        res = run_one_k(cols, Xtr_full, ytr_full, Xva, yva,
                        args.neg_per_pos, seeds, args.thresholds)
        results[k] = res
        print(f"k={k:2d}  val PR-AUC={res['ap_mean']:.4f} "
              f"+/-{res['ap_std']:.4f}  ROC-AUC={res['roc_mean']:.4f}")
        for t in args.thresholds:
            m = res["thr"][t]
            print(f"        thr={t:.1f}  recall={m['recall']:.3f}  "
                  f"precision={m['precision']:.3f}  F1={m['f1']:.3f}")
        print()

    # Summary: PR-AUC mean +/- std. A k whose mean is within one std of the best
    # is a statistical tie -- prefer the smallest such k (fewer features =
    # cheaper, safer 16-bit hardware).
    print("=" * 70)
    print("Summary (headline = val PR-AUC, mean +/- std over seeds)")
    print("=" * 70)
    print(f"{'k':>3} {'PR-AUC':>8} {'+/-std':>8} {'ROC-AUC':>8}  vs-best")
    best_k = max(results, key=lambda k: results[k]["ap_mean"])
    best_ap = results[best_k]["ap_mean"]
    best_std = results[best_k]["ap_std"]
    tie_ks = []
    for k in ks:
        ap_mean = results[k]["ap_mean"]
        delta = ap_mean - best_ap
        if k == best_k:
            flag = "  <== best"
        else:
            is_tie = -delta <= best_std
            if is_tie:
                tie_ks.append(k)
            flag = f"  {delta:+.4f}" + ("  (tie, within best's std)" if is_tie else "")
        print(f"{k:>3} {ap_mean:>8.4f} {results[k]['ap_std']:>8.4f} "
              f"{results[k]['roc_mean']:>8.4f}{flag}")

    smallest_tie = min(tie_ks + [best_k])
    print(f"\nBest k={best_k} (PR-AUC={best_ap:.4f}). Smallest k statistically "
          f"tied with best: k={smallest_tie}.")
    print("Test split was not selected on or scored. Pick k, then re-run with "
          "--select k to lock it.")

    # --- Optional: lock a choice and write the machine-readable artifact -----
    if args.select is not None:
        k = args.select
        if k not in results:
            # Score the requested k if it was not in the sweep list.
            if not (1 <= k <= n_stable):
                sys.exit(f"--select {k} out of range 1..{n_stable}")
            cols = [name_to_idx[names[i]] for i in range(k)]
            results[k] = run_one_k(cols, Xtr_full, ytr_full, Xva, yva,
                                   args.neg_per_pos, seeds, args.thresholds)
        chosen = names[:k]
        chosen_idx = [name_to_idx[n] for n in chosen]
        artifact = {
            "created": datetime.now().isoformat(timespec="seconds"),
            "k": k,
            "feature_names": chosen,
            "feature_indices": chosen_idx,      # columns into dataset.npz features
            "selection": {
                "source_ranking": str(Path(args.ranking).as_posix()),
                "rule": "stable cluster representatives (is_rep==1 & "
                        "perm_mean>perm_std>0), top-k by permutation importance",
                "n_stable_available": n_stable,
            },
            "val_metrics": {
                "pr_auc_mean": results[k]["ap_mean"],
                "pr_auc_std": results[k]["ap_std"],
                "roc_auc_mean": results[k]["roc_mean"],
                "thresholds": {str(t): results[k]["thr"][t]
                               for t in args.thresholds},
            },
            "params": {
                "neg_per_pos": args.neg_per_pos,
                "seed": args.seed,
                "n_seeds": args.n_seeds,
                "mlp": "hidden=(32,16), relu, alpha=1e-3, early_stopping",
            },
            "per_feature": shortlist[:k],
        }
        outdir = project_root / args.outdir
        outdir.mkdir(parents=True, exist_ok=True)
        out_path = outdir / "selected_features.json"
        out_path.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
        print(f"\nLocked k={k}. Wrote {out_path}")
        print("Selected features:")
        for i, n in enumerate(chosen, 1):
            print(f"  {i:>2}. {n}  (dataset col {name_to_idx[n]})")


if __name__ == "__main__":
    main()
