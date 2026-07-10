#!/usr/bin/env python3
"""
Step 8: Seizure-LEVEL evaluation on VAL, for one purpose only -- pick the
decision threshold.

Why this exists
    Window-level PR-AUC understates the model's clinical value: a seizure lasts
    many windows, and we only need to fire ONCE inside it to "catch" it. The
    quantity we actually deploy against is a threshold on the logit/probability,
    and the sensible way to choose it is the seizure-level trade-off it produces:

        detection rate (caught seizures / all seizures)   -- want high
        false alarms per hour                              -- want low

    This script sweeps thresholds on VAL and prints that trade-off so we can
    lock a threshold. It is NOT the paper-grade seizure-detection report (latency
    distributions, per-patient breakdowns, persistence smoothing, etc.) -- that
    is deferred. TEST is never touched here.

Time reconstruction (no explicit timestamps in dataset.npz)
    Windows are stored contiguously and in order within each edf_file (verified:
    val has 73 edf runs, each contiguous, no interleaving). Windowing used a 4 s
    window at 50% overlap => hop = 2 s. So within one edf, window i starts at
    i * 2 s. That is enough to (a) group windows into recordings, (b) rebuild
    ground-truth seizure events as maximal runs of label==1, and (c) turn window
    counts into wall-clock hours for the false-alarm rate.

Definitions (kept simple and defensible, since the goal is only thresholding)
    - Ground-truth seizure event = a maximal run of label==1 windows within one
      edf. Detection rate is over these events.
    - A seizure is DETECTED if >=1 window inside its label==1 span fires
      (score >= thr). Catch-once semantics.
    - A FALSE ALARM = a run of fired windows lying entirely outside any seizure
      event; consecutive fired windows are merged into one alarm, and separate
      runs are merged if the gap between them is <= --merge-gap seconds (default
      30 s), so one jittery stretch is not counted as many alarms. Rate =
      alarms / total_val_hours.
    - Latency = time from the event's first window to its first fired window
      (only over detected events).

Everything is computed on VAL real windows only (val already has no augmentation,
but we filter defensively). The model, scaler, and selected columns are read from
results/mlp/ so this stays in lock-step with 06_train_mlp.py.

Usage
    python scripts/07_seizure_level_eval.py
    python scripts/07_seizure_level_eval.py --thresholds 0.5 0.6 0.7 0.8 0.9 0.95
    python scripts/07_seizure_level_eval.py --merge-gap 30
"""

import argparse
import json
import sys
from itertools import groupby
from pathlib import Path

import numpy as np

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

HOP_SECONDS = 2.0        # 4 s window, 50% overlap -> 2 s hop (windowing params)


def build_mlp(n_in, hidden=(16, 8), p_drop=0.1):
    """Mirror of 06_train_mlp.build_mlp so the state_dict loads cleanly."""
    import torch.nn as nn
    layers, prev = [], n_in
    for h in hidden:
        layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(p_drop)]
        prev = h
    layers += [nn.Linear(prev, 1)]
    return nn.Sequential(*layers)


def contiguous_runs(mask):
    """Yield (start, stop) index pairs of maximal True runs in a bool array."""
    i, n = 0, len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            yield i, j
            i = j
        else:
            i += 1


def score_val(model_path, scaler_path, dataset_path, device):
    """Return per-edf ordered arrays of (scores, labels) for VAL real windows."""
    import torch

    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    cols = list(ckpt["feature_indices"])
    n_in = ckpt["arch"]["n_in"]
    p_drop = ckpt["arch"]["dropout"]

    scaler = json.loads(Path(scaler_path).read_text(encoding="utf-8"))
    mean = np.asarray(scaler["mean"], dtype=np.float64)
    scale = np.asarray(scaler["scale"], dtype=np.float64)
    # sanity: scaler and model must reference the same columns
    if list(scaler["feature_indices"]) != cols:
        sys.exit("scaler.json and mlp_model.pt disagree on feature_indices.")

    d = np.load(dataset_path, allow_pickle=True)
    feat_names = [str(n) for n in d["feature_names"]]
    got = [feat_names[i] for i in cols]
    if got != list(ckpt["feature_names"]):
        sys.exit("dataset.npz columns no longer match the model's feature_names; "
                 "re-run 05/06.")

    va = (d["split"] == "val") & ~d["is_augmented"]
    X = d["features"][va][:, cols].astype(np.float64)
    y = d["labels"][va].astype(np.int64)
    edf = d["edf_file"][va]

    Xs = ((X - mean) / scale).astype(np.float32)

    model = build_mlp(n_in, p_drop=p_drop).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    with torch.no_grad():
        logit = model(torch.from_numpy(Xs).to(device)).squeeze(1).cpu().numpy()
    score = 1.0 / (1.0 + np.exp(-logit))

    # Group into contiguous edf runs, preserving stored order (= time order).
    groups = []
    pos = 0
    for key, grp in groupby(edf):
        length = sum(1 for _ in grp)
        sl = slice(pos, pos + length)
        groups.append((str(key), score[sl], y[sl]))
        pos += length
    return groups


def evaluate(groups, thr, merge_gap_windows):
    """Seizure-level metrics at one threshold.

    Returns dict with n_seizures, n_detected, detection_rate, false_alarms,
    fa_per_hour, total_hours, median/mean latency (seconds, over detected).
    """
    n_seizures = n_detected = 0
    n_windows_total = 0
    false_alarm_count = 0
    latencies = []

    for _edf, score, y in groups:
        n_windows_total += len(score)
        fired = score >= thr
        seizure_spans = list(contiguous_runs(y == 1))

        # mark which windows are inside ANY seizure span
        in_seizure = (y == 1)

        # detection + latency per seizure event
        for s, e in seizure_spans:
            n_seizures += 1
            hit = np.where(fired[s:e])[0]
            if hit.size:
                n_detected += 1
                latencies.append(hit[0] * HOP_SECONDS)  # first fired offset

        # false alarms: fired windows outside seizures, merged into alarms
        fp = fired & ~in_seizure
        fp_runs = list(contiguous_runs(fp))
        # merge runs separated by <= merge_gap_windows into one alarm
        merged = 0
        prev_end = None
        for s, e in fp_runs:
            if prev_end is not None and (s - prev_end) <= merge_gap_windows:
                prev_end = e            # same alarm, extend
            else:
                merged += 1
                prev_end = e
        false_alarm_count += merged

    total_hours = n_windows_total * HOP_SECONDS / 3600.0
    return {
        "thr": thr,
        "n_seizures": n_seizures,
        "n_detected": n_detected,
        "detection_rate": n_detected / n_seizures if n_seizures else 0.0,
        "false_alarms": false_alarm_count,
        "fa_per_hour": false_alarm_count / total_hours if total_hours else 0.0,
        "total_hours": total_hours,
        "latency_median_s": float(np.median(latencies)) if latencies else None,
        "latency_mean_s": float(np.mean(latencies)) if latencies else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="data/processed/dataset.npz")
    ap.add_argument("--model", default="results/mlp/mlp_model.pt")
    ap.add_argument("--scaler", default="results/mlp/scaler.json")
    ap.add_argument("--outdir", default="results/mlp")
    ap.add_argument("--thresholds", type=float, nargs="+",
                    default=[0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95])
    ap.add_argument("--merge-gap", type=float, default=30.0,
                    help="Seconds; fired runs within this gap = one alarm.")
    ap.add_argument("--select-threshold", type=float, default=None,
                    help="Lock this decision threshold into the artifact "
                         "(chosen_threshold + criterion + selected_from_split).")
    args = ap.parse_args()

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"

    groups = score_val(project_root / args.model, project_root / args.scaler,
                       project_root / args.dataset, device)

    merge_gap_windows = int(round(args.merge_gap / HOP_SECONDS))
    total_hours = sum(len(s) for _, s, _ in groups) * HOP_SECONDS / 3600.0
    n_seiz = sum(1 for _, _, y in groups for _ in contiguous_runs(y == 1))

    print("=" * 72)
    print("Seizure-level evaluation on VAL (threshold-selection only)")
    print("=" * 72)
    print(f"device: {device}")
    print(f"val recordings (edf): {len(groups)}   total: {total_hours:.1f} h")
    print(f"ground-truth seizure events: {n_seiz}")
    print(f"merge-gap: {args.merge_gap:.0f}s ({merge_gap_windows} windows)  "
          f"hop: {HOP_SECONDS:.0f}s\n")

    print(f"{'thr':>5} {'det/seiz':>10} {'det-rate':>9} {'FA/h':>8} "
          f"{'lat-med':>8} {'lat-mean':>9}")
    rows = []
    for t in args.thresholds:
        r = evaluate(groups, t, merge_gap_windows)
        rows.append(r)
        lat_med = f"{r['latency_median_s']:.0f}s" if r['latency_median_s'] is not None else "-"
        lat_mn = f"{r['latency_mean_s']:.0f}s" if r['latency_mean_s'] is not None else "-"
        print(f"{t:>5.2f} {r['n_detected']:>4d}/{r['n_seizures']:<5d} "
              f"{r['detection_rate']:>9.3f} {r['fa_per_hour']:>8.2f} "
              f"{lat_med:>8} {lat_mn:>9}")

    print("\nReading the table: pick the highest threshold that still keeps a")
    print("clinically acceptable detection rate while dropping FA/h. Detection")
    print("rate is catch-once per event; FA/h merges jittery fires into alarms.")

    # --- lock a chosen threshold into a machine-readable field ---------------
    # Downstream (quantisation / inference) must not re-eyeball or hand-copy the
    # number: it reads chosen.threshold from this JSON. --select-threshold picks
    # the row; we store the rule and the split it was chosen on for provenance.
    chosen = None
    if args.select_threshold is not None:
        match = [r for r in rows if abs(r["thr"] - args.select_threshold) < 1e-9]
        if not match:
            sys.exit(f"--select-threshold {args.select_threshold} is not among the "
                     f"swept thresholds {args.thresholds}; add it and re-run.")
        r = match[0]
        chosen = {
            "threshold": args.select_threshold,
            "selected_from_split": "val",
            "criterion": "highest threshold whose detection rate plateaus while "
                         "FA/h keeps dropping (knee of the det-rate vs FA/h curve)",
            "detection_rate": r["detection_rate"],
            "fa_per_hour": r["fa_per_hour"],
            "latency_median_s": r["latency_median_s"],
            "n_detected": r["n_detected"],
            "n_seizures": r["n_seizures"],
        }
        print(f"\nLocked decision threshold = {args.select_threshold} "
              f"(val det-rate={r['detection_rate']:.3f}, FA/h={r['fa_per_hour']:.2f}, "
              f"lat-med={r['latency_median_s']}s)")

    out = project_root / args.outdir / "seizure_level_val.json"
    out.write_text(json.dumps({
        "purpose": "threshold selection only (not paper-grade seizure report)",
        "split": "val",
        "hop_seconds": HOP_SECONDS,
        "merge_gap_seconds": args.merge_gap,
        "total_hours": total_hours,
        "n_seizure_events": n_seiz,
        "chosen": chosen,          # null unless --select-threshold was passed
        "sweep": rows,
    }, indent=2), encoding="utf-8")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
