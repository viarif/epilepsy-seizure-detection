#!/usr/bin/env python3
"""
Step 5b: Merge per-recording feature files into one analysis-ready dataset.

Each EDF was windowed and turned into a ``{stem}_features.npz`` by
scripts/02_extract_features.py. Iterating over dozens of those files (and
re-deriving patient grouping, augmentation flags and the train/val/test split)
in every downstream model script is error-prone. This step does it once and
writes a single ``dataset.npz`` with every column aligned row-for-row:

    features      [N, 28] float32   feature matrix
    labels        [N]     int8      0 = normal, 1 = seizure
    patient_id    [N]     str       e.g. 'chb17' (normalised, no a/b/c suffix)
    edf_file      [N]     str       source recording, e.g. 'chb17a_03.edf'
    is_augmented  [N]     bool      True for time-shift augmentation copies
    split         [N]     str       'train' | 'val' | 'test' (patient-level)

Plus ``feature_names`` [28] and a ``meta`` dict (split definition, seed, counts).

Why keep augmented rows instead of dropping them here?
    Augmentation copies balance the minority (seizure) class for *training*.
    But they must never inflate val/test metrics, and window-level evaluation
    should run on real windows only. Keeping the ``is_augmented`` flag lets each
    consumer decide: training can use everything in the train split, evaluation
    filters to ``~is_augmented``. Dropping them now would throw that choice away.

Leakage guard:
    The split is by *patient*, so all windows from a patient (real + augmented)
    land in exactly one of train/val/test. Feature selection (Random Forest) and
    MLP training must read only rows where split == 'train'. See
    feature-selection-strategy in project memory.

Usage:
    python scripts/03_build_dataset.py
    python scripts/03_build_dataset.py --n-val 3 --n-test 3 --seed 42
"""

import argparse
import sys
from pathlib import Path

import numpy as np

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.data_split import make_patient_split, assign_split


def load_feature_file(path):
    """Load one ``*_features.npz`` and return aligned per-window arrays.

    Returns a dict of column arrays (all length = n_windows) or None if the
    file is malformed. Per-window ``is_augmented`` comes from the window
    metadata written during windowing; if absent (older files) we fall back to
    all-False, which is correct for un-augmented data.
    """
    data = np.load(path, allow_pickle=True)

    if 'features' not in data or 'labels' not in data:
        print(f"  WARNING: {path.name} missing features/labels, skipping")
        return None

    features = data['features']
    labels = data['labels']
    n = len(features)

    if len(labels) != n:
        print(f"  WARNING: {path.name} feature/label length mismatch, skipping")
        return None

    meta = data['metadata'].item() if 'metadata' in data else {}
    patient_id = meta.get('patient_id', 'unknown')
    edf_file = meta.get('edf_file', path.stem.replace('_features', '') + '.edf')

    # Per-window augmentation flags from the window metadata list.
    window_meta = meta.get('window_metadata', [])
    if len(window_meta) == n:
        is_aug = np.array([bool(w.get('is_augmented', False)) for w in window_meta])
    else:
        # Length mismatch or missing -> treat all as real windows.
        is_aug = np.zeros(n, dtype=bool)

    return {
        'features': features.astype(np.float32),
        'labels': labels.astype(np.int8),
        'patient_id': np.array([patient_id] * n),
        'edf_file': np.array([edf_file] * n),
        'is_augmented': is_aug,
    }


def main():
    parser = argparse.ArgumentParser(
        description='Merge per-recording feature files into one dataset.npz')
    parser.add_argument('--n-val', type=int, default=3,
                        help='Patients held out for validation (default: 3)')
    parser.add_argument('--n-test', type=int, default=3,
                        help='Patients held out for test (default: 3)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Seed for the patient-level split (default: 42)')
    parser.add_argument('--out', type=str, default='dataset.npz',
                        help='Output filename under data/processed (default: dataset.npz)')
    args = parser.parse_args()

    processed_dir = project_root / 'data' / 'processed'
    feature_files = sorted(processed_dir.glob('*_features.npz'))

    print("=" * 70)
    print("Build Dataset - merge per-recording features")
    print("=" * 70)

    if not feature_files:
        print(f"No *_features.npz found in {processed_dir}")
        print("Run scripts/02_extract_features.py --all-files first.")
        return

    print(f"Found {len(feature_files)} feature file(s)")

    # Load and collect every recording's columns.
    cols = {k: [] for k in
            ('features', 'labels', 'patient_id', 'edf_file', 'is_augmented')}
    feature_names = None

    for fpath in feature_files:
        rec = load_feature_file(fpath)
        if rec is None:
            continue
        for k in cols:
            cols[k].append(rec[k])
        if feature_names is None:
            d = np.load(fpath, allow_pickle=True)
            if 'feature_names' in d:
                feature_names = d['feature_names']

    if not cols['features']:
        print("No usable feature files. Aborting.")
        return

    # Concatenate into single aligned arrays.
    features = np.concatenate(cols['features'], axis=0)
    labels = np.concatenate(cols['labels'], axis=0)
    patient_id = np.concatenate(cols['patient_id'], axis=0)
    edf_file = np.concatenate(cols['edf_file'], axis=0)
    is_augmented = np.concatenate(cols['is_augmented'], axis=0)

    # Patient-level split (deterministic).
    patients = sorted(set(patient_id.tolist()))
    split_def = make_patient_split(patients, n_val=args.n_val,
                                   n_test=args.n_test, seed=args.seed)
    split = np.array([assign_split(p, split_def) for p in patient_id])

    # Drop augmented copies from val/test. Augmentation is a training-time
    # oversampling trick for the minority (seizure) class; it must never appear
    # in the evaluation sets or reported metrics would be inflated. Windowing
    # augments before the split exists, so those copies leak into val/test
    # patients here -- we remove them now so the saved dataset is safe to use
    # even if a downstream consumer forgets to filter on is_augmented.
    drop = is_augmented & (split != 'train')
    if drop.any():
        n_drop = int(drop.sum())
        keep = ~drop
        features = features[keep]
        labels = labels[keep]
        patient_id = patient_id[keep]
        edf_file = edf_file[keep]
        is_augmented = is_augmented[keep]
        split = split[keep]
        print(f"\nDropped {n_drop} augmented windows from val/test "
              f"(training-only augmentation).")

    # Report.
    print(f"\nTotal windows: {len(features)}")
    print(f"  Patients: {len(patients)}")
    print(f"  Seizure: {int((labels == 1).sum())}, "
          f"Normal: {int((labels == 0).sum())}")
    print(f"  Augmented: {int(is_augmented.sum())}, "
          f"Real: {int((~is_augmented).sum())}")
    print(f"\nSplit (patient-level, seed={args.seed}):")
    for name in ('train', 'val', 'test'):
        mask = split == name
        real = mask & ~is_augmented
        print(f"  {name:5s}: {len(split_def[name])} patients, "
              f"{int(mask.sum())} windows "
              f"({int((labels[real] == 1).sum())} real seizure, "
              f"{int((labels[real] == 0).sum())} real normal)")
        print(f"         patients: {split_def[name]}")

    # Save single analysis-ready file.
    out_path = processed_dir / args.out
    meta = {
        'split': split_def,
        'split_seed': args.seed,
        'n_val': args.n_val,
        'n_test': args.n_test,
        'n_windows': int(len(features)),
        'n_patients': len(patients),
        'patients': patients,
        'feature_dim': int(features.shape[1]),
    }
    np.savez_compressed(
        out_path,
        features=features,
        labels=labels,
        patient_id=patient_id,
        edf_file=edf_file,
        is_augmented=is_augmented,
        split=split,
        feature_names=feature_names if feature_names is not None else np.array([]),
        meta=meta,
    )
    print(f"\nSaved: {out_path}")
    print(f"File size: {out_path.stat().st_size / 1024 / 1024:.2f} MB")

    print("\nDownstream usage:")
    print("  d = np.load('data/processed/dataset.npz', allow_pickle=True)")
    print("  tr = (d['split'] == 'train')              # train mask")
    print("  Xtr, ytr = d['features'][tr], d['labels'][tr]")
    print("  # evaluation on real windows only:")
    print("  ev = (d['split'] == 'test') & ~d['is_augmented']")


if __name__ == '__main__':
    main()
