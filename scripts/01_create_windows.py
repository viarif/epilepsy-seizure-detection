#!/usr/bin/env python3
"""
Step 2: Data Windowing for CHB-MIT Dataset

Usage:
    python scripts/01_create_windows.py

Creates sliding windows from EDF files for seizure detection.
"""

import numpy as np
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.preprocessing.windowing import process_edf_file


def main():
    # Configuration
    DATA_DIR = project_root / 'data'
    RAW_DIR = DATA_DIR / 'raw'
    PROCESSED_DIR = DATA_DIR / 'processed'

    # Target channels (T7-P7, T8-P8, FZ-CZ or alternative)
    TARGET_CHANNELS = ['T7-P7', 'T8-P8', 'FZ-CZ']

    # Windowing parameters
    WINDOW_DURATION = 4.0  # seconds
    OVERLAP_RATIO = 0.5    # 50% overlap
    SEIZURE_THRESHOLD = 0.25  # 25% overlap to label as seizure

    # Test on one file first: chb01_03.edf (has seizures)
    patient_id = 'chb01'
    edf_file = 'chb01_03.edf'

    patient_dir = RAW_DIR / patient_id
    edf_path = patient_dir / edf_file
    summary_file = patient_dir / f'{patient_id}-summary.txt'

    print("="*70)
    print("EEG Data Windowing - CHB-MIT Dataset")
    print("="*70)
    print(f"Patient: {patient_id}")
    print(f"Test file: {edf_file}")
    print(f"Target channels: {TARGET_CHANNELS}")
    print(f"Window: {WINDOW_DURATION}s, Overlap: {OVERLAP_RATIO*100:.0f}%")
    print(f"Seizure threshold: {SEIZURE_THRESHOLD*100:.0f}% overlap")
    print("="*70)
    print()

    # Process the file
    X, y, info = process_edf_file(
        edf_path=edf_path,
        summary_file=summary_file,
        target_channels=TARGET_CHANNELS,
        window_duration=WINDOW_DURATION,
        overlap_ratio=OVERLAP_RATIO,
        seizure_threshold=SEIZURE_THRESHOLD,
        augment_seizures=True
    )

    print()
    print("="*70)
    print("Processing Complete")
    print("="*70)
    print(f"Output shape: X={X.shape}, y={y.shape}")
    print(f"  X: [n_windows={X.shape[0]}, n_channels={X.shape[1]}, samples_per_window={X.shape[2]}]")
    print(f"  y: [n_windows={y.shape[0]}] with values {np.unique(y, return_counts=True)}")
    print()

    # Save processed data
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    output_file = PROCESSED_DIR / f'{patient_id}_03_windows.npz'
    np.savez_compressed(
        output_file,
        X=X,
        y=y,
        channels=info['channels'],
        sfreq=info['sfreq']
    )

    print(f"Saved to: {output_file}")
    print(f"File size: {output_file.stat().st_size / 1024 / 1024:.2f} MB")
    print()

    # Data quality check
    print("="*70)
    print("Data Quality Check")
    print("="*70)
    print(f"X statistics:")
    print(f"  Mean: {X.mean():.4f}, Std: {X.std():.4f}")
    print(f"  Min: {X.min():.4f}, Max: {X.max():.4f}")
    print(f"  NaN count: {np.isnan(X).sum()}")
    print(f"  Inf count: {np.isinf(X).sum()}")
    print()
    print(f"Label distribution:")
    print(f"  Class 0 (normal): {np.sum(y==0)} ({np.sum(y==0)/len(y)*100:.1f}%)")
    print(f"  Class 1 (seizure): {np.sum(y==1)} ({np.sum(y==1)/len(y)*100:.1f}%)")
    print(f"  Imbalance ratio: 1:{np.sum(y==0)/max(np.sum(y==1),1):.1f}")
    print()

    print("[OK] Windowing validation complete!")
    print()
    print("Next steps:")
    print("  1. Visualize some windows to verify correctness")
    print("  2. Extract features from windows (Step 4-5)")
    print("  3. Batch process all patient files")


if __name__ == '__main__':
    main()
