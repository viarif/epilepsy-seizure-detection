"""
Script to extract features from windowed EEG data.

This script:
1. Loads windowed data from data/processed/
2. Extracts 28 baseline features per window
3. Saves features and labels to data/processed/

Usage:
    # Extract features from single file
    python scripts/02_extract_features.py --file chb01_03_windows.npz

    # Extract features from all window files
    python scripts/02_extract_features.py --all-files
"""

import numpy as np
from pathlib import Path
import sys
import time
import argparse

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features import FeatureExtractor


def main():
    """Extract features from all processed window files."""

    parser = argparse.ArgumentParser(description='Extract features from windowed EEG data')
    parser.add_argument('--file', type=str, help='Specific window file to process')
    parser.add_argument('--all-files', action='store_true',
                       help='Process all window files')

    args = parser.parse_args()

    # Paths
    project_root = Path(__file__).parent.parent
    processed_dir = project_root / 'data' / 'processed'

    print("=" * 80)
    print("Feature Extraction - Step 4")
    print("=" * 80)
    print()

    # Find window files to process
    if args.file:
        window_files = [processed_dir / args.file]
    elif args.all_files:
        window_files = sorted(processed_dir.glob('*_windows.npz'))
    else:
        # Default: find all window files
        window_files = sorted(processed_dir.glob('*_windows.npz'))
        if not window_files:
            print("No window files found. Use --help for options.")
            return

    if not window_files:
        print(f"Error: No window files found in {processed_dir}")
        print("Please run scripts/01_create_windows.py first.")
        return

    print(f"Found {len(window_files)} window file(s):")
    for wf in window_files:
        print(f"  - {wf.name}")
    print()

    # Initialize feature extractor
    print("Initializing feature extractor...")
    extractor = FeatureExtractor(sfreq=256, fz_cz_channel_idx=2)
    print(f"  Total features: {extractor.n_features}")
    print(f"  Sampling frequency: {extractor.sfreq} Hz")
    print(f"  FZ-CZ channel index: {extractor.fz_cz_channel_idx}")
    print()

    # Process each window file
    for window_file in window_files:
        print(f"Processing: {window_file.name}")
        print("-" * 80)

        # Load windowed data
        print("Loading windowed data...")
        data = np.load(window_file, allow_pickle=True)

        # Check data format
        if 'X' in data:
            windows = data['X']
            labels = data['y']

            # Handle metadata
            if 'metadata' in data:
                metadata = data['metadata'].item()  # New format with full metadata
            else:
                # Old format: reconstruct minimal metadata
                metadata = {
                    'edf_file': window_file.stem.replace('_windows', ''),
                    'channels': data['channels'].tolist() if 'channels' in data else ['T7-P7', 'T8-P8', 'FZ-CZ'],
                    'sfreq': float(data['sfreq']) if 'sfreq' in data else 256.0,
                    'n_windows_total': len(windows),
                    'n_seizure': int(np.sum(labels == 1)),
                    'n_normal': int(np.sum(labels == 0)),
                }
        else:
            print(f"  Error: Unexpected data format in {window_file.name}")
            continue

        print(f"  Windows shape: {windows.shape}")
        print(f"  Labels shape: {labels.shape}")
        print(f"  Patient ID: {metadata.get('patient_id', 'unknown')}")
        print(f"  EDF file: {metadata.get('edf_file', 'unknown')}")
        print(f"  Channels: {metadata.get('channels', 'unknown')}")
        print(f"  Sampling frequency: {metadata.get('sfreq', 256)} Hz")
        print(f"  Seizure windows: {metadata.get('n_seizure', np.sum(labels==1))}")
        print(f"  Normal windows: {metadata.get('n_normal', np.sum(labels==0))}")
        print()

        # Extract features
        start_time = time.time()
        features = extractor.extract_batch(windows, verbose=True, batch_size=100)
        elapsed_time = time.time() - start_time

        print(f"  Time elapsed: {elapsed_time:.2f} seconds")
        print(f"  Speed: {len(windows) / elapsed_time:.1f} windows/second")
        print()

        # Compute feature statistics
        print("Feature statistics:")
        print(f"  Feature matrix shape: {features.shape}")
        print(f"  Feature range: [{features.min():.4f}, {features.max():.4f}]")
        print(f"  Feature mean: {features.mean():.4f}")
        print(f"  Feature std: {features.std():.4f}")
        print(f"  NaN count: {np.isnan(features).sum()}")
        print(f"  Inf count: {np.isinf(features).sum()}")
        print()

        # Check for problematic values
        if np.isnan(features).any() or np.isinf(features).any():
            print("Warning: NaN or Inf values detected in features!")
            print("  This may indicate issues with the input data or feature computation.")
            print()

        # Prepare output filename
        base_name = window_file.stem.replace('_windows', '')
        output_file = processed_dir / f'{base_name}_features.npz'
        feature_info_file = processed_dir / f'{base_name}_feature_info.txt'

        # Save features
        print(f"Saving features to: {output_file.name}")
        np.savez_compressed(
            output_file,
            features=features,
            labels=labels,
            feature_names=extractor.get_feature_names(),
            metadata=metadata  # Preserve all metadata
        )
        print(f"  Saved: {output_file}")

        # Save feature information
        print(f"Saving feature info to: {feature_info_file.name}")
        extractor.save_feature_info(feature_info_file)
        print(f"  Saved: {feature_info_file}")
        print()

        # Print per-class statistics
        print("Per-class feature statistics:")
        seizure_features = features[labels == 1]
        normal_features = features[labels == 0]

        if len(seizure_features) > 0:
            print(f"  Seizure windows ({len(seizure_features)}):")
            print(f"    Mean: {seizure_features.mean():.4f}")
            print(f"    Std: {seizure_features.std():.4f}")

        if len(normal_features) > 0:
            print(f"  Normal windows ({len(normal_features)}):")
            print(f"    Mean: {normal_features.mean():.4f}")
            print(f"    Std: {normal_features.std():.4f}")
        print()

        print("=" * 80)
        print()

    print("Feature extraction completed!")
    print()
    print("Next steps:")
    print("  1. Merge all features: python scripts/merge_all_features.py")
    print("  2. Run feature selection (Random Forest) - scripts/03_select_features.py")
    print("  3. Train MLP classifier - scripts/04_train_mlp.py")
    print()


if __name__ == '__main__':
    main()
