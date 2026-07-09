#!/usr/bin/env python3
"""
Step 2: Data Windowing for CHB-MIT Dataset

Usage:
    # Process single file (for testing)
    python scripts/01_create_windows.py --patient chb01 --file chb01_03.edf

    # Process all files for one patient
    python scripts/01_create_windows.py --patient chb01 --all-files

    # Process all patients (all files with seizures)
    python scripts/01_create_windows.py --all-patients

Creates sliding windows from EDF files for seizure detection.
"""

import numpy as np
from pathlib import Path
import sys
import argparse
import re

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.preprocessing.windowing import process_edf_file
from src.utils.annotation_parser import parse_seizure_times


def find_files_with_seizures(patient_dir, summary_file):
    """Find all EDF files that contain seizures for a given patient."""
    files_with_seizures = []

    with open(summary_file, 'r') as f:
        content = f.read()

    # Find all file sections
    file_sections = re.findall(r'File Name: (chb\d+_\d+\.edf).*?(?=File Name:|$)', content, re.DOTALL)

    for edf_file in file_sections:
        # Check if this file has seizures
        seizures = parse_seizure_times(summary_file, edf_file)
        if seizures:
            edf_path = patient_dir / edf_file
            if edf_path.exists():
                files_with_seizures.append((edf_file, seizures))

    return files_with_seizures


def process_single_file(patient_id, edf_file, config):
    """Process a single EDF file."""
    patient_dir = config['RAW_DIR'] / patient_id
    edf_path = patient_dir / edf_file
    summary_file = patient_dir / f'{patient_id}-summary.txt'

    if not edf_path.exists():
        print(f"Error: File not found: {edf_path}")
        return None

    print(f"\nProcessing: {patient_id}/{edf_file}")
    print("-" * 70)

    # Process the file
    X, y, metadata = process_edf_file(
        edf_path=edf_path,
        summary_file=summary_file,
        target_channels=config['TARGET_CHANNELS'],
        window_duration=config['WINDOW_DURATION'],
        overlap_ratio=config['OVERLAP_RATIO'],
        seizure_threshold=config['SEIZURE_THRESHOLD'],
        augment_seizures=config['AUGMENT_SEIZURES'],
        random_seed=config['RANDOM_SEED']
    )

    # Save processed data
    config['PROCESSED_DIR'].mkdir(parents=True, exist_ok=True)

    # Extract recording number from filename (e.g., chb01_03.edf -> 03)
    recording_num = edf_file.replace('.edf', '').split('_')[-1]
    output_file = config['PROCESSED_DIR'] / f'{patient_id}_{recording_num}_windows.npz'

    np.savez_compressed(
        output_file,
        X=X,
        y=y,
        metadata=metadata  # Save complete metadata
    )

    print(f"\nSaved to: {output_file}")
    print(f"File size: {output_file.stat().st_size / 1024 / 1024:.2f} MB")
    print(f"Windows: {len(X)} ({np.sum(y==1)} seizure, {np.sum(y==0)} normal)")

    return output_file


def main():
    parser = argparse.ArgumentParser(description='Create windows from CHB-MIT EEG data')
    parser.add_argument('--patient', type=str, help='Patient ID (e.g., chb01)')
    parser.add_argument('--file', type=str, help='Specific EDF file to process')
    parser.add_argument('--all-files', action='store_true',
                       help='Process all files with seizures for specified patient')
    parser.add_argument('--all-patients', action='store_true',
                       help='Process all patients (all files with seizures)')
    parser.add_argument('--window-duration', type=float, default=4.0,
                       help='Window duration in seconds (default: 4.0)')
    parser.add_argument('--overlap', type=float, default=0.5,
                       help='Overlap ratio (default: 0.5 = 50%%)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility (default: 42)')

    args = parser.parse_args()

    # Configuration
    config = {
        'DATA_DIR': project_root / 'data',
        'RAW_DIR': project_root / 'data' / 'raw',
        'PROCESSED_DIR': project_root / 'data' / 'processed',
        'TARGET_CHANNELS': ['T7-P7', 'T8-P8', 'FZ-CZ'],
        'WINDOW_DURATION': args.window_duration,
        'OVERLAP_RATIO': args.overlap,
        'SEIZURE_THRESHOLD': 0.25,
        'AUGMENT_SEIZURES': True,
        'RANDOM_SEED': args.seed
    }

    print("=" * 70)
    print("EEG Data Windowing - CHB-MIT Dataset")
    print("=" * 70)
    print(f"Target channels: {config['TARGET_CHANNELS']}")
    print(f"Window: {config['WINDOW_DURATION']}s, Overlap: {config['OVERLAP_RATIO']*100:.0f}%")
    print(f"Seizure threshold: {config['SEIZURE_THRESHOLD']*100:.0f}% overlap")
    print(f"Random seed: {config['RANDOM_SEED']}")
    print("=" * 70)

    processed_files = []

    if args.all_patients:
        # Process all patients
        print("\nMode: Process all patients")
        patient_dirs = sorted([d for d in config['RAW_DIR'].iterdir()
                              if d.is_dir() and d.name.startswith('chb')])

        for patient_dir in patient_dirs:
            patient_id = patient_dir.name
            summary_file = patient_dir / f'{patient_id}-summary.txt'

            if not summary_file.exists():
                print(f"\nSkipping {patient_id}: no summary file")
                continue

            # Find files with seizures
            files_with_seizures = find_files_with_seizures(patient_dir, summary_file)

            if not files_with_seizures:
                print(f"\nSkipping {patient_id}: no files with seizures")
                continue

            print(f"\n{'='*70}")
            print(f"Patient: {patient_id} - {len(files_with_seizures)} files with seizures")
            print('='*70)

            for edf_file, seizures in files_with_seizures:
                output_file = process_single_file(patient_id, edf_file, config)
                if output_file:
                    processed_files.append(output_file)

    elif args.patient:
        if args.all_files:
            # Process all files for one patient
            print(f"\nMode: Process all files for {args.patient}")
            patient_dir = config['RAW_DIR'] / args.patient
            summary_file = patient_dir / f'{args.patient}-summary.txt'

            files_with_seizures = find_files_with_seizures(patient_dir, summary_file)
            print(f"Found {len(files_with_seizures)} files with seizures")

            for edf_file, seizures in files_with_seizures:
                output_file = process_single_file(args.patient, edf_file, config)
                if output_file:
                    processed_files.append(output_file)

        elif args.file:
            # Process single file
            print(f"\nMode: Process single file")
            output_file = process_single_file(args.patient, args.file, config)
            if output_file:
                processed_files.append(output_file)
        else:
            print("Error: Must specify --file or --all-files with --patient")
            return

    else:
        # Default: test on chb01_03
        print("\nMode: Test on chb01_03 (default)")
        print("Use --help to see all options")
        output_file = process_single_file('chb01', 'chb01_03.edf', config)
        if output_file:
            processed_files.append(output_file)

    # Summary
    print("\n" + "=" * 70)
    print("Processing Complete")
    print("=" * 70)
    print(f"Total files processed: {len(processed_files)}")
    print(f"Output directory: {config['PROCESSED_DIR']}")
    print()
    print("Next steps:")
    print("  1. Extract features: python scripts/02_extract_features.py --all-files")
    print("  2. Feature selection: python scripts/03_select_features.py")


if __name__ == '__main__':
    main()
