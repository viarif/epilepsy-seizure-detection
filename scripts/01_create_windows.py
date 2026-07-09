#!/usr/bin/env python3
"""
Step 2: Data Windowing for CHB-MIT Dataset

Usage:
    # Process single file (for testing)
    python scripts/01_create_windows.py --patient chb01 --file chb01_03.edf

    # Process every file for one patient (seizure + background)
    python scripts/01_create_windows.py --patient chb01 --all-files

    # Process every file for every patient
    python scripts/01_create_windows.py --all-patients

Creates sliding windows from EDF files for seizure detection.

All files are windowed, including background (interictal) recordings with no
seizures -- they supply the normal EEG needed for a realistic false-alarm rate.
Each edf is saved to its own *_windows.npz so recording gaps between files are
never bridged; sequence-level detection is built strictly within one file
(see src/utils/sequence_builder.py).
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
from src.utils.eeg_loader import ChannelNotFoundError


def list_patient_edf_files(patient_dir, summary_file):
    """List every EDF file for a patient, each paired with its seizures.

    Returns (edf_file, seizures) tuples for ALL files present on disk, not just
    those containing seizures. Background (interictal) files come back with an
    empty seizure list -- they still carry valuable normal EEG that improves
    specificity / false-alarm rate during continuous monitoring, so we window
    them too. Files listed in the summary but missing on disk are skipped.
    """
    files = []

    with open(summary_file, 'r') as f:
        content = f.read()

    # Each file section starts with "File Name:". Match any .edf token, not
    # just chbNN_NN.edf: real data also has chb17a_03.edf / chb17b_63.edf
    # (letter-suffixed montage sessions) and chb02_16+.edf (a '+' continuation
    # file). A stricter pattern silently dropped 22 files (4 with seizures).
    file_names = re.findall(r'File Name:\s*(\S+\.edf)', content)

    for edf_file in file_names:
        edf_path = patient_dir / edf_file
        if not edf_path.exists():
            continue
        seizures = parse_seizure_times(summary_file, edf_file)
        files.append((edf_file, seizures))

    return files


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

    # Process the file. Files whose montage lacks the bipolar target channels
    # (chb12's CS2 referential and single-electrode recordings) raise
    # ChannelNotFoundError -- skip them without aborting the whole batch.
    try:
        X, y, metadata = process_edf_file(
            edf_path=edf_path,
            summary_file=summary_file,
            target_channels=config['TARGET_CHANNELS'],
            window_duration=config['WINDOW_DURATION'],
            overlap_ratio=config['OVERLAP_RATIO'],
            seizure_threshold=config['SEIZURE_THRESHOLD'],
            augment_seizures=config['AUGMENT_SEIZURES'],
            random_seed=config['RANDOM_SEED'],
            patient_id=patient_id  # Directory name, so 'chb17a_03.edf' -> 'chb17'
        )
    except ChannelNotFoundError as e:
        print(f"  SKIPPED: incompatible montage - missing {e.missing}")
        return None

    # Save processed data
    config['PROCESSED_DIR'].mkdir(parents=True, exist_ok=True)

    # Name outputs by the full EDF stem, not patient_id + trailing number.
    # Multi-part recordings share a number across parts ('chb17a_03.edf' and
    # 'chb17c_03.edf' both end in '03'); using the stem keeps them distinct
    # and avoids one file silently overwriting another.
    edf_stem = edf_file.replace('.edf', '')
    output_file = config['PROCESSED_DIR'] / f'{edf_stem}_windows.npz'

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
                       help='Process all edf files (seizure + background) for specified patient')
    parser.add_argument('--all-patients', action='store_true',
                       help='Process all patients (all edf files, seizure + background)')
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
    skipped_files = []  # (patient/file, reason) for files we could not process

    def _run(patient_id, edf_file):
        """Process one file, recording success or skip reason."""
        result = process_single_file(patient_id, edf_file, config)
        if result is None:
            skipped_files.append((f"{patient_id}/{edf_file}", "channel/load error"))
        else:
            processed_files.append(result)

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

            # List ALL edf files (seizure + background). Background files
            # provide the interictal data needed for a realistic false-alarm
            # rate; sequence-level detection is built per-file downstream so
            # the recording gaps between files never bridge into a sequence.
            edf_files = list_patient_edf_files(patient_dir, summary_file)

            if not edf_files:
                print(f"\nSkipping {patient_id}: no usable edf files")
                continue

            n_sz = sum(1 for _, s in edf_files if s)
            print(f"\n{'='*70}")
            print(f"Patient: {patient_id} - {len(edf_files)} files "
                  f"({n_sz} with seizures, {len(edf_files) - n_sz} background)")
            print('='*70)

            for edf_file, seizures in edf_files:
                _run(patient_id, edf_file)

    elif args.patient:
        if args.all_files:
            # Process all files for one patient
            print(f"\nMode: Process all files for {args.patient}")
            patient_dir = config['RAW_DIR'] / args.patient
            summary_file = patient_dir / f'{args.patient}-summary.txt'

            edf_files = list_patient_edf_files(patient_dir, summary_file)
            n_sz = sum(1 for _, s in edf_files if s)
            print(f"Found {len(edf_files)} files "
                  f"({n_sz} with seizures, {len(edf_files) - n_sz} background)")

            for edf_file, seizures in edf_files:
                _run(args.patient, edf_file)

        elif args.file:
            # Process single file
            print(f"\nMode: Process single file")
            _run(args.patient, args.file)
        else:
            print("Error: Must specify --file or --all-files with --patient")
            return

    else:
        # Default: test on chb01_03
        print("\nMode: Test on chb01_03 (default)")
        print("Use --help to see all options")
        _run('chb01', 'chb01_03.edf')

    # Summary
    print("\n" + "=" * 70)
    print("Processing Complete")
    print("=" * 70)
    print(f"Total files processed: {len(processed_files)}")
    print(f"Files skipped: {len(skipped_files)}")
    if skipped_files:
        print("\nSkipped files (incompatible channel montage or load error):")
        for name, reason in skipped_files:
            print(f"  - {name}: {reason}")
    print(f"\nOutput directory: {config['PROCESSED_DIR']}")
    print()
    print("Next steps:")
    print("  1. Extract features: python scripts/02_extract_features.py --all-files")
    print("  2. Feature selection: python scripts/03_select_features.py")


if __name__ == '__main__':
    main()
