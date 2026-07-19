"""Preprocess EDF recordings without materializing overlapping windows."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.preprocessing.channel_selection import list_edf_files
from src.preprocessing.config import DEFAULT_SPLIT_CONFIG, DatasetSplit
from src.preprocessing.pipeline import (
    manifest_row_from_metadata,
    preprocess_recording,
    write_manifest,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=Path, default=Path('data/raw'))
    parser.add_argument(
        '--output-root',
        type=Path,
        default=Path('data/processed/selected4'),
    )
    parser.add_argument('--split-config', type=Path, default=DEFAULT_SPLIT_CONFIG)
    parser.add_argument(
        '--channels-json',
        type=Path,
        default=Path('data/processed/channel_selection.json'),
        help='Reviewed training-only four-channel selection JSON.',
    )
    parser.add_argument(
        '--splits',
        nargs='+',
        choices=('train', 'val', 'test'),
        default=('train', 'val', 'test'),
    )
    parser.add_argument('--patients', nargs='+')
    parser.add_argument(
        '--edf-names',
        nargs='+',
        help='Optional exact EDF filenames for targeted format smoke tests.',
    )
    parser.add_argument('--max-files-per-patient', type=int)
    parser.add_argument('--overwrite', action='store_true')
    return parser.parse_args()


def load_channels(args, split):
    with open(args.channels_json, 'r', encoding='utf-8') as handle:
        selection = json.load(handle)
    if not selection.get('complete'):
        raise SystemExit(
            'Refusing incomplete channel selection. Run the full '
            'training-only channel selection first.'
        )
    if selection.get('split_name') != split.name:
        raise SystemExit(
            'Channel selection split mismatch: '
            f"selection={selection.get('split_name')!r}, "
            f'preprocessing={split.name!r}.'
        )
    if tuple(selection.get('train_patient_ids', ())) != split.train:
        raise SystemExit(
            'Channel selection does not contain the exact configured '
            'training-patient list.'
        )
    return tuple(selection['selected_channels'])


def main():
    args = parse_args()
    split = DatasetSplit.from_json(args.split_config)
    channels = load_channels(args, split)
    requested = tuple(args.patients or split.patients_for(args.splits))
    invalid = sorted(
        patient
        for patient in requested
        if split.role_for(patient) not in args.splits
    )
    if invalid:
        raise SystemExit(
            f'Requested patients are outside --splits {args.splits}: {invalid}'
        )

    rows = []
    found_edf_names = set()
    for patient_id in requested:
        role = split.role_for(patient_id)
        patient_dir = args.data_root / patient_id
        summary_path = patient_dir / f'{patient_id}-summary.txt'
        edf_files = list(list_edf_files(args.data_root, patient_id))
        if args.edf_names:
            requested_names = set(args.edf_names)
            edf_files = [path for path in edf_files if path.name in requested_names]
            found_edf_names.update(path.name for path in edf_files)
        if args.max_files_per_patient is not None:
            edf_files = edf_files[:args.max_files_per_patient]
        for edf_path in edf_files:
            metadata_path = (
                args.output_root / role / patient_id / f'{edf_path.stem}.json'
            )
            try:
                metadata = preprocess_recording(
                    edf_path,
                    patient_id,
                    role,
                    summary_path,
                    args.output_root,
                    channels,
                    overwrite=args.overwrite,
                )
                rows.append(manifest_row_from_metadata(
                    metadata,
                    metadata_path,
                ))
                print(
                    f"OK {role}/{patient_id}/{edf_path.name}: "
                    f"{metadata['window_count']} windows, "
                    f"{metadata['positive_window_count']} positive"
                )
            except Exception as exc:
                rows.append({
                    'status': 'error',
                    'split': role,
                    'patient_id': patient_id,
                    'edf_name': edf_path.name,
                    'signal_file': '',
                    'metadata_file': str(metadata_path.resolve()),
                    'channel_count': len(channels),
                    'n_samples': '',
                    'all_window_count': '',
                    'window_count': '',
                    'all_positive_window_count': '',
                    'positive_window_count': '',
                    'discarded_warmup_window_count': '',
                    'discarded_warmup_positive_window_count': '',
                    'seizure_event_count': '',
                    'seizure_event_centers_in_discarded_prefix': '',
                    'seizure_events_overlapping_discarded_prefix': '',
                    'error': f'{type(exc).__name__}: {exc}',
                })
                print(f'ERROR {role}/{patient_id}/{edf_path.name}: {exc}')

    if args.edf_names:
        missing_edf_names = sorted(set(args.edf_names) - found_edf_names)
        if missing_edf_names:
            raise SystemExit(f'EDF filenames not found: {missing_edf_names}')

    error_rows = [row for row in rows if row['status'] == 'error']
    error_keys = {
        (row['split'], row['patient_id'], row['edf_name'])
        for row in error_rows
    }
    manifest_rows = []
    for metadata_path in sorted(args.output_root.glob('*/*/*.json')):
        with open(metadata_path, 'r', encoding='utf-8') as handle:
            metadata = json.load(handle)
        key = (
            metadata['split'],
            metadata['patient_id'],
            Path(metadata['source_edf']).name,
        )
        if key not in error_keys:
            manifest_rows.append(manifest_row_from_metadata(
                metadata,
                metadata_path,
            ))
    manifest_rows.extend(error_rows)
    manifest_path = args.output_root / 'preprocess_manifest.csv'
    write_manifest(manifest_rows, manifest_path)
    print(f'Wrote manifest: {manifest_path.resolve()}')
    if error_rows:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
