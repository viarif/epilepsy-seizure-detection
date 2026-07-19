"""Audit the selected-four model cache and independently rebuild its index."""

import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.preprocessing.channel_selection import list_edf_files
from src.preprocessing.config import (
    ALL_PATIENTS,
    DEFAULT_SPLIT_CONFIG,
    DatasetSplit,
    PreprocessingConfig,
)
from src.preprocessing.pipeline import (
    METADATA_SCHEMA_VERSION,
    build_window_index_metadata,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=Path, default=Path('data/raw'))
    parser.add_argument(
        '--output-root',
        type=Path,
        default=Path('data/processed/selected4'),
    )
    parser.add_argument(
        '--channels-json',
        type=Path,
        default=Path('data/processed/channel_selection.json'),
    )
    parser.add_argument('--split-config', type=Path, default=DEFAULT_SPLIT_CONFIG)
    parser.add_argument('--sample-points', type=int, default=257)
    return parser.parse_args()


def main():
    args = parse_args()
    split = DatasetSplit.from_json(args.split_config)
    config = PreprocessingConfig()
    with open(args.channels_json, 'r', encoding='utf-8') as handle:
        selection = json.load(handle)
    if not selection.get('complete'):
        raise SystemExit('Channel selection is not complete.')
    expected_channels = tuple(selection['selected_channels'])
    selected_indices = np.asarray(
        selection['selected_channel_indices'],
        dtype=np.int64,
    )
    expected_edfs = {
        path.resolve()
        for patient_id in ALL_PATIENTS
        for path in list_edf_files(args.data_root, patient_id)
    }
    manifest_path = args.output_root / 'preprocess_manifest.csv'
    with open(manifest_path, 'r', encoding='utf-8', newline='') as handle:
        manifest_rows = list(csv.DictReader(handle))

    errors = []
    seen_sources = set()
    split_stats = defaultdict(Counter)
    patient_stats = defaultdict(Counter)
    metadata_files = sorted(args.output_root.glob('*/*/*.json'))

    for metadata_path in metadata_files:
        with open(metadata_path, 'r', encoding='utf-8') as handle:
            metadata = json.load(handle)
        patient_id = metadata['patient_id']
        role = metadata['split']
        source_edf = Path(metadata['source_edf']).resolve()
        signal_path = Path(metadata['signal_file'])
        seen_sources.add(source_edf)

        if split.role_for(patient_id) != role:
            errors.append(f'{metadata_path}: split mismatch')
        if metadata.get('schema_version', 0) < METADATA_SCHEMA_VERSION:
            errors.append(f'{metadata_path}: old metadata schema')
        if metadata.get('warmup_windows_discarded') is not True:
            errors.append(f'{metadata_path}: warmup exclusion is not mandatory')
        if tuple(metadata['canonical_channels']) != expected_channels:
            errors.append(f'{metadata_path}: selected channel order mismatch')
        if len(metadata['resolved_channels']) != len(expected_channels):
            errors.append(f'{metadata_path}: resolved channel count mismatch')
        if not signal_path.exists():
            errors.append(f'{metadata_path}: missing {signal_path}')
            continue

        rebuilt = build_window_index_metadata(
            int(metadata['n_samples']),
            float(metadata['sfreq']),
            metadata['seizure_intervals_sec'],
            config,
        )
        for key in (
            'window_policy',
            'discard_initial_samples',
            'first_retained_window_start_sample',
            'all_window_count',
            'window_count',
            'all_positive_window_count',
            'positive_window_count',
            'positive_window_indices',
            'discarded_warmup_window_count',
            'discarded_warmup_positive_window_count',
            'discarded_warmup_positive_window_indices',
        ):
            if metadata.get(key) != rebuilt[key]:
                errors.append(f'{metadata_path}: independently rebuilt {key} mismatch')

        array = np.load(signal_path, mmap_mode='r', allow_pickle=False)
        expected_shape = (len(expected_channels), int(metadata['n_samples']))
        if array.shape != expected_shape:
            errors.append(
                f'{signal_path}: shape={array.shape}, expected={expected_shape}'
            )
        if array.dtype != np.float32:
            errors.append(f'{signal_path}: dtype={array.dtype}, expected=float32')
        if array.shape[1]:
            sample_indices = np.linspace(
                0,
                array.shape[1] - 1,
                num=min(args.sample_points, array.shape[1]),
                dtype=np.int64,
            )
            sample = np.asarray(array[:, sample_indices])
            if not np.isfinite(sample).all():
                errors.append(f'{signal_path}: non-finite sampled value')
            if sample.min() < -1.0 or sample.max() > 1.0:
                errors.append(f'{signal_path}: sampled value outside [-1, 1]')

        values = {
            'files': 1,
            'all_windows': int(metadata['all_window_count']),
            'retained_windows': int(metadata['window_count']),
            'discarded_windows': int(
                metadata['discarded_warmup_window_count']
            ),
            'all_positive_windows': int(
                metadata['all_positive_window_count']
            ),
            'retained_positive_windows': int(
                metadata['positive_window_count']
            ),
            'discarded_positive_windows': int(
                metadata['discarded_warmup_positive_window_count']
            ),
            'seizure_events': int(metadata['seizure_event_count']),
            'samples': int(metadata['n_samples']),
        }
        split_stats[role].update(values)
        patient_stats[patient_id].update(values)

    missing_sources = sorted(str(path) for path in expected_edfs - seen_sources)
    unexpected_sources = sorted(str(path) for path in seen_sources - expected_edfs)
    if missing_sources:
        errors.append(f'missing source EDF outputs: {missing_sources}')
    if unexpected_sources:
        errors.append(f'unexpected source EDF outputs: {unexpected_sources}')
    if len(manifest_rows) != len(expected_edfs):
        errors.append(
            f'manifest rows={len(manifest_rows)}, expected={len(expected_edfs)}'
        )
    bad_manifest = [row for row in manifest_rows if row['status'] != 'ok']
    if bad_manifest:
        errors.append(f'manifest contains {len(bad_manifest)} non-ok rows')

    report = {
        'schema_version': 2,
        'split_name': split.name,
        'selection_file': str(args.channels_json.resolve()),
        'selected_channels': list(expected_channels),
        'selected_channel_indices': selected_indices.astype(int).tolist(),
        'expected_edf_count': len(expected_edfs),
        'metadata_count': len(metadata_files),
        'manifest_row_count': len(manifest_rows),
        'window_policy': 'discard every window with start < 600 seconds',
        'split_stats': {key: dict(value) for key, value in split_stats.items()},
        'patient_stats': {key: dict(value) for key, value in patient_stats.items()},
        'errors': errors,
        'passed': not errors,
    }
    report_path = args.output_root / 'audit_report.json'
    with open(report_path, 'w', encoding='utf-8') as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write('\n')

    retained = sum(v['retained_windows'] for v in split_stats.values())
    discarded = sum(v['discarded_windows'] for v in split_stats.values())
    retained_positive = sum(
        v['retained_positive_windows'] for v in split_stats.values()
    )
    discarded_positive = sum(
        v['discarded_positive_windows'] for v in split_stats.values()
    )
    print(f"Audit passed: {report['passed']}")
    print(f"EDF outputs: {len(metadata_files)}/{len(expected_edfs)}")
    print(f'Retained/discarded windows: {retained}/{discarded}')
    print(
        'Retained/discarded positive windows: '
        f'{retained_positive}/{discarded_positive}'
    )
    print(f'Wrote: {report_path.resolve()}')
    if errors:
        for error in errors:
            print(f'ERROR: {error}')
        raise SystemExit(1)


if __name__ == '__main__':
    main()
