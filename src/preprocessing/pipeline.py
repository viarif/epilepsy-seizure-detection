import csv
import json
import os
from pathlib import Path

import numpy as np

from src.utils.annotation_parser import parse_seizure_times
from src.utils.eeg_loader import load_edf_channels

from .config import PreprocessingConfig
from .signal import preprocess_continuous
from .windowing import label_window_starts, window_start_samples


METADATA_SCHEMA_VERSION = 3
WINDOW_POLICY_NAME = 'discard_first_10_minutes'


def build_window_index_metadata(n_samples, sfreq, seizure_intervals, config=None):
    """Build the only window index allowed for train/validation/test.

    A retained window must start at or after ``discard_initial_sec``. Using
    the start rather than the center guarantees that every sample in every
    retained window has a complete ten-minute rolling-statistics history.
    """
    config = config or PreprocessingConfig()
    config.validate_sfreq(sfreq)
    counts = config.sample_counts()
    all_starts = window_start_samples(
        n_samples,
        counts['window'],
        counts['hop'],
    )
    retained_starts = window_start_samples(
        n_samples,
        counts['window'],
        counts['hop'],
        min_start_sample=counts['discard_initial'],
    )
    all_labels = label_window_starts(
        all_starts,
        sfreq,
        counts['window'],
        seizure_intervals,
    )
    retained_labels = label_window_starts(
        retained_starts,
        sfreq,
        counts['window'],
        seizure_intervals,
    )
    discarded_count = int(all_starts.size - retained_starts.size)
    discarded_labels = all_labels[:discarded_count]
    positive_indices = np.flatnonzero(retained_labels).astype(int).tolist()
    discarded_positive_indices = np.flatnonzero(
        discarded_labels
    ).astype(int).tolist()
    event_centers_in_prefix = sum(
        1
        for start_sec, end_sec in seizure_intervals
        if (start_sec + end_sec) / 2.0 < config.discard_initial_sec
    )
    events_overlapping_prefix = sum(
        1
        for start_sec, _end_sec in seizure_intervals
        if start_sec < config.discard_initial_sec
    )
    first_start = (
        int(retained_starts[0]) if retained_starts.size else None
    )
    return {
        'window_policy': WINDOW_POLICY_NAME,
        'window_exclusion_rule': (
            'retain only windows with window_start >= discard_initial_sec'
        ),
        'warmup_windows_discarded': True,
        'discard_initial_sec': config.discard_initial_sec,
        'discard_initial_samples': counts['discard_initial'],
        'window_samples': counts['window'],
        'hop_samples': counts['hop'],
        'first_retained_window_start_sample': first_start,
        'first_retained_window_start_sec': (
            first_start / sfreq if first_start is not None else None
        ),
        'all_window_count': int(all_starts.size),
        'window_count': int(retained_starts.size),
        'all_positive_window_count': int(np.count_nonzero(all_labels)),
        'positive_window_count': len(positive_indices),
        # Indices are relative to the retained sequence. Index zero starts at
        # first_retained_window_start_sample, never at EDF sample zero.
        'positive_window_indices': positive_indices,
        'discarded_warmup_window_count': discarded_count,
        'discarded_warmup_positive_window_count': len(
            discarded_positive_indices
        ),
        'discarded_warmup_positive_window_indices': (
            discarded_positive_indices
        ),
        'window_label_rule': 'center in [seizure_start, seizure_end)',
        'seizure_event_count': len(seizure_intervals),
        'seizure_event_centers_in_discarded_prefix': event_centers_in_prefix,
        'seizure_events_overlapping_discarded_prefix': events_overlapping_prefix,
    }


def _atomic_save_npy(path, array):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with open(temporary, 'wb') as handle:
        np.save(handle, array, allow_pickle=False)
    os.replace(temporary, path)


def save_metadata_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with open(temporary, 'w', encoding='utf-8') as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write('\n')
    os.replace(temporary, path)


def preprocess_recording(
    edf_path,
    patient_id,
    split_role,
    summary_path,
    output_root,
    channels,
    config=None,
    overwrite=False,
):
    config = config or PreprocessingConfig()
    edf_path = Path(edf_path)
    output_dir = Path(output_root) / split_role / patient_id
    signal_path = output_dir / f'{edf_path.stem}.npy'
    metadata_path = output_dir / f'{edf_path.stem}.json'

    if signal_path.exists() and metadata_path.exists() and not overwrite:
        with open(metadata_path, 'r', encoding='utf-8') as handle:
            existing = json.load(handle)
        if (
            existing.get('schema_version', 0) >= METADATA_SCHEMA_VERSION
            and existing.get('canonical_channels') == list(channels)
            and existing.get('warmup_windows_discarded') is True
            and existing.get('preprocessing') == config.to_dict()
        ):
            return existing

    raw_data, sfreq, resolved_channels = load_edf_channels(
        edf_path,
        channels,
        allow_fz_cz_alternative=False,
    )
    config.validate_sfreq(sfreq)
    transformed = preprocess_continuous(raw_data, sfreq, config)
    del raw_data

    seizures = parse_seizure_times(summary_path, edf_path.name)
    window_metadata = build_window_index_metadata(
        transformed.shape[1],
        sfreq,
        seizures,
        config,
    )

    _atomic_save_npy(signal_path, transformed)
    metadata = {
        'schema_version': METADATA_SCHEMA_VERSION,
        'patient_id': patient_id,
        'split': split_role,
        'source_edf': str(edf_path.resolve()),
        'signal_file': str(signal_path.resolve()),
        'canonical_channels': list(channels),
        'resolved_channels': resolved_channels,
        'sfreq': sfreq,
        'n_samples': int(transformed.shape[1]),
        'dtype': str(transformed.dtype),
        'seizure_intervals_sec': [list(pair) for pair in seizures],
        'rolling_std_warmup_samples': min(
            config.sample_counts()['rolling_std'],
            int(transformed.shape[1]),
        ),
        'preprocessing': config.to_dict(),
        **window_metadata,
    }
    save_metadata_json(metadata_path, metadata)
    return metadata


def manifest_row_from_metadata(metadata, metadata_path):
    return {
        'status': 'ok',
        'split': metadata['split'],
        'patient_id': metadata['patient_id'],
        'edf_name': Path(metadata['source_edf']).name,
        'signal_file': metadata['signal_file'],
        'metadata_file': str(Path(metadata_path).resolve()),
        'channel_count': len(metadata['canonical_channels']),
        'n_samples': metadata['n_samples'],
        'all_window_count': metadata['all_window_count'],
        'window_count': metadata['window_count'],
        'all_positive_window_count': metadata['all_positive_window_count'],
        'positive_window_count': metadata['positive_window_count'],
        'discarded_warmup_window_count': (
            metadata['discarded_warmup_window_count']
        ),
        'discarded_warmup_positive_window_count': (
            metadata['discarded_warmup_positive_window_count']
        ),
        'seizure_event_count': metadata['seizure_event_count'],
        'seizure_event_centers_in_discarded_prefix': (
            metadata['seizure_event_centers_in_discarded_prefix']
        ),
        'seizure_events_overlapping_discarded_prefix': (
            metadata['seizure_events_overlapping_discarded_prefix']
        ),
        'error': '',
    }


def write_manifest(rows, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        'status', 'split', 'patient_id', 'edf_name', 'signal_file',
        'metadata_file', 'channel_count', 'n_samples', 'all_window_count',
        'window_count', 'all_positive_window_count',
        'positive_window_count', 'discarded_warmup_window_count',
        'discarded_warmup_positive_window_count', 'seizure_event_count',
        'seizure_event_centers_in_discarded_prefix',
        'seizure_events_overlapping_discarded_prefix', 'error',
    )
    temporary = path.with_suffix(path.suffix + '.tmp')
    with open(temporary, 'w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)
