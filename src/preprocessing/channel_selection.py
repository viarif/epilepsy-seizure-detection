from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from src.utils.annotation_parser import parse_seizure_times
from src.utils.eeg_loader import (
    inspect_edf,
    load_edf_channels,
    resolve_channel_sources,
)

from .config import PreprocessingConfig
from .signal import filter_continuous
from .windowing import label_window_starts, window_start_samples


STANDARD_BIPOLAR_CANDIDATES = (
    'FP1-F7', 'F7-T7', 'T7-P7', 'P7-O1',
    'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1',
    'FZ-CZ', 'CZ-PZ',
    'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2',
    'FP2-F8', 'F8-T8', 'T8-P8', 'P8-O2',
)


def list_edf_files(data_root, patient_id):
    patient_dir = Path(data_root) / patient_id
    if not patient_dir.is_dir():
        raise FileNotFoundError(f'Missing patient directory: {patient_dir}')
    return tuple(sorted(patient_dir.glob('*.edf')))


def audit_candidate_resolvability(
    data_root,
    patient_ids,
    candidates=STANDARD_BIPOLAR_CANDIDATES,
    expected_sfreq=256.0,
):
    """Header-only audit; signal samples and annotations are not accessed."""
    failures = defaultdict(list)
    file_count = 0
    for patient_id in patient_ids:
        for edf_path in list_edf_files(data_root, patient_id):
            file_count += 1
            header = inspect_edf(edf_path)
            if abs(header['sfreq'] - expected_sfreq) > 1e-6:
                failures['__sampling_frequency__'].append(
                    f"{edf_path.name}:{header['sfreq']:g}"
                )
            for candidate in candidates:
                terms, _ = resolve_channel_sources(
                    candidate,
                    header['channels'],
                )
                if terms is None:
                    failures[candidate].append(str(edf_path))

    usable = tuple(
        candidate for candidate in candidates if not failures[candidate]
    )
    return {
        'file_count': file_count,
        'usable_candidates': usable,
        'failures': {key: value for key, value in failures.items() if value},
    }


def _ictal_window_starts(seizure_intervals, sfreq, n_samples, config):
    counts = config.sample_counts()
    all_starts = window_start_samples(
        n_samples,
        counts['window'],
        counts['hop'],
        min_start_sample=0,
    )
    labels = label_window_starts(
        all_starts,
        sfreq,
        counts['window'],
        seizure_intervals,
    )
    return all_starts[labels == 1]


def compute_patient_line_length(
    data_root,
    patient_id,
    candidates,
    config=None,
    max_seizure_files=None,
):
    """Compute one training patient's mean ictal line length per channel.

    Filtering always starts at EDF sample zero, exactly like formal
    preprocessing. Rolling-std normalization and approx-tanh are deliberately
    not applied: this signal exists only for relative channel ranking and is
    not the tensor consumed by the classifier. All ictal windows are
    eligible here, including those in the first ten minutes; the mandatory
    warmup exclusion applies only to model train/validation/test windows.
    """
    config = config or PreprocessingConfig()
    patient_dir = Path(data_root) / patient_id
    summary_path = patient_dir / f'{patient_id}-summary.txt'
    if not summary_path.exists():
        raise FileNotFoundError(f'Missing summary: {summary_path}')

    line_length_sum = np.zeros(len(candidates), dtype=np.float64)
    window_count = 0
    event_count = 0
    seizure_file_count = 0

    for edf_path in list_edf_files(data_root, patient_id):
        seizures = parse_seizure_times(summary_path, edf_path.name)
        if not seizures:
            continue
        if (
            max_seizure_files is not None
            and seizure_file_count >= max_seizure_files
        ):
            break
        seizure_file_count += 1

        event_count += len(seizures)
        signal, sfreq, _ = load_edf_channels(edf_path, candidates)
        config.validate_sfreq(sfreq)
        filtered = filter_continuous(signal, sfreq, config)
        del signal
        global_starts = _ictal_window_starts(
            seizures,
            sfreq,
            filtered.shape[1],
            config,
        )
        window_samples = config.sample_counts()['window']
        for window_start in global_starts:
            window = filtered[
                :,
                window_start:window_start + window_samples,
            ]
            line_length_sum += np.mean(
                np.abs(np.diff(window, axis=1)),
                axis=1,
            )
        window_count += int(global_starts.size)

    if window_count == 0:
        raise ValueError(f'No ictal windows found for training patient {patient_id}.')
    scores = line_length_sum / window_count
    order = np.argsort(-scores, kind='stable')
    ranks = np.empty(len(candidates), dtype=np.int64)
    ranks[order] = np.arange(1, len(candidates) + 1)
    return {
        'patient_id': patient_id,
        'scores': {channel: float(scores[index]) for index, channel in enumerate(candidates)},
        'ranks': {channel: int(ranks[index]) for index, channel in enumerate(candidates)},
        'ictal_window_count': window_count,
        'seizure_event_count': event_count,
        'seizure_file_count': seizure_file_count,
    }


def aggregate_patient_results(per_patient, candidates):
    rank_matrix = np.asarray([
        [result['ranks'][channel] for channel in candidates]
        for result in per_patient
    ], dtype=np.float64)
    median_rank = np.median(rank_matrix, axis=0)
    mean_rank = np.mean(rank_matrix, axis=0)

    order = sorted(
        range(len(candidates)),
        key=lambda index: (
            median_rank[index],
            mean_rank[index],
            candidates[index],
        ),
    )

    aggregate = {
        channel: {
            'median_rank': float(median_rank[index]),
            'mean_rank': float(mean_rank[index]),
        }
        for index, channel in enumerate(candidates)
    }
    return order, aggregate


def select_channels(
    data_root,
    train_patients,
    availability_patients,
    top_k=4,
    config=None,
    max_seizure_files_per_patient=None,
):
    config = config or PreprocessingConfig()
    audit = audit_candidate_resolvability(
        data_root,
        availability_patients,
        expected_sfreq=config.target_sfreq,
    )
    if audit['failures'].get('__sampling_frequency__'):
        raise ValueError(
            'Unexpected EDF sampling frequencies: '
            f"{audit['failures']['__sampling_frequency__']}"
        )
    candidates = audit['usable_candidates']
    if len(candidates) < top_k:
        raise ValueError(
            f'Only {len(candidates)} common channels are safely resolvable; '
            f'cannot select top {top_k}.'
        )

    per_patient = [
        compute_patient_line_length(
            data_root,
            patient_id,
            candidates,
            config=config,
            max_seizure_files=max_seizure_files_per_patient,
        )
        for patient_id in train_patients
    ]
    order, aggregate = aggregate_patient_results(
        per_patient,
        candidates,
    )
    selected_ranked = [candidates[index] for index in order[:top_k]]
    selected_set = set(selected_ranked)
    selected_model_order = [
        channel for channel in candidates if channel in selected_set
    ]
    selected_indices = [
        candidates.index(channel) for channel in selected_model_order
    ]
    return {
        'schema_version': 2,
        'algorithm': 'train-only patient-aggregated ictal line length',
        'aggregation': 'median patient rank, with mean rank and channel name as tie-breakers',
        'line_length_signal': {
            'purpose': 'channel ranking only; not a classifier input tensor',
            'filter_history': 'entire EDF filtered causally from sample zero',
            'window_scope': (
                'all ictal windows, including the first ten minutes; model '
                'train/validation/test warmup exclusion does not apply'
            ),
            'applied': ['0.1 Hz causal high-pass', '60 Hz causal notch'],
            'not_applied': ['causal rolling std', 'approx-tanh'],
            'model_input_relation': (
                'The causal filtered prefix matches formal preprocessing; '
                'formal model input then additionally applies rolling-std '
                'normalization and approx-tanh.'
            ),
        },
        'availability_patient_ids': list(availability_patients),
        'train_patient_ids': list(train_patients),
        'candidate_channels': list(candidates),
        'selected_channels_ranked': selected_ranked,
        'selected_channels': selected_model_order,
        'selected_channel_indices': selected_indices,
        # The CLI verifies the exact configured train-patient scope before it
        # upgrades this flag to True. A library caller cannot accidentally
        # label a subset run as final.
        'complete': False,
        'audit_file_count': audit['file_count'],
        'audit_failures': audit['failures'],
        'aggregate': aggregate,
        'per_patient': per_patient,
    }


def save_selection(result, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + '.tmp')
    with open(temporary, 'w', encoding='utf-8') as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
        handle.write('\n')
    temporary.replace(output_path)
