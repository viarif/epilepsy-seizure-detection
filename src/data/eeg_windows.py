"""Streaming access to retained four-channel EEG windows."""

from bisect import bisect_right
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from src.preprocessing.recording_index import RecordingWindowIndex


_SPLIT_ALIASES = {
    'train': 'train',
    'val': 'val',
    'validation': 'val',
    'test': 'test',
}


@dataclass(frozen=True)
class _Recording:
    index: RecordingWindowIndex
    patient_id: str
    recording_id: str
    offset: int
    positive_indices: np.ndarray
    positive_set: frozenset

    @property
    def window_count(self):
        return self.index.window_count

    @property
    def negative_count(self):
        return self.window_count - self.positive_indices.size


class EEGWindowDataset(Dataset):
    """Expose retained one-second windows without materializing a giant NPZ.

    By default each item is ``(signal, label)``. ``signal`` is float32 with
    shape ``[1, 4, 256]`` for the classifier, and ``label`` is a scalar float32
    tensor suitable for ``BCEWithLogitsLoss``. Set ``return_metadata=True``
    when patient/recording identifiers are needed during evaluation.
    """

    def __init__(
        self,
        cache_root='data/processed/selected4',
        split='train',
        patients=None,
        return_metadata=False,
        expected_channels=4,
        expected_window_samples=256,
        max_open_recordings=8,
    ):
        try:
            self.split = _SPLIT_ALIASES[split]
        except KeyError as error:
            raise ValueError(
                f'Unknown split {split!r}; expected train, val/validation, or test.'
            ) from error

        self.cache_root = Path(cache_root)
        self.return_metadata = bool(return_metadata)
        self.expected_channels = int(expected_channels)
        self.expected_window_samples = int(expected_window_samples)
        self.max_open_recordings = int(max_open_recordings)
        if self.max_open_recordings <= 0:
            raise ValueError('max_open_recordings must be positive.')
        requested_patients = None if patients is None else frozenset(patients)

        split_root = self.cache_root / self.split
        metadata_paths = sorted(split_root.glob('*/*.json'))
        if not metadata_paths:
            raise FileNotFoundError(
                f'No recording metadata found under {split_root.resolve()}.'
            )

        records = []
        offset = 0
        canonical_channels = None
        discovered_patients = set()
        skipped_empty_recordings = 0
        for metadata_path in metadata_paths:
            index = RecordingWindowIndex.from_json(metadata_path)
            metadata = index.metadata
            patient_id = metadata['patient_id']
            if requested_patients is not None and patient_id not in requested_patients:
                continue
            discovered_patients.add(patient_id)

            if metadata.get('split') != self.split:
                raise ValueError(
                    f'{metadata_path}: metadata split {metadata.get("split")!r} '
                    f'does not match directory {self.split!r}.'
                )
            channels = tuple(metadata['canonical_channels'])
            if len(channels) != self.expected_channels:
                raise ValueError(
                    f'{metadata_path}: expected {self.expected_channels} channels, '
                    f'got {len(channels)}.'
                )
            if canonical_channels is None:
                canonical_channels = channels
            elif channels != canonical_channels:
                raise ValueError(
                    f'{metadata_path}: channel order differs from other recordings.'
                )
            if int(metadata['window_samples']) != self.expected_window_samples:
                raise ValueError(
                    f'{metadata_path}: expected {self.expected_window_samples} samples '
                    f'per window, got {metadata["window_samples"]}.'
                )
            if index.window_count == 0:
                skipped_empty_recordings += 1
                continue

            positives = np.asarray(index.positive_window_indices, dtype=np.int64)
            if (
                np.any(positives < 0)
                or np.any(positives >= index.window_count)
                or np.unique(positives).size != positives.size
                or np.any(np.diff(positives) <= 0)
            ):
                raise ValueError(
                    f'{metadata_path}: positive window indices must be unique, '
                    f'strictly increasing, and within the retained sequence.'
                )
            records.append(
                _Recording(
                    index=index,
                    patient_id=patient_id,
                    recording_id=metadata_path.stem,
                    offset=offset,
                    positive_indices=positives,
                    positive_set=frozenset(positives.tolist()),
                )
            )
            offset += index.window_count

        if requested_patients is not None:
            missing = sorted(requested_patients - discovered_patients)
            if missing:
                raise ValueError(
                    f'Patient(s) absent from {self.split} cache: {missing}.'
                )
        if not records:
            raise ValueError(
                f'No retained windows are available for split {self.split!r}.'
            )

        self.canonical_channels = canonical_channels
        self.skipped_empty_recordings = skipped_empty_recordings
        self._records = tuple(records)
        self._offsets = tuple(record.offset for record in records) + (offset,)
        self._length = offset
        self._signal_cache = OrderedDict()

    def __getstate__(self):
        state = self.__dict__.copy()
        # Spawned DataLoader workers must open their own read-only memory maps.
        state['_signal_cache'] = OrderedDict()
        return state

    def close(self):
        """Close memory maps held by the current process/worker."""
        while self._signal_cache:
            _recording_index, signal = self._signal_cache.popitem(last=False)
            memory_map = getattr(signal, '_mmap', None)
            if memory_map is not None:
                memory_map.close()

    def __del__(self):
        signal_cache = getattr(self, '_signal_cache', None)
        if signal_cache is not None:
            self.close()

    def __len__(self):
        return self._length

    @property
    def patient_ids(self):
        return tuple(sorted({record.patient_id for record in self._records}))

    @property
    def positive_count(self):
        return sum(record.positive_indices.size for record in self._records)

    @property
    def negative_count(self):
        return len(self) - self.positive_count

    @property
    def positive_indices(self):
        if self.positive_count == 0:
            return np.empty(0, dtype=np.int64)
        return np.concatenate(
            [record.offset + record.positive_indices for record in self._records]
        )

    def locate_index(self, dataset_index):
        """Map a dataset index to ``(recording_index, local_window_index)``."""
        if dataset_index < 0:
            dataset_index += len(self)
        if not 0 <= dataset_index < len(self):
            raise IndexError(dataset_index)
        recording_index = bisect_right(self._offsets, dataset_index) - 1
        local_index = dataset_index - self._records[recording_index].offset
        return recording_index, local_index

    def label_at(self, dataset_index):
        recording_index, local_index = self.locate_index(dataset_index)
        return int(local_index in self._records[recording_index].positive_set)

    def sample_metadata(self, dataset_index):
        recording_index, local_index = self.locate_index(dataset_index)
        record = self._records[recording_index]
        return {
            'patient_id': record.patient_id,
            'recording_id': record.recording_id,
            'window_index': local_index,
            'start_sample': record.index.start_sample(local_index),
        }

    def _signal_path(self, record):
        configured = Path(record.index.metadata['signal_file'])
        if configured.is_file():
            return configured
        portable = record.index.metadata_path.with_suffix('.npy')
        if portable.is_file():
            return portable
        raise FileNotFoundError(
            f'Signal file is missing: {configured} (also tried {portable}).'
        )

    def _load_signal(self, recording_index):
        signal = self._signal_cache.get(recording_index)
        if signal is None:
            record = self._records[recording_index]
            signal_path = self._signal_path(record)
            signal = np.load(signal_path, mmap_mode='r', allow_pickle=False)
            expected_shape = (
                self.expected_channels,
                int(record.index.metadata['n_samples']),
            )
            if signal.shape != expected_shape:
                raise ValueError(
                    f'{signal_path}: expected shape {expected_shape}, got {signal.shape}.'
                )
            if signal.dtype != np.float32:
                raise ValueError(
                    f'{signal_path}: expected float32, got {signal.dtype}.'
                )
            self._signal_cache[recording_index] = signal
            if len(self._signal_cache) > self.max_open_recordings:
                _evicted_index, evicted = self._signal_cache.popitem(last=False)
                memory_map = getattr(evicted, '_mmap', None)
                if memory_map is not None:
                    memory_map.close()
        else:
            self._signal_cache.move_to_end(recording_index)
        return signal

    def __getitem__(self, dataset_index):
        recording_index, local_index = self.locate_index(dataset_index)
        record = self._records[recording_index]
        start = record.index.start_sample(local_index)
        stop = start + self.expected_window_samples
        window = np.array(
            self._load_signal(recording_index)[:, start:stop],
            dtype=np.float32,
            copy=True,
        )
        if window.shape != (self.expected_channels, self.expected_window_samples):
            raise ValueError(
                f'{record.index.metadata_path}: incomplete window {local_index}; '
                f'got shape {window.shape}.'
            )
        signal = torch.from_numpy(window).unsqueeze(0)
        label = torch.tensor(
            float(local_index in record.positive_set),
            dtype=torch.float32,
        )
        if not self.return_metadata:
            return signal, label
        return {
            'signal': signal,
            'label': label,
            **self.sample_metadata(dataset_index),
        }


def _balanced_counts(total, group_count, rng):
    counts = np.full(group_count, total // group_count, dtype=np.int64)
    remainder = total % group_count
    if remainder:
        counts[rng.permutation(group_count)[:remainder]] += 1
    return counts


def _negative_ranks_to_window_indices(ranks, positive_indices):
    """Map ranks in the negative-only sequence to recording window indices."""
    if positive_indices.size == 0:
        return ranks
    shifted_positives = positive_indices - np.arange(
        positive_indices.size,
        dtype=np.int64,
    )
    return ranks + np.searchsorted(
        shifted_positives,
        ranks,
        side='right',
    )

