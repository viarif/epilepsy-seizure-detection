"""Safe access to retained windows in the common continuous cache."""

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from .pipeline import METADATA_SCHEMA_VERSION, WINDOW_POLICY_NAME


@dataclass(frozen=True)
class RecordingWindowIndex:
    metadata_path: Path
    metadata: dict

    @classmethod
    def from_json(cls, metadata_path):
        metadata_path = Path(metadata_path)
        with open(metadata_path, 'r', encoding='utf-8') as handle:
            metadata = json.load(handle)
        cls._validate(metadata, metadata_path)
        return cls(metadata_path=metadata_path, metadata=metadata)

    @staticmethod
    def _validate(metadata, metadata_path='<metadata>'):
        if metadata.get('schema_version', 0) < METADATA_SCHEMA_VERSION:
            raise ValueError(
                f'{metadata_path}: metadata predates mandatory warmup exclusion.'
            )
        if metadata.get('window_policy') != WINDOW_POLICY_NAME:
            raise ValueError(f'{metadata_path}: unexpected window policy.')
        if metadata.get('warmup_windows_discarded') is not True:
            raise ValueError(f'{metadata_path}: warmup windows are not discarded.')
        first_start = metadata.get('first_retained_window_start_sample')
        if (
            first_start is not None
            and first_start < metadata['discard_initial_samples']
        ):
            raise ValueError(f'{metadata_path}: retained window enters warmup.')
        if len(metadata['positive_window_indices']) != metadata['positive_window_count']:
            raise ValueError(f'{metadata_path}: positive index count mismatch.')

    @property
    def window_count(self):
        return int(self.metadata['window_count'])

    @property
    def positive_window_indices(self):
        return tuple(int(value) for value in self.metadata['positive_window_indices'])

    def start_sample(self, window_index):
        if not 0 <= window_index < self.window_count:
            raise IndexError(window_index)
        return (
            int(self.metadata['first_retained_window_start_sample'])
            + window_index * int(self.metadata['hop_samples'])
        )

    def load_window(self, window_index, channel_indices=None):
        signal = np.load(
            self.metadata['signal_file'],
            mmap_mode='r',
            allow_pickle=False,
        )
        start = self.start_sample(window_index)
        stop = start + int(self.metadata['window_samples'])
        if channel_indices is None:
            return np.asarray(signal[:, start:stop])
        return np.asarray(signal[np.asarray(channel_indices), start:stop])
