from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessing.config import (
    DEFAULT_SPLIT_CONFIG,
    DatasetSplit,
    PreprocessingConfig,
)
from src.preprocessing.channel_selection import (
    _ictal_window_starts,
    aggregate_patient_results,
)
from src.preprocessing.pipeline import (
    METADATA_SCHEMA_VERSION,
    WINDOW_POLICY_NAME,
    build_window_index_metadata,
)
from src.preprocessing.recording_index import RecordingWindowIndex
from src.preprocessing.signal import (
    approx_tanh,
    causal_rolling_std,
    filter_continuous,
)
from src.preprocessing.windowing import label_window_starts, window_start_samples
from src.utils.annotation_parser import label_window_center


class SplitTests(unittest.TestCase):
    def test_requested_patient_split_is_locked(self):
        split = DatasetSplit.from_json(DEFAULT_SPLIT_CONFIG)
        self.assertEqual(split.test, ('chb10', 'chb11', 'chb12'))
        self.assertEqual(split.val, ('chb13', 'chb14', 'chb15'))
        self.assertEqual(len(split.train), 18)
        self.assertEqual(split.role_for('chb24'), 'train')


class SignalTests(unittest.TestCase):
    def test_approx_tanh_is_piecewise_linear_and_clipped(self):
        values = np.asarray([-2.0, -0.6, 0.0, 0.6, 2.0])
        actual = approx_tanh(values)
        np.testing.assert_allclose(actual, [-1.0, -0.5, 0.0, 0.5, 1.0])

    def test_causal_rolling_std_does_not_look_into_future(self):
        prefix = np.asarray([[1.0, 2.0, 4.0, 8.0]])
        extended = np.asarray([[1.0, 2.0, 4.0, 8.0, 1e9]])
        prefix_std = causal_rolling_std(prefix, window_samples=3)
        extended_std = causal_rolling_std(extended, window_samples=3)
        np.testing.assert_allclose(prefix_std, extended_std[:, :4])

    def test_causal_rolling_std_matches_known_values(self):
        actual = causal_rolling_std(
            np.asarray([[1.0, 2.0, 3.0, 4.0]]),
            window_samples=3,
        )
        expected = np.asarray([[0.0, 0.5, np.sqrt(2 / 3), np.sqrt(2 / 3)]])
        np.testing.assert_allclose(actual, expected)

    def test_notch_strongly_reduces_60_hz_component(self):
        sfreq = 256.0
        time = np.arange(int(20 * sfreq)) / sfreq
        signal = np.sin(2 * np.pi * 60 * time) + 0.2 * np.sin(2 * np.pi * 10 * time)
        filtered = filter_continuous(signal[None, :], sfreq)[0]
        stable = slice(int(5 * sfreq), None)

        def amplitude(values, frequency):
            local_time = time[stable]
            basis = np.exp(-2j * np.pi * frequency * local_time)
            return 2 * abs(np.dot(values[stable], basis)) / values[stable].size

        self.assertLess(amplitude(filtered, 60), 0.1 * amplitude(signal, 60))
        self.assertGreater(amplitude(filtered, 10), 0.8 * amplitude(signal, 10))


class WindowTests(unittest.TestCase):
    def test_window_count_and_hop(self):
        starts = window_start_samples(512, 256, 128)
        np.testing.assert_array_equal(starts, [0, 128, 256])

    def test_model_windows_start_only_after_full_ten_minutes(self):
        config = PreprocessingConfig()
        counts = config.sample_counts()
        starts = window_start_samples(
            602 * 256,
            counts['window'],
            counts['hop'],
            min_start_sample=counts['discard_initial'],
        )
        self.assertEqual(starts[0], 600 * 256)

    def test_window_metadata_discards_positive_and_negative_warmup_windows(self):
        metadata = build_window_index_metadata(
            n_samples=602 * 256,
            sfreq=256,
            seizure_intervals=[(600, 601)],
        )
        self.assertTrue(metadata['warmup_windows_discarded'])
        self.assertEqual(metadata['discarded_warmup_window_count'], 1200)
        self.assertEqual(metadata['discarded_warmup_positive_window_count'], 1)
        self.assertEqual(metadata['positive_window_count'], 1)
        self.assertEqual(metadata['first_retained_window_start_sec'], 600.0)

    def test_center_rule_uses_half_open_seizure_interval(self):
        self.assertEqual(label_window_center(9.5, 1.0, [(10, 11)]), 1)
        self.assertEqual(label_window_center(10.5, 1.0, [(10, 11)]), 0)

    def test_vectorized_center_labels_match_boundary_rule(self):
        starts = np.asarray([9 * 256 + 128, 10 * 256 + 128])
        labels = label_window_starts(starts, 256, 256, [(10, 11)])
        np.testing.assert_array_equal(labels, [1, 0])


class ChannelSelectionTests(unittest.TestCase):
    def test_channel_selection_still_uses_ictal_windows_before_ten_minutes(self):
        starts = _ictal_window_starts(
            seizure_intervals=[(10, 11)],
            sfreq=256,
            n_samples=602 * 256,
            config=PreprocessingConfig(),
        )
        self.assertTrue(np.any(starts < 600 * 256))

    def test_median_rank_aggregation_is_deterministic(self):
        candidates = ('A', 'B', 'C')
        per_patient = [
            {
                'scores': {'A': 3.0, 'B': 2.0, 'C': 1.0},
                'ranks': {'A': 1, 'B': 2, 'C': 3},
            },
            {
                'scores': {'A': 2.0, 'B': 3.0, 'C': 1.0},
                'ranks': {'A': 2, 'B': 1, 'C': 3},
            },
            {
                'scores': {'A': 4.0, 'B': 2.0, 'C': 1.0},
                'ranks': {'A': 1, 'B': 2, 'C': 3},
            },
        ]
        order, aggregate = aggregate_patient_results(per_patient, candidates)
        self.assertEqual([candidates[index] for index in order], ['A', 'B', 'C'])
        self.assertEqual(aggregate['A']['median_rank'], 1.0)


class RecordingIndexTests(unittest.TestCase):
    def test_index_zero_maps_to_600_seconds(self):
        metadata = {
            'schema_version': METADATA_SCHEMA_VERSION,
            'window_policy': WINDOW_POLICY_NAME,
            'warmup_windows_discarded': True,
            'discard_initial_samples': 600 * 256,
            'first_retained_window_start_sample': 600 * 256,
            'window_count': 3,
            'hop_samples': 128,
            'window_samples': 256,
            'positive_window_count': 1,
            'positive_window_indices': [1],
        }
        index = RecordingWindowIndex(Path('metadata.json'), metadata)
        RecordingWindowIndex._validate(metadata)
        self.assertEqual(index.start_sample(0), 600 * 256)
        self.assertEqual(index.start_sample(2), 600 * 256 + 256)

    def test_legacy_metadata_is_rejected(self):
        with self.assertRaises(ValueError):
            RecordingWindowIndex._validate({'schema_version': 2})


if __name__ == '__main__':
    unittest.main()
