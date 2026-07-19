import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import EEGWindowDataset
from src.data.training_data import (
    HardNegativeReplaySampler,
    build_hierarchical_positive_weights,
    create_training_dataloaders,
)
from src.preprocessing.pipeline import METADATA_SCHEMA_VERSION, WINDOW_POLICY_NAME
from src.training.trainer import (
    hierarchical_weighted_bce,
    macro_patient_metrics,
)


CHANNELS = ["F7-T7", "T7-P7", "F8-T8", "T8-P8"]


def write_recording(root, split, patient, name, window_count, positives):
    output_dir = Path(root) / split / patient
    output_dir.mkdir(parents=True, exist_ok=True)
    n_samples = (window_count - 1) * 128 + 256
    signal_path = output_dir / f"{name}.npy"
    np.save(
        signal_path,
        np.zeros((4, n_samples), dtype=np.float32),
        allow_pickle=False,
    )
    metadata = {
        "schema_version": METADATA_SCHEMA_VERSION,
        "patient_id": patient,
        "split": split,
        "signal_file": str(signal_path.resolve()),
        "canonical_channels": CHANNELS,
        "n_samples": n_samples,
        "window_policy": WINDOW_POLICY_NAME,
        "warmup_windows_discarded": True,
        "discard_initial_samples": 0,
        "first_retained_window_start_sample": 0,
        "window_count": window_count,
        "hop_samples": 128,
        "window_samples": 256,
        "positive_window_count": len(positives),
        "positive_window_indices": positives,
    }
    with open(output_dir / f"{name}.json", "w", encoding="utf-8") as handle:
        json.dump(metadata, handle)


class HierarchicalWeightTests(unittest.TestCase):
    def test_equalizes_patients_and_bouts_without_changing_mean_weight(self):
        with tempfile.TemporaryDirectory() as temporary:
            write_recording(temporary, "train", "p1", "r1", 6, [0, 1, 4])
            write_recording(temporary, "train", "p2", "r1", 6, [0, 1, 2, 3])
            dataset = EEGWindowDataset(temporary, "train")

            weights = build_hierarchical_positive_weights(dataset)
            patient_totals = {"p1": 0.0, "p2": 0.0}
            for dataset_index, weight in weights.items():
                patient = dataset.sample_metadata(dataset_index)["patient_id"]
                patient_totals[patient] += weight

            self.assertAlmostEqual(np.mean(list(weights.values())), 1.0)
            self.assertAlmostEqual(patient_totals["p1"], 3.5)
            self.assertAlmostEqual(patient_totals["p2"], 3.5)
            self.assertAlmostEqual(weights[0] + weights[1], weights[4])
            dataset.close()


class HardNegativeReplayTests(unittest.TestCase):
    def test_bank_keeps_top_per_recording_and_replays_half_the_negatives(self):
        with tempfile.TemporaryDirectory() as temporary:
            write_recording(temporary, "train", "p1", "r1", 6, [0])
            write_recording(temporary, "train", "p2", "r1", 6, [0])
            dataset = EEGWindowDataset(temporary, "train")
            sampler = HardNegativeReplaySampler(
                dataset,
                positive_fraction=0.20,
                replay_fraction=0.50,
                hard_per_recording=2,
                seed=7,
            )
            sampler.update_hard_negatives(
                np.asarray([1, 2, 3, 7, 8, 9]),
                np.asarray([0.1, 0.9, 0.5, -0.2, 0.8, 0.4]),
            )

            bank_indices, _ = sampler.hard_bank_arrays()
            components = sampler.sampled_components_for_epoch(1)

            self.assertEqual(sampler.hard_bank_size, 4)
            self.assertEqual(set(bank_indices.tolist()), {2, 3, 8, 9})
            self.assertEqual(components["positive"].size, 2)
            self.assertEqual(components["random_negative"].size, 4)
            self.assertEqual(components["hard_negative"].size, 4)
            self.assertTrue(
                set(components["hard_negative"].tolist()).issubset(
                    set(bank_indices.tolist())
                )
            )
            dataset.close()

    def test_factory_does_not_require_or_construct_test_split(self):
        with tempfile.TemporaryDirectory() as temporary:
            write_recording(temporary, "train", "p1", "train", 8, [0, 1])
            write_recording(temporary, "val", "p2", "val", 8, [0, 1])

            loaders = create_training_dataloaders(
                temporary,
                batch_size=4,
                positive_fraction=0.25,
                num_workers=0,
                pin_memory=False,
            )

            self.assertEqual(loaders.train.dataset.base_dataset.split, "train")
            self.assertEqual(loaders.val.dataset.split, "val")
            loaders.close()


class TrainingLossTests(unittest.TestCase):
    def test_hierarchical_bce_matches_manual_weighted_mean(self):
        logits = torch.tensor([1.0, -0.5, 0.25])
        labels = torch.tensor([1.0, 1.0, 0.0])
        positive_weights = torch.tensor([2.0, 0.5, 9.0])

        actual = hierarchical_weighted_bce(logits, labels, positive_weights)
        per_sample = F.binary_cross_entropy_with_logits(
            logits, labels, reduction="none"
        )
        expected = (
            per_sample * torch.tensor([2.0, 0.5, 1.0])
        ).sum() / 3.5

        self.assertAlmostEqual(actual.item(), expected.item(), places=7)

    def test_macro_patient_metrics_do_not_weight_by_window_count(self):
        report = {
            "per_patient": {
                "p1": {"sensitivity": 1.0, "specificity": 0.9},
                "p2": {"sensitivity": 0.0, "specificity": 1.0},
            }
        }

        macro = macro_patient_metrics(report)

        self.assertAlmostEqual(macro["macro_sensitivity"], 0.5)
        self.assertAlmostEqual(macro["macro_specificity"], 0.95)
        self.assertAlmostEqual(macro["min_patient_sensitivity"], 0.0)


if __name__ == "__main__":
    unittest.main()
