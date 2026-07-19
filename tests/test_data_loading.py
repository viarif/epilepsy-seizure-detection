import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import EEGWindowDataset, create_training_dataloaders
from src.preprocessing.pipeline import METADATA_SCHEMA_VERSION, WINDOW_POLICY_NAME


CHANNELS = ["F7-T7", "T7-P7", "F8-T8", "T8-P8"]


def write_recording(root, split, patient, name, window_count, positives):
    output_dir = Path(root) / split / patient
    output_dir.mkdir(parents=True, exist_ok=True)
    first_start = 0 if window_count else None
    n_samples = 0 if not window_count else (window_count - 1) * 128 + 256
    signal = np.arange(4 * n_samples, dtype=np.float32).reshape(4, n_samples)
    signal_path = output_dir / f"{name}.npy"
    np.save(signal_path, signal, allow_pickle=False)
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
        "first_retained_window_start_sample": first_start,
        "window_count": window_count,
        "hop_samples": 128,
        "window_samples": 256,
        "positive_window_count": len(positives),
        "positive_window_indices": positives,
    }
    with open(output_dir / f"{name}.json", "w", encoding="utf-8") as handle:
        json.dump(metadata, handle)


class DatasetTests(unittest.TestCase):
    def test_dataset_returns_model_shape_and_float_label(self):
        with tempfile.TemporaryDirectory() as temporary:
            write_recording(temporary, "train", "chb01", "rec", 3, [1])
            dataset = EEGWindowDataset(temporary, "train")
            signal, label = dataset[1]
            self.assertEqual(tuple(signal.shape), (1, 4, 256))
            self.assertEqual(signal.dtype, torch.float32)
            self.assertEqual(label.dtype, torch.float32)
            self.assertEqual(label.item(), 1.0)
            self.assertEqual(dataset.sample_metadata(0)["start_sample"], 0)
            dataset.close()

    def test_empty_recording_is_skipped(self):
        with tempfile.TemporaryDirectory() as temporary:
            write_recording(temporary, "train", "chb01", "empty", 0, [])
            write_recording(temporary, "train", "chb01", "usable", 2, [0])
            dataset = EEGWindowDataset(temporary, "train")
            self.assertEqual(len(dataset), 2)
            self.assertEqual(dataset.skipped_empty_recordings, 1)
            dataset.close()


class DataLoaderTests(unittest.TestCase):
    def _write_train_and_validation(self, root):
        write_recording(root, "train", "p1", "train_a", 20, [0, 1])
        write_recording(root, "train", "p2", "train_b", 20, [0, 1])
        write_recording(root, "val", "p3", "val", 5, [0])

    def test_factory_streams_train_and_complete_validation_without_test(self):
        with tempfile.TemporaryDirectory() as temporary:
            self._write_train_and_validation(temporary)
            loaders = create_training_dataloaders(
                temporary,
                batch_size=4,
                positive_fraction=0.20,
                num_workers=0,
                pin_memory=False,
            )
            train_labels = torch.cat(
                [batch["label"] for batch in loaders.train]
            )
            val_labels = torch.cat([labels for _, labels in loaders.val])
            self.assertEqual(train_labels.sum().item(), 4)
            self.assertEqual(train_labels.numel(), 20)
            self.assertEqual(val_labels.numel(), 5)
            self.assertFalse((Path(temporary) / "test").exists())
            loaders.close()

    def test_factory_supports_spawned_workers(self):
        with tempfile.TemporaryDirectory() as temporary:
            self._write_train_and_validation(temporary)
            loaders = create_training_dataloaders(
                temporary,
                batch_size=4,
                positive_fraction=0.20,
                num_workers=2,
                pin_memory=False,
            )
            batch = next(iter(loaders.train))
            self.assertEqual(tuple(batch["signal"].shape), (4, 1, 4, 256))
            self.assertEqual(tuple(batch["label"].shape), (4,))
            loaders.close()


if __name__ == "__main__":
    unittest.main()
