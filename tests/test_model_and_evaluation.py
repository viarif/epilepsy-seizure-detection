from pathlib import Path
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation import (
    evaluate_predictions,
    select_threshold_at_specificity,
)
from src.models import SeizureNetLite


class SeizureNetLiteTests(unittest.TestCase):
    def test_locked_architecture_shape_and_parameter_count(self):
        model = SeizureNetLite()
        shapes = {}
        handles = []
        for name in ("conv1", "pool1", "conv2", "pool2", "conv3", "pool3", "conv4", "output"):
            layer = getattr(model, name)
            handles.append(
                layer.register_forward_hook(
                    lambda _module, _inputs, output, name=name: shapes.__setitem__(
                        name, tuple(output.shape)
                    )
                )
            )

        logits = model(torch.randn(3, 1, 4, 256))
        for handle in handles:
            handle.remove()

        self.assertEqual(model.parameter_count, 2991)
        self.assertEqual(tuple(logits.shape), (3,))
        self.assertEqual(shapes["conv1"], (3, 16, 1, 240))
        self.assertEqual(shapes["pool1"], (3, 16, 1, 60))
        self.assertEqual(shapes["conv2"], (3, 10, 1, 56))
        self.assertEqual(shapes["pool2"], (3, 10, 1, 14))
        self.assertEqual(shapes["conv3"], (3, 10, 1, 10))
        self.assertEqual(shapes["pool3"], (3, 10, 1, 5))
        self.assertEqual(shapes["conv4"], (3, 10, 1, 1))
        self.assertEqual(shapes["output"], (3, 1, 1, 1))

    def test_wrong_input_geometry_is_rejected(self):
        with self.assertRaises(ValueError):
            SeizureNetLite()(torch.randn(2, 4, 256))


class EvaluationTests(unittest.TestCase):
    def test_threshold_maximizes_sensitivity_under_specificity_constraint(self):
        labels = np.asarray([1, 1, 0, 0, 0, 0])
        scores = np.asarray([0.9, 0.6, 0.8, 0.2, 0.1, 0.0])

        selected = select_threshold_at_specificity(
            scores, labels, target_specificity=0.75
        )

        self.assertAlmostEqual(selected["threshold_logit"], 0.6)
        self.assertAlmostEqual(selected["sensitivity"], 1.0)
        self.assertAlmostEqual(selected["specificity"], 0.75)

    def test_threshold_never_splits_equal_scores(self):
        labels = np.asarray([1, 0, 0, 1])
        scores = np.asarray([0.5, 0.5, 0.1, 0.0])

        selected = select_threshold_at_specificity(
            scores, labels, target_specificity=0.5
        )

        self.assertAlmostEqual(selected["threshold_logit"], 0.5)
        self.assertAlmostEqual(selected["specificity"], 0.5)

    def test_event_metrics_count_contiguous_false_alarm_runs(self):
        labels = np.asarray([0, 0, 1, 1, 0, 0, 0, 0])
        scores = np.asarray([1, 1, 1, -1, -1, 1, 1, -1], dtype=float)
        report = evaluate_predictions(
            scores,
            labels,
            threshold_logit=0.0,
            patient_ids=np.asarray(["p1"] * 8),
            recording_ids=np.asarray(["r1"] * 8),
            window_indices=np.arange(8),
            hop_sec=0.5,
            window_sec=1.0,
        )

        self.assertEqual(report["event_metrics"]["false_alarms"], 1)
        self.assertEqual(report["event_metrics"]["seizure_bouts"], 1)
        self.assertEqual(report["event_metrics"]["detected_seizure_bouts"], 1)


if __name__ == "__main__":
    unittest.main()
