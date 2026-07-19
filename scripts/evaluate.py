"""Evaluate the locked classifier without retraining or test-time tuning."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch.utils.data import DataLoader

from src.data import EEGWindowDataset
from src.evaluation import collect_dataset_metadata, evaluate_predictions
from src.models import SeizureNetLite
from src.training import load_checkpoint, macro_patient_metrics, predict_loader


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate the locked seizure-window classifier."
    )
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("results/model/best.pt")
    )
    parser.add_argument("--cache-root", type=Path, default=Path("data/processed/selected4"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/model"))
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--splits", nargs="+", choices=("val", "test"), default=("val", "test"))
    return parser.parse_args()


def resolve_device(value):
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is unavailable.")
    return device


def save_json(value, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)


def main():
    args = parse_args()
    if args.batch_size <= 0 or args.num_workers < 0:
        raise SystemExit("batch size must be positive and num_workers non-negative.")
    device = resolve_device(args.device)
    model = SeizureNetLite()
    checkpoint = load_checkpoint(model, args.checkpoint, device)
    threshold = checkpoint["threshold"]["threshold_logit"]
    criterion = torch.nn.BCEWithLogitsLoss()
    loaders = []
    try:
        for split in args.splits:
            dataset = EEGWindowDataset(args.cache_root, split=split)
            loader = DataLoader(
                dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=device.type == "cuda",
                persistent_workers=args.num_workers > 0,
                drop_last=False,
            )
            loaders.append((split, dataset, loader))
            scores, labels, loss = predict_loader(
                model,
                loader,
                device,
                criterion=criterion,
                use_amp=False,
            )
            metadata = collect_dataset_metadata(dataset)
            report = evaluate_predictions(
                scores,
                labels,
                threshold,
                patient_ids=metadata["patient_ids"],
                recording_ids=metadata["recording_ids"],
                window_indices=metadata["window_indices"],
                hop_sec=0.5,
                window_sec=1.0,
            )
            report.update(macro_patient_metrics(report))
            report["loss"] = float(loss)
            report["threshold_source"] = "locked checkpoint validation threshold"
            report["checkpoint_epoch"] = checkpoint["epoch"]
            output_split = "validation" if split == "val" else split
            save_json(report, args.output_dir / f"{output_split}_report.json")
            save_json(
                report.get("per_patient", {}),
                args.output_dir / f"{output_split}_per_patient.json",
            )
            print(
                json.dumps(
                    {
                        "split": split,
                        "sensitivity": report["sensitivity"],
                        "specificity": report["specificity"],
                        "pr_auc": report["pr_auc"],
                        "false_alarms_per_hour": report["event_metrics"]["false_alarms_per_hour"],
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    finally:
        for _split, dataset, _loader in loaders:
            dataset.close()


if __name__ == "__main__":
    main()
