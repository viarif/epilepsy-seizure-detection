"""Train the locked cross-patient seizure-window classifier."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import platform
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from src.data import create_training_dataloaders
from src.evaluation import collect_dataset_metadata, evaluate_predictions
from src.models import SeizureNetLite
from src.training import (
    TrainingConfig,
    fit_model,
    macro_patient_metrics,
    predict_loader,
    seed_everything,
)


POSITIVE_FRACTION = 0.05
REPLAY_FRACTION = 0.50
HARD_NEGATIVES_PER_RECORDING = 256
TARGET_SPECIFICITY = 0.97


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Train the locked patient/bout-balanced hard-negative replay "
            "protocol. Model selection reads train and validation only."
        )
    )
    parser.add_argument(
        "--cache-root", type=Path, default=Path("data/processed/selected4")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/model")
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260718)
    parser.add_argument("--gradient-clip-norm", type=float, default=5.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--deterministic", action="store_true")
    return parser.parse_args()


def resolve_device(value):
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is unavailable.")
    return device


def save_json(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)


def main():
    args = parse_args()
    if args.batch_size <= 0 or args.num_workers < 0:
        raise SystemExit("batch size must be positive and num_workers non-negative.")
    if args.max_epochs <= 0 or args.patience <= 0:
        raise SystemExit("max_epochs and patience must be positive.")

    device = resolve_device(args.device)
    seed_everything(args.seed, deterministic=args.deterministic)
    loaders = create_training_dataloaders(
        cache_root=args.cache_root,
        batch_size=args.batch_size,
        positive_fraction=POSITIVE_FRACTION,
        replay_fraction=REPLAY_FRACTION,
        hard_per_recording=HARD_NEGATIVES_PER_RECORDING,
        num_workers=args.num_workers,
        seed=args.seed,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    try:
        sampled_prior_bias = math.log(
            POSITIVE_FRACTION / (1.0 - POSITIVE_FRACTION)
        )
        model = SeizureNetLite(output_bias=sampled_prior_bias)
        if model.parameter_count != 2991:
            raise RuntimeError(
                f"The locked model must have 2,991 parameters, got "
                f"{model.parameter_count}."
            )

        train_dataset = loaders.train.dataset.base_dataset
        run_manifest = {
            "protocol": "patient_bout_balanced_hard_negative_replay",
            "status": "model selection uses train and validation only",
            "model": "SeizureNetLite",
            "parameter_count": model.parameter_count,
            "canonical_channels": list(train_dataset.canonical_channels),
            "cache_root": str(args.cache_root.resolve()),
            "locked_training": {
                "positive_fraction": POSITIVE_FRACTION,
                "replay_fraction_after_warmup": REPLAY_FRACTION,
                "hard_negatives_per_recording": HARD_NEGATIVES_PER_RECORDING,
                "target_specificity": TARGET_SPECIFICITY,
            },
            "split_counts": {
                "train_windows_full": len(train_dataset),
                "train_positive_windows": train_dataset.positive_count,
                "train_windows_per_epoch": len(loaders.train_sampler),
                "validation_windows": len(loaders.val.dataset),
                "validation_positive_windows": loaders.val.dataset.positive_count,
            },
            "runtime": {
                "python": sys.version,
                "platform": platform.platform(),
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "device": str(device),
                "device_name": (
                    torch.cuda.get_device_name(device)
                    if device.type == "cuda"
                    else platform.processor()
                ),
            },
            "arguments": vars(args)
            | {
                "cache_root": str(args.cache_root),
                "output_dir": str(args.output_dir),
            },
        }
        save_json(run_manifest, args.output_dir / "run_manifest.json")
        print(json.dumps(run_manifest, indent=2, ensure_ascii=False), flush=True)

        config = TrainingConfig(
            max_epochs=args.max_epochs,
            patience=args.patience,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            positive_fraction=POSITIVE_FRACTION,
            replay_fraction=REPLAY_FRACTION,
            hard_per_recording=HARD_NEGATIVES_PER_RECORDING,
            target_specificity=TARGET_SPECIFICITY,
            gradient_clip_norm=args.gradient_clip_norm,
            use_amp=not args.no_amp,
            seed=args.seed,
        )
        train_report = fit_model(
            model,
            loaders,
            device,
            args.output_dir,
            config=config,
        )

        threshold = train_report["best_validation"]["threshold"]["threshold_logit"]
        scores, labels, loss = predict_loader(
            model,
            loaders.val,
            device,
            criterion=torch.nn.BCEWithLogitsLoss(),
            use_amp=False,
        )
        metadata = collect_dataset_metadata(loaders.val.dataset)
        validation_report = evaluate_predictions(
            scores,
            labels,
            threshold,
            patient_ids=metadata["patient_ids"],
            recording_ids=metadata["recording_ids"],
            window_indices=metadata["window_indices"],
            hop_sec=0.5,
            window_sec=1.0,
        )
        validation_report.update(macro_patient_metrics(validation_report))
        validation_report["loss"] = float(loss)
        validation_report["threshold_source"] = (
            "best validation epoch; selected before test evaluation"
        )
        validation_report["best_epoch"] = train_report["best_epoch"]
        validation_report["test_evaluated_during_selection"] = False
        save_json(validation_report, args.output_dir / "validation_report.json")
        save_json(
            validation_report["per_patient"],
            args.output_dir / "validation_per_patient.json",
        )
        summary = {
            "best_epoch": train_report["best_epoch"],
            "macro_sensitivity_at_target": validation_report["macro_sensitivity"],
            "aggregate_sensitivity_at_target": validation_report["sensitivity"],
            "specificity": validation_report["specificity"],
            "pr_auc": validation_report["pr_auc"],
            "min_patient_sensitivity": validation_report[
                "min_patient_sensitivity"
            ],
            "hard_bank_size": train_report["hard_negative_bank"]["window_count"],
            "output_dir": str(args.output_dir.resolve()),
        }
        print("FINAL_TRAINING_SUMMARY", json.dumps(summary, ensure_ascii=False), flush=True)
    finally:
        loaders.close()


if __name__ == "__main__":
    main()
