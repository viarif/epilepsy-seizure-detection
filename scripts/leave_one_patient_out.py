"""Leave-one-patient-out evaluation for the locked original protocol.

Each fold trains SeizureNetLite on 23 patients and evaluates the held-out
patient with a fixed threshold from the original locked checkpoint. No
validation data from the fold is used for threshold or checkpoint selection.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
import shutil
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.data import EEGWindowDataset, HardNegativeReplaySampler, WeightedTrainingDataset
from src.evaluation import collect_dataset_metadata, evaluate_predictions
from src.models import SeizureNetLite
from src.preprocessing.config import ALL_PATIENTS
from src.training.runtime import predict_loader, seed_everything
from src.training.trainer import hierarchical_weighted_bce, macro_patient_metrics


POSITIVE_FRACTION = 0.05
REPLAY_FRACTION = 0.50
HARD_NEGATIVES_PER_RECORDING = 256
TARGET_SPECIFICITY = 0.97
HOP_SEC = 0.5
WINDOW_SEC = 1.0


@dataclass(frozen=True)
class FoldConfig:
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    gradient_clip_norm: float
    seed: int
    use_amp: bool
    progress_interval: int


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run leave-one-patient-out training with the original locked "
            "SeizureNetLite protocol, a fixed epoch count, and a fixed threshold."
        )
    )
    parser.add_argument(
        "--cache-root", type=Path, default=Path("data/processed/selected4")
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/experiments/loso_original"),
    )
    parser.add_argument(
        "--threshold-checkpoint", type=Path, default=Path("results/model/best.pt")
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Fixed training epochs per fold. Defaults to checkpoint epoch.",
    )
    parser.add_argument("--folds", nargs="*", default=None)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--gradient-clip-norm", type=float, default=5.0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-open-recordings", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260718)
    parser.add_argument("--progress-interval", type=int, default=200)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip completed folds with test_report.json and test_scores.npz.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only build fold cache views and print dataset counts.",
    )
    return parser.parse_args()


def resolve_device(value):
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is unavailable.")
    return device


def read_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)


def write_csv(rows, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _safe_rmtree(path, allowed_root):
    path = Path(path).resolve()
    allowed_root = Path(allowed_root).resolve()
    if path == allowed_root or allowed_root not in path.parents:
        raise RuntimeError(f"Refusing to remove unexpected path: {path}")
    shutil.rmtree(path)


def _autocast_context(device, enabled):
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def _cpu_state_dict(model):
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }


def discover_metadata(cache_root):
    cache_root = Path(cache_root)
    metadata_by_patient = {patient_id: [] for patient_id in ALL_PATIENTS}
    for metadata_path in sorted(cache_root.glob("*/*/*.json")):
        metadata = read_json(metadata_path)
        patient_id = metadata.get("patient_id")
        if patient_id in metadata_by_patient:
            metadata_by_patient[patient_id].append(metadata_path)
    missing = [
        patient_id
        for patient_id, paths in metadata_by_patient.items()
        if not paths
    ]
    if missing:
        raise FileNotFoundError(f"Missing cached metadata for patients: {missing}")
    return metadata_by_patient


def _resolve_signal_file(metadata, source_metadata_path):
    configured = Path(metadata["signal_file"])
    if configured.is_file():
        return configured.resolve()
    portable = Path(source_metadata_path).with_suffix(".npy")
    if portable.is_file():
        return portable.resolve()
    raise FileNotFoundError(
        f"Signal file is missing: {configured} (also tried {portable})"
    )


def prepare_fold_cache(metadata_by_patient, fold_cache_root, test_patient, output_dir):
    fold_cache_root = Path(fold_cache_root)
    if fold_cache_root.exists():
        _safe_rmtree(fold_cache_root, output_dir)

    counts = {
        "train_recordings": 0,
        "test_recordings": 0,
        "train_windows": 0,
        "test_windows": 0,
        "train_positive_windows": 0,
        "test_positive_windows": 0,
    }
    for patient_id in ALL_PATIENTS:
        split = "test" if patient_id == test_patient else "train"
        for source_path in metadata_by_patient[patient_id]:
            metadata = read_json(source_path)
            original_split = metadata["split"]
            metadata["split"] = split
            metadata["loso_source_split"] = original_split
            metadata["signal_file"] = str(_resolve_signal_file(metadata, source_path))
            destination = fold_cache_root / split / patient_id / source_path.name
            write_json(metadata, destination)
            counts[f"{split}_recordings"] += 1
            counts[f"{split}_windows"] += int(metadata["window_count"])
            counts[f"{split}_positive_windows"] += int(
                metadata["positive_window_count"]
            )
    return counts


def create_train_loader(fold_cache_root, args, device):
    train_base = EEGWindowDataset(
        fold_cache_root,
        split="train",
        return_metadata=False,
        max_open_recordings=args.max_open_recordings,
    )
    train_dataset = WeightedTrainingDataset(train_base)
    sampler = HardNegativeReplaySampler(
        train_base,
        positive_fraction=POSITIVE_FRACTION,
        replay_fraction=REPLAY_FRACTION,
        hard_per_recording=HARD_NEGATIVES_PER_RECORDING,
        seed=args.seed,
    )
    loader = DataLoader(
        train_dataset,
        sampler=sampler,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        drop_last=False,
    )
    return train_base, train_dataset, sampler, loader


def create_eval_loader(fold_cache_root, args, device):
    dataset = EEGWindowDataset(
        fold_cache_root,
        split="test",
        return_metadata=False,
        max_open_recordings=args.max_open_recordings,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        drop_last=False,
    )
    return dataset, loader


def shutdown_loader(loader):
    iterator = getattr(loader, "_iterator", None)
    if iterator is not None:
        iterator._shutdown_workers()
        loader._iterator = None


def train_fixed_epochs(model, train_loader, sampler, device, config, fold_label):
    model.to(device)
    model.train()
    optimizer = torch.optim.Adam(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=float(config.learning_rate),
        weight_decay=float(config.weight_decay),
    )
    amp_enabled = bool(config.use_amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    history = []

    for epoch in range(config.epochs):
        epoch_start = time.perf_counter()
        sampler.set_epoch(epoch)
        sampled_components = sampler.sampled_components_for_epoch(epoch)
        sampled_random_count = int(sampled_components["random_negative"].size)
        sampled_hard_count = int(sampled_components["hard_negative"].size)

        total_loss = 0.0
        total_count = 0
        observed_negative_indices = []
        observed_negative_scores = []
        model.train()

        for batch in train_loader:
            batch_number = total_count // int(config.batch_size) + 1
            signals = batch["signal"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True).reshape(-1)
            positive_weights = batch["positive_weight"].to(
                device, non_blocking=True
            ).reshape(-1)
            dataset_indices = batch["dataset_index"].reshape(-1)

            optimizer.zero_grad(set_to_none=True)
            with _autocast_context(device, amp_enabled):
                logits = model(signals)
                loss = hierarchical_weighted_bce(logits, labels, positive_weights)
            scaler.scale(loss).backward()
            if config.gradient_clip_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), float(config.gradient_clip_norm)
                )
            scaler.step(optimizer)
            scaler.update()

            negative_mask = labels < 0.5
            if torch.any(negative_mask):
                observed_negative_indices.append(
                    dataset_indices[negative_mask.cpu()].numpy()
                )
                observed_negative_scores.append(
                    logits.detach()[negative_mask].float().cpu().numpy()
                )

            count = int(labels.numel())
            total_loss += float(loss.detach().cpu()) * count
            total_count += count

            if (
                config.progress_interval > 0
                and batch_number % config.progress_interval == 0
            ):
                print(
                    f"fold={fold_label} epoch={epoch + 1:03d}/{config.epochs:03d} "
                    f"batch={batch_number:05d}/{len(train_loader):05d} "
                    f"seen={total_count}",
                    flush=True,
                )

        if observed_negative_indices:
            sampler.update_hard_negatives(
                np.concatenate(observed_negative_indices),
                np.concatenate(observed_negative_scores),
            )

        train_loss = total_loss / max(total_count, 1)
        row = {
            "epoch": epoch + 1,
            "train_loss": float(train_loss),
            "sampled_random_negatives": sampled_random_count,
            "sampled_hard_negatives": sampled_hard_count,
            "hard_bank_size": sampler.hard_bank_size,
            "hard_bank_recordings": sampler.hard_bank_recording_count,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "elapsed_sec": float(time.perf_counter() - epoch_start),
        }
        history.append(row)
        print(
            f"fold={fold_label} epoch={epoch + 1:03d}/{config.epochs:03d} "
            f"train_loss={train_loss:.6f} "
            f"hard={sampled_hard_count}/{sampler.hard_bank_size} "
            f"elapsed_sec={row['elapsed_sec']:.1f}",
            flush=True,
        )
    return history, optimizer


def evaluate_fold(model, fold_cache_root, args, device, threshold_logit):
    test_dataset, test_loader = create_eval_loader(fold_cache_root, args, device)
    try:
        scores, labels, loss = predict_loader(
            model,
            test_loader,
            device,
            criterion=nn.BCEWithLogitsLoss(),
            use_amp=False,
        )
        metadata = collect_dataset_metadata(test_dataset)
        report = evaluate_predictions(
            scores,
            labels,
            threshold_logit,
            patient_ids=metadata["patient_ids"],
            recording_ids=metadata["recording_ids"],
            window_indices=metadata["window_indices"],
            hop_sec=HOP_SEC,
            window_sec=WINDOW_SEC,
        )
        report.update(macro_patient_metrics(report))
        report["loss"] = float(loss)
        return report, scores.astype(np.float32, copy=False), labels
    finally:
        shutdown_loader(test_loader)
        test_dataset.close()


def fold_row(report):
    event = report["event_metrics"]
    return {
        "patient": report["left_out_patient"],
        "sensitivity": report["sensitivity"],
        "specificity": report["specificity"],
        "precision": report["precision"],
        "f1": report["f1"],
        "balanced_accuracy": report["balanced_accuracy"],
        "pr_auc": report["pr_auc"],
        "roc_auc": report["roc_auc"],
        "true_positives": report["true_positives"],
        "false_positives": report["false_positives"],
        "true_negatives": report["true_negatives"],
        "false_negatives": report["false_negatives"],
        "positive_count": report["positive_count"],
        "negative_count": report["negative_count"],
        "window_count": report["window_count"],
        "false_alarms": event["false_alarms"],
        "evaluated_hours": event["evaluated_hours"],
        "false_alarms_per_hour": event["false_alarms_per_hour"],
        "detected_seizure_bouts": event["detected_seizure_bouts"],
        "seizure_bouts": event["seizure_bouts"],
        "seizure_bout_sensitivity": event["seizure_bout_sensitivity"],
        "train_windows_full": report["train_windows_full"],
        "train_positive_windows": report["train_positive_windows"],
        "train_windows_per_epoch": report["train_windows_per_epoch"],
        "epochs": report["fixed_epochs"],
        "seed": report["seed"],
    }


def _rate(numerator, denominator):
    return float(numerator / denominator) if denominator else None


def build_summary(fold_reports, score_arrays, label_arrays, threshold_logit):
    rows = [fold_row(report) for report in fold_reports]
    tp = sum(int(row["true_positives"]) for row in rows)
    fp = sum(int(row["false_positives"]) for row in rows)
    tn = sum(int(row["true_negatives"]) for row in rows)
    fn = sum(int(row["false_negatives"]) for row in rows)
    positives = tp + fn
    negatives = tn + fp
    false_alarms = sum(int(row["false_alarms"]) for row in rows)
    evaluated_hours = sum(float(row["evaluated_hours"]) for row in rows)
    seizure_bouts = sum(int(row["seizure_bouts"]) for row in rows)
    detected_bouts = sum(int(row["detected_seizure_bouts"]) for row in rows)

    scores = np.concatenate(score_arrays) if score_arrays else np.empty(0)
    labels = np.concatenate(label_arrays) if label_arrays else np.empty(0)
    auc_payload = {}
    if labels.size and np.unique(labels).size == 2:
        auc_payload = {
            "pr_auc": float(average_precision_score(labels, scores)),
            "roc_auc": float(roc_auc_score(labels, scores)),
        }

    overall = {
        "threshold_logit": float(threshold_logit),
        "threshold_probability": float(
            1.0 / (1.0 + np.exp(-np.clip(float(threshold_logit), -50.0, 50.0)))
        ),
        "sensitivity": _rate(tp, positives),
        "specificity": _rate(tn, negatives),
        "precision": _rate(tp, tp + fp),
        "f1": _rate(2 * tp, 2 * tp + fp + fn),
        "accuracy": _rate(tp + tn, positives + negatives),
        "balanced_accuracy": None,
        "macro_sensitivity": float(np.mean([row["sensitivity"] for row in rows])),
        "macro_specificity": float(np.mean([row["specificity"] for row in rows])),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
        "positive_count": int(positives),
        "negative_count": int(negatives),
        "window_count": int(positives + negatives),
        "false_alarms": int(false_alarms),
        "evaluated_hours": float(evaluated_hours),
        "false_alarms_per_hour": _rate(false_alarms, evaluated_hours),
        "detected_seizure_bouts": int(detected_bouts),
        "seizure_bouts": int(seizure_bouts),
        "seizure_bout_sensitivity": _rate(detected_bouts, seizure_bouts),
    }
    overall["balanced_accuracy"] = (
        None
        if overall["sensitivity"] is None or overall["specificity"] is None
        else float((overall["sensitivity"] + overall["specificity"]) / 2.0)
    )
    overall.update(auc_payload)
    return rows, overall


def write_markdown_report(path, rows, overall, manifest):
    def fmt(value):
        if value is None:
            return "NA"
        if isinstance(value, (int, np.integer)):
            return str(int(value))
        if isinstance(value, (float, np.floating)):
            return f"{float(value):.6f}"
        return str(value)

    lines = [
        "# Leave-one-patient-out original protocol",
        "",
        f"- Fixed threshold logit: `{overall['threshold_logit']:.12f}`",
        f"- Fixed threshold probability: `{overall['threshold_probability']:.12f}`",
        f"- Fixed epochs per fold: `{manifest['epochs']}`",
        f"- Completed folds: `{len(rows)}`",
        "",
        "## Per-patient held-out results",
        "",
        "| Patient | Sensitivity | Specificity | TP | FN | TN | FP | Windows | Bout sens | FA/hour |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["patient"]),
                    fmt(row["sensitivity"]),
                    fmt(row["specificity"]),
                    fmt(row["true_positives"]),
                    fmt(row["false_negatives"]),
                    fmt(row["true_negatives"]),
                    fmt(row["false_positives"]),
                    fmt(row["window_count"]),
                    fmt(row["seizure_bout_sensitivity"]),
                    fmt(row["false_alarms_per_hour"]),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Overall held-out results",
            "",
            "| Metric | Value |",
            "|---|---:|",
            f"| Window sensitivity | {fmt(overall['sensitivity'])} |",
            f"| Window specificity | {fmt(overall['specificity'])} |",
            f"| Macro patient sensitivity | {fmt(overall['macro_sensitivity'])} |",
            f"| Macro patient specificity | {fmt(overall['macro_specificity'])} |",
            f"| PR-AUC | {fmt(overall.get('pr_auc'))} |",
            f"| ROC-AUC | {fmt(overall.get('roc_auc'))} |",
            f"| Seizure-bout sensitivity | {fmt(overall['seizure_bout_sensitivity'])} |",
            f"| False alarms/hour | {fmt(overall['false_alarms_per_hour'])} |",
            f"| TP / FN / TN / FP | {overall['true_positives']} / {overall['false_negatives']} / {overall['true_negatives']} / {overall['false_positives']} |",
            f"| Evaluated windows | {overall['window_count']} |",
        ]
    )
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_summary(output_dir, fold_reports, score_arrays, label_arrays, threshold_logit, manifest):
    rows, overall = build_summary(
        fold_reports,
        score_arrays,
        label_arrays,
        threshold_logit,
    )
    write_csv(rows, output_dir / "loso_per_patient.csv")
    write_json(
        {
            "manifest": manifest,
            "overall": overall,
            "per_patient": rows,
        },
        output_dir / "loso_summary.json",
    )
    write_markdown_report(output_dir / "loso_report.md", rows, overall, manifest)
    return rows, overall


def run_fold(args, test_patient, metadata_by_patient, threshold_payload, config, device):
    output_dir = Path(args.output_dir)
    fold_dir = output_dir / "folds" / test_patient
    fold_cache_root = output_dir / "_cache_views" / test_patient
    fold_dir.mkdir(parents=True, exist_ok=True)
    counts = prepare_fold_cache(
        metadata_by_patient,
        fold_cache_root,
        test_patient,
        output_dir,
    )
    if args.dry_run:
        print(
            f"dry_run fold={test_patient} train_windows={counts['train_windows']} "
            f"train_pos={counts['train_positive_windows']} "
            f"test_windows={counts['test_windows']} "
            f"test_pos={counts['test_positive_windows']}",
            flush=True,
        )
        return None

    start = time.perf_counter()
    seed_everything(args.seed, deterministic=args.deterministic)
    train_base = None
    train_dataset = None
    train_loader = None
    test_report = None
    scores = None
    labels = None
    try:
        train_base, train_dataset, sampler, train_loader = create_train_loader(
            fold_cache_root,
            args,
            device,
        )
        sampled_prior_bias = math.log(POSITIVE_FRACTION / (1.0 - POSITIVE_FRACTION))
        model = SeizureNetLite(output_bias=sampled_prior_bias)
        if model.parameter_count != 2991:
            raise RuntimeError(
                f"The locked model must have 2,991 parameters, got "
                f"{model.parameter_count}."
            )

        history, optimizer = train_fixed_epochs(
            model,
            train_loader,
            sampler,
            device,
            config,
            test_patient,
        )
        write_csv(history, fold_dir / "training_curve.csv")
        torch.save(
            {
                "model_name": "SeizureNetLite",
                "model_architecture": "SeizureNetLite",
                "model_state": _cpu_state_dict(model),
                "optimizer_state": optimizer.state_dict(),
                "epoch": int(config.epochs),
                "threshold": threshold_payload,
                "training_config": asdict(config),
                "loso_left_out_patient": test_patient,
                "parameter_count": model.parameter_count,
            },
            fold_dir / "final.pt",
        )

        test_report, scores, labels = evaluate_fold(
            model,
            fold_cache_root,
            args,
            device,
            float(threshold_payload["threshold_logit"]),
        )
        elapsed_sec = float(time.perf_counter() - start)
        test_report.update(
            {
                "left_out_patient": test_patient,
                "threshold_source": str(args.threshold_checkpoint),
                "fixed_epochs": int(config.epochs),
                "seed": int(args.seed),
                "positive_fraction": POSITIVE_FRACTION,
                "replay_fraction": REPLAY_FRACTION,
                "hard_negatives_per_recording": HARD_NEGATIVES_PER_RECORDING,
                "target_specificity": TARGET_SPECIFICITY,
                "train_windows_full": int(len(train_base)),
                "train_positive_windows": int(train_base.positive_count),
                "train_windows_per_epoch": int(len(sampler)),
                "train_patients": int(len(train_base.patient_ids)),
                "train_recordings": int(len(train_base._records)),
                "test_recordings": int(counts["test_recordings"]),
                "elapsed_sec": elapsed_sec,
            }
        )
        train_report = {
            "left_out_patient": test_patient,
            "fixed_epochs": int(config.epochs),
            "history": history,
            "hard_negative_bank": {
                "window_count": sampler.hard_bank_size,
                "recording_count": sampler.hard_bank_recording_count,
                "per_recording_cap": sampler.hard_per_recording,
            },
            "train_windows_full": int(len(train_base)),
            "train_positive_windows": int(train_base.positive_count),
            "train_windows_per_epoch": int(len(sampler)),
            "train_patients": int(len(train_base.patient_ids)),
            "train_recordings": int(len(train_base._records)),
            "config": asdict(config),
        }
        write_json(train_report, fold_dir / "train_report.json")
        write_json(test_report, fold_dir / "test_report.json")
        write_json(test_report["per_patient"], fold_dir / "test_per_patient.json")
        np.savez_compressed(
            fold_dir / "test_scores.npz",
            scores=scores,
            labels=labels.astype(np.int8, copy=False),
        )
        print(
            f"fold={test_patient} result "
            f"sens={test_report['sensitivity']:.6f} "
            f"spec={test_report['specificity']:.6f} "
            f"windows={test_report['window_count']} "
            f"elapsed_sec={elapsed_sec:.1f}",
            flush=True,
        )
        return test_report, scores, labels
    finally:
        if train_loader is not None:
            shutdown_loader(train_loader)
        if train_dataset is not None:
            train_dataset.close()
        elif train_base is not None:
            train_base.close()
        del train_loader
        del train_dataset
        del train_base
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()


def load_completed_fold(fold_dir):
    report_path = fold_dir / "test_report.json"
    scores_path = fold_dir / "test_scores.npz"
    if not report_path.is_file() or not scores_path.is_file():
        return None
    report = read_json(report_path)
    arrays = np.load(scores_path, allow_pickle=False)
    return report, arrays["scores"], arrays["labels"]


def main():
    args = parse_args()
    if args.batch_size <= 0:
        raise SystemExit("batch-size must be positive.")
    if args.num_workers < 0:
        raise SystemExit("num-workers must be non-negative.")
    if args.epochs is not None and args.epochs <= 0:
        raise SystemExit("epochs must be positive.")

    device = resolve_device(args.device)
    checkpoint = torch.load(
        args.threshold_checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    threshold_payload = checkpoint["threshold"]
    epochs = int(args.epochs if args.epochs is not None else checkpoint["epoch"])
    selected_folds = tuple(args.folds) if args.folds else ALL_PATIENTS
    unknown = sorted(set(selected_folds) - set(ALL_PATIENTS))
    if unknown:
        raise SystemExit(f"Unknown patient fold(s): {unknown}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_by_patient = discover_metadata(args.cache_root)
    config = FoldConfig(
        epochs=epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        gradient_clip_norm=args.gradient_clip_norm,
        seed=args.seed,
        use_amp=not args.no_amp,
        progress_interval=args.progress_interval,
    )
    manifest = {
        "protocol": "leave_one_patient_out_original_fixed_threshold",
        "model": "SeizureNetLite",
        "parameter_count": 2991,
        "fold_rule": "train on 23 patients, test on the held-out patient",
        "checkpoint_selection": "fixed epoch count; no fold validation selection",
        "threshold_selection": "fixed from original checkpoint; not reselected per fold",
        "threshold_checkpoint": str(args.threshold_checkpoint),
        "threshold": threshold_payload,
        "epochs": epochs,
        "positive_fraction": POSITIVE_FRACTION,
        "replay_fraction": REPLAY_FRACTION,
        "hard_negatives_per_recording": HARD_NEGATIVES_PER_RECORDING,
        "target_specificity": TARGET_SPECIFICITY,
        "seed": args.seed,
        "device": str(device),
        "cuda_device_name": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else None
        ),
        "config": asdict(config),
        "folds": list(selected_folds),
    }
    write_json(manifest, output_dir / "loso_manifest.json")
    print(json.dumps(manifest, indent=2, ensure_ascii=False), flush=True)

    if args.dry_run:
        for test_patient in selected_folds:
            counts = prepare_fold_cache(
                metadata_by_patient,
                output_dir / "_cache_views" / test_patient,
                test_patient,
                output_dir,
            )
            print(
                f"dry_run fold={test_patient} train_windows={counts['train_windows']} "
                f"train_pos={counts['train_positive_windows']} "
                f"test_windows={counts['test_windows']} "
                f"test_pos={counts['test_positive_windows']}",
                flush=True,
            )
        return

    fold_reports = []
    score_arrays = []
    label_arrays = []
    for test_patient in selected_folds:
        fold_dir = output_dir / "folds" / test_patient
        if args.resume:
            completed = load_completed_fold(fold_dir)
            if completed is not None:
                report, scores, labels = completed
                print(
                    f"fold={test_patient} skipped resume "
                    f"sens={report['sensitivity']:.6f} "
                    f"spec={report['specificity']:.6f}",
                    flush=True,
                )
                fold_reports.append(report)
                score_arrays.append(scores)
                label_arrays.append(labels)
                save_summary(
                    output_dir,
                    fold_reports,
                    score_arrays,
                    label_arrays,
                    float(threshold_payload["threshold_logit"]),
                    manifest,
                )
                continue

        result = run_fold(
            args,
            test_patient,
            metadata_by_patient,
            threshold_payload,
            config,
            device,
        )
        if result is None:
            continue
        report, scores, labels = result
        fold_reports.append(report)
        score_arrays.append(scores)
        label_arrays.append(labels)
        _rows, overall = save_summary(
            output_dir,
            fold_reports,
            score_arrays,
            label_arrays,
            float(threshold_payload["threshold_logit"]),
            manifest,
        )
        print(
            f"completed_folds={len(fold_reports)}/{len(selected_folds)} "
            f"overall_sens={overall['sensitivity']:.6f} "
            f"overall_spec={overall['specificity']:.6f}",
            flush=True,
        )

    rows, overall = save_summary(
        output_dir,
        fold_reports,
        score_arrays,
        label_arrays,
        float(threshold_payload["threshold_logit"]),
        manifest,
    )
    print(
        "FINAL_LOSO_SUMMARY "
        + json.dumps(
            {
                "folds": len(rows),
                "sensitivity": overall["sensitivity"],
                "specificity": overall["specificity"],
                "macro_sensitivity": overall["macro_sensitivity"],
                "macro_specificity": overall["macro_specificity"],
                "window_count": overall["window_count"],
                "output_dir": str(output_dir.resolve()),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
