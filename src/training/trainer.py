"""Training loop for the locked cross-patient protocol."""

from __future__ import annotations

import csv
import json
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from src.evaluation import (
    collect_dataset_metadata,
    evaluate_predictions,
    select_threshold_at_specificity,
)
from src.training.runtime import predict_loader


@dataclass
class TrainingConfig:
    max_epochs: int = 50
    patience: int = 5
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    positive_fraction: float = 0.05
    replay_fraction: float = 0.50
    hard_per_recording: int = 256
    target_specificity: float = 0.97
    gradient_clip_norm: float = 5.0
    use_amp: bool = True
    seed: int = 20260717
    min_delta: float = 1e-6


def hierarchical_weighted_bce(logits, labels, positive_weights):
    """BCE with precomputed patient/bout-balanced positive weights."""
    logits = logits.reshape(-1)
    labels = labels.reshape(-1).to(logits.dtype)
    positive_weights = positive_weights.reshape(-1).to(logits.dtype)
    if logits.shape != labels.shape or logits.shape != positive_weights.shape:
        raise ValueError("logits, labels, and positive_weights must have equal shape.")
    if torch.any(positive_weights <= 0):
        raise ValueError("positive_weights must be strictly positive.")
    per_sample = F.binary_cross_entropy_with_logits(
        logits, labels, reduction="none"
    )
    sample_weights = torch.where(
        labels >= 0.5,
        positive_weights,
        torch.ones_like(positive_weights),
    )
    return torch.sum(per_sample * sample_weights) / torch.sum(sample_weights)


def macro_patient_metrics(report):
    """Macro-average per-patient rates at one shared global threshold."""
    per_patient = report.get("per_patient", {})
    sensitivities = [
        metrics["sensitivity"]
        for metrics in per_patient.values()
        if metrics.get("sensitivity") is not None
    ]
    specificities = [
        metrics["specificity"]
        for metrics in per_patient.values()
        if metrics.get("specificity") is not None
    ]
    if not sensitivities or not specificities:
        raise ValueError("Macro patient metrics require both classes across patients.")
    return {
        "macro_sensitivity": float(np.mean(sensitivities)),
        "macro_specificity": float(np.mean(specificities)),
        "min_patient_sensitivity": float(np.min(sensitivities)),
        "patient_count": len(per_patient),
    }


def _autocast_context(device, enabled):
    if not enabled or device.type != "cuda":
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=torch.float16)


def _cpu_state_dict(model):
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }


def _is_better(current, best, min_delta):
    if best is None:
        return True
    for current_value, best_value in zip(current, best):
        if current_value > best_value + min_delta:
            return True
        if current_value < best_value - min_delta:
            return False
    return False


def _write_history_csv(history, path):
    if not history:
        return
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)


def fit_model(
    model,
    loaders,
    device,
    output_dir,
    *,
    config=None,
    verbose=True,
):
    """Train on train patients and select only with complete validation data."""
    config = config or TrainingConfig()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(device)
    model = model.to(device)
    if getattr(model, "parameter_count", 0) != 2991:
        raise ValueError("Training requires the locked 2,991-parameter model.")

    sampler = loaders.train_sampler
    if not np.isclose(
        sampler.actual_positive_fraction,
        config.positive_fraction,
        atol=1e-12,
    ):
        raise ValueError("Config positive_fraction does not match the sampler.")
    if not np.isclose(sampler.replay_fraction, config.replay_fraction, atol=1e-12):
        raise ValueError("Config replay_fraction does not match the sampler.")
    if sampler.hard_per_recording != int(config.hard_per_recording):
        raise ValueError("Config hard_per_recording does not match the sampler.")

    optimizer = torch.optim.Adam(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=float(config.learning_rate),
        weight_decay=float(config.weight_decay),
    )
    validation_criterion = nn.BCEWithLogitsLoss()
    amp_enabled = bool(config.use_amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    validation_metadata = collect_dataset_metadata(loaders.val.dataset)

    history = []
    best_state = None
    best_epoch = None
    best_threshold = None
    best_validation = None
    best_key = None
    epochs_without_improvement = 0

    for epoch in range(int(config.max_epochs)):
        sampler.set_epoch(epoch)
        model.train()
        total_loss = 0.0
        total_count = 0
        observed_negative_indices = []
        observed_negative_scores = []

        sampled_components = sampler.sampled_components_for_epoch(epoch)
        sampled_random_count = int(sampled_components["random_negative"].size)
        sampled_hard_count = int(sampled_components["hard_negative"].size)

        for batch in loaders.train:
            signals = batch["signal"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True).reshape(-1)
            positive_weights = batch["positive_weight"].to(
                device, non_blocking=True
            ).reshape(-1)
            dataset_indices = batch["dataset_index"].reshape(-1)

            optimizer.zero_grad(set_to_none=True)
            with _autocast_context(device, amp_enabled):
                logits = model(signals)
                bce_loss = hierarchical_weighted_bce(
                    logits, labels, positive_weights
                )
                loss = bce_loss
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

        sampler.update_hard_negatives(
            np.concatenate(observed_negative_indices),
            np.concatenate(observed_negative_scores),
        )
        train_loss = total_loss / max(total_count, 1)

        validation_scores, validation_labels, validation_loss = predict_loader(
            model,
            loaders.val,
            device,
            criterion=validation_criterion,
            use_amp=False,
        )
        threshold = select_threshold_at_specificity(
            validation_scores,
            validation_labels,
            target_specificity=config.target_specificity,
        )
        validation_report = evaluate_predictions(
            validation_scores,
            validation_labels,
            threshold["threshold_logit"],
            patient_ids=validation_metadata["patient_ids"],
        )
        macro = macro_patient_metrics(validation_report)
        history_row = {
            "epoch": epoch + 1,
            "train_loss": float(train_loss),
            "train_weighted_bce": float(train_loss),
            "val_loss": float(validation_loss),
            "val_macro_sensitivity_at_target": macro["macro_sensitivity"],
            "val_macro_specificity_at_target": macro["macro_specificity"],
            "val_min_patient_sensitivity": macro["min_patient_sensitivity"],
            "val_aggregate_sensitivity_at_target": validation_report["sensitivity"],
            "val_aggregate_specificity_at_target": validation_report["specificity"],
            "val_pr_auc": validation_report["pr_auc"],
            "val_roc_auc": validation_report["roc_auc"],
            "val_threshold_logit": float(threshold["threshold_logit"]),
            "sampled_random_negatives": sampled_random_count,
            "sampled_hard_negatives": sampled_hard_count,
            "hard_bank_size": sampler.hard_bank_size,
            "hard_bank_recordings": sampler.hard_bank_recording_count,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(history_row)
        if verbose:
            print(
                f"epoch={epoch + 1:03d} train_loss={train_loss:.6f} "
                f"macro_sens@{100 * config.target_specificity:.0f}spec="
                f"{macro['macro_sensitivity']:.6f} "
                f"agg_sens={validation_report['sensitivity']:.6f} "
                f"spec={validation_report['specificity']:.6f} "
                f"pr_auc={validation_report['pr_auc']:.6f} "
                f"hard={sampled_hard_count}/{sampler.hard_bank_size}",
                flush=True,
            )

        current_key = (
            macro["macro_sensitivity"],
            float(validation_report["sensitivity"]),
            float(validation_report["pr_auc"]),
        )
        if _is_better(current_key, best_key, float(config.min_delta)):
            best_key = current_key
            best_epoch = epoch + 1
            best_threshold = threshold
            best_state = _cpu_state_dict(model)
            best_validation = {
                "macro_sensitivity_at_target_specificity": current_key[0],
                "aggregate_sensitivity_at_target_specificity": current_key[1],
                "pr_auc": current_key[2],
                "macro_specificity_at_target": macro["macro_specificity"],
                "min_patient_sensitivity": macro["min_patient_sensitivity"],
                "threshold": threshold,
                "per_patient": validation_report["per_patient"],
            }
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_name": "SeizureNetLite",
                    "model_architecture": "SeizureNetLite",
                    "model_state": best_state,
                    "optimizer_state": optimizer.state_dict(),
                    "epoch": best_epoch,
                    "threshold": best_threshold,
                    "training_config": asdict(config),
                    "selection_key": {
                        "macro_patient_sensitivity": current_key[0],
                        "aggregate_sensitivity": current_key[1],
                        "pr_auc": current_key[2],
                    },
                    "parameter_count": int(
                        sum(parameter.numel() for parameter in model.parameters())
                    ),
                },
                output_dir / "best.pt",
            )
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= int(config.patience):
            break

    if best_state is None:
        raise RuntimeError("Training produced no checkpoint.")
    model.load_state_dict(best_state)
    _write_history_csv(history, output_dir / "training_curve.csv")
    bank_indices, bank_scores = sampler.hard_bank_arrays()
    np.savez_compressed(
        output_dir / "hard_negative_bank.npz",
        dataset_indices=bank_indices,
        logits=bank_scores.astype(np.float32),
    )
    report = {
        "model_name": "SeizureNetLite",
        "model_architecture": "SeizureNetLite",
        "parameter_count": int(sum(p.numel() for p in model.parameters())),
        "device": str(device),
        "amp_enabled": amp_enabled,
        "best_epoch": int(best_epoch),
        "epochs_completed": len(history),
        "early_stopped": len(history) < int(config.max_epochs),
        "best_validation": best_validation,
        "hard_negative_bank": {
            "window_count": sampler.hard_bank_size,
            "recording_count": sampler.hard_bank_recording_count,
            "per_recording_cap": sampler.hard_per_recording,
        },
        "training_config": asdict(config),
        "history": history,
    }
    with open(output_dir / "train_report.json", "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
    return report
