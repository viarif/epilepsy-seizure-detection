"""Reproducible seeding, inference, and checkpoint loading."""

from __future__ import annotations

import random
from contextlib import nullcontext

import numpy as np
import torch


def seed_everything(seed: int, deterministic: bool = False) -> None:
    """Seed Python, NumPy, and PyTorch without changing the environment."""
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


def _unpack_batch(batch):
    if isinstance(batch, dict):
        return batch["signal"], batch["label"]
    if isinstance(batch, (tuple, list)) and len(batch) >= 2:
        return batch[0], batch[1]
    raise TypeError("A batch must be (signal, label) or a mapping with those keys.")


def _autocast_context(device, enabled):
    if not enabled or device.type != "cuda":
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=torch.float16)


def predict_loader(model, loader, device, *, criterion=None, use_amp=False):
    """Run deterministic inference over a complete data loader."""
    model.eval()
    all_scores = []
    all_labels = []
    total_loss = 0.0
    total_count = 0
    with torch.inference_mode():
        for batch in loader:
            signals, labels = _unpack_batch(batch)
            signals = signals.to(device, non_blocking=True)
            device_labels = labels.to(device, non_blocking=True).reshape(-1)
            with _autocast_context(device, use_amp):
                scores = model(signals)
                loss = None if criterion is None else criterion(scores, device_labels)
            all_scores.append(scores.float().cpu())
            all_labels.append(labels.float().cpu())
            if loss is not None:
                count = int(device_labels.numel())
                total_loss += float(loss.detach().cpu()) * count
                total_count += count
    scores = torch.cat(all_scores).numpy().astype(np.float64, copy=False)
    labels = torch.cat(all_labels).numpy().astype(np.int8, copy=False)
    mean_loss = None if criterion is None else total_loss / max(total_count, 1)
    return scores, labels, mean_loss


def load_checkpoint(model, checkpoint_path, device="cpu"):
    """Load the locked model checkpoint and return its metadata payload."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    return checkpoint
