"""Train-only hierarchical weighting and hard-negative replay."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from .eeg_windows import (
    EEGWindowDataset,
    _balanced_counts,
    _negative_ranks_to_window_indices,
)


def build_hierarchical_positive_weights(dataset: EEGWindowDataset) -> dict[int, float]:
    """Give every patient and every seizure bout equal total positive weight."""
    if dataset.split != "train":
        raise ValueError("Hierarchical positive weights are train-only.")
    if dataset.positive_count <= 0:
        raise ValueError("Training dataset contains no positive windows.")

    bouts_by_patient: dict[str, list[np.ndarray]] = {}
    for record in dataset._records:
        positives = np.asarray(record.positive_indices, dtype=np.int64)
        if positives.size == 0:
            continue
        split_points = np.flatnonzero(np.diff(positives) != 1) + 1
        for local_bout in np.split(positives, split_points):
            bouts_by_patient.setdefault(record.patient_id, []).append(
                record.offset + local_bout
            )

    if not bouts_by_patient:
        raise ValueError("Training dataset contains no positive seizure bouts.")
    patient_count = len(bouts_by_patient)
    positive_count = dataset.positive_count
    weights: dict[int, float] = {}
    for patient_bouts in bouts_by_patient.values():
        bout_count = len(patient_bouts)
        for bout in patient_bouts:
            window_weight = positive_count / (
                patient_count * bout_count * int(bout.size)
            )
            for dataset_index in bout:
                weights[int(dataset_index)] = float(window_weight)

    if len(weights) != positive_count:
        raise RuntimeError(
            "Positive weight count does not match the dataset positive count."
        )
    mean_weight = float(np.mean(tuple(weights.values())))
    if not np.isclose(mean_weight, 1.0, atol=1e-12):
        raise RuntimeError(
            f"Positive weights should have mean 1, got {mean_weight}."
        )
    return weights


class WeightedTrainingDataset(Dataset):
    """Attach dataset indices and hierarchical weights to train windows."""

    def __init__(self, base_dataset: EEGWindowDataset):
        if base_dataset.split != "train":
            raise ValueError("WeightedTrainingDataset requires the train split.")
        if base_dataset.return_metadata:
            raise ValueError("The wrapped dataset must use return_metadata=False.")
        self.base_dataset = base_dataset
        self.positive_weights = build_hierarchical_positive_weights(base_dataset)

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, dataset_index):
        signal, label = self.base_dataset[int(dataset_index)]
        return {
            "signal": signal,
            "label": label,
            "dataset_index": torch.tensor(int(dataset_index), dtype=torch.int64),
            "positive_weight": torch.tensor(
                self.positive_weights.get(int(dataset_index), 1.0),
                dtype=torch.float32,
            ),
        }

    def close(self):
        self.base_dataset.close()


class HardNegativeReplaySampler(Sampler):
    """Keep all positives and mix balanced random/replayed hard negatives."""

    def __init__(
        self,
        dataset: EEGWindowDataset,
        *,
        positive_fraction=0.05,
        replay_fraction=0.50,
        hard_per_recording=256,
        seed=0,
    ):
        if dataset.split != "train":
            raise ValueError("Hard-negative replay is only allowed on train.")
        if not 0.0 < positive_fraction < 1.0:
            raise ValueError("positive_fraction must be strictly between 0 and 1.")
        if not 0.0 <= replay_fraction <= 1.0:
            raise ValueError("replay_fraction must be in [0, 1].")
        if int(hard_per_recording) <= 0:
            raise ValueError("hard_per_recording must be positive.")
        if dataset.positive_count <= 0 or dataset.negative_count <= 0:
            raise ValueError("Training requires both positive and negative windows.")

        self.dataset = dataset
        self.positive_fraction = float(positive_fraction)
        self.replay_fraction = float(replay_fraction)
        self.hard_per_recording = int(hard_per_recording)
        self.seed = int(seed)
        self.num_positive = dataset.positive_count
        self.num_negative = int(
            round(
                self.num_positive
                * (1.0 - self.positive_fraction)
                / self.positive_fraction
            )
        )
        self.actual_positive_fraction = self.num_positive / (
            self.num_positive + self.num_negative
        )
        self._next_epoch = 0
        self._hard_bank: dict[int, tuple[np.ndarray, np.ndarray]] = {}

        records_by_patient: dict[str, list[int]] = {}
        for recording_index, record in enumerate(dataset._records):
            if record.negative_count > 0:
                records_by_patient.setdefault(record.patient_id, []).append(
                    recording_index
                )
        self._records_by_patient = tuple(
            (patient_id, tuple(recording_indices))
            for patient_id, recording_indices in sorted(records_by_patient.items())
        )
        if not self._records_by_patient:
            raise ValueError("Training cache contains no usable negative recordings.")

    def __len__(self):
        return self.num_positive + self.num_negative

    @property
    def hard_bank_size(self):
        return int(sum(indices.size for indices, _scores in self._hard_bank.values()))

    @property
    def hard_bank_recording_count(self):
        return len(self._hard_bank)

    def set_epoch(self, epoch):
        self._next_epoch = int(epoch)

    def _sample_random_negatives(self, total, rng):
        if total <= 0:
            return np.empty(0, dtype=np.int64)
        sampled = []
        patient_counts = _balanced_counts(total, len(self._records_by_patient), rng)
        for patient_count, (_patient_id, recording_indices) in zip(
            patient_counts,
            self._records_by_patient,
        ):
            recording_counts = _balanced_counts(
                int(patient_count), len(recording_indices), rng
            )
            for sample_count, recording_index in zip(
                recording_counts, recording_indices
            ):
                if sample_count == 0:
                    continue
                record = self.dataset._records[recording_index]
                ranks = rng.choice(
                    record.negative_count,
                    size=int(sample_count),
                    replace=sample_count > record.negative_count,
                ).astype(np.int64, copy=False)
                local_indices = _negative_ranks_to_window_indices(
                    ranks, record.positive_indices
                )
                sampled.append(record.offset + local_indices)
        return np.concatenate(sampled).astype(np.int64, copy=False)

    def _bank_records_by_patient(self):
        bank_records = []
        for patient_id, recording_indices in self._records_by_patient:
            available = tuple(
                recording_index
                for recording_index in recording_indices
                if recording_index in self._hard_bank
                and self._hard_bank[recording_index][0].size > 0
            )
            if available:
                bank_records.append((patient_id, available))
        return tuple(bank_records)

    def _sample_hard_negatives(self, total, rng):
        if total <= 0:
            return np.empty(0, dtype=np.int64)
        bank_records = self._bank_records_by_patient()
        if not bank_records:
            return np.empty(0, dtype=np.int64)
        sampled = []
        patient_counts = _balanced_counts(total, len(bank_records), rng)
        for patient_count, (_patient_id, recording_indices) in zip(
            patient_counts, bank_records
        ):
            recording_counts = _balanced_counts(
                int(patient_count), len(recording_indices), rng
            )
            for sample_count, recording_index in zip(
                recording_counts, recording_indices
            ):
                if sample_count == 0:
                    continue
                bank_indices = self._hard_bank[recording_index][0]
                sampled.append(
                    rng.choice(
                        bank_indices,
                        size=int(sample_count),
                        replace=sample_count > bank_indices.size,
                    ).astype(np.int64, copy=False)
                )
        return np.concatenate(sampled).astype(np.int64, copy=False)

    def sampled_components_for_epoch(self, epoch):
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, int(epoch)]))
        requested_hard = (
            int(round(self.num_negative * self.replay_fraction))
            if self.hard_bank_size
            else 0
        )
        hard = self._sample_hard_negatives(requested_hard, rng)
        random_count = self.num_negative - int(hard.size)
        random = self._sample_random_negatives(random_count, rng)
        return {
            "positive": self.dataset.positive_indices.copy(),
            "random_negative": random,
            "hard_negative": hard,
        }

    def indices_for_epoch(self, epoch):
        components = self.sampled_components_for_epoch(epoch)
        indices = np.concatenate(tuple(components.values()))
        rng = np.random.default_rng(
            np.random.SeedSequence([self.seed, int(epoch), 0xA11CE])
        )
        rng.shuffle(indices)
        return indices

    def __iter__(self):
        epoch = self._next_epoch
        self._next_epoch += 1
        return iter(self.indices_for_epoch(epoch).tolist())

    def update_hard_negatives(self, dataset_indices, scores):
        """Merge observed train negatives and keep the top scores per recording."""
        dataset_indices = np.asarray(dataset_indices, dtype=np.int64)
        scores = np.asarray(scores, dtype=np.float64)
        if dataset_indices.ndim != 1 or scores.ndim != 1:
            raise ValueError("dataset_indices and scores must be one-dimensional.")
        if dataset_indices.size != scores.size:
            raise ValueError("dataset_indices and scores must have equal length.")
        if dataset_indices.size == 0:
            return
        if not np.isfinite(scores).all():
            raise ValueError("Hard-negative scores must be finite.")
        if np.any(dataset_indices < 0) or np.any(dataset_indices >= len(self.dataset)):
            raise ValueError("Hard-negative indices are outside the dataset.")
        if any(self.dataset.label_at(int(index)) for index in dataset_indices):
            raise ValueError("The hard-negative bank cannot contain positive windows.")

        offsets = np.asarray(self.dataset._offsets[1:], dtype=np.int64)
        recording_indices = np.searchsorted(
            offsets, dataset_indices, side="right"
        )
        for recording_index in np.unique(recording_indices):
            mask = recording_indices == recording_index
            candidate_indices = dataset_indices[mask]
            candidate_scores = scores[mask]
            existing = self._hard_bank.get(int(recording_index))
            if existing is not None:
                candidate_indices = np.concatenate([existing[0], candidate_indices])
                candidate_scores = np.concatenate([existing[1], candidate_scores])

            best_by_index: dict[int, float] = {}
            for dataset_index, score in zip(candidate_indices, candidate_scores):
                key = int(dataset_index)
                best_by_index[key] = max(best_by_index.get(key, -np.inf), float(score))
            ranked = sorted(
                best_by_index.items(), key=lambda item: (-item[1], item[0])
            )[: self.hard_per_recording]
            self._hard_bank[int(recording_index)] = (
                np.asarray([item[0] for item in ranked], dtype=np.int64),
                np.asarray([item[1] for item in ranked], dtype=np.float64),
            )

    def hard_bank_arrays(self):
        if not self._hard_bank:
            return (
                np.empty(0, dtype=np.int64),
                np.empty(0, dtype=np.float64),
            )
        indices = np.concatenate(
            [value[0] for _key, value in sorted(self._hard_bank.items())]
        )
        scores = np.concatenate(
            [value[1] for _key, value in sorted(self._hard_bank.items())]
        )
        return indices, scores


@dataclass(frozen=True)
class TrainingDataLoaders:
    train: DataLoader
    val: DataLoader
    train_sampler: HardNegativeReplaySampler

    def close(self):
        self.train.dataset.close()
        self.val.dataset.close()


def create_training_dataloaders(
    cache_root="data/processed/selected4",
    *,
    batch_size=256,
    positive_fraction=0.05,
    replay_fraction=0.50,
    hard_per_recording=256,
    num_workers=0,
    seed=0,
    pin_memory=None,
    persistent_workers=False,
    max_open_recordings=8,
):
    """Create train/validation loaders without constructing or reading test."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if num_workers < 0:
        raise ValueError("num_workers must be non-negative.")
    if persistent_workers and num_workers == 0:
        raise ValueError("persistent_workers requires num_workers > 0.")
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    train_base = EEGWindowDataset(
        cache_root=cache_root,
        split="train",
        return_metadata=False,
        max_open_recordings=max_open_recordings,
    )
    train_dataset = WeightedTrainingDataset(train_base)
    val_dataset = EEGWindowDataset(
        cache_root=cache_root,
        split="val",
        return_metadata=False,
        max_open_recordings=max_open_recordings,
    )
    sampler = HardNegativeReplaySampler(
        train_base,
        positive_fraction=positive_fraction,
        replay_fraction=replay_fraction,
        hard_per_recording=hard_per_recording,
        seed=seed,
    )
    common = {
        "batch_size": int(batch_size),
        "num_workers": int(num_workers),
        "pin_memory": bool(pin_memory),
        "persistent_workers": bool(persistent_workers),
        "drop_last": False,
    }
    return TrainingDataLoaders(
        train=DataLoader(train_dataset, sampler=sampler, **common),
        val=DataLoader(val_dataset, shuffle=False, **common),
        train_sampler=sampler,
    )
