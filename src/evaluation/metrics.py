"""Threshold selection and reporting for the seizure-window classifier."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def _as_1d(values, *, dtype=None, name="values") -> np.ndarray:
    array = np.asarray(values, dtype=dtype)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got {array.shape}.")
    return array


def _validate_scores_and_labels(scores, labels):
    scores = _as_1d(scores, dtype=np.float64, name="scores")
    labels = _as_1d(labels, dtype=np.int8, name="labels")
    if scores.size != labels.size:
        raise ValueError("scores and labels must have the same length.")
    if scores.size == 0:
        raise ValueError("scores and labels cannot be empty.")
    if not np.isfinite(scores).all():
        raise ValueError("scores contain NaN or infinite values.")
    if not np.isin(labels, (0, 1)).all():
        raise ValueError("labels must contain only 0 and 1.")
    if labels.sum() == 0 or labels.sum() == labels.size:
        raise ValueError("both positive and negative labels are required.")
    return scores, labels


def select_threshold_at_specificity(
    scores,
    labels,
    target_specificity=0.97,
):
    """Select the most sensitive threshold satisfying a specificity floor.

    Scores are raw logits.  Predictions use ``score >= threshold``.  Among
    equal-sensitivity candidates, the least conservative threshold that still
    satisfies the specificity floor is selected.  This avoids accidentally
    choosing an all-negative operating point when a finite threshold is tied.
    """
    if not 0.0 < target_specificity <= 1.0:
        raise ValueError("target_specificity must be in (0, 1].")
    scores, labels = _validate_scores_and_labels(scores, labels)

    order = np.argsort(-scores, kind="stable")
    sorted_scores = scores[order]
    sorted_labels = labels[order]
    cumulative_tp = np.cumsum(sorted_labels, dtype=np.int64)
    cumulative_fp = np.cumsum(1 - sorted_labels, dtype=np.int64)

    # A score tie must be included as a whole because prediction uses >=.
    distinct_ends = np.flatnonzero(
        np.r_[sorted_scores[1:] != sorted_scores[:-1], True]
    )
    thresholds = np.r_[np.inf, sorted_scores[distinct_ends]]
    true_positives = np.r_[0, cumulative_tp[distinct_ends]]
    false_positives = np.r_[0, cumulative_fp[distinct_ends]]

    positive_count = int(labels.sum())
    negative_count = int(labels.size - positive_count)
    sensitivities = true_positives / positive_count
    specificities = 1.0 - false_positives / negative_count
    feasible = specificities >= float(target_specificity) - 1e-12
    best_sensitivity = sensitivities[feasible].max()
    candidates = np.flatnonzero(
        feasible & np.isclose(sensitivities, best_sensitivity, atol=1e-12)
    )
    best_specificity = specificities[candidates].min()
    candidates = candidates[
        np.isclose(specificities[candidates], best_specificity, atol=1e-12)
    ]
    index = int(candidates[np.argmin(thresholds[candidates])])
    threshold = float(thresholds[index])
    threshold_probability = (
        1.0 if np.isposinf(threshold)
        else float(1.0 / (1.0 + np.exp(-np.clip(threshold, -50.0, 50.0))))
    )
    return {
        "target_specificity": float(target_specificity),
        "threshold_logit": threshold,
        "threshold_probability": threshold_probability,
        "sensitivity": float(sensitivities[index]),
        "specificity": float(specificities[index]),
        "true_positives": int(true_positives[index]),
        "false_positives": int(false_positives[index]),
        "positive_count": positive_count,
        "negative_count": negative_count,
    }


def compute_binary_metrics(scores, labels, threshold_logit):
    scores, labels = _validate_scores_and_labels(scores, labels)
    predictions = scores >= float(threshold_logit)
    positive = labels == 1
    negative = ~positive
    true_positive = int(np.count_nonzero(predictions & positive))
    false_positive = int(np.count_nonzero(predictions & negative))
    true_negative = int(np.count_nonzero(~predictions & negative))
    false_negative = int(np.count_nonzero(~predictions & positive))

    sensitivity = true_positive / (true_positive + false_negative)
    specificity = true_negative / (true_negative + false_positive)
    precision_denominator = true_positive + false_positive
    precision = (
        true_positive / precision_denominator
        if precision_denominator
        else 0.0
    )
    f1_denominator = 2 * true_positive + false_positive + false_negative
    f1 = 2 * true_positive / f1_denominator if f1_denominator else 0.0
    threshold_probability = (
        1.0 if np.isposinf(threshold_logit)
        else float(
            1.0
            / (1.0 + np.exp(-np.clip(float(threshold_logit), -50.0, 50.0)))
        )
    )
    return {
        "threshold_logit": float(threshold_logit),
        "threshold_probability": threshold_probability,
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "precision": float(precision),
        "f1": float(f1),
        "accuracy": float((true_positive + true_negative) / labels.size),
        "balanced_accuracy": float((sensitivity + specificity) / 2.0),
        "pr_auc": float(average_precision_score(labels, scores)),
        "roc_auc": float(roc_auc_score(labels, scores)),
        "true_positives": true_positive,
        "false_positives": false_positive,
        "true_negatives": true_negative,
        "false_negatives": false_negative,
        "positive_count": int(positive.sum()),
        "negative_count": int(negative.sum()),
        "window_count": int(labels.size),
    }


def collect_dataset_metadata(dataset):
    """Return metadata arrays in the deterministic dataset iteration order."""
    patient_ids = np.empty(len(dataset), dtype=object)
    recording_ids = np.empty(len(dataset), dtype=object)
    window_indices = np.empty(len(dataset), dtype=np.int64)
    start_samples = np.empty(len(dataset), dtype=np.int64)
    cursor = 0
    for record in dataset._records:
        stop = cursor + record.window_count
        patient_ids[cursor:stop] = record.patient_id
        recording_ids[cursor:stop] = record.recording_id
        local_indices = np.arange(record.window_count, dtype=np.int64)
        window_indices[cursor:stop] = local_indices
        start_samples[cursor:stop] = (
            int(record.index.metadata["first_retained_window_start_sample"])
            + local_indices * int(record.index.metadata["hop_samples"])
        )
        cursor = stop
    if cursor != len(dataset):
        raise RuntimeError("Dataset metadata length does not match dataset length.")
    return {
        "patient_ids": patient_ids,
        "recording_ids": recording_ids,
        "window_indices": window_indices,
        "start_samples": start_samples,
    }


def _safe_group_metrics(scores, labels, threshold):
    """Compute count metrics for a group that may contain one class only."""
    predictions = scores >= threshold
    positive = labels == 1
    negative = ~positive
    tp = int(np.count_nonzero(predictions & positive))
    fp = int(np.count_nonzero(predictions & negative))
    tn = int(np.count_nonzero(~predictions & negative))
    fn = int(np.count_nonzero(~predictions & positive))
    positive_count = tp + fn
    negative_count = tn + fp
    return {
        "sensitivity": float(tp / positive_count) if positive_count else None,
        "specificity": float(tn / negative_count) if negative_count else None,
        "true_positives": tp,
        "false_positives": fp,
        "true_negatives": tn,
        "false_negatives": fn,
        "positive_count": positive_count,
        "negative_count": negative_count,
        "window_count": int(labels.size),
    }


def _run_statistics(
    predictions,
    labels,
    recording_ids,
    window_indices,
    *,
    hop_sec,
    window_sec,
):
    groups = defaultdict(list)
    for index, recording_id in enumerate(recording_ids):
        groups[str(recording_id)].append(index)

    false_alarms = 0
    seizure_bouts = 0
    detected_seizure_bouts = 0
    evaluated_seconds = 0.0
    for indices in groups.values():
        indices = np.asarray(indices, dtype=np.int64)
        order = np.argsort(window_indices[indices], kind="stable")
        indices = indices[order]
        local_windows = window_indices[indices]
        local_predictions = predictions[indices]
        local_labels = labels[indices]
        if local_windows.size:
            contiguous_span = int(local_windows[-1] - local_windows[0])
            evaluated_seconds += contiguous_span * hop_sec + window_sec

        prediction_starts = np.flatnonzero(
            local_predictions
            & np.r_[True, (~local_predictions[:-1]) | (np.diff(local_windows) != 1)]
        )
        for start in prediction_starts:
            stop = start + 1
            while (
                stop < local_predictions.size
                and local_predictions[stop]
                and local_windows[stop] == local_windows[stop - 1] + 1
            ):
                stop += 1
            if not local_labels[start:stop].any():
                false_alarms += 1

        seizure_starts = np.flatnonzero(
            local_labels
            & np.r_[True, (~local_labels[:-1]) | (np.diff(local_windows) != 1)]
        )
        for start in seizure_starts:
            stop = start + 1
            while (
                stop < local_labels.size
                and local_labels[stop]
                and local_windows[stop] == local_windows[stop - 1] + 1
            ):
                stop += 1
            seizure_bouts += 1
            detected_seizure_bouts += int(local_predictions[start:stop].any())

    evaluated_hours = evaluated_seconds / 3600.0
    return {
        "false_alarms": int(false_alarms),
        "evaluated_hours": float(evaluated_hours),
        "false_alarms_per_hour": (
            float(false_alarms / evaluated_hours) if evaluated_hours else None
        ),
        "seizure_bouts": int(seizure_bouts),
        "detected_seizure_bouts": int(detected_seizure_bouts),
        "seizure_bout_sensitivity": (
            float(detected_seizure_bouts / seizure_bouts)
            if seizure_bouts
            else None
        ),
        "definition": (
            "A false alarm is a contiguous predicted-positive run within one "
            "recording that contains no positive-labeled window. A seizure bout "
            "is a contiguous run of positive-labeled windows."
        ),
    }


def evaluate_predictions(
    scores,
    labels,
    threshold_logit,
    *,
    patient_ids=None,
    recording_ids=None,
    window_indices=None,
    hop_sec=0.5,
    window_sec=1.0,
):
    """Compute global, per-patient, and optional event/run-level metrics."""
    scores, labels = _validate_scores_and_labels(scores, labels)
    report = compute_binary_metrics(scores, labels, threshold_logit)

    if patient_ids is not None:
        patient_ids = _as_1d(patient_ids, name="patient_ids")
        if patient_ids.size != labels.size:
            raise ValueError("patient_ids length does not match labels.")
        per_patient = {}
        for patient_id in sorted({str(value) for value in patient_ids}):
            mask = patient_ids.astype(str) == patient_id
            per_patient[patient_id] = _safe_group_metrics(
                scores[mask], labels[mask], float(threshold_logit)
            )
        report["per_patient"] = per_patient

    supplied_run_metadata = (
        recording_ids is not None or window_indices is not None
    )
    if supplied_run_metadata:
        if recording_ids is None or window_indices is None:
            raise ValueError(
                "recording_ids and window_indices must be supplied together."
            )
        recording_ids = _as_1d(recording_ids, name="recording_ids")
        window_indices = _as_1d(
            window_indices, dtype=np.int64, name="window_indices"
        )
        if recording_ids.size != labels.size or window_indices.size != labels.size:
            raise ValueError("run metadata length does not match labels.")
        report["event_metrics"] = _run_statistics(
            scores >= float(threshold_logit),
            labels,
            recording_ids,
            window_indices,
            hop_sec=float(hop_sec),
            window_sec=float(window_sec),
        )
    return report
