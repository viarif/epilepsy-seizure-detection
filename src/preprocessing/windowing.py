import numpy as np


def window_start_samples(
    n_samples,
    window_samples,
    hop_samples,
    min_start_sample=0,
):
    if window_samples <= 0 or hop_samples <= 0:
        raise ValueError('window_samples and hop_samples must be positive.')
    if min_start_sample < 0:
        raise ValueError('min_start_sample must be non-negative.')
    if n_samples < window_samples:
        return np.empty(0, dtype=np.int64)
    first_index = (int(min_start_sample) + hop_samples - 1) // hop_samples
    first_start = first_index * hop_samples
    last_start = n_samples - window_samples
    if first_start > last_start:
        return np.empty(0, dtype=np.int64)
    return np.arange(
        first_start,
        last_start + 1,
        hop_samples,
        dtype=np.int64,
    )


def label_window_starts(
    starts,
    sfreq,
    window_samples,
    seizure_intervals,
):
    """Label windows whose centers lie in any [seizure_start, seizure_end)."""
    starts = np.asarray(starts, dtype=np.int64)
    centers_sec = (starts + window_samples / 2.0) / float(sfreq)
    labels = np.zeros(starts.shape, dtype=np.uint8)
    for seizure_start, seizure_end in seizure_intervals:
        labels |= (
            (centers_sec >= seizure_start)
            & (centers_sec < seizure_end)
        )
    return labels
