import numpy as np
from scipy.signal import butter, iirnotch, lfilter, lfilter_zi, sosfilt, sosfilt_zi

from .config import PreprocessingConfig


def approx_tanh(values, divisor=1.2):
    """Hardware-friendly tanh approximation fixed by README.md."""
    return np.clip(np.asarray(values) / divisor, -1.0, 1.0)


def _sosfilt_rows(data, sos):
    output = np.empty_like(data, dtype=np.float64)
    base_zi = sosfilt_zi(sos)
    for row_index, row in enumerate(data):
        zi = base_zi * float(row[0])
        output[row_index], _ = sosfilt(sos, row, zi=zi)
    return output


def _lfilter_rows(data, numerator, denominator):
    output = np.empty_like(data, dtype=np.float64)
    base_zi = lfilter_zi(numerator, denominator)
    for row_index, row in enumerate(data):
        zi = base_zi * float(row[0])
        output[row_index], _ = lfilter(
            numerator,
            denominator,
            row,
            zi=zi,
        )
    return output


def filter_continuous(data, sfreq, config=None):
    """Apply causal 0.1 Hz high-pass and causal 60 Hz notch filters."""
    config = config or PreprocessingConfig()
    config.validate_sfreq(sfreq)
    values = np.asarray(data, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] == 0:
        raise ValueError('Expected non-empty [channels, samples] EEG array.')

    highpass_sos = butter(
        config.highpass_order,
        config.highpass_hz,
        btype='highpass',
        fs=sfreq,
        output='sos',
    )
    filtered = _sosfilt_rows(values, highpass_sos)

    notch_b, notch_a = iirnotch(
        config.notch_hz,
        config.notch_q,
        fs=sfreq,
    )
    return _lfilter_rows(filtered, notch_b, notch_a)


def _causal_rolling_std_1d(row, window_samples):
    row = np.asarray(row, dtype=np.float64)
    n_samples = row.size
    window_samples = min(int(window_samples), n_samples)
    expanding_count = np.arange(1, window_samples, dtype=np.float64)

    cumulative = np.empty(n_samples + 1, dtype=np.float64)
    cumulative[0] = 0.0
    np.cumsum(row, dtype=np.float64, out=cumulative[1:])
    rolling_sum = cumulative[1:].copy()
    if n_samples > window_samples:
        rolling_sum[window_samples:] -= cumulative[1:-window_samples]

    cumulative_sq = np.empty(n_samples + 1, dtype=np.float64)
    cumulative_sq[0] = 0.0
    np.cumsum(row * row, dtype=np.float64, out=cumulative_sq[1:])
    rolling_sq_sum = cumulative_sq[1:].copy()
    if n_samples > window_samples:
        rolling_sq_sum[window_samples:] -= cumulative_sq[1:-window_samples]

    means = rolling_sum
    mean_squares = rolling_sq_sum
    if window_samples > 1:
        means[:window_samples - 1] /= expanding_count
        mean_squares[:window_samples - 1] /= expanding_count
    means[window_samples - 1:] /= window_samples
    mean_squares[window_samples - 1:] /= window_samples

    variances = mean_squares - means * means
    np.maximum(variances, 0.0, out=variances)
    np.sqrt(variances, out=variances)
    return variances


def causal_rolling_std(data, window_samples):
    """Population std over [max(0, t-window+1), t] for every sample.

    The first ten minutes use an expanding causal history.  No future samples
    and no neighboring EDF recordings are used.
    """
    values = np.asarray(data, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] == 0:
        raise ValueError('Expected non-empty [channels, samples] array.')
    if window_samples <= 0:
        raise ValueError('window_samples must be positive.')

    output = np.empty(values.shape, dtype=np.float64)
    for row_index, row in enumerate(values):
        output[row_index] = _causal_rolling_std_1d(row, window_samples)

    return output


def preprocess_continuous(data, sfreq, config=None):
    """Run the shared continuous transform before any window extraction."""
    config = config or PreprocessingConfig()
    filtered = filter_continuous(data, sfreq, config)
    rolling_samples = int(round(config.rolling_std_sec * sfreq))
    transformed = np.empty(filtered.shape, dtype=np.float32)
    for row_index, row in enumerate(filtered):
        rolling_std = _causal_rolling_std_1d(row, rolling_samples)
        denominator = np.maximum(rolling_std, config.epsilon_volts)
        normalized = config.scale * row / denominator
        transformed[row_index] = approx_tanh(
            normalized,
            config.tanh_divisor,
        )
    return transformed
