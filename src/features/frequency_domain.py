"""
Frequency-domain feature extraction.

Implements 15 frequency-domain features based on Paper 2:
- 5 frequency bands: δ(4-8Hz), θ(8-13Hz), β(13-30Hz), γ_low(30-50Hz), Broadband(4-50Hz)
- 3 channels: T7-P7, T8-P8, FZ-CZ
- Total: 3 channels × 5 bands = 15 features

Uses FFT-based power spectral density estimation.
"""

import numpy as np
from scipy import signal as scipy_signal


# Frequency band definitions (in Hz)
FREQ_BANDS = {
    'delta': (4, 8),      # Deep sleep, brain injury
    'theta': (8, 13),     # Drowsiness, emotion
    'beta': (13, 30),     # Awake, anxiety
    'gamma_low': (30, 50), # Cognition, seizure feature (Paper 2 key finding)
    'broadband': (4, 50),  # Total power for normalization
}

# Order for feature extraction (matches memory plan)
BAND_ORDER = ['delta', 'theta', 'beta', 'gamma_low', 'broadband']


def extract_frequency_domain_features(window, sfreq=256):
    """
    Extract frequency-domain features from a multi-channel window.

    Args:
        window: numpy array [n_channels, n_samples] - typically [3, 1024]
        sfreq: int - sampling frequency (default 256 Hz)

    Returns:
        features: list of 15 float values (3 channels × 5 bands)
                  Order: [ch0_delta, ch0_theta, ..., ch0_broadband,
                          ch1_delta, ch1_theta, ..., ch1_broadband,
                          ch2_delta, ch2_theta, ..., ch2_broadband]
    """
    n_channels = window.shape[0]
    features = []

    for ch_idx in range(n_channels):
        # Compute power spectral density for this channel
        freqs, psd = compute_psd(window[ch_idx], sfreq)

        # Extract power in each frequency band
        for band_name in BAND_ORDER:
            freq_min, freq_max = FREQ_BANDS[band_name]
            band_power = compute_band_power(freqs, psd, freq_min, freq_max)
            features.append(band_power)

    return features


def compute_psd(signal, sfreq=256):
    """
    Compute Power Spectral Density using Welch's method.

    Welch's method reduces noise by averaging over multiple overlapping segments.

    Args:
        signal: numpy array [n_samples]
        sfreq: int - sampling frequency

    Returns:
        freqs: numpy array - frequency values in Hz
        psd: numpy array - power spectral density values
    """
    # Use Welch's method for robust PSD estimation
    # nperseg: length of each segment (512 samples = 2 seconds at 256 Hz)
    # noverlap: overlap between segments (50%)
    nperseg = min(512, len(signal))
    noverlap = nperseg // 2

    freqs, psd = scipy_signal.welch(
        signal,
        fs=sfreq,
        nperseg=nperseg,
        noverlap=noverlap,
        scaling='density'
    )

    return freqs, psd


def compute_band_power(freqs, psd, freq_min, freq_max):
    """
    Compute total power in a specific frequency band.

    Uses trapezoidal integration over the frequency range.
    Returns log-scale power for better numerical stability.

    Args:
        freqs: numpy array - frequency values
        psd: numpy array - power spectral density values
        freq_min: float - lower bound of frequency band (Hz)
        freq_max: float - upper bound of frequency band (Hz)

    Returns:
        band_power: float - log10 of integrated power in the band
    """
    # Find indices corresponding to the frequency band
    band_mask = (freqs >= freq_min) & (freqs <= freq_max)

    if not np.any(band_mask):
        # Band not found in frequency range
        return 0.0

    # Integrate power over the frequency band using trapezoidal rule
    band_freqs = freqs[band_mask]
    band_psd = psd[band_mask]

    # Sum power (trapezoidal integration)
    # Use trapezoid (numpy >= 2.0) or trapz (numpy < 2.0)
    try:
        power = np.trapezoid(band_psd, band_freqs)
    except AttributeError:
        power = np.trapz(band_psd, band_freqs)

    # Return log-scale power for better numerical properties
    # Add small epsilon to avoid log(0)
    log_power = np.log10(power + 1e-12)

    return log_power


def get_frequency_feature_names(channel_names=None):
    """
    Generate feature names for frequency-domain features.

    Args:
        channel_names: list of str - channel names (default: ['T7-P7', 'T8-P8', 'FZ-CZ'])

    Returns:
        feature_names: list of str - 15 feature names
    """
    if channel_names is None:
        channel_names = ['T7-P7', 'T8-P8', 'FZ-CZ']

    feature_names = []
    for ch_name in channel_names:
        for band_name in BAND_ORDER:
            feature_names.append(f'{ch_name}_{band_name}_power')

    return feature_names


# Feature names for debugging (3 channels × 5 bands)
FREQUENCY_FEATURE_NAMES = get_frequency_feature_names()
