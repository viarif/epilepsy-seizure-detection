"""
Spectral power ratio feature extraction.

Implements 6 spectral power ratio features based on Paper 3:
- δ/θ (Delta-Theta Ratio)
- θ/β (Theta-Beta Ratio)
- (δ+θ)/β (Slow/Fast Ratio)
- β/γ_low (Beta-Gamma Ratio)
- δ/Broadband (Delta relative power)
- γ_low/Broadband (Gamma relative power)

All ratios computed from FZ-CZ channel only (most stable for ratios).
Uses log-scale power, so ratios become simple subtractions.
"""

import numpy as np


def compute_spectral_ratios(fz_cz_band_powers):
    """
    Compute 6 spectral power ratio features from frequency band powers.

    Args:
        fz_cz_band_powers: list or numpy array of 5 values
                           Order: [delta, theta, beta, gamma_low, broadband]
                           Values should be in log scale (log10 of power)

    Returns:
        ratios: list of 6 float values
    """
    # Extract individual band powers (in log scale)
    delta = fz_cz_band_powers[0]
    theta = fz_cz_band_powers[1]
    beta = fz_cz_band_powers[2]
    gamma_low = fz_cz_band_powers[3]
    broadband = fz_cz_band_powers[4]

    ratios = []

    # 1. δ/θ (Delta-Theta Ratio)
    # In log scale: log(δ/θ) = log(δ) - log(θ)
    delta_theta_ratio = delta - theta
    ratios.append(delta_theta_ratio)

    # 2. θ/β (Theta-Beta Ratio)
    theta_beta_ratio = theta - beta
    ratios.append(theta_beta_ratio)

    # 3. (δ+θ)/β (Slow/Fast Ratio)
    # In log scale: log((δ+θ)/β) = log(δ+θ) - log(β)
    # Need to convert back from log scale for addition
    delta_linear = 10 ** delta
    theta_linear = 10 ** theta
    slow_sum = delta_linear + theta_linear
    slow_fast_ratio = np.log10(slow_sum + 1e-12) - beta
    ratios.append(slow_fast_ratio)

    # 4. β/γ_low (Beta-Gamma Ratio)
    beta_gamma_ratio = beta - gamma_low
    ratios.append(beta_gamma_ratio)

    # 5. δ/Broadband (Delta relative power)
    delta_relative = delta - broadband
    ratios.append(delta_relative)

    # 6. γ_low/Broadband (Gamma relative power)
    gamma_relative = gamma_low - broadband
    ratios.append(gamma_relative)

    return ratios


def compute_spectral_ratios_from_window(window, sfreq=256):
    """
    Compute spectral ratios directly from a multi-channel window.

    Convenience function that extracts frequency features and computes ratios
    for the FZ-CZ channel (assumed to be channel index 2).

    Args:
        window: numpy array [n_channels, n_samples]
        sfreq: int - sampling frequency

    Returns:
        ratios: list of 6 float values
    """
    from .frequency_domain import extract_frequency_domain_features

    # Extract frequency features (15 total: 3 channels × 5 bands)
    freq_features = extract_frequency_domain_features(window, sfreq)

    # Get FZ-CZ band powers (channel index 2 → features 10-14)
    # Channel 0: features 0-4, Channel 1: features 5-9, Channel 2: features 10-14
    fz_cz_band_powers = freq_features[10:15]

    # Compute ratios
    ratios = compute_spectral_ratios(fz_cz_band_powers)

    return ratios


# Feature names for debugging and interpretation
SPECTRAL_RATIO_FEATURE_NAMES = [
    'delta_theta_ratio',      # δ/θ
    'theta_beta_ratio',       # θ/β
    'slow_fast_ratio',        # (δ+θ)/β
    'beta_gamma_ratio',       # β/γ_low
    'delta_relative_power',   # δ/Broadband
    'gamma_relative_power',   # γ_low/Broadband
]
