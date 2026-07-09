"""
Time-domain feature extraction.

Implements 7 time-domain features based on Paper 1:
- Standard Deviation
- Skewness
- Hjorth Activity
- Hjorth Mobility
- Hjorth Complexity
- Peak-to-Peak Amplitude
- Zero Crossing Rate
"""

import numpy as np
from scipy import stats


def extract_time_domain_features(signal):
    """
    Extract 7 time-domain features from a single-channel signal.

    Args:
        signal: numpy array [n_samples] - single channel EEG data

    Returns:
        features: list of 7 float values
    """
    features = []

    # 1. Standard Deviation
    std = np.std(signal)
    features.append(std)

    # 2. Skewness (measure of asymmetry)
    # Handle edge case: constant signal returns NaN from scipy.stats.skew
    if np.std(signal) < 1e-10:
        skewness = 0.0
    else:
        skewness = stats.skew(signal)
    features.append(skewness)

    # 3-5. Hjorth parameters
    hjorth_activity, hjorth_mobility, hjorth_complexity = compute_hjorth_parameters(signal)
    features.append(hjorth_activity)
    features.append(hjorth_mobility)
    features.append(hjorth_complexity)

    # 6. Peak-to-Peak Amplitude
    ptp = np.ptp(signal)  # max - min
    features.append(ptp)

    # 7. Zero Crossing Rate
    zcr = compute_zero_crossing_rate(signal)
    features.append(zcr)

    return features


def compute_hjorth_parameters(signal):
    """
    Compute Hjorth parameters: Activity, Mobility, and Complexity.

    These parameters characterize statistical properties of the signal:
    - Activity: variance of the signal (power)
    - Mobility: mean frequency (related to standard deviation of slope)
    - Complexity: change in frequency (deviation from pure sinusoid)

    Args:
        signal: numpy array [n_samples]

    Returns:
        activity: float - variance of signal
        mobility: float - square root of (variance of 1st derivative / variance of signal)
        complexity: float - mobility of 1st derivative / mobility of signal

    Reference: Paper 1 - Most informative time-domain features for seizure detection
    """
    # First derivative (approximated by diff)
    first_deriv = np.diff(signal)

    # Second derivative
    second_deriv = np.diff(first_deriv)

    # Activity: variance of signal
    activity = np.var(signal)

    # Mobility: sqrt(var(1st derivative) / var(signal))
    var_first_deriv = np.var(first_deriv)
    mobility = np.sqrt(var_first_deriv / activity) if activity > 0 else 0.0

    # Complexity: mobility(1st derivative) / mobility(signal)
    var_second_deriv = np.var(second_deriv)
    mobility_deriv = np.sqrt(var_second_deriv / var_first_deriv) if var_first_deriv > 0 else 0.0
    complexity = mobility_deriv / mobility if mobility > 0 else 0.0

    return activity, mobility, complexity


def compute_zero_crossing_rate(signal):
    """
    Compute Zero Crossing Rate - number of times signal crosses zero.

    Normalized by signal length. High ZCR indicates high-frequency components.

    Args:
        signal: numpy array [n_samples]

    Returns:
        zcr: float - zero crossing rate (0 to 1)
    """
    # Sign changes indicate zero crossings
    sign_changes = np.diff(np.sign(signal))
    zero_crossings = np.sum(np.abs(sign_changes)) / 2  # Each crossing gives ±2

    # Normalize by signal length
    zcr = zero_crossings / len(signal)

    return zcr


# Feature names for debugging and interpretation
TIME_DOMAIN_FEATURE_NAMES = [
    'std',
    'skewness',
    'hjorth_activity',
    'hjorth_mobility',
    'hjorth_complexity',
    'peak_to_peak',
    'zero_crossing_rate',
]
