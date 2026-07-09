"""
Feature extraction module for EEG seizure detection.

This module implements 28 baseline features:
- 7 time-domain features (FZ-CZ channel only)
- 15 frequency-domain features (3 channels × 5 bands)
- 6 spectral power ratio features (FZ-CZ channel only)
"""

from .time_domain import extract_time_domain_features
from .frequency_domain import extract_frequency_domain_features
from .spectral_ratios import compute_spectral_ratios
from .feature_extractor import FeatureExtractor, extract_features_batch

__all__ = [
    'extract_time_domain_features',
    'extract_frequency_domain_features',
    'compute_spectral_ratios',
    'FeatureExtractor',
    'extract_features_batch',
]
