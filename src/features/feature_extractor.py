"""
Main feature extraction interface.

Combines all feature extraction modules into a single interface.
Extracts 28 baseline features from multi-channel EEG windows.
"""

import numpy as np
from typing import List, Tuple, Optional
from pathlib import Path

from .time_domain import (
    extract_time_domain_features,
    TIME_DOMAIN_FEATURE_NAMES
)
from .frequency_domain import (
    extract_frequency_domain_features,
    FREQUENCY_FEATURE_NAMES
)
from .spectral_ratios import (
    compute_spectral_ratios,
    SPECTRAL_RATIO_FEATURE_NAMES
)


class FeatureExtractor:
    """
    Feature extractor for EEG seizure detection.

    Extracts 28 baseline features:
    - 7 time-domain features (FZ-CZ channel only)
    - 15 frequency-domain features (3 channels × 5 bands)
    - 6 spectral power ratio features (FZ-CZ channel only)
    """

    def __init__(self, sfreq=256, fz_cz_channel_idx=2):
        """
        Initialize feature extractor.

        Args:
            sfreq: int - sampling frequency in Hz
            fz_cz_channel_idx: int - index of FZ-CZ channel in window array
        """
        self.sfreq = sfreq
        self.fz_cz_channel_idx = fz_cz_channel_idx
        self.n_features = 28

        # Build feature names
        self.feature_names = self._build_feature_names()

    def _build_feature_names(self) -> List[str]:
        """
        Build list of all 28 feature names.

        Returns:
            feature_names: list of 28 strings
        """
        feature_names = []

        # Time-domain features (7) - FZ-CZ only
        feature_names.extend([f'FZ-CZ_{name}' for name in TIME_DOMAIN_FEATURE_NAMES])

        # Frequency-domain features (15) - all 3 channels
        feature_names.extend(FREQUENCY_FEATURE_NAMES)

        # Spectral ratio features (6) - FZ-CZ only
        feature_names.extend([f'FZ-CZ_{name}' for name in SPECTRAL_RATIO_FEATURE_NAMES])

        return feature_names

    def extract(self, window: np.ndarray) -> np.ndarray:
        """
        Extract 28 features from a single window.

        Args:
            window: numpy array [n_channels, n_samples] - typically [3, 1024]
                    Expected channel order: T7-P7, T8-P8, FZ-CZ

        Returns:
            features: numpy array [28] - feature vector
        """
        features = []

        # 1. Time-domain features (7) - FZ-CZ only
        fz_cz_signal = window[self.fz_cz_channel_idx]
        time_features = extract_time_domain_features(fz_cz_signal)
        features.extend(time_features)  # 7 features

        # 2. Frequency-domain features (15) - all 3 channels
        freq_features = extract_frequency_domain_features(window, self.sfreq)
        features.extend(freq_features)  # 15 features

        # 3. Spectral power ratios (6) - FZ-CZ only
        # Extract FZ-CZ band powers from frequency features
        # Channel 2 (FZ-CZ) corresponds to features 10-14 (5 bands)
        fz_cz_band_powers = freq_features[10:15]
        ratio_features = compute_spectral_ratios(fz_cz_band_powers)
        features.extend(ratio_features)  # 6 features

        return np.array(features, dtype=np.float32)

    def extract_batch(self, windows: np.ndarray,
                     verbose: bool = True,
                     batch_size: int = 100) -> np.ndarray:
        """
        Extract features from a batch of windows.

        Args:
            windows: numpy array [n_windows, n_channels, n_samples]
            verbose: bool - print progress messages
            batch_size: int - process in batches for memory efficiency

        Returns:
            features: numpy array [n_windows, 28]
        """
        n_windows = windows.shape[0]
        feature_matrix = np.zeros((n_windows, self.n_features), dtype=np.float32)

        if verbose:
            print(f"Extracting features from {n_windows} windows...")
            print(f"  Window shape: {windows.shape}")
            print(f"  Batch size: {batch_size}")

        # Process in batches to show progress
        n_batches = (n_windows + batch_size - 1) // batch_size

        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, n_windows)

            # Extract features for this batch
            for i in range(start_idx, end_idx):
                feature_matrix[i] = self.extract(windows[i])

            if verbose and (batch_idx + 1) % 5 == 0:
                progress = (end_idx / n_windows) * 100
                print(f"  Progress: {end_idx}/{n_windows} ({progress:.1f}%)")

        if verbose:
            print(f"  Completed: {n_windows}/{n_windows} (100.0%)")
            print(f"  Feature matrix shape: {feature_matrix.shape}")

        return feature_matrix

    def get_feature_names(self) -> List[str]:
        """Get list of all feature names."""
        return self.feature_names.copy()

    def save_feature_info(self, output_path: Path):
        """
        Save feature information to a text file.

        Args:
            output_path: Path to output file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("# Feature Extraction Configuration\n\n")
            f.write(f"Total features: {self.n_features}\n")
            f.write(f"Sampling frequency: {self.sfreq} Hz\n")
            f.write(f"FZ-CZ channel index: {self.fz_cz_channel_idx}\n\n")

            f.write("## Feature Breakdown\n\n")
            f.write("### Time-domain features (7) - FZ-CZ only:\n")
            for i, name in enumerate(TIME_DOMAIN_FEATURE_NAMES, 1):
                f.write(f"{i}. {name}\n")

            f.write("\n### Frequency-domain features (15) - 3 channels × 5 bands:\n")
            for i, name in enumerate(FREQUENCY_FEATURE_NAMES, 1):
                f.write(f"{i}. {name}\n")

            f.write("\n### Spectral ratio features (6) - FZ-CZ only:\n")
            for i, name in enumerate(SPECTRAL_RATIO_FEATURE_NAMES, 1):
                f.write(f"{i}. {name}\n")

            f.write("\n## Complete Feature List (28 total)\n\n")
            for i, name in enumerate(self.feature_names, 1):
                f.write(f"{i}. {name}\n")


def extract_features_batch(windows: np.ndarray,
                          labels: Optional[np.ndarray] = None,
                          sfreq: int = 256,
                          fz_cz_channel_idx: int = 2,
                          verbose: bool = True) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Convenience function to extract features from a batch of windows.

    Args:
        windows: numpy array [n_windows, n_channels, n_samples]
        labels: optional numpy array [n_windows] - seizure labels
        sfreq: int - sampling frequency
        fz_cz_channel_idx: int - index of FZ-CZ channel
        verbose: bool - print progress

    Returns:
        features: numpy array [n_windows, 28]
        labels: numpy array [n_windows] or None
    """
    extractor = FeatureExtractor(sfreq=sfreq, fz_cz_channel_idx=fz_cz_channel_idx)
    features = extractor.extract_batch(windows, verbose=verbose)

    return features, labels
