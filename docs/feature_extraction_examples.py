"""
Quick reference for feature extraction usage.

Example 1: Extract features from a single window
-------------------------------------------------
"""
import numpy as np
from src.features import FeatureExtractor

# Initialize extractor
extractor = FeatureExtractor(sfreq=256, fz_cz_channel_idx=2)

# Load a window (shape: [3 channels, 1024 samples])
window = np.random.randn(3, 1024)  # Replace with real data

# Extract 28 features
features = extractor.extract(window)
print(f"Features shape: {features.shape}")  # (28,)
print(f"Feature names: {extractor.get_feature_names()[:5]}")


"""
Example 2: Batch extraction from windowed data
-----------------------------------------------
"""
# Load windowed data
data = np.load('data/processed/chb01_03_windows.npz')
windows = data['X']  # Shape: (1841, 3, 1024)
labels = data['y']   # Shape: (1841,)

# Extract features for all windows
features = extractor.extract_batch(windows, verbose=True, batch_size=100)
print(f"Feature matrix shape: {features.shape}")  # (1841, 28)


"""
Example 3: Load extracted features
-----------------------------------
"""
# Load pre-extracted features
feature_data = np.load('data/processed/chb01_03_features.npz', allow_pickle=True)

features = feature_data['features']        # Shape: (1841, 28)
labels = feature_data['labels']            # Shape: (1841,)
feature_names = feature_data['feature_names']  # Shape: (28,)

print(f"Loaded {len(features)} feature vectors")
print(f"Feature names:\n{feature_names}")


"""
Example 4: Access individual feature categories
------------------------------------------------
"""
# Time-domain features: indices 0-6 (7 features)
time_features = features[:, 0:7]

# Frequency-domain features: indices 7-21 (15 features)
freq_features = features[:, 7:22]

# Spectral ratio features: indices 22-27 (6 features)
ratio_features = features[:, 22:28]

print(f"Time-domain: {time_features.shape}")
print(f"Frequency-domain: {freq_features.shape}")
print(f"Spectral ratios: {ratio_features.shape}")


"""
Example 5: Filter features by channel
--------------------------------------
"""
# Get T8-P8 frequency features (indices 12-16: 5 bands)
t8p8_freq_features = features[:, 12:17]

# Get FZ-CZ frequency features (indices 17-21: 5 bands)
fzcz_freq_features = features[:, 17:22]

# Get all FZ-CZ features (time + freq + ratios)
# Time: 0-6, Freq: 17-21, Ratios: 22-27
fzcz_all_features = np.concatenate([
    features[:, 0:7],    # Time-domain
    features[:, 17:22],  # Frequency-domain
    features[:, 22:28],  # Spectral ratios
], axis=1)

print(f"T8-P8 frequency features: {t8p8_freq_features.shape}")
print(f"FZ-CZ all features: {fzcz_all_features.shape}")
