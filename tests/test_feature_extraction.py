"""
Unit tests for feature extraction modules.

Run with: python tests/test_feature_extraction.py
"""

import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features import (
    extract_time_domain_features,
    extract_frequency_domain_features,
    compute_spectral_ratios,
    FeatureExtractor
)


def test_time_domain_features():
    """Test time-domain feature extraction."""
    print("Testing time-domain features...")

    # Create synthetic signal
    signal = np.random.randn(1024)

    # Extract features
    features = extract_time_domain_features(signal)

    # Validate
    assert len(features) == 7, f"Expected 7 features, got {len(features)}"
    assert all(np.isfinite(f) for f in features), "Features contain NaN or Inf"

    print("  [OK] 7 features extracted")
    print("  [OK] No NaN or Inf values")
    print(f"  Feature range: [{min(features):.3f}, {max(features):.3f}]")


def test_frequency_domain_features():
    """Test frequency-domain feature extraction."""
    print("\nTesting frequency-domain features...")

    # Create synthetic multi-channel window
    window = np.random.randn(3, 1024)

    # Extract features
    features = extract_frequency_domain_features(window, sfreq=256)

    # Validate
    assert len(features) == 15, f"Expected 15 features, got {len(features)}"
    assert all(np.isfinite(f) for f in features), "Features contain NaN or Inf"

    print("  [OK] 15 features extracted (3 channels x 5 bands)")
    print("  [OK] No NaN or Inf values")
    print(f"  Feature range: [{min(features):.3f}, {max(features):.3f}]")


def test_spectral_ratios():
    """Test spectral ratio computation."""
    print("\nTesting spectral ratio features...")

    # Create synthetic band powers (in log scale)
    band_powers = [-6.0, -5.5, -5.0, -4.5, -4.0]  # delta, theta, beta, gamma_low, broadband

    # Compute ratios
    ratios = compute_spectral_ratios(band_powers)

    # Validate
    assert len(ratios) == 6, f"Expected 6 ratios, got {len(ratios)}"
    assert all(np.isfinite(r) for r in ratios), "Ratios contain NaN or Inf"

    print("  [OK] 6 spectral ratios computed")
    print("  [OK] No NaN or Inf values")
    print(f"  Ratio range: [{min(ratios):.3f}, {max(ratios):.3f}]")


def test_feature_extractor():
    """Test full feature extraction pipeline."""
    print("\nTesting FeatureExtractor class...")

    # Initialize extractor
    extractor = FeatureExtractor(sfreq=256, fz_cz_channel_idx=2)

    # Test single window extraction
    window = np.random.randn(3, 1024)
    features = extractor.extract(window)

    # Validate
    assert features.shape == (28,), f"Expected shape (28,), got {features.shape}"
    assert features.dtype == np.float32, f"Expected dtype float32, got {features.dtype}"
    assert not np.any(np.isnan(features)), "Features contain NaN"
    assert not np.any(np.isinf(features)), "Features contain Inf"

    print("  [OK] Single window extraction works")
    print(f"  Feature shape: {features.shape}")
    print(f"  Feature dtype: {features.dtype}")

    # Test batch extraction
    windows = np.random.randn(10, 3, 1024)
    feature_matrix = extractor.extract_batch(windows, verbose=False)

    # Validate
    assert feature_matrix.shape == (10, 28), f"Expected shape (10, 28), got {feature_matrix.shape}"
    assert not np.any(np.isnan(feature_matrix)), "Feature matrix contains NaN"
    assert not np.any(np.isinf(feature_matrix)), "Feature matrix contains Inf"

    print("  [OK] Batch extraction works")
    print(f"  Feature matrix shape: {feature_matrix.shape}")


def test_feature_names():
    """Test feature name generation."""
    print("\nTesting feature names...")

    extractor = FeatureExtractor(sfreq=256, fz_cz_channel_idx=2)
    feature_names = extractor.get_feature_names()

    # Validate
    assert len(feature_names) == 28, f"Expected 28 names, got {len(feature_names)}"
    assert len(set(feature_names)) == 28, "Feature names not unique"

    print("  [OK] 28 feature names generated")
    print("  [OK] All names are unique")
    print(f"  First 5 names: {feature_names[:5]}")
    print(f"  Last 5 names: {feature_names[-5:]}")


def test_real_data_extraction():
    """Test extraction on real data if available."""
    print("\nTesting on real data...")

    # Try to load real data
    data_file = Path(__file__).parent.parent / 'data' / 'processed' / 'chb01_03_windows.npz'

    if not data_file.exists():
        print("  [WARN] Real data not found, skipping test")
        return

    # Load data
    data = np.load(data_file)
    windows = data['X']
    labels = data['y']

    print(f"  Loaded {len(windows)} windows")

    # Extract features from first 10 windows
    extractor = FeatureExtractor(sfreq=256, fz_cz_channel_idx=2)
    features = extractor.extract_batch(windows[:10], verbose=False)

    # Validate
    assert features.shape == (10, 28), f"Expected shape (10, 28), got {features.shape}"
    assert not np.any(np.isnan(features)), "Features contain NaN"
    assert not np.any(np.isinf(features)), "Features contain Inf"

    print("  [OK] Real data extraction successful")
    print(f"  Feature range: [{features.min():.3f}, {features.max():.3f}]")
    print(f"  Feature mean: {features.mean():.3f}")
    print(f"  Feature std: {features.std():.3f}")


def test_edge_cases():
    """Test edge cases and robustness."""
    print("\nTesting edge cases...")

    extractor = FeatureExtractor(sfreq=256, fz_cz_channel_idx=2)

    # Test 1: Zero signal
    window_zeros = np.zeros((3, 1024))
    features_zeros = extractor.extract(window_zeros)
    assert np.all(np.isfinite(features_zeros)), "Zero signal produces NaN/Inf"
    print("  [OK] Zero signal handled")

    # Test 2: Constant signal
    window_const = np.ones((3, 1024)) * 5.0
    features_const = extractor.extract(window_const)
    assert np.all(np.isfinite(features_const)), "Constant signal produces NaN/Inf"
    print("  [OK] Constant signal handled")

    # Test 3: Very large values
    window_large = np.random.randn(3, 1024) * 1000
    features_large = extractor.extract(window_large)
    assert np.all(np.isfinite(features_large)), "Large values produce NaN/Inf"
    print("  [OK] Large values handled")

    # Test 4: Very small values
    window_small = np.random.randn(3, 1024) * 1e-6
    features_small = extractor.extract(window_small)
    assert np.all(np.isfinite(features_small)), "Small values produce NaN/Inf"
    print("  [OK] Small values handled")


def run_all_tests():
    """Run all tests."""
    print("=" * 80)
    print("Feature Extraction Unit Tests")
    print("=" * 80)

    try:
        test_time_domain_features()
        test_frequency_domain_features()
        test_spectral_ratios()
        test_feature_extractor()
        test_feature_names()
        test_edge_cases()
        test_real_data_extraction()

        print("\n" + "=" * 80)
        print("ALL TESTS PASSED [OK]")
        print("=" * 80)
        return True

    except AssertionError as e:
        print("\n" + "=" * 80)
        print(f"TEST FAILED [FAIL]: {e}")
        print("=" * 80)
        return False

    except Exception as e:
        print("\n" + "=" * 80)
        print(f"UNEXPECTED ERROR [ERROR]: {e}")
        print("=" * 80)
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
