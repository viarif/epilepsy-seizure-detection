# Feature Extraction Documentation

## Overview

This module implements the feature extraction pipeline for EEG-based seizure detection, extracting 28 baseline features from windowed EEG data.

## Architecture

### Module Structure

```
src/features/
├── __init__.py              # Module exports
├── time_domain.py           # Time-domain features (7)
├── frequency_domain.py      # Frequency-domain features (15)
├── spectral_ratios.py       # Spectral power ratios (6)
└── feature_extractor.py     # Main orchestration class
```

### Feature Breakdown

**Total: 28 features**

1. **Time-domain (7 features)** - FZ-CZ channel only
   - Standard Deviation
   - Skewness
   - Hjorth Activity
   - Hjorth Mobility
   - Hjorth Complexity
   - Peak-to-Peak Amplitude
   - Zero Crossing Rate

2. **Frequency-domain (15 features)** - 3 channels × 5 bands
   - Channels: T7-P7, T8-P8, FZ-CZ
   - Bands: δ (4-8Hz), θ (8-13Hz), β (13-30Hz), γ_low (30-50Hz), Broadband (4-50Hz)
   - Method: Welch's PSD with log-scale power

3. **Spectral Ratios (6 features)** - FZ-CZ channel only
   - δ/θ (Delta-Theta Ratio)
   - θ/β (Theta-Beta Ratio)
   - (δ+θ)/β (Slow/Fast Ratio)
   - β/γ_low (Beta-Gamma Ratio)
   - δ/Broadband (Delta relative power)
   - γ_low/Broadband (Gamma relative power)

## Usage

### Basic Usage

```python
from src.features import FeatureExtractor

# Initialize extractor
extractor = FeatureExtractor(sfreq=256, fz_cz_channel_idx=2)

# Extract features from single window
window = np.array([[...], [...], [...]])  # Shape: [3, 1024]
features = extractor.extract(window)  # Returns: [28]

# Extract features from batch
windows = np.array([...])  # Shape: [n_windows, 3, 1024]
features = extractor.extract_batch(windows)  # Returns: [n_windows, 28]
```

### Batch Processing Script

```bash
# Extract features from all windowed data files
python scripts/02_extract_features.py
```

**Input:** `data/processed/*_windows.npz` (windowed EEG data)
**Output:** 
- `data/processed/*_features.npz` (feature matrix + labels)
- `data/processed/*_feature_info.txt` (feature documentation)

### Validation Script

```bash
# Validate and visualize extracted features
python scripts/validate_features.py
```

**Generates:**
- Feature distribution plots (seizure vs normal)
- Top features by effect size (Cohen's d)
- Feature correlation matrix
- Quality check report

## Implementation Details

### Time-Domain Features

Implemented in [time_domain.py](../../src/features/time_domain.py)

**Hjorth Parameters:**
- Activity = var(signal)
- Mobility = sqrt(var(1st_derivative) / var(signal))
- Complexity = mobility(1st_derivative) / mobility(signal)

**Zero Crossing Rate:**
- Normalized by signal length
- Indicates high-frequency content

### Frequency-Domain Features

Implemented in [frequency_domain.py](../../src/features/frequency_domain.py)

**Power Spectral Density:**
- Method: Welch's method (scipy.signal.welch)
- Segment length: 512 samples (2 seconds @ 256Hz)
- Overlap: 50%
- Returns: Log-scale power (log10)

**Band Power Computation:**
- Integration: Trapezoidal rule
- Log-scale: Improves numerical stability and normalization

### Spectral Ratios

Implemented in [spectral_ratios.py](../../src/features/spectral_ratios.py)

**Hardware-Friendly Design:**
- Log-scale ratios = log(A/B) = log(A) - log(B)
- Simple subtraction instead of division
- Only one exception: (δ+θ)/β requires addition then log

## Performance

**Benchmark** (1841 windows, Intel Core i7):
- Extraction speed: ~700 windows/second
- Total time: ~2.6 seconds
- Memory usage: ~100KB for feature matrix (float32)

## Validation Results

### Data Quality

From `chb01_03` recording:
- ✓ No NaN or Inf values
- ✓ No constant features (except FZ-CZ_hjorth_activity*)
- ⚠ 3 highly correlated pairs (|r| > 0.95) - expected, will be handled by Random Forest

*Note: Hjorth Activity is highly correlated with Standard Deviation (r=0.968). This is expected as Activity = variance. Random Forest will handle redundancy.

### Top Features by Effect Size

**Cohen's d** (seizure mean - normal mean / pooled std):

| Rank | Feature | Cohen's d |
|------|---------|-----------|
| 1 | T8-P8_broadband_power | 4.271 |
| 2 | T8-P8_beta_power | 4.066 |
| 3 | T8-P8_gamma_low_power | 3.694 |
| 4 | FZ-CZ_delta_power | 3.000 |
| 5 | T8-P8_theta_power | 2.990 |
| 6 | FZ-CZ_slow_fast_ratio | 2.978 |
| 7 | T8-P8_delta_power | 2.916 |
| 8 | FZ-CZ_broadband_power | 2.463 |
| 9 | T7-P7_broadband_power | 2.430 |
| 10 | FZ-CZ_delta_theta_ratio | 2.258 |

**Key Insights:**
- T8-P8 (right temporal) shows strongest discriminative power
- Frequency-domain features dominate top 10
- Spectral ratios (slow_fast, delta_theta) show good separation
- Effect sizes > 2.0 indicate strong class separation

### Feature Correlations

**Highly Correlated Pairs (|r| > 0.95):**
1. FZ-CZ_std ↔ FZ-CZ_hjorth_activity (r=0.968)
2. T7-P7_delta_power ↔ T7-P7_broadband_power (r=0.976)
3. T8-P8_delta_power ↔ T8-P8_broadband_power (r=0.955)

These correlations are expected:
- Hjorth Activity = variance ≈ std²
- Delta power dominates broadband in low-frequency recordings

Random Forest feature selection (Step 6) will automatically handle redundancy.

## Design Rationale

### Why These Features?

Based on 3 reference papers:
1. **Paper 1**: Hjorth parameters most informative for seizure detection
2. **Paper 2**: Low gamma (30-50Hz) critical for seizures; multi-channel improves accuracy
3. **Paper 3**: Power ratios provide strong discriminative power with minimal computation

### Why 28 Features?

- **Quality over quantity**: Each feature has physical meaning
- **Hardware-friendly**: Small matrix (28 × n_windows)
- **Redundancy management**: Let Random Forest select optimal subset (12-16)
- **Interpretability**: Easy to visualize and debug

### Channel Strategy

- **Time-domain**: FZ-CZ only (most stable, low inter-subject variability)
- **Frequency-domain**: All 3 channels (captures spatial patterns)
- **Ratios**: FZ-CZ only (Paper 3 showed single-channel ratios outperform multi-channel)

## Next Steps

1. **Feature Selection (Step 6)**: Use Random Forest to select 12-16 most important features
2. **MLP Training (Step 7-8)**: Train classifier on selected features
3. **Quantization (Step 9-10)**: 16-bit quantization for hardware deployment

## File Outputs

### Feature Matrix (`*_features.npz`)

```python
{
    'features': np.ndarray,      # Shape: [n_windows, 28], dtype: float32
    'labels': np.ndarray,        # Shape: [n_windows], dtype: int
    'feature_names': np.ndarray, # Shape: [28], dtype: str
    'info': dict                 # Metadata (channels, sfreq, etc.)
}
```

### Feature Info (`*_feature_info.txt`)

Human-readable documentation of:
- Configuration parameters
- Feature breakdown by category
- Complete feature name list

## Troubleshooting

### Issue: NaN or Inf Values

**Cause**: Usually from zero-variance signals or frequency bands outside PSD range

**Solution**: 
- Check input window quality
- Verify sampling frequency matches data
- Check frequency band definitions

### Issue: Constant Features

**Cause**: All windows have identical values for a feature

**Solution**:
- Check if input data is corrupted
- Verify channel indexing is correct
- Remove constant features before training

### Issue: High Correlation

**Expected**: Some features are naturally correlated (e.g., std ↔ activity)

**Solution**: Random Forest handles this automatically - no manual intervention needed

## References

1. Sharanreddy & Kulkarni (2013): Time-domain features and classification
2. Ramgopal et al. (2014): STFT + SVM hardware design
3. Direito et al. (2017): Low-complexity seizure prediction with spectral ratios
