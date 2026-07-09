# Step 6: Random Forest Feature Selection - Implementation Guide

## Critical: Data Splitting Strategy

### ⚠️ IMPORTANT: Use ONLY Training Set for Feature Selection

**Data Leakage Prevention**:
- ❌ WRONG: Train RF on all data (including test)
- ✅ CORRECT: Train RF on training set ONLY
- Test set must be completely isolated until final evaluation (Step 11)

### Recommended Split Strategy

**Option 1: Patient-Level Split (Recommended for Cross-Patient Generalization)**
```
Train: Patients 1-18 (75%)
Val:   Patients 19-21 (12.5%)  
Test:  Patients 22-24 (12.5%)
```

**Option 2: Recording-Level Split (More Training Data)**
```
Stratified split within each patient:
70% train / 15% val / 15% test
```

## Implementation Steps

### 1. Process All Data First

**Current Status**: Only chb01_03 processed (1841 windows)
**Required**: Process all 24 patients, all recordings with seizures

```bash
# Run windowing for all files
python scripts/01_create_windows.py --all-patients

# Extract features from all windows
python scripts/02_extract_features.py --all-files
```

### 2. Merge All Features

```python
# scripts/03_merge_all_features.py
import numpy as np
from pathlib import Path

def merge_all_features(processed_dir):
    """Merge all *_features.npz files into one dataset."""
    
    feature_files = sorted(processed_dir.glob('*_features.npz'))
    
    all_features = []
    all_labels = []
    file_ids = []  # Track which file each window comes from
    
    for i, fpath in enumerate(feature_files):
        data = np.load(fpath, allow_pickle=True)
        features = data['features']
        labels = data['labels']
        
        all_features.append(features)
        all_labels.append(labels)
        file_ids.extend([i] * len(features))
    
    merged = {
        'features': np.vstack(all_features),
        'labels': np.concatenate(all_labels),
        'file_ids': np.array(file_ids),
        'file_names': [f.name for f in feature_files],
        'feature_names': data['feature_names']  # Same for all files
    }
    
    return merged
```

### 3. Split Data (Patient-Level)

```python
# scripts/03_select_features.py
from sklearn.model_selection import GroupShuffleSplit

def split_by_patient(features, labels, patient_ids):
    """
    Split data by patient to ensure no patient appears in multiple sets.
    
    Args:
        features: [n_windows, 28]
        labels: [n_windows]
        patient_ids: [n_windows] - patient ID for each window
    
    Returns:
        train_idx, val_idx, test_idx
    """
    unique_patients = np.unique(patient_ids)
    n_patients = len(unique_patients)
    
    # 75% train, 12.5% val, 12.5% test
    n_train = int(n_patients * 0.75)
    n_val = int(n_patients * 0.125)
    
    np.random.seed(42)
    shuffled_patients = np.random.permutation(unique_patients)
    
    train_patients = shuffled_patients[:n_train]
    val_patients = shuffled_patients[n_train:n_train+n_val]
    test_patients = shuffled_patients[n_train+n_val:]
    
    train_idx = np.isin(patient_ids, train_patients)
    val_idx = np.isin(patient_ids, val_patients)
    test_idx = np.isin(patient_ids, test_patients)
    
    return train_idx, val_idx, test_idx
```

### 4. Train Random Forest (Train Set Only!)

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Split data
X_train = features[train_idx]
y_train = labels[train_idx]
X_val = features[val_idx]
y_val = labels[val_idx]
# X_test, y_test = features[test_idx], labels[test_idx]  # LOCKED until Step 11

print(f"Train: {len(X_train)} windows")
print(f"Val: {len(X_val)} windows")
print(f"Test: {len(X_test)} windows (LOCKED)")

# Train RF on TRAIN SET ONLY
rf = RandomForestClassifier(
    n_estimators=500,
    max_depth=20,
    min_samples_split=10,
    class_weight='balanced',  # Handle class imbalance
    random_state=42,
    n_jobs=-1
)

print("Training Random Forest on TRAIN SET...")
rf.fit(X_train, y_train)

# Evaluate on train (sanity check)
train_score = rf.score(X_train, y_train)
print(f"Train accuracy: {train_score:.4f}")

# Evaluate on validation (NOT used for selection)
val_score = rf.score(X_val, y_val)
print(f"Val accuracy: {val_score:.4f}")
```

### 5. Extract and Rank Feature Importance

```python
# Get feature importance from TRAIN performance
importance_scores = rf.feature_importances_
feature_names = merged['feature_names']

# Sort by importance
importance_ranking = sorted(
    zip(feature_names, importance_scores),
    key=lambda x: x[1],
    reverse=True
)

print("\nFeature Importance Ranking:")
for i, (name, score) in enumerate(importance_ranking, 1):
    print(f"{i:2d}. {name:35s} {score:.6f}")
```

### 6. Select Top K Features

```python
# Select top 12-16 features
TOP_K = 16  # Can try [12, 14, 16] and compare on VAL

selected_features = [name for name, score in importance_ranking[:TOP_K]]
selected_indices = [
    i for i, name in enumerate(feature_names) 
    if name in selected_features
]

print(f"\nSelected {TOP_K} features:")
for i, name in enumerate(selected_features, 1):
    print(f"{i:2d}. {name}")

# Save selection
np.savez(
    'results/selected_features.npz',
    feature_names=np.array(selected_features),
    feature_indices=np.array(selected_indices),
    importance_scores=importance_scores,
    rf_params={'n_estimators': 500, 'max_depth': 20}
)
```

### 7. Verify Performance with Selected Features

```python
# Extract selected features
X_train_selected = X_train[:, selected_indices]
X_val_selected = X_val[:, selected_indices]

# Retrain RF with selected features only (on train)
rf_selected = RandomForestClassifier(n_estimators=500, random_state=42)
rf_selected.fit(X_train_selected, y_train)

# Compare performance
train_score_selected = rf_selected.score(X_train_selected, y_train)
val_score_selected = rf_selected.score(X_val_selected, y_val)

print(f"\nPerformance Comparison:")
print(f"All features (28):     Train={train_score:.4f}, Val={val_score:.4f}")
print(f"Selected features ({TOP_K}): Train={train_score_selected:.4f}, Val={val_score_selected:.4f}")
print(f"Performance drop:      {(val_score - val_score_selected)*100:.2f}%")

# Success criterion: <5% drop on validation
assert val_score_selected >= val_score * 0.95, "Performance drop too large!"
```

## Expected Outputs

1. **`results/selected_features.npz`**:
   - Selected feature names (12-16)
   - Feature indices in original 28-feature array
   - Importance scores

2. **`results/figures/feature_importance_all_data.png`**:
   - Bar chart of all 28 features ranked by importance

3. **`results/feature_selection_report.txt`**:
   - Train/val/test split info
   - Performance comparison (28 vs selected)
   - Selected feature list

## File Naming Convention

Going forward, use this naming convention:

**Scripts**: `NN_verb_noun.py`
- `01_create_windows.py`
- `02_extract_features.py`
- `03_select_features.py`
- `04_train_mlp.py`

**Validation scripts**: `validate_<noun>.py`
- `validate_features.py` (single-file quality check)
- `validate_model.py` (future)

**Docs**: `UPPER_CASE.md` for major docs, `lower_case.md` for notes
- `FEATURE_EXTRACTION.md`
- `WINDOWING_SUMMARY.md`

**Results**: descriptive names with context
- `chb01_03_features.npz` (single file)
- `all_features_merged.npz` (all data)
- `selected_features.npz` (final selection)

## Summary

✅ **Must Do**:
1. Process ALL patient data (not just chb01_03)
2. Split data BEFORE any analysis
3. Train RF on train set ONLY
4. Select features based on train importance
5. Lock test set until Step 11

❌ **Must NOT Do**:
1. Use test data in feature selection
2. Use test performance to choose number of features
3. Draw conclusions from single-file analysis

**Next Steps**: 
- Process all CHB-MIT data
- Implement `03_select_features.py`
- Generate all-data feature importance plot
