# Project Critique and Corrections - 2026-07-08

## Issues Found and Fixed

### 1. ⚠️ Single-Sample Bias (Critical)
**Problem**: Only chb01_03 processed, but documentation presented results as general findings.

**Fixed**:
- Updated `validate_features.py` with clear warnings
- Changed plot title to "SINGLE FILE ONLY"
- Added disclaimer text in outputs
- Updated all documentation to clarify sample limitations

**Lesson**: Never draw conclusions from single-file analysis. Wait for full dataset.

---

### 2. ⚠️ Data Leakage Risk (Critical for Step 6)
**Problem**: Unclear whether to use all data or train-only for Random Forest feature selection.

**Answer**: **ONLY TRAINING SET**

**Why**:
- Using test data in feature selection = data leakage
- Test performance will be overestimated
- Model won't generalize to real patients

**Correct Workflow**:
```
1. Split data: train (75%) / val (12.5%) / test (12.5%)
2. Train RF on TRAIN ONLY
3. Select features based on TRAIN importance
4. Lock test set until final evaluation (Step 11)
```

**Created**: 
- `memory/feature-selection-strategy.md` - Detailed strategy
- `docs/STEP_06_FEATURE_SELECTION_GUIDE.md` - Implementation guide

---

### 3. File Naming Inconsistency
**Problem**: Mixed naming conventions
- `FEATURE_EXTRACTION_SUMMARY.txt` (temp file in root)
- Inconsistent case usage

**Fixed**:
- Removed temporary summary from root
- Established naming convention:
  - Scripts: `NN_verb_noun.py`
  - Validation: `validate_<noun>.py`
  - Docs: `UPPER_CASE.md` (major), `lower_case.md` (notes)
  - Results: descriptive with context

---

### 4. Output File Naming
**Problem**: `top_features_effect_size.png` implies general conclusion

**Fixed**: Renamed to `single_file_feature_effect_size.png` with warning text

---

## Key Corrections to Documentation

### Updated Files:
1. **`scripts/validate_features.py`**:
   - Added warning in docstring
   - Changed title and output filename
   - Added disclaimer to plots
   - Clarified results are single-file only

2. **`memory/feature-selection-strategy.md`** (NEW):
   - Critical rule: train-only for feature selection
   - Correct workflow documented
   - Common mistakes to avoid

3. **`docs/STEP_06_FEATURE_SELECTION_GUIDE.md`** (NEW):
   - Complete implementation guide
   - Data splitting strategy
   - Code examples with correct workflow

---

## File Naming Convention (Going Forward)

### Scripts (`scripts/`)
```
NN_verb_noun.py
├─ 01_create_windows.py
├─ 02_extract_features.py
├─ 03_select_features.py  (to be implemented)
└─ 04_train_mlp.py
```

### Validation Scripts
```
validate_<noun>.py
├─ validate_features.py  (single-file quality check)
└─ validate_model.py     (future)
```

### Documentation (`docs/`)
```
Major docs: UPPER_CASE.md
├─ FEATURE_EXTRACTION.md
├─ WINDOWING_SUMMARY.md
└─ STEP_06_FEATURE_SELECTION_GUIDE.md

Notes: lower_case.md
├─ evaluation_metrics.md
└─ labeling_strategy_analysis.md
```

### Results
```
Descriptive names with context:
├─ chb01_03_features.npz           (single file)
├─ all_features_merged.npz         (all data - future)
├─ selected_features.npz           (final selection - future)
└─ single_file_feature_effect_size.png
```

---

## Action Items Before Step 6

**Must Complete**:
1. ✅ Update validate script with warnings
2. ✅ Create feature selection strategy docs
3. ✅ Establish file naming convention
4. ⏳ Process ALL CHB-MIT data (not just chb01_03)
5. ⏳ Implement proper train/val/test split
6. ⏳ Implement `03_select_features.py` with correct workflow

---

## Questions Answered

**Q: Should Random Forest use all data or just training set?**
**A: ONLY TRAINING SET. Using test data causes data leakage.**

**Q: When to use test set?**
**A: ONLY in Step 11 (final evaluation). Test set must be locked until then.**

**Q: How to split data?**
**A: Patient-level split recommended (75% train / 12.5% val / 12.5% test) for cross-patient generalization.**

---

## Takeaways

✅ **What went well**:
- Modular, clean code implementation
- Comprehensive documentation
- Caught issues before they became problems

⚠️ **What to improve**:
- Don't draw conclusions from insufficient data
- Always clarify train/test boundaries upfront
- Maintain consistent naming from the start
- Add warnings when results are preliminary

🎯 **Critical for ML projects**:
- Data leakage is silent but deadly
- Test set isolation is non-negotiable
- Sample size matters for conclusions
- Clear documentation prevents mistakes

---

**Status**: Issues identified and corrected. Ready to proceed with full dataset processing.
