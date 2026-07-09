# 时间平移增强 + 25%阳性标准的适配性分析

## 当前实现
- **阳性标准**: 窗口内癫痫占比 ≥25%
- **时间平移**: ±0.5秒 (±128采样点 @ 256Hz)
- **窗口大小**: 4秒 (1024采样点)

## 场景分析

### 场景1: 癫痫在窗口中心 (安全)
```
原始窗口:  |----[====癫痫====]----| (50%占比)
左移0.5s:  |---[====癫痫====]----|  (43.75%占比) ✓ 仍>25%
右移0.5s:  |----|[====癫痫====]---|  (43.75%占比) ✓ 仍>25%
```
**结论**: 标签保持一致，增强有效 ✅

### 场景2: 癫痫在窗口边缘 (危险)
```
原始窗口:  |[==癫痫==]----------| (25%占比，刚好阳性)
左移0.5s:  [==癫痫==]----------|  (癫痫部分被裁掉，<25%) ✗ 标签变化！
右移0.5s:  |--[==癫痫==]--------| (18.75%占比) ✗ 标签变化！
```
**问题**: 增强后的样本标签应该是0，但我们标记为1 ❌

### 场景3: 短时癫痫发作
```
癫痫发作持续时间: 10秒
窗口1 (0-4s):   |[癫痫]--------| (25%占比) → label=1
窗口2 (2-6s):   |--[癫痫癫痫]--| (50%占比) → label=1
窗口3 (4-8s):   |----[癫痫]----| (25%占比) → label=1
窗口4 (6-10s):  |------[癫痫]--| (25%占比) → label=1

如果对窗口1左移0.5s:
新窗口(-0.5-3.5s): [正常正常癫痫]  (可能<25%) → 标签应该是0，但我们用的是1
```

## 潜在问题总结

### ❌ 问题1: 标签污染
```python
# 当前代码
for window, label in zip(windows, labels):
    if label == 1:  # 癫痫窗口
        augmented = augment_seizure_window(window, n_augments=2)
        X_list.extend(augmented)
        y_list.extend([1] * 2)  # 假设增强后仍是癫痫！
```

**实际情况**: 增强后的窗口可能不满足25%标准，但我们强行标记为1

### ❌ 问题2: 边界效应
- 25%标准意味着窗口只需要1秒癫痫数据 (4秒 × 25% = 1秒)
- ±0.5秒平移可能完全改变这1秒的位置
- 最坏情况: 癫痫数据被平移到窗口外

## 🔧 修复方案

### 方案A: 重新计算增强样本标签 (最准确)
```python
def augment_with_relabel(window, label, seizure_mask, n_augments=2):
    """
    window: [n_channels, window_size]
    seizure_mask: [window_size] 布尔数组，标记哪些采样点是癫痫
    """
    augmented_windows = []
    augmented_labels = []
    
    for _ in range(n_augments):
        shift = np.random.randint(-128, 129)  # ±0.5s
        
        # 平移窗口和mask
        shifted_window = apply_shift(window, shift)
        shifted_mask = apply_shift(seizure_mask, shift)
        
        # 重新计算标签
        new_overlap_ratio = shifted_mask.sum() / len(shifted_mask)
        new_label = 1 if new_overlap_ratio >= 0.25 else 0
        
        augmented_windows.append(shifted_window)
        augmented_labels.append(new_label)
    
    return augmented_windows, augmented_labels
```

### 方案B: 限制平移范围 (保守)
```python
# 只在癫痫发作的"核心区域"增强
def augment_seizure_window_safe(window, seizure_overlap_ratio, max_shift=64):
    """
    max_shift = 64 samples = 0.25s @ 256Hz
    
    如果overlap_ratio = 50%，平移0.25s后:
      最差情况: 50% - 6.25% = 43.75% > 25% ✓
    
    如果overlap_ratio = 30%，平移0.25s后:
      最差情况: 30% - 6.25% = 23.75% < 25% ✗ (不应该增强)
    """
    # 只增强overlap > 35%的窗口，确保平移后仍>25%
    if seizure_overlap_ratio < 0.35:
        return []
    
    # 缩小平移范围
    augmented = []
    for _ in range(n_augments):
        shift = np.random.randint(-max_shift, max_shift + 1)
        shifted = apply_shift(window, shift)
        augmented.append(shifted)
    
    return augmented
```

### 方案C: 只对"明确癫痫"增强 (简单有效) ⭐推荐
```python
def augment_seizure_window_conservative(window, seizure_overlap_ratio, 
                                        threshold=0.5, max_shift=128):
    """
    只增强overlap ≥ 50%的窗口
    这样即使平移±0.5s，overlap仍 ≥ 37.5% > 25%
    """
    if seizure_overlap_ratio < threshold:
        return []
    
    augmented = []
    for _ in range(n_augments):
        shift = np.random.randint(-max_shift, max_shift + 1)
        shifted = apply_shift(window, shift)
        augmented.append(shifted)
    
    return augmented
```

## 🎯 推荐的实现

### 修改 `src/preprocessing/windowing.py`

```python
def process_edf_file(...):
    # ... 前面代码不变 ...
    
    # 生成标签时，同时记录每个窗口的overlap_ratio
    labels = []
    overlap_ratios = []
    
    for start_idx in window_starts:
        start_sec = start_idx / sfreq
        label, overlap_ratio = check_window_seizure_label_with_ratio(
            start_sec, window_duration, seizure_intervals, seizure_threshold
        )
        labels.append(label)
        overlap_ratios.append(overlap_ratio)
    
    # 数据增强：只增强overlap≥50%的窗口
    if augment_seizures:
        X_list = []
        y_list = []
        
        for window, label, overlap_ratio in zip(windows, labels, overlap_ratios):
            X_list.append(window)
            y_list.append(label)
            
            # 只增强"明确癫痫"窗口
            if label == 1 and overlap_ratio >= 0.5:
                augmented = augment_seizure_window(window, n_augments=2)
                X_list.extend(augmented)
                y_list.extend([1] * len(augmented))
        
        X = np.array(X_list)
        y = np.array(y_list)
```

## 总结

### 当前实现的问题
- ❌ 25%阳性 + ±0.5s平移 → 标签不一定准确
- ❌ 边界窗口增强后可能变成阴性，但仍标记为阳性

### 推荐修改
1. **方案1**: 只增强overlap≥50%的窗口 (简单有效)
2. **方案2**: 缩小平移范围到±0.25s
3. **方案3**: 增强后重新计算标签 (最准确但复杂)

**我建议用方案1**，既保证标签准确性，又能有效增强数据。
