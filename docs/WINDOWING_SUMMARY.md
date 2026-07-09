# 数据切窗阶段总结报告

## 📊 执行结果

### 测试文件
- **文件名**: chb01_03.edf
- **时长**: 3600秒 (1小时)
- **采样率**: 256 Hz
- **癫痫发作**: 1次 (2996-3036秒，持续40秒)

### 生成数据
```
输出形状: X=(1841, 3, 1024), y=(1841,)
├── 1841 个窗口
├── 3 个通道 (T7-P7, T8-P8, FZ-CZ)
└── 每窗口 1024 个采样点 (4秒)

保存位置: data/processed/chb01_03_windows.npz
文件大小: 4.93 MB
```

### 标签分布
- **正常样本 (class 0)**: 1778个 (96.6%)
- **癫痫样本 (class 1)**: 63个 (3.4%)
- **类别不平衡比**: 1:28.2

### 数据质量
- 数值范围: [-0.000721, 0.000595]
- 均值: 0.000000, 标准差: 0.000049
- 无NaN值，无Inf值 ✓

---

## 🎯 实现的关键特性

### 1. 窗口切割参数
- **窗口大小**: 4秒 (1024采样点 @ 256Hz)
- **步长**: 2秒 (512采样点，50%重叠)
- **基础窗口数**: 1799个

### 2. 标签策略
- **阳性标准**: 窗口内癫痫占比 ≥25%
- **原始癫痫窗口**: 21个
- **理由**: 平衡敏感性和标签准确性

### 3. 数据增强策略（已优化）
- **增强对象**: 仅增强 overlap ≥50% 的癫痫窗口
- **增强方法**: 时间平移 ±0.5秒 (±128采样点)
- **增强倍数**: 每个窗口生成2个副本
- **增强窗口数**: 21个窗口 × 3 = 63个癫痫样本
- **改善效果**: 不平衡比从 1:84.7 → 1:28.2

**为什么只增强50%以上的窗口？**
```
原始窗口 overlap=50%:  |----[====癫痫====]----|
左移0.5s后:            |---[====癫痫====]-----|  (overlap≈43.75% > 25%) ✓
右移0.5s后:            |-----|[====癫痫====]---|  (overlap≈43.75% > 25%) ✓

如果overlap=25% (边界情况):
原始窗口:              |[==癫痫==]-----------|
左移0.5s后:            [==]---------------|     (overlap<25%) ✗ 标签错误！

结论: 只增强overlap≥50%的窗口，确保增强后标签仍准确
```

### 4. 通道处理
- **目标通道**: T7-P7, T8-P8, FZ-CZ
- **实际加载**: 全部3个通道 ✓
- **T8-P8问题**: 数据集中有重复通道名，自动使用 `T8-P8-0`
- **FZ-CZ**: 成功识别为中线参考通道

---

## 🔧 解决的技术问题

### 问题1: T8-P8通道重复
**现象**: CHB-MIT数据集中T8-P8出现2次 (T8-P8-0, T8-P8-1)

**解决方案**:
```python
# src/utils/eeg_loader.py
# 检测并使用第一个重复通道
duplicates = [ch for ch in available_channels if ch.startswith(f"{target_ch}-")]
if duplicates:
    channels_to_pick.append(duplicates[0])  # 使用 T8-P8-0
```

### 问题2: 时间平移 + 25%标签的兼容性
**问题**: 边界窗口(overlap=25%)在平移后可能<25%，但仍被标记为癫痫

**解决方案**:
```python
# 只增强overlap≥50%的窗口
if label == 1 and overlap_ratio >= 0.5:
    augmented = augment_seizure_window(window, n_augments=2)
```

**效果**: 
- 保证增强后标签准确性
- 21个癫痫窗口全部满足50%标准，都被增强
- 增强后仍能将不平衡比改善到 1:28.2

---

## 📝 评估策略设计

根据你的问题，我们设计了两层评估体系：

### Level 1: 窗口级评估 (Window-Level)
**用途**: 模型性能评估、论文对比

**指标**:
- Accuracy, Precision, Recall, F1-Score
- 基于25%标准的ground truth

**计算方式**:
```python
y_pred = model.predict(X_test)
accuracy = (y_pred == y_test).mean()
```

### Level 2: 发作检测级评估 (Seizure-Level) ⭐核心指标
**用途**: 临床应用价值评估

**指标**:
1. **Seizure Detection Rate (检出率)**
   - 定义: 成功检测的发作次数 / 总发作次数
   - 目标: ≥95%
   - 只要在发作期间任意时刻检测到即算成功

2. **Detection Latency (检测延迟)**
   - 定义: 发作开始 → 首次检测的时间
   - 目标: <10秒
   - 越早检测越有价值

3. **False Alarm Rate (误报率)**
   - 定义: 每小时误报次数
   - 目标: <0.5次/小时
   - 太多误报会降低可信度

**计算方式**:
```python
# 伪代码
for seizure_start, seizure_end in true_seizures:
    detected = any(
        pred_window overlaps with [seizure_start, seizure_end]
        for pred_window in positive_predictions
    )
    if detected:
        detection_count += 1
        latency = first_detection_time - seizure_start

sensitivity = detection_count / total_seizures
```

**最终报告示例**:
```
Window-Level (论文对比):
  - Accuracy: 96.5%
  - Sensitivity: 92.3%
  - F1-Score: 0.89

Seizure-Level (临床价值):
  - Detection Rate: 95.2% (检测到40/42次发作)
  - Mean Latency: 7.3±3.1 seconds
  - False Alarm Rate: 0.32/hour
```

---

## 🤔 关于你的标签策略问题

### 你的提议
- 训练时: 25%标准 → 标记为阳性
- 测试时: 50%标准 → 标记为阳性

### 分析结果
**不推荐**，原因:
1. 训练/测试标签不一致会导致评估指标失真
2. 25%-50%区间的窗口会被判定为"预测错误"，但实际模型可能更好

### 推荐方案
**统一标签 + 调整决策阈值**:
```python
# 数据标注: 统一用25%标准
y_label = (overlap_ratio >= 0.25)

# 训练: 输出概率
y_prob = model.predict_proba(X_test)[:, 1]

# 测试: 调整决策阈值
decision_threshold = 0.7  # 提高阈值 = 更保守
y_pred = (y_prob >= decision_threshold)

# 评估: 基于原始25%标准
accuracy = (y_pred == y_test).mean()

# 多阈值评估
for thresh in [0.3, 0.5, 0.7, 0.9]:
    y_pred = (y_prob >= thresh)
    print(f"Threshold {thresh}: Sens={sens}, Spec={spec}")
```

**优点**:
- 训练/测试一致性
- 可灵活调整敏感性/特异性平衡
- 不影响评估指标的可解释性

---

## 📦 NPZ文件结构说明

NPZ是NumPy的压缩归档格式，类似字典：

```python
# 保存
np.savez_compressed('file.npz', 
    X=windows_array,      # (1841, 3, 1024)
    y=labels_array,       # (1841,)
    channels=channel_list,
    sfreq=256.0
)

# 加载
data = np.load('file.npz')
X = data['X']
y = data['y']

# 优点
- 单文件存储多个数组
- 自动压缩 (4.93MB vs 原始>15MB)
- 快速随机访问
- 兼容 sklearn, PyTorch, TensorFlow
- 训练时可直接shuffle
```

---

## ✅ 项目文件结构

```
summer/
├── data/
│   ├── raw/
│   │   └── chb01/
│   │       ├── chb01-summary.txt
│   │       └── chb01_03.edf
│   └── processed/
│       └── chb01_03_windows.npz  ← 新生成
├── src/
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── eeg_loader.py          ← 修复T8-P8重复
│   │   └── annotation_parser.py   ← 返回overlap_ratio
│   └── preprocessing/
│       ├── __init__.py
│       └── windowing.py            ← 优化增强策略
├── scripts/
│   └── 01_create_windows.py       ← 执行脚本
├── docs/
│   ├── labeling_strategy_analysis.md
│   ├── evaluation_metrics.md
│   └── augmentation_compatibility.md
└── README.md
```

---

## 🚀 下一步工作

根据12步实施计划（README.md 第22行），现在处于 **Step 3 完成**：

### ✅ 已完成
- [x] Step 1: 环境搭建
- [x] Step 2-3: 数据下载 + 切窗 + 增强

### 🔜 接下来
**Step 4-5: 特征提取**

提取候选特征（目标：41-45个）：

1. **时域特征 (7个)**:
   - 均值、方差、标准差、偏度、峰度
   - 过零率、峰值

2. **频域特征 (16个)**:
   - δ (0.5-4 Hz)
   - θ (4-8 Hz)
   - α (8-13 Hz)
   - β (13-30 Hz)
   - γ (30-50 Hz)
   - 各频段: 绝对功率、相对功率、功率比

3. **谱功率比 (12-16个)**:
   - δ/θ, δ/α, θ/α, etc.
   - 参考论文3: Low-Complexity Seizure Prediction

4. **小波特征 (6个)**:
   - db4小波分解
   - 各层系数统计特征

**输出**: `data/processed/chb01_03_features.npz`
- 形状: (1841, 41-45)
- 每行一个窗口的特征向量

---

## 💡 关键决策总结

| 决策点 | 选择 | 理由 |
|--------|------|------|
| 窗口大小 | 4秒 | 平衡时间分辨率和特征稳定性 |
| 重叠率 | 50% | 增加样本数，减少边界效应 |
| 阳性标准 | 25%重叠 | 提高敏感性，早期检测 |
| 增强对象 | overlap≥50% | 保证增强后标签准确性 |
| 增强方法 | 时间平移±0.5s | 轻量级，保持信号特性 |
| 通道选择 | T7-P7, T8-P8, FZ-CZ | 双侧颞叶+中线参考 |
| 保存格式 | NPZ | 压缩、快速、便于shuffle |
| 评估策略 | 双层评估 | 论文对比+临床价值 |

---

## 📚 参考文档

详细分析文档已创建在 `docs/` 目录：

1. **labeling_strategy_analysis.md**
   - 训练/测试标签策略分析
   - 软标签、多阈值方案

2. **evaluation_metrics.md**
   - 窗口级 vs 发作检测级评估
   - 具体实现代码

3. **augmentation_compatibility.md**
   - 时间平移与标签策略的兼容性
   - 三种修复方案对比

---

## ✨ 最终验证

```bash
python -c "
import numpy as np
data = np.load('data/processed/chb01_03_windows.npz')
assert data['X'].shape == (1841, 3, 1024)
assert data['y'].shape == (1841,)
assert len(data['channels']) == 3
assert data['sfreq'] == 256.0
assert np.sum(data['y'] == 1) == 63
assert np.isnan(data['X']).sum() == 0
print('All checks passed! ✓')
"
```

**数据切窗阶段完成！准备进入特征提取阶段。**
