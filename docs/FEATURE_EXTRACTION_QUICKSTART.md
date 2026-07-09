# Feature Extraction Module - Quick Start Guide

## 概览

这是一个用于癫痫检测的EEG特征提取模块，从3通道EEG窗口中提取28个基线特征。

## 快速使用

### 1. 批量特征提取（推荐）

```bash
# 从窗口化数据提取特征
python scripts/02_extract_features.py
```

**输入**: `data/processed/*_windows.npz`  
**输出**: `data/processed/*_features.npz`

### 2. 验证和可视化

```bash
# 生成特征分析报告和可视化
python scripts/validate_features.py
```

**输出**: `results/figures/` 下的可视化图表

### 3. 单元测试

```bash
# 运行完整性测试
python tests/test_feature_extraction.py
```

## 代码示例

### 提取单个窗口的特征

```python
from src.features import FeatureExtractor
import numpy as np

# 初始化提取器
extractor = FeatureExtractor(sfreq=256, fz_cz_channel_idx=2)

# 窗口形状: [3通道, 1024采样点]
window = np.random.randn(3, 1024)

# 提取28个特征
features = extractor.extract(window)
print(features.shape)  # (28,)
```

### 批量提取

```python
# 加载窗口化数据
data = np.load('data/processed/chb01_03_windows.npz')
windows = data['X']  # (1841, 3, 1024)

# 批量提取
features = extractor.extract_batch(windows, verbose=True)
print(features.shape)  # (1841, 28)
```

### 加载已提取的特征

```python
# 加载特征
data = np.load('data/processed/chb01_03_features.npz', allow_pickle=True)
features = data['features']        # (1841, 28)
labels = data['labels']            # (1841,)
feature_names = data['feature_names']  # (28,)
```

## 特征说明

### 28个特征组成

| 类别 | 数量 | 通道 | 说明 |
|------|------|------|------|
| 时域 | 7 | FZ-CZ | 标准差、偏度、Hjorth参数、峰峰值、过零率 |
| 频域 | 15 | T7-P7, T8-P8, FZ-CZ | 5频段×3通道的功率谱 |
| 谱比 | 6 | FZ-CZ | δ/θ, θ/β, (δ+θ)/β, β/γ, δ/BB, γ/BB |

### 频段定义

- **δ (delta)**: 4-8 Hz
- **θ (theta)**: 8-13 Hz  
- **β (beta)**: 13-30 Hz
- **γ_low (gamma_low)**: 30-50 Hz
- **Broadband**: 4-50 Hz

## 性能指标

- **提取速度**: ~700 窗口/秒
- **内存占用**: ~100KB (1841窗口的特征矩阵)
- **数据质量**: 无NaN/Inf值

## 验证结果

### Top 5 判别特征 (Cohen's d)

1. **T8-P8_broadband_power** (d=4.271) - 右颞叶宽频功率
2. **T8-P8_beta_power** (d=4.066) - 右颞叶β波功率
3. **T8-P8_gamma_low_power** (d=3.694) - 右颞叶低γ功率
4. **FZ-CZ_delta_power** (d=3.000) - 中央δ波功率
5. **T8-P8_theta_power** (d=2.990) - 右颞叶θ波功率

**关键发现**: T8-P8通道（右颞叶）在癫痫检测中显示最强判别能力

## 文件结构

```
src/features/
├── time_domain.py          # 时域特征提取
├── frequency_domain.py     # 频域特征提取
├── spectral_ratios.py      # 谱功率比特征
├── feature_extractor.py    # 主提取器类
└── __init__.py

scripts/
├── 02_extract_features.py  # 批量提取脚本
└── validate_features.py    # 验证脚本

tests/
└── test_feature_extraction.py  # 单元测试

docs/
├── FEATURE_EXTRACTION.md   # 完整技术文档
└── feature_extraction_examples.py  # 代码示例
```

## 故障排查

### 问题: 特征包含NaN/Inf

**原因**: 输入数据异常（全零、全常数）  
**解决**: 已在代码中处理边缘情况，检查输入数据质量

### 问题: 提取速度慢

**原因**: 处理大量窗口  
**解决**: 使用批处理模式（`batch_size=100`），默认已优化

### 问题: 特征相关性高

**原因**: 某些特征数学上相关（如std和Hjorth Activity）  
**解决**: 正常现象，Random Forest会自动处理冗余

## 下一步

完成特征提取后：

1. **Step 6**: 运行Random Forest特征选择（从28选12-16个特征）
2. **Step 7-8**: 使用选定特征训练MLP分类器
3. **Step 9-10**: 16-bit量化用于硬件部署

## 更多信息

详细技术文档: [docs/FEATURE_EXTRACTION.md](FEATURE_EXTRACTION.md)  
完成报告: [docs/STEP_4_5_COMPLETION_REPORT.md](STEP_4_5_COMPLETION_REPORT.md)
