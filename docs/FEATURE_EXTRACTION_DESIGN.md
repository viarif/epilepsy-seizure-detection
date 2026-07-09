# 特征提取设计方案（基于3篇核心文献）

## 📚 文献综合分析

根据3篇核心论文，我们设计以下特征提取方案：

### 总体特征集（41-45个候选特征）

从3个通道 (T7-P7, T8-P8, FZ-CZ) 提取特征，然后用Random Forest选择12-16个。

---

## 🎯 特征分类

### 类别1: 时域特征 (7个) - 来自论文1

**每个通道提取7个时域特征** → 3通道 × 7 = **21个时域特征**

1. **Mean (均值)** - 信号平均值
2. **Standard Deviation (标准差)** - 信号波动幅度
3. **Modulus of Difference (差值模)** - 捕捉尖锐变化
4. **Skewness (偏度)** - 分布不对称性
5. **Hjorth Activity** - 信号能量（方差）
6. **Hjorth Mobility** - 频率偏移
7. **Hjorth Complexity** - 与谐波的相似度

**依据**: 论文1证明Hjorth参数是最有信息量的时域特征

---

### 类别2: 频域特征 (12-16个) - 来自论文2+3

基于**STFT**，窗口4秒，FFT 1024点 @ 256Hz

#### 方案A: 12频段能量 (论文2)
每个通道提取12个频段的绝对能量 → 3通道 × 12 = **36个频域特征**

频段划分（每个4Hz宽）:
1. δ1: 0.25-4 Hz
2. δ2/θ1: 4-8 Hz
3. θ2: 8-12 Hz
4. α: 12-16 Hz
5-12. β到低γ: 16-48 Hz (每4Hz一个频段)

#### 方案B: 8频段 + 相对功率 + 比值 (论文3)
**更推荐此方案** - 论文3证明比值特征更有区分度

每个通道提取:
- 8个绝对功率
- 8个相对功率（功率/总功率）
- 28个功率比值 (C(8,2) = 28)

频段划分（论文3）:
1. δ: 4-8 Hz
2. θ: 8-13 Hz
3. β: 13-30 Hz
4. γ1: 30-50 Hz
5. γ2: 50-70 Hz (可选)
6. γ3: 70-90 Hz (可选)
7. γ4: 90-110 Hz (可选)
8. γ5: 110-128 Hz (可选)

**关键**: 低γ频段 (30-50Hz) 被论文2证明有用！

---

### 类别3: 小波特征 (6个) - 来自论文1

**Daubechies db4小波分解**

每个通道提取6个小波系数统计量 → 3通道 × 6 = **18个小波特征**

1. **Level 1-3 细节系数能量**
2. **Level 1-3 近似系数能量**

或者提取各层系数的统计特征（均值、标准差）

**依据**: 论文1证明DWT系数是最有区分度的特征之一

---

## 🎨 推荐特征提取方案

### 方案1: 平衡方案（推荐）⭐
**总特征数**: 45个

- **时域** (21个): 3通道 × 7特征
- **频域** (18个): 3通道 × 6频段能量（δ, θ, α, β, γ低, γ高）
- **小波** (6个): 1个通道 × 6小波系数

**优点**:
- 覆盖时域+频域+时频域
- 特征数适中（45个）
- 计算复杂度可控

**频段选择**（6个）:
1. δ (4-8 Hz)
2. θ (8-13 Hz)
3. α (13-20 Hz)
4. β (20-30 Hz)
5. γ低 (30-50 Hz) ← 论文2证明重要
6. γ高 (50-100 Hz)

---

### 方案2: 频域重点方案
**总特征数**: 43个

- **时域** (7个): 仅1个通道 × 7特征
- **频域** (30个): 3通道 × 10频段（更细分）
- **小波** (6个): 1个通道 × 6小波系数

**优点**:
- 重点关注频域（论文2+3都强调频域）
- 更细的频率分辨率
- 可以计算谱功率比

---

### 方案3: 简化方案（MVP）
**总特征数**: 33个

- **时域** (7个): 1个通道 × 7特征
- **频域** (24个): 3通道 × 8频段
- **小波** (0个): 暂不实现

**优点**:
- 最简单，快速验证
- 足够覆盖核心特征
- 易于调试

---

## 🔧 实现细节

### 时域特征计算
```python
def extract_time_domain_features(window):
    # window: [n_channels, n_samples]
    features = []
    for ch in range(n_channels):
        signal = window[ch]
        
        # 1. Mean
        mean = np.mean(signal)
        
        # 2. Std
        std = np.std(signal)
        
        # 3. Modulus of difference
        mod_diff = np.sum(np.abs(np.diff(signal)))
        
        # 4. Skewness
        skewness = scipy.stats.skew(signal)
        
        # 5. Hjorth Activity
        activity = np.var(signal)
        
        # 6. Hjorth Mobility
        diff1 = np.diff(signal)
        mobility = np.sqrt(np.var(diff1) / activity)
        
        # 7. Hjorth Complexity
        diff2 = np.diff(diff1)
        mobility2 = np.sqrt(np.var(diff2) / np.var(diff1))
        complexity = mobility2 / mobility
        
        features.extend([mean, std, mod_diff, skewness, 
                        activity, mobility, complexity])
    
    return features  # 21 features for 3 channels
```

### 频域特征计算
```python
def extract_frequency_features(window, sfreq=256):
    # window: [n_channels, n_samples]
    features = []
    
    # Define frequency bands
    bands = {
        'delta': (4, 8),
        'theta': (8, 13),
        'alpha': (13, 20),
        'beta': (20, 30),
        'gamma_low': (30, 50),
        'gamma_high': (50, 100)
    }
    
    for ch in range(n_channels):
        # FFT
        fft_vals = np.fft.rfft(window[ch])
        fft_freq = np.fft.rfftfreq(len(window[ch]), 1/sfreq)
        psd = np.abs(fft_vals) ** 2
        
        # Extract band powers
        for band_name, (low, high) in bands.items():
            idx = np.logical_and(fft_freq >= low, fft_freq < high)
            band_power = np.log(np.sum(psd[idx]) + 1e-10)
            features.append(band_power)
    
    return features  # 18 features (3 channels × 6 bands)
```

### 小波特征计算
```python
import pywt

def extract_wavelet_features(window):
    # Use only one channel (e.g., FZ-CZ)
    signal = window[2]  # FZ-CZ
    
    # Daubechies db4 wavelet, 3 levels
    coeffs = pywt.wavedec(signal, 'db4', level=3)
    
    features = []
    for coeff in coeffs:
        # Energy of each level
        energy = np.sum(coeff ** 2)
        features.append(np.log(energy + 1e-10))
    
    return features  # 4 features (cA3, cD3, cD2, cD1)
```

---

## 📊 特征选择流程

根据论文2和论文3，采用**两阶段特征选择**:

### 阶段1: Random Forest特征重要性排序
```python
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=500, random_state=42)
rf.fit(X_train, y_train)

# 获取特征重要性
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]
```

### 阶段2: 选择Top 12-16特征
```python
# 选择重要性最高的12-16个特征
n_features_to_select = 12  # or 16
selected_features = indices[:n_features_to_select]
X_selected = X[:, selected_features]
```

**依据**: 论文2证明从36个特征选8-10个就能达到最佳性能

---

## ⚡ 计算复杂度

### 每个窗口的计算量:
1. **时域特征**: O(n) - 线性扫描
2. **频域特征**: O(n log n) - FFT
3. **小波特征**: O(n) - 小波变换

**总复杂度**: O(n log n)，被FFT主导

**硬件友好性**: ✅
- FFT有高效硬件实现
- 所有特征计算都是固定点运算
- 适合16-bit量化

---

## 🎯 最终推荐

**使用方案1（平衡方案）**:
- 45个候选特征（21时域 + 18频域 + 6小波）
- 用Random Forest选择12-16个
- **优先保留**: Hjorth参数、低频功率(δ,θ,α)、低γ功率(30-50Hz)

**原因**:
1. 论文1证明Hjorth+小波最有效
2. 论文2证明低γ频段(30-50Hz)重要
3. 论文3证明功率比值有高区分度
4. 平衡计算复杂度和准确度

---

下一步: 开始实现特征提取代码？
