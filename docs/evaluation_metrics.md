# 癫痫检测评估指标设计

## 问题：什么才算"检测成功"？

### 方案1: 窗口级评估 (Window-Level)
```python
# 每个4秒窗口单独评估
y_true = [0, 0, 1, 1, 1, 0, 0]  # 7个窗口
y_pred = [0, 0, 1, 0, 1, 0, 0]  # 模型预测

accuracy = (y_true == y_pred).mean()  # 5/7 = 71.4%
sensitivity = TP / (TP + FN)  # 2/3 = 66.7%
```

**优点**:
- 简单直观
- 符合机器学习常规评估
- 可用于论文对比

**缺点**:
- **不符合临床需求**！只要检测到发作就算成功，不需要每个窗口都对
- 对时间边界过于苛刻

---

### 方案2: 发作检测级评估 (Seizure-Level) ⭐推荐

#### 核心思想
只要在发作期间**任意时刻**检测到癫痫，就算成功检测这次发作。

#### 评估方法
```python
# 真实发作事件
seizure_events = [
    (2996, 3036),  # 第1次发作: 40秒
    (4500, 4540),  # 第2次发作: 40秒
]

# 模型预测的阳性窗口时间
predicted_windows = [
    (3000, 3004),  # 检测到发作1
    (3008, 3012),  # 检测到发作1
    (5000, 5004),  # 假阳性
]

# 评估逻辑
def evaluate_seizure_detection(seizure_events, predicted_windows, tolerance=5):
    """
    tolerance: 允许的时间误差（秒）
    """
    detected_seizures = []
    
    for seizure_start, seizure_end in seizure_events:
        detected = False
        
        # 检查是否有预测窗口与发作期重叠
        for pred_start, pred_end in predicted_windows:
            # 允许一定的时间偏移
            if (pred_start - tolerance <= seizure_end) and \
               (pred_end + tolerance >= seizure_start):
                detected = True
                break
        
        detected_seizures.append(detected)
    
    # Sensitivity: 成功检测的发作比例
    sensitivity = sum(detected_seizures) / len(seizure_events)
    
    # False Alarm Rate: 每小时误报次数
    false_alarms = count_false_alarms(seizure_events, predicted_windows)
    total_hours = get_total_duration_hours()
    false_alarm_rate = false_alarms / total_hours
    
    return sensitivity, false_alarm_rate

# 例子
sensitivity = 2/2 = 100%  # 2次发作都检测到了
false_alarm_rate = 1次误报 / 总时长
```

#### 关键指标

1. **Sensitivity (检出率)**
   - 定义: 成功检测的发作次数 / 总发作次数
   - 目标: ≥ 90%
   - 临床意义: 不能漏掉癫痫发作

2. **Latency (检测延迟)**
   - 定义: 发作开始 → 首次检测到的时间
   - 目标: < 10秒
   - 临床意义: 早期预警更有价值

3. **False Alarm Rate (误报率)**
   - 定义: 每小时误报次数
   - 目标: < 0.5次/小时
   - 临床意义: 太多误报会让医护人员麻木

4. **Precision (精确率)**
   - 定义: 正确报警次数 / 总报警次数
   - 目标: > 80%

---

### 方案3: 混合评估 ⭐⭐最佳实践

同时报告两种指标：

#### 论文/技术报告
```
Window-Level Metrics (基于25%标准):
  - Accuracy: 96.5%
  - Sensitivity: 92.3%
  - Specificity: 96.8%
  - F1-Score: 0.89

Seizure-Level Metrics (临床相关):
  - Seizure Detection Rate: 95.2% (40/42次发作被检测到)
  - Mean Detection Latency: 7.3 ± 3.1 seconds
  - False Alarm Rate: 0.32 per hour
  - Positive Predictive Value: 85.7%
```

#### 为什么两种都要？
- **Window-Level**: 方便与其他论文对比，证明模型质量
- **Seizure-Level**: 证明临床实用价值

---

## 具体实现建议

### 训练阶段
```python
# 使用25%标准生成窗口标签
label = (overlap_ratio >= 0.25).astype(int)

# 训练时优化窗口级准确率
model.fit(X_train, y_train)
```

### 评估阶段
```python
# 1. 窗口级评估
y_pred = model.predict(X_test)
window_acc = accuracy_score(y_test, y_pred)
window_sens = recall_score(y_test, y_pred)

# 2. 发作级评估
def evaluate_event_detection(model, test_files):
    seizure_detected = 0
    total_seizures = 0
    false_alarms = 0
    detection_latencies = []
    
    for file_path, seizure_times in test_files:
        X_test, window_times = load_windows(file_path)
        y_pred = model.predict(X_test)
        
        # 找到所有预测为阳性的窗口
        positive_windows = [window_times[i] for i, pred in enumerate(y_pred) if pred == 1]
        
        # 评估每次发作
        for seizure_start, seizure_end in seizure_times:
            total_seizures += 1
            
            # 检查是否检测到
            detected = False
            earliest_detection = None
            
            for win_start, win_end in positive_windows:
                if win_start <= seizure_end and win_end >= seizure_start:
                    detected = True
                    if earliest_detection is None or win_start < earliest_detection:
                        earliest_detection = win_start
            
            if detected:
                seizure_detected += 1
                latency = max(0, earliest_detection - seizure_start)
                detection_latencies.append(latency)
        
        # 计算误报
        false_alarms += count_false_positives(positive_windows, seizure_times)
    
    sensitivity = seizure_detected / total_seizures
    mean_latency = np.mean(detection_latencies)
    
    return sensitivity, mean_latency, false_alarms

# 3. 报告
print("Window-Level Performance:")
print(f"  Accuracy: {window_acc:.3f}")
print(f"  Sensitivity: {window_sens:.3f}")
print()
print("Seizure-Level Performance:")
print(f"  Detection Rate: {event_sens:.3f}")
print(f"  Mean Latency: {mean_lat:.1f}s")
print(f"  False Alarm Rate: {fa_rate:.2f}/hour")
```

---

## 最终推荐

**训练标签**: 25%重叠标准（已实现）

**最终报告**: 
1. 主要指标: **Seizure-Level** (临床相关)
2. 辅助指标: **Window-Level** (论文对比)

**成功标准**:
- ✅ Seizure Detection Rate ≥ 95%
- ✅ Detection Latency < 10s
- ✅ False Alarm Rate < 0.5/hour
- ✅ Window-Level Accuracy ≥ 90% (bonus)
