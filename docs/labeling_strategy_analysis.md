# 标签策略分析：训练25% vs 测试50%

## 你的提议
- **训练集**: 窗口内癫痫占比 ≥25% → 标记为阳性
- **测试集**: 窗口内癫痫占比 ≥50% → 标记为阳性

## 分析

### ✅ 优点
1. **提高模型敏感性**: 训练时包含"部分癫痫"窗口，模型学会识别早期癫痫特征
2. **减少漏检**: 模型对弱信号更敏感
3. **符合医学需求**: 早期检测比精确定位更重要

### ⚠️ 潜在问题
1. **标签不一致**: 训练和测试用不同标准会导致：
   - 25%-50%重叠的窗口在训练时是阳性，测试时是阴性
   - 模型confusion：我预测是癫痫，但ground truth说不是
   
2. **评估指标失真**:
   ```
   训练标签 (25%):  [1, 1, 1, 1, 1] (5个阳性)
   测试标签 (50%):  [0, 0, 1, 1, 1] (3个阳性)
   模型预测:        [1, 1, 1, 1, 1]
   
   结果: 准确率下降，但实际上模型更好！
   ```

### 🎯 更好的策略

#### **方案A: 统一标准 + 软标签**
```python
# 训练时保留重叠比例作为软标签
overlap_ratio = 0.35  # 35%重叠
label = overlap_ratio  # 软标签 [0, 1]

# 测试时用阈值
pred_prob = model.predict()
if pred_prob > 0.5:
    pred_label = 1
```

#### **方案B: 三类标签**
```python
if overlap < 0.25:
    label = 0  # 正常
elif overlap < 0.75:
    label = 1  # 疑似癫痫
else:
    label = 2  # 明确癫痫

# 训练时: 3类分类
# 测试时: 合并1和2为阳性
```

#### **方案C: 多阈值评估 (推荐)**
```python
# 训练: 统一用25%标准
# 测试: 报告多个阈值的结果

eval_thresholds = [0.1, 0.25, 0.5, 0.75]
for thresh in eval_thresholds:
    acc, sens, spec = evaluate(predictions, labels, threshold=thresh)
    print(f"Threshold {thresh}: Acc={acc}, Sens={sens}, Spec={spec}")

# 最终报告选择最适合临床需求的阈值
```

## 推荐做法

保持**训练和测试标准一致**，但在后处理阶段调整决策阈值：

```python
# 数据标注: 统一用25%
y_label = (overlap_ratio >= 0.25).astype(int)

# 模型训练: 输出概率
model.fit(X_train, y_train)
y_prob = model.predict_proba(X_test)[:, 1]

# 测试阶段: 调整决策阈值
decision_threshold = 0.7  # 更保守，要求更高的概率
y_pred = (y_prob >= decision_threshold).astype(int)

# 评估: 基于25%标准的ground truth
accuracy = (y_pred == y_test).mean()
```

这样既保证了训练测试一致性，又能通过调整决策阈值来平衡敏感性/特异性。
