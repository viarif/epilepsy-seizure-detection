# 最终模型测试报告

运行日期：2026-07-17  
模型：`SeizureNetLite`，2,991个训练参数  
输入：四通道、1秒、256 Hz EEG窗口  
工作点：validation锁定的单一全局阈值，目标specificity ≥ 97%

## 最终采用结果

最终采用无患者适配、无时间投票的纯跨患者模型。报告范围是预先固定的非重叠测试片段：排除三位测试患者各自的一小时参考块、前置保护区和边界重叠窗口，但不使用这些参考块更新模型或阈值。

| 指标 | 结果 |
|---|---:|
| Window sensitivity | 0.35059 |
| Window specificity | 0.98357 |
| Macro patient sensitivity | 0.45472 |
| Macro patient specificity | 0.98092 |
| Macro patient PR-AUC | 0.32558 |
| Seizure-bout sensitivity | 0.825（33/40） |
| False alarms/hour | 60.70 |
| 评估窗口 | 658,788 |

### 逐患者结果

| 患者 | Sensitivity | Specificity | PR-AUC | Seizure-bout sensitivity | False alarms/hour | 窗口数 |
|---|---:|---:|---:|---:|---:|---:|
| chb10 | 0.87136 | 0.99588 | 0.81287 | 1.000（6/6） | 19.47 | 321,743 |
| chb11 | 0.31516 | 0.96474 | 0.08777 | 1.000（1/1） | 125.28 | 201,280 |
| chb12 | 0.17763 | 0.98215 | 0.07611 | 0.78788（26/33） | 65.17 | 135,765 |

## 完整保留测试集补充结果

若不排除一小时参考块和保护区，对全部680,398个保留test窗口直接评估：

| 指标 | 结果 |
|---|---:|
| Window sensitivity | 0.34806 |
| Window specificity | 0.98337 |
| Macro patient sensitivity | 0.45268 |
| Macro patient specificity | 0.98071 |
| PR-AUC | 0.19566 |
| ROC-AUC | 0.82067 |
| Seizure-bout sensitivity | 0.84091（37/44） |
| False alarms/hour | 62.08 |

## Validation与模型锁定

模型使用seed `20260718`训练，最佳checkpoint为epoch 3。validation阈值logit为`-2.632594`，sigmoid概率约`0.067070`。

| 指标 | Validation结果 |
|---|---:|
| Aggregate sensitivity | 0.58705 |
| Specificity | 0.97000 |
| Macro patient sensitivity | 0.31873 |
| Macro patient specificity | 0.97303 |
| PR-AUC | 0.30818 |
| ROC-AUC | 0.85608 |
| Seizure-bout sensitivity | 0.74286（26/35） |
| False alarms/hour | 108.06 |

## 解释与限制

- 模型达到并超过了97% specificity目标，但window sensitivity仍较低。
- 患者间差异很大：chb10表现较好，chb11和chb12是主要困难患者。
- 发作级灵敏度高于窗口级灵敏度，因为一个发作段只要至少一个窗口被命中就算检测成功。
- false alarms/hour按每条recording中的连续预测阳性段计数；一个连续误报段只计一次。
- 当前划分是固定development split，不是patient-level outer cross-validation结果，不能直接作为临床泛化结论。
- 模型没有使用患者适配、时间上下文、投票平滑、refractory period或测试患者专属阈值。
