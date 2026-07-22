# 项目记忆：四通道跨患者癫痫窗口检测

这是当前项目的工作记录。README 只保留简练最终报告，详细方案、参数和历史实验放在这里。

## 当前最终口径

- 数据集：CHB-MIT，24 位患者。
- 任务：1 秒 EEG 窗口二分类，窗口中心落在 seizure interval 内为阳性。
- 输入：固定四通道 bipolar EEG，`F7-T7, T7-P7, F8-T8, T8-P8`。
- 模型：`SeizureNetLite`，2,991 个参数。
- 最终评估：leave-one-subject-out，24 折。
- 每折：23 位患者训练，剩下 1 位患者测试。
- 阈值：固定使用原验证集锁定阈值，logit `-2.632594108582`，sigmoid 约 `0.067069950728`。
- 不在每折测试患者上重新选阈值。
- 不使用投票、平滑、refractory period 或患者适配。

项目用于科研实验，不是临床诊断或告警系统。

## 最终 LOSO 结果

完整 24 折窗口级结果：

| 指标 | 数值 |
|---|---:|
| Window sensitivity | 0.509261 |
| Window specificity | 0.976026 |
| Macro patient sensitivity | 0.486245 |
| Macro patient specificity | 0.977417 |
| PR-AUC | 0.179868 |
| ROC-AUC | 0.835630 |
| Seizure-bout sensitivity | 0.817647 |
| False alarms/hour | 92.179163 |
| TP / FN / TN / FP | 10421 / 10042 / 6083361 / 149425 |
| Evaluated windows | 6,253,249 |

`Window sensitivity` 是按窗口数量加权的全局指标：

```text
sum(TP) / (sum(TP) + sum(FN))
= 10421 / (10421 + 10042)
= 0.509261
```

`Window specificity` 同理按阴性窗口数量加权：

```text
sum(TN) / (sum(TN) + sum(FP))
= 6083361 / (6083361 + 149425)
= 0.976026
```

逐患者结果：

| Patient | Sensitivity | Specificity | TP | FN | TN | FP | Windows |
|---|---:|---:|---:|---:|---:|---:|---:|
| chb01 | 0.697708 | 0.992505 | 487 | 211 | 239032 | 1805 | 241535 |
| chb02 | 0.066667 | 0.994466 | 12 | 168 | 209337 | 1165 | 210682 |
| chb03 | 0.786477 | 0.985265 | 442 | 120 | 224061 | 3351 | 227974 |
| chb04 | 0.588624 | 0.986788 | 445 | 311 | 1058301 | 14169 | 1073226 |
| chb05 | 0.733634 | 0.983792 | 650 | 236 | 229317 | 3778 | 233981 |
| chb06 | 0.058333 | 0.961896 | 14 | 226 | 441158 | 17476 | 458874 |
| chb07 | 0.807692 | 0.975047 | 525 | 125 | 447846 | 11461 | 459957 |
| chb08 | 0.300326 | 0.985345 | 552 | 1286 | 116456 | 1732 | 120026 |
| chb09 | 0.963768 | 0.925114 | 532 | 20 | 430460 | 34845 | 465857 |
| chb10 | 0.848993 | 0.996449 | 759 | 135 | 328080 | 1169 | 330143 |
| chb11 | 0.307398 | 0.980035 | 482 | 1086 | 202780 | 4131 | 208479 |
| chb12 | 0.111064 | 0.987591 | 262 | 2097 | 137687 | 1730 | 141776 |
| chb13 | 0.186905 | 0.978547 | 157 | 683 | 192898 | 4229 | 197967 |
| chb14 | 0.002959 | 0.994693 | 1 | 337 | 154810 | 826 | 155974 |
| chb15 | 0.750828 | 0.921335 | 2721 | 903 | 217811 | 18597 | 240032 |
| chb16 | 0.046667 | 0.962462 | 7 | 143 | 109558 | 4273 | 113981 |
| chb17 | 0.315700 | 0.948031 | 185 | 401 | 118922 | 6519 | 126027 |
| chb18 | 0.566390 | 0.972173 | 273 | 209 | 206929 | 5923 | 213334 |
| chb19 | 0.731013 | 0.993034 | 231 | 85 | 177898 | 1248 | 179462 |
| chb20 | 0.536957 | 0.970259 | 247 | 213 | 158582 | 4861 | 163903 |
| chb21 | 0.311558 | 0.994912 | 124 | 274 | 195348 | 999 | 196745 |
| chb22 | 0.752451 | 0.994100 | 307 | 101 | 184488 | 1095 | 185991 |
| chb23 | 0.581683 | 0.985312 | 470 | 338 | 176965 | 2638 | 180411 |
| chb24 | 0.616092 | 0.988853 | 536 | 334 | 124637 | 1405 | 126912 |

结果文件：

- `results/final_report.md`
- `results/experiments/loso_original/loso_report.md`
- `results/experiments/loso_original/loso_summary.json`
- `results/experiments/loso_original/loso_per_patient.csv`

## 数据预处理

预处理仍使用已有四通道缓存 `data/processed/selected4`。

关键规则：

- 采样率固定为 256 Hz。
- 每条 EDF 从开头连续滤波，不能逐窗口单独滤波。
- 0.1 Hz causal high-pass。
- 60 Hz notch，Q=30。
- causal rolling standard deviation，历史窗口 600 秒。
- 前 600 秒窗口丢弃，保证每个保留窗口都有完整 rolling history。
- 幅值缩放：`z = 0.2 * x_filtered / max(rolling_std, 1e-8)`。
- 缓存使用近似 tanh：`clip(z / 1.2, -1, 1)`。
- 窗口长度 1 秒，即 256 点。
- hop 0.5 秒，即 128 点。
- 模型输入形状 `[B, 1, 4, 256]`。

固定通道：

```text
F7-T7, T7-P7, F8-T8, T8-P8
```

通道选择原理：只用训练患者 seizure 区间的 ictal line length 排名，跨患者取稳定靠前的四个通道。当前 LOSO 评估直接复用已经生成好的四通道缓存，没有为每折重新选通道。

## 模型

当前最终模型是 `SeizureNetLite`。

结构摘要：

| 层 | 操作 | 参数 |
|---|---|---:|
| Conv1 | `1 -> 16, kernel=(4,17), bias=False` | 1,088 |
| BN1 | BatchNorm2d(16) | 32 |
| Conv2 | `16 -> 10, kernel=(1,5), bias=False` | 800 |
| BN2 | BatchNorm2d(10) | 20 |
| Conv3 | `10 -> 10, kernel=(1,5), bias=False` | 500 |
| BN3 | BatchNorm2d(10) | 20 |
| Conv4 | `10 -> 10, kernel=(1,5), bias=False` | 500 |
| BN4 | BatchNorm2d(10) | 20 |
| Output | `10 -> 1, kernel=(1,1), bias=True` | 11 |
| 总计 | raw logit 输出 | 2,991 |

模型内部不做 sigmoid。训练和评估都使用 raw logit。

## 训练协议

每折训练使用原始方案：

- optimizer：Adam。
- learning rate：`1e-3`。
- weight decay：0。
- batch size：256。
- AMP：CUDA 上启用。
- gradient clipping：global norm 5.0。
- seed：`20260718`。
- 固定 epoch：3。
- 阳性采样比例：每个 epoch 约 5%。
- hard-negative replay：第 1 轮后启用，随机背景和 hard negative 各 50%。
- 每条 recording 的 hard negative 上限：256 个窗口。

阳性样本使用患者/发作层次权重：

```text
w(p,b) = N_+ / (P * B_p * L_b)
```

含义：

- 每位含发作患者的阳性总权重相同。
- 同一患者内每个 seizure bout 的阳性总权重相同。
- bout 内部平均分配。
- 所有层次阳性权重均值为 1。
- 阴性窗口权重固定为 1。

训练损失直接使用这套层次权重，没有额外正类倍率。

## LOSO 脚本

入口：

```powershell
$py = 'C:\Users\ASUS\anaconda3\envs\unet\python.exe'

& $py scripts/leave_one_patient_out.py `
  --device cuda:0 `
  --num-workers 4 `
  --resume
```

脚本行为：

1. 扫描 `data/processed/selected4` 下所有 split 里的 24 位患者 metadata。
2. 为每一折生成轻量 `_cache_views/<patient>` JSON 视图，不复制 `.npy` 大文件。
3. 把留出患者 metadata 的 split 改成 `test`，其余 23 位改成 `train`。
4. 每折训练 3 epoch。
5. 用固定阈值 `-2.632594108582` 在留出患者上评估。
6. 每折写入 `folds/<patient>/test_report.json`、`test_scores.npz`、`final.pt`。
7. 汇总写入 `loso_summary.json`、`loso_per_patient.csv`、`loso_report.md`。

注意：脚本支持 `--resume`，如果某折已有 `test_report.json` 和 `test_scores.npz`，会跳过该折。

## 历史实验

### 固定 3 位 test 患者旧结果

旧方案使用固定 split：

- train：18 位患者。
- validation：`chb13/chb14/chb15`。
- development test：`chb10/chb11/chb12`。

旧 README 曾报告固定 3 位 test 患者：

- window sensitivity 约 0.35059。
- window specificity 约 0.98357。

这个口径已经不作为最终结果。对应旧测试入口 `scripts/evaluate.py` 和旧 test JSON 已删除。保留 `results/model/best.pt`、`validation_report.json` 等训练/阈值来源文件。

## 当前限制

- 总体 window sensitivity 约 0.51，仍远低于理想目标。
- 患者间差异很大，`chb02/chb06/chb14/chb16` 等患者 sensitivity 很低。
- `chb09/chb15` sensitivity 高但 specificity 明显下降，误报较多。
- 固定四通道和小模型对跨患者泛化仍不足。
- 当前 LOSO 使用固定阈值，不做每折阈值重选；这符合本次实验要求，但不是最优 operating point 搜索。
