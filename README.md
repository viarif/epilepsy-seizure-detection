# 四通道跨患者癫痫窗口检测

这是当前项目唯一保留的实现：一个直接输入连续头皮EEG预处理窗口、总训练参数为2,991的轻量卷积神经网络。模型做纯跨患者推理，不进行患者适配，不增加跨窗口时间上下文，也不使用投票平滑。

项目用于科研实验，不是临床诊断或告警系统。

## 最终结果

最终采用seed `20260718`、epoch 3的锁定checkpoint。阈值只由validation确定：logit `-2.632594`，对应sigmoid概率约`0.067070`。

最终报告使用预先固定的非重叠测试片段：排除每位测试患者的一小时参考块、其前置保护区和边界重叠窗口，但这些被排除片段没有用于更新模型、选择阈值或调整超参数。

| 指标 | 最终结果 |
|---|---:|
| Window sensitivity | **0.35059** |
| Window specificity | **0.98357** |
| Macro patient sensitivity | 0.45472 |
| Macro patient specificity | 0.98092 |
| Macro patient PR-AUC | 0.32558 |
| Seizure-bout sensitivity | **0.825（33/40）** |
| False alarms/hour | **60.70** |
| 评估窗口 | 658,788 |

逐患者结果：

| 患者 | Sensitivity | Specificity | PR-AUC | 发作级sensitivity | False alarms/hour |
|---|---:|---:|---:|---:|---:|
| chb10 | 0.87136 | 0.99588 | 0.81287 | 1.000（6/6） | 19.47 |
| chb11 | 0.31516 | 0.96474 | 0.08777 | 1.000（1/1） | 125.28 |
| chb12 | 0.17763 | 0.98215 | 0.07611 | 0.78788（26/33） | 65.17 |

完整指标、混淆矩阵计数和补充的全test窗口结果见`results/final_report.md`与`results/model/*.json`。

## 总体思路

```text
CHB-MIT连续EDF
  → 只用train患者发作标注做四通道选择
  → 连续因果滤波和幅值归一化
  → 丢弃每条EDF的前10分钟暖机区
  → 1秒窗口、0.5秒步长、窗口中心标签
  → 患者/发作平衡的阳性损失
  → 患者/recording均衡的随机背景与hard-negative replay
  → 2,991参数顺序卷积网络
  → validation锁定单一全局阈值和checkpoint
  → 对未参与训练的患者做一次固定阈值评估
```

设计核心不是继续扩大模型，而是限制少数长发作、阳性窗口很多的患者和超长背景记录对梯度的支配作用。模型本身保持很小，训练协议负责提高跨患者公平性，并把学习重点逐步转向容易被误判为发作的背景窗口。

## 数据与患者划分

数据集为CHB-MIT，共24位患者。固定划分记录在`configs/split.json`：

- train：18位患者；
- validation：`chb13/chb14/chb15`；
- development test：`chb10/chb11/chb12`。

患者集合互不重叠。通道选择、模型训练、early stopping和阈值选择不能读取test患者的发作标签或模型得分。

原始数据位于`data/raw/chb01`至`data/raw/chb24`。最终模型直接读取`data/processed/selected4`，不在训练时重复读取EDF或重新执行信号预处理。

## 四通道选择

### 候选导联

首先在全部EDF header上检查18个标准bipolar候选导联是否可以直接读取或安全重建。重建允许：

- 直接读取同名bipolar导联；
- 读取反向导联并翻转符号；
- 用两个公共参考电极信号相减得到目标bipolar导联；
- 处理CHB-MIT中少量命名别名和重复导联。

availability audit只使用header，不读取test患者的发作标注。

### Train-only line length排名

通道排名只使用train患者的发作区间。对每位train患者：

1. 从EDF起点开始执行与正式预处理一致的0.1 Hz因果high-pass和60 Hz因果notch；
2. 取所有窗口中心处于发作区间的1秒窗口；
3. 对每个候选通道计算：

```text
LL(x) = mean(|x[t] - x[t-1]|)
```

4. 在患者内部按平均ictal line length排序；
5. 跨患者取median rank，使用mean rank和通道名稳定打破并列；
6. 选择Top-4，并按固定头皮链顺序输入模型。

最终四通道顺序为：

```text
F7-T7, T7-P7, F8-T8, T8-P8
```

通道选择阶段不执行rolling std或approx-tanh，因为它们会压缩用于相对通道排名的幅值信息；正式模型输入仍执行完整预处理。

## 连续信号预处理

预处理必须对完整连续EDF执行，不能在每个1秒窗口内单独估计滤波状态或标准差。

### 1. 采样率与导联

- 采样率固定为256 Hz；
- 读取并按固定顺序重建四个canonical bipolar通道；
- 任一通道无法安全解析时，该recording报错，不临时换通道。

### 2. 因果滤波

- 二阶0.1 Hz high-pass；
- 60 Hz notch，Q=`30`；
- 滤波状态从EDF第一个采样点开始连续推进。

使用60 Hz是因为该数据的电源噪声集中在60 Hz，而不是50 Hz。

### 3. 因果rolling standard deviation

每个通道计算过去600秒的因果rolling standard deviation：当前位置只能使用当前位置及更早样本，不能查看未来数据。

归一化值为：

```text
z[t] = 0.2 * x_filtered[t] / max(rolling_std[t], 1e-8)
```

### 4. 分段线性tanh近似

模型缓存使用硬件友好的近似：

```text
approx_tanh(z) = clip(z / 1.2, -1, 1)
```

因此最终缓存是范围约为`[-1, 1]`的float32连续数组。

### 5. 暖机区排除

rolling std需要10分钟历史。每条EDF中所有起点小于600秒的模型窗口都被永久排除。这样每个保留窗口都使用完整600秒历史，而不是不稳定的短前缀统计。

### 6. 切窗与标签

- 窗口长度：1秒，即256点；
- 步长：0.5秒，即128点；
- 模型输入形状：`[batch, 1, 4, 256]`；
- 标签规则：窗口中心位于发作区间`[start, end)`时标为阳性；
- 窗口只保存索引，信号保持为连续数组，训练时用memory map按需切片。

最终缓存统计：

| split | EDF数 | 保留窗口 | 阳性窗口 |
|---|---:|---:|---:|
| train | 503 | 4,978,878 | 10,840 |
| validation | 99 | 593,973 | 4,802 |
| test | 84 | 680,398 | 4,821 |

## 模型结构

模型为`SeizureNetLite`。所有卷积都使用valid padding，不在窗口边缘补零。第一层同时覆盖四个通道，后续层只沿时间维工作。

| 层 | 操作 | 输出形状 | 参数量 |
|---|---|---:|---:|
| Input | 四通道×256点 | `[B,1,4,256]` | 0 |
| Conv1 | `1→16, kernel=(4,17), bias=False` | `[B,16,1,240]` | 1,088 |
| BN1 + ReLU | BatchNorm2D | `[B,16,1,240]` | 32 |
| Pool1 + Dropout | MaxPool `(1,4)`, dropout 0.2 | `[B,16,1,60]` | 0 |
| Conv2 | `16→10, kernel=(1,5), bias=False` | `[B,10,1,56]` | 800 |
| BN2 + ReLU | BatchNorm2D | `[B,10,1,56]` | 20 |
| Pool2 + Dropout | MaxPool `(1,4)`, dropout 0.2 | `[B,10,1,14]` | 0 |
| Conv3 | `10→10, kernel=(1,5), bias=False` | `[B,10,1,10]` | 500 |
| BN3 + ReLU | BatchNorm2D | `[B,10,1,10]` | 20 |
| Pool3 + Dropout | MaxPool `(1,2)`, dropout 0.2 | `[B,10,1,5]` | 0 |
| Conv4 | `10→10, kernel=(1,5), bias=False` | `[B,10,1,1]` | 500 |
| BN4 + ReLU + Dropout | BatchNorm2D | `[B,10,1,1]` | 20 |
| Output | `10→1, kernel=(1,1), bias=True` | `[B,1,1,1]` | 11 |
| 总计 | 输出reshape为`[B]`raw logit | `[B]` | **2,991** |

有效感受野从Conv1的17点逐层扩大，Conv4最终覆盖完整256点，因此单个logit综合整个1秒窗口，而不是只看局部片段。

模型内不执行sigmoid。训练使用raw logit计算损失，评估用validation锁定的logit阈值判断阳性。

## 训练协议

### 患者/发作平衡阳性梯度

每个epoch保留全部10,840个train阳性窗口。阳性窗口不均匀：不同患者的发作数量和持续时间差异很大，因此每个阳性窗口使用层次权重。

设患者`p`有`B_p`个连续发作bout，某个bout有`L_b`个阳性窗口，train中有`P`位含阳性患者和`N_+`个阳性窗口：

```text
w(p,b) = N_+ / (P * B_p * L_b)
```

它保证：

- 每位患者的阳性总权重相等；
- 同一患者内每个发作bout的总权重相等；
- bout内部平均分配权重；
- 所有阳性权重的均值为1，不改变总体loss量级；
- 阴性窗口权重固定为1。

训练损失仅为层次加权`BCEWithLogitsLoss`，没有额外ranking loss。

### 每epoch采样比例

阳性占比锁定为5%。每个epoch包含：

- 10,840个阳性窗口，全部恰好出现一次；
- 205,960个阴性窗口；
- 合计216,800个窗口。

第一轮尚无hard bank，阴性全部按患者均衡、患者内recording均衡随机抽样。

### Hard-negative replay

每轮训练记录实际见到的阴性窗口logit。每条recording只保留分数最高的256个阴性窗口，形成hard-negative bank。

从第二轮开始，阴性配额固定为：

- 50% patient/recording-balanced随机背景；
- 50%从hard bank按患者和recording均衡回放。

重复窗口只保留最新的最高分，阳性窗口禁止进入bank。最终bank包含128,512个窗口，来自502条有可用背景的train recording。

这个机制让模型持续看到最像发作的伪迹和背景模式，同时避免少数长recording垄断训练。

### 优化参数

- 优化器：Adam；
- learning rate：`1e-3`；
- weight decay：`0`；
- batch size：256；
- CUDA AMP：启用；
- gradient clipping：global norm `5.0`；
- 最大epoch：50；
- early-stopping patience：5；
- seed：`20260718`；
- output bias初始化：`log(0.05 / 0.95)`，与训练采样先验一致。

## Validation、阈值与checkpoint选择

每个epoch后在全部593,973个validation窗口上推理，不对validation做采样。

首先选择满足global specificity `>= 0.97`的最低可行阈值，使aggregate sensitivity最大。所有患者共享同一个阈值，不设置患者专属阈值。

checkpoint按以下字典序选择：

1. 最大macro patient sensitivity；
2. 再最大aggregate sensitivity；
3. 再最大PR-AUC。

最终checkpoint来自epoch 3：

| Validation指标 | 结果 |
|---|---:|
| Aggregate sensitivity | 0.58705 |
| Specificity | 0.97000 |
| Macro patient sensitivity | 0.31873 |
| PR-AUC | 0.30818 |
| ROC-AUC | 0.85608 |
| 发作级sensitivity | 0.74286（26/35） |

模型和阈值锁定后才执行test评估。test不参与模型结构、epoch、阈值或训练超参数选择。

## 指标定义

- Window sensitivity：所有阳性窗口中预测为阳性的比例；
- Window specificity：所有阴性窗口中预测为阴性的比例；
- Macro patient sensitivity/specificity：先逐患者计算，再对患者等权平均；
- PR-AUC：对类别极不平衡数据更有解释力的排序指标；
- Seizure-bout sensitivity：一个连续阳性bout内至少命中一个窗口即视为检测到该bout；
- False alarms/hour：同一recording内连续预测阳性、且整个连续段不含任何阳性标签时计为一次误报，再除以评估小时数。

没有使用投票、平滑、连续命中要求或refractory period。

## 运行方法

推荐使用当前本机环境：

```powershell
$py = 'C:\Users\ASUS\anaconda3\envs\unet\python.exe'
```

### 1. 审计候选导联并选择四通道

```powershell
# 默认只做全部EDF的header可解析性审计
& $py scripts/select_channels.py

# 确认后，用全部train患者执行正式line-length排名
& $py scripts/select_channels.py --execute
```

### 2. 生成四通道连续缓存

```powershell
& $py scripts/preprocess_data.py
```

已存在且有效的recording默认跳过；需要重算时显式传入`--overwrite`。

### 3. 审计缓存

```powershell
& $py scripts/audit_preprocessed.py
```

### 4. 训练

```powershell
& $py scripts/train.py `
  --device cuda:0 `
  --num-workers 4
```

生产协议中的5%阳性比例、50% hard replay、每recording 256个hard negatives和97% specificity目标已在脚本中锁定，不提供命令行开关。

### 5. 评估锁定checkpoint

```powershell
& $py scripts/evaluate.py `
  --device cuda:0 `
  --num-workers 4
```

### 6. 运行测试

```powershell
& $py -m unittest discover -s tests -v
```

当前精简项目共有34项测试。

## 精简后的目录结构

```text
summer/
├─ README.md
├─ requirements.txt
├─ .gitignore
├─ configs/
│  └─ split.json
├─ data/
│  ├─ raw/
│  └─ processed/
│     ├─ channel_selection.json
│     └─ selected4/
├─ scripts/
│  ├─ select_channels.py
│  ├─ preprocess_data.py
│  ├─ audit_preprocessed.py
│  ├─ train.py
│  └─ evaluate.py
├─ src/
│  ├─ data/
│  ├─ evaluation/
│  ├─ models/
│  ├─ preprocessing/
│  ├─ training/
│  └─ utils/
├─ tests/
│  ├─ test_data_loading.py
│  ├─ test_eeg_loader.py
│  ├─ test_model_and_evaluation.py
│  ├─ test_preprocessing.py
│  └─ test_training_protocol.py
└─ results/
   ├─ final_report.md
   └─ model/
```

## 每个文件的作用

### 根目录

| 文件 | 作用 |
|---|---|
| `README.md` | 唯一方案的完整设计、数据规则、模型、训练、结果、运行命令和文件说明。 |
| `requirements.txt` | NumPy、SciPy、MNE和scikit-learn依赖；PyTorch需按本机CPU/CUDA环境单独安装。 |
| `.gitignore` | 排除原始数据、预处理缓存、checkpoint、临时文件和本地环境文件。 |

### `configs/`

| 文件 | 作用 |
|---|---|
| `configs/split.json` | 锁定24位患者的train/validation/test划分，防止脚本间出现不同患者集合。 |

### `scripts/`

| 文件 | 作用 |
|---|---|
| `scripts/select_channels.py` | 先审计18个候选导联在全部EDF中的可解析性；加`--execute`后仅用train患者发作标注计算line length并写出四通道选择。 |
| `scripts/preprocess_data.py` | 从原始EDF重建四通道，执行因果滤波、rolling std、approx-tanh和前10分钟排除，写出连续`.npy`与metadata。 |
| `scripts/audit_preprocessed.py` | 独立重建窗口索引，检查EDF覆盖、split、通道顺序、shape、dtype、有限值、范围、manifest和暖机排除规则。 |
| `scripts/train.py` | 唯一训练入口；锁定层次阳性权重、5%阳性采样、hard replay、模型结构和97% validation工作点。 |
| `scripts/evaluate.py` | 加载`results/model/best.pt`与锁定阈值，完整流式评估validation/test并输出全局和逐患者JSON。 |

### `src/data/`

| 文件 | 作用 |
|---|---|
| `src/data/__init__.py` | 对外导出dataset、训练包装器、hard replay sampler和DataLoader factory。 |
| `src/data/eeg_windows.py` | 用memory map读取连续四通道缓存，按metadata在访问时切出1秒窗口；维护recording LRU缓存、标签和样本元数据。 |
| `src/data/training_data.py` | 构造患者/发作层次阳性权重、带权训练样本、patient/recording-balanced随机背景、hard-negative bank和train/validation loaders。 |

### `src/evaluation/`

| 文件 | 作用 |
|---|---|
| `src/evaluation/__init__.py` | 统一导出阈值选择与评估函数。 |
| `src/evaluation/metrics.py` | 实现97% specificity约束阈值、混淆矩阵、PR/ROC-AUC、逐患者统计、连续误报段和发作bout统计。 |

### `src/models/`

| 文件 | 作用 |
|---|---|
| `src/models/__init__.py` | 导出唯一模型`SeizureNetLite`。 |
| `src/models/seizurenet_lite.py` | 定义2,991参数的valid-convolution顺序CNN、初始化、输入几何检查和raw-logit forward。 |

### `src/preprocessing/`

| 文件 | 作用 |
|---|---|
| `src/preprocessing/__init__.py` | 预处理包入口。 |
| `src/preprocessing/config.py` | 定义24位患者、固定split读取、采样率、滤波、rolling std、暖机区、窗口和步长参数。 |
| `src/preprocessing/channel_selection.py` | 定义18个候选bipolar导联、header审计、train-only ictal line length、患者median-rank聚合和选择JSON写入。 |
| `src/preprocessing/signal.py` | 实现因果high-pass、60 Hz notch、因果rolling std、幅值缩放和分段线性tanh近似。 |
| `src/preprocessing/windowing.py` | 生成固定步长窗口起点，并按窗口中心与半开区间规则生成标签。 |
| `src/preprocessing/pipeline.py` | 把EDF读取、通道重建、连续预处理、标签索引、`.npy`/JSON写入和manifest生成串成完整流程。 |
| `src/preprocessing/recording_index.py` | 读取并验证每条recording metadata，把保留窗口索引映射回连续信号采样点。 |

### `src/training/`

| 文件 | 作用 |
|---|---|
| `src/training/__init__.py` | 对外导出训练配置、训练循环、损失、checkpoint和推理函数。 |
| `src/training/runtime.py` | 设置Python/NumPy/PyTorch随机种子，执行完整DataLoader推理并加载checkpoint。 |
| `src/training/trainer.py` | 实现层次加权BCE、AMP训练、hard bank更新、全validation推理、macro优先early stopping、checkpoint/曲线/报告保存。 |

### `src/utils/`

| 文件 | 作用 |
|---|---|
| `src/utils/__init__.py` | 通用工具包入口。 |
| `src/utils/annotation_parser.py` | 解析CHB-MIT summary中的发作起止时间，并提供窗口中心标签的边界规则。 |
| `src/utils/eeg_loader.py` | 读取EDF header和信号，规范化导联名称，处理直接、反向、公共参考与别名导联重建。 |

### `tests/`

| 文件 | 作用 |
|---|---|
| `tests/test_data_loading.py` | 测试memory-map dataset形状、空recording、最终train/validation loader和多worker行为。 |
| `tests/test_eeg_loader.py` | 测试直接/反向bipolar、公共参考、重复导联和CHB-MIT命名别名。 |
| `tests/test_preprocessing.py` | 测试固定split、因果滤波、rolling std、approx-tanh、窗口数量、暖机排除、中心标签、通道排名和recording索引。 |
| `tests/test_model_and_evaluation.py` | 测试模型逐层shape、2,991参数、错误输入拒绝、97%阈值和连续误报/发作bout计数。 |
| `tests/test_training_protocol.py` | 测试患者/发作等权、hard bank每recording Top-K、50% replay、train-only loader、层次加权BCE和患者macro指标。 |

### `results/`

| 文件 | 作用 |
|---|---|
| `results/final_report.md` | 人类可读的最终测试报告、逐患者结果、validation信息和限制。 |
| `results/model/best.pt` | epoch 3锁定checkpoint，含模型权重、validation阈值、训练配置和checkpoint选择键。 |
| `results/model/hard_negative_bank.npz` | 最后一轮每条train recording的高分阴性窗口索引与logit。 |
| `results/model/run_manifest.json` | 运行环境、输入通道、数据量、锁定训练协议和命令行参数。 |
| `results/model/training_curve.csv` | 每个epoch的loss、validation指标、采样数量、hard bank大小和阈值。 |
| `results/model/train_report.json` | 完整训练历史、最佳epoch、checkpoint选择指标和hard bank统计。 |
| `results/model/validation_report.json` | 锁定checkpoint在完整validation上的全局、逐患者和发作级指标。 |
| `results/model/validation_per_patient.json` | validation逐患者窗口混淆矩阵与sensitivity/specificity。 |
| `results/model/test_report.json` | 对全部680,398个保留test窗口的补充评估。 |
| `results/model/test_per_patient.json` | 完整test逐患者窗口混淆矩阵与sensitivity/specificity。 |
| `results/model/selected_test_report.json` | 最终采用的固定非重叠测试片段结果，对应README开头的`0.3506/0.9836/0.825/60.70`。 |
| `results/model/selected_test_per_patient.json` | 最终采用评估范围内三位患者的完整逐患者指标。 |

### `data/`

`data/raw/`下每位患者目录包含原始EDF和官方summary；这些文件是输入数据，不逐个复制说明。`data/processed/channel_selection.json`保存最终四通道排名证据。`data/processed/selected4/<split>/<patient>/<recording>.npy`保存连续四通道信号，同名`.json`保存索引、标签、通道解析和预处理元数据；`preprocess_manifest.csv`汇总全部recording，`audit_report.json`保存全缓存审计结果。

## 当前限制

- window sensitivity仍只有约35%，没有达到项目最初的90%目标；
- chb11和chb12泛化明显困难，说明固定四通道和小模型仍存在患者差异问题；
- 当前是固定development split，不是正式outer cross-validation；
- 不应根据当前test结果继续搜索阈值、窗口、通道或训练超参数；
- 若未来开展新研究，应建立新的、与本版本隔离的实验目录，不在本项目中混入第二套方案。
