# 癫痫检测 MLP 硬件落地项目

暑期任务：训练轻量级、可 **16bit 量化**、可硬件落地的癫痫（epileptic seizure）检测模型。
硬件落地载体不限（Memristor / MCU / FPGA 均可），**memristor 只是可能载体之一，非重点**。

## 项目目标

**Pipeline**: 特征提取 → Random Forest 选择 → MLP(FP32) → QAT/PTQ 量化 → **16bit 权重导出**

- **输入**: 41-45 个 EEG 特征（时域 + 频域 + 小波）
- **特征选择**: Random Forest 筛选至 12-16 个
- **模型**: MLP + ReLU + Dropout/L2 正则化（FP32 训练）
- **量化**: 独立最后步骤，QAT 或 PTQ，**位宽 = 16bit**（非 8bit）
- **输出**: 16bit 量化权重（`.npz` / `.h` / `.txt`）+ 特征列表 + 性能报告
- **数据集**: CHB-MIT（3 通道：T7-P7 / T8-P8 / FZ-CZ，256 Hz）；MVP 阶段可先用 Bonn

## 12 步实施计划（精简版）

| 阶段 | 步骤 | 内容 | 状态 |
|------|------|------|------|
| 1 环境准备 | Step 1 | 搭环境 + 项目结构 | ✅ |
| 2 数据准备 | Step 2-3 | 下载数据集 + 切窗(4s,50% overlap) + 轻量增强(时间平移) | ✅ |
| 3 特征提取 | Step 4-5 | 28 基线特征（时域7 + 频域15 + 谱比6）批量提取 | ✅ |
| 4 特征选择 | Step 6 | Random Forest(500树) 重要性排序 → 选 top 12-16 | 🔜 |
| 5 MLP 训练 | Step 7-8 | 构建 MLP + FP32 训练 + 5-fold CV + 过拟合处理(Dropout/L2/EarlyStop) | ⏳ |
| 6 量化训练 | Step 9-10 | **16bit** QAT/PTQ + 导出权重(int16) | ⏳ |
| 7 验证分析 | Step 11-12 | 端到端流水线测试 + 文档 | ⏳ |

**核心交付物**：16bit 量化权重 (`mlp_weights_int16.*`) + `selected_features.json` + `final_metrics.txt` + 代码库 + 文档

**成功标准**：MLP ≥ 90% 准确率；16bit 量化损失 < 1%；最终目标 acc ≥ 95%、sensitivity ≥ 90%、specificity ≥ 90%

> 完整逐步细节见 `.claude/memory/project-implementation-plan.md`

## 参考论文

项目根目录下 5 篇 PDF：
1. Classification by Simple ML（时域特征参考）
2. STFT + SVM Hardware Design（频域特征参考）
3. Low-Complexity Seizure Prediction（谱功率比特征）
4. Long-term data article
5. Parallel Memristor CNN（仅作硬件对比参考，非采用方案）

## 安装依赖

```bash
cd /c/Users/ASUS/Desktop/summer
pip install -r requirements.txt
```

**注**: PyTorch 默认 CPU 版本，若有 GPU 请按官网安装 CUDA 版本（本机 RTX 5070 + torch cu118）

## 项目结构

```
summer/
├── data/
│   ├── raw/           # 原始 EEG 数据 (.edf)
│   └── processed/     # 处理后数据
│       └── chb01_03_windows.npz  # ✅ 已完成
├── notebooks/         # Jupyter 探索笔记
├── src/
│   ├── features/      # 特征提取脚本
│   ├── models/        # MLP 定义 + 训练循环
│   ├── preprocessing/ # ✅ 切窗脚本
│   └── utils/         # ✅ 数据加载、标注解析
├── scripts/
│   └── 01_create_windows.py  # ✅ 切窗执行脚本
├── results/
│   ├── figures/       # 训练曲线、混淆矩阵
│   └── models/        # checkpoint.pth / quantized 权重
├── docs/              # ✅ 技术文档
│   ├── WINDOWING_SUMMARY.md
│   ├── evaluation_metrics.md
│   └── LABELING_THRESHOLD_EXPLAINED.md
├── papers/            # (PDF 已在项目根目录)
├── requirements.txt
└── README.md
```

## 当前进度

### ✅ 已完成 (Step 1-5)
- 环境搭建 + 依赖安装
- CHB-MIT 数据集下载
- **数据切窗 + 增强**:
  - 输出: `data/processed/chb01_03_windows.npz`
  - 形状: (1841, 3, 1024) - 1841窗口, 3通道, 4秒/窗口
  - 通道: T7-P7, T8-P8, FZ-CZ
  - 标签: 1778正常 / 63癫痫 (比例 1:28.2)
  - **双阈值策略**: 25%标签阈值 + 50%增强阈值
- **特征提取**:
  - 输出: `data/processed/chb01_03_features.npz`
  - 形状: (1841, 28) - 28个基线特征
  - 速度: ~700窗口/秒
  - **特征组成**: 7时域(FZ-CZ) + 15频域(3通道×5频段) + 6谱比(FZ-CZ)
  - **Top特征**: T8-P8频域特征 (Cohen's d > 4.0)

### 🔜 进行中 (Step 6)
- 特征选择 (Random Forest 重要性排序)

### ⏳ 待完成 (Step 6-12)
- 特征选择 (Random Forest) - Step 6
- MLP 训练 (Step 7-8)
- 量化训练 (Step 9-10)
- 端到端验证 (Step 11-12)

## 预计时长

15-18 天，或分阶段：MVP 8 天 + 训练 4 天 + 验证 3 天
