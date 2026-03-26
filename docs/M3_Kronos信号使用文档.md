# M3 Kronos 信号管线 使用文档

## 目录

- [模块简介](#模块简介)
- [环境准备](#环境准备)
- [快速开始](#快速开始)
- [脚本详细说明](#脚本详细说明)
  - [kronos_data.py 数据准备](#kronos_datapy-数据准备)
  - [kronos_finetune.py 微调训练](#kronos_finetunepy-微调训练)
  - [kronos_predict.py 信号预测](#kronos_predictpy-信号预测)
  - [kronos_experiment.py 方案对比](#kronos_experimentpy-方案对比)
  - [kronos_recipe_cli.py 方案管理](#kronos_recipe_clipy-方案管理)
- [Python API 参考](#python-api-参考)
  - [FinetuneRecipe 微调方案](#finetunerecipe-微调方案)
  - [KronosFinetuner 微调执行器](#kronosfinetuner-微调执行器)
  - [KronosInference 推理执行器](#kronosinference-推理执行器)
  - [KronosSignalPipeline 信号生产](#kronossignalpipeline-信号生产)
  - [FinetuneExperiment 实验管理](#finetuneexperiment-实验管理)
- [Kronos 模型原理简介](#kronos-模型原理简介)
- [微调方案设计指南](#微调方案设计指南)
- [配置说明](#配置说明)
- [运行测试](#运行测试)
- [常见问题](#常见问题)

---

## 模块简介

M3 Kronos 信号管线是 QuantLab 三条信号管线之一（管线 B），负责：

1. **数据准备** — 下载预训练模型、将 Qlib 数据预处理为 Kronos 训练格式
2. **微调训练** — 基于可配置的 recipe（方案）对 Kronos 模型执行微调
3. **信号预测** — 批量推理全市场，产出预测收益 + 不确定性度量
4. **方案实验** — 对比多种微调方案的效果，找出最优配置

所有脚本位于 `quantlab/signal/` 目录下。

**与其他模块的关系：**

```
M1 DataManager ──→ M3 Kronos 信号 ──→ M5 信号融合
                        ↓
                   return_1d（预测收益）
                   return_5d（5日预测）
                   uncertainty（不确定性）
```

**与管线 A（M2 Alpha158）的区别：**

| | M2 Alpha158 | M3 Kronos |
|---|---|---|
| 模型 | LightGBM（表格模型） | Kronos（Transformer） |
| 输入 | 158 个人工因子 | 原始 OHLCV K 线 |
| 输出 | 截面评分 | 预测收益 + 不确定性 |
| 训练方式 | 每 20 天滚动重训 | 每日快速微调（1-5 epoch） |
| 优势 | 可解释、因子可扩展 | 端到端、自动学特征 |

---

## 环境准备

M1 环境已装好的前提下，额外安装：

```bash
pip install torch torchvision  # PyTorch（需匹配 CUDA 版本）
pip install huggingface_hub    # 下载预训练模型
```

确认 GPU 可用（推荐）：

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

确认 M1 数据已就绪：

```bash
cd quantlab/data
python data_status.py
```

---

## 快速开始

```bash
cd quantlab/signal

# 第一步：下载 Kronos 预训练模型
python kronos_data.py download-models

# 第二步：准备训练数据（从 Qlib 数据生成 pkl）
python kronos_data.py prepare --market csi300

# 第三步：查看数据状态
python kronos_data.py status

# 第四步：查看可用的微调方案
python kronos_recipe_cli.py list

# 第五步：执行微调训练
python kronos_finetune.py --recipe conservative --save-model --evaluate

# 第六步：单日预测（微调 + 推理一体化）
python kronos_predict.py --anchor-date 2024-06-28

# 第七步：零样本推理（不微调，直接用预训练模型）
python kronos_predict.py --anchor-date 2024-06-28 --recipe zero_shot

# 第八步：方案对比实验
python kronos_experiment.py --all --start 2024-07-01 --end 2024-12-31
```

---

## 脚本详细说明

### kronos_data.py 数据准备

负责两件事：下载预训练模型、将 Qlib 二进制数据预处理为 Kronos 训练所需的 pkl 格式。

**下载预训练模型：**

```bash
# 下载默认模型（NeoQuasar/Kronos-Tokenizer-base + NeoQuasar/Kronos-base）
python kronos_data.py download-models

# 下载到指定目录
python kronos_data.py download-models --save-dir ./data/models

# 下载指定版本
python kronos_data.py download-models \
    --tokenizer NeoQuasar/Kronos-Tokenizer-base \
    --predictor NeoQuasar/Kronos-base
```

模型下载后同时保存到 HuggingFace 缓存（`~/.cache/huggingface/hub/`）和指定目录，后续加载时自动使用缓存。

**预处理训练数据：**

```bash
# 使用默认参数（csi300，2011-2025，lookback=90，horizon=10）
python kronos_data.py prepare

# 自定义参数
python kronos_data.py prepare \
    --market csi300 \
    --start 2011-01-01 \
    --end 2025-01-01 \
    --lookback 90 \
    --horizon 10 \
    --output ./data/kronos

# 自定义训练/验证/测试集切分
python kronos_data.py prepare \
    --train-end 2023-06-30 \
    --val-start 2023-04-01 \
    --val-end 2024-06-30 \
    --test-start 2024-04-01
```

输出文件：

```
data/kronos/
├── train_data.pkl      训练集（按股票分的 DataFrame 字典）
├── val_data.pkl        验证集
├── test_data.pkl       测试集
└── meta.json           元信息（市场、时间区间、股票数等）
```

默认切分：

| 数据集 | 时间区间 | 用途 |
|--------|----------|------|
| train | 2011-01-01 ~ 2022-12-31 | 微调训练 |
| val | 2022-09-01 ~ 2024-06-30 | 早停验证 |
| test | 2024-04-01 ~ 2025-01-01 | 回测评估 |

> 验证集与训练集有时间重叠，这是因为回看窗口（lookback=90）需要额外的历史数据作为输入。

**查看数据状态：**

```bash
python kronos_data.py status

# 指定目录
python kronos_data.py status --data-path ./data/kronos
```

输出示例：

```
数据集元信息:
  market: csi300
  start: 2011-01-01
  end: 2025-01-01
  n_train_symbols: 498
  n_val_symbols: 495
  n_test_symbols: 492

  train: 498 只股票, 1352400 行, 平均 2716 行/股, 312.5 MB
  val: 495 只股票, 287100 行, 平均 580 行/股, 67.2 MB
  test: 492 只股票, 102200 行, 平均 208 行/股, 24.1 MB

  HuggingFace 缓存中的 Kronos 模型:
    models--NeoQuasar--Kronos-Tokenizer-base: 45.2 MB
    models--NeoQuasar--Kronos-base: 128.7 MB
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--data-dir` | `~/.qlib/qlib_data/cn_data` | Qlib 数据目录 |
| `--market` | `csi300` | 股票池 |
| `--start` | `2011-01-01` | 数据起始日期 |
| `--end` | `2025-01-01` | 数据截止日期 |
| `--lookback` | `90` | 回看窗口天数 |
| `--horizon` | `10` | 预测窗口天数 |
| `--output` | `data/kronos` | 输出目录 |

---

### kronos_finetune.py 微调训练

基于 recipe 配置执行微调训练。支持两种数据模式：

- **pkl 模式（默认）**：使用 `kronos_data.py prepare` 预处理好的数据，适合离线实验
- **Qlib 实时模式**：从 Qlib 直接加载数据，适合日频微调

**基本用法：**

```bash
# 使用 conservative 方案微调
python kronos_finetune.py --recipe conservative

# 微调并保存模型
python kronos_finetune.py --recipe conservative --save-model

# 微调并在验证集上评估
python kronos_finetune.py --recipe conservative --save-model --evaluate
```

输出示例：

```
微调方案: conservative
  描述: 只微调 predictor 最后2层，安全稳健
  Tokenizer: 冻结 (none)
  Predictor: 微调 (last_n)
  Epochs: 3, LR: 2e-05
  采样策略: uniform

设备: cuda
加载 train 数据: 498 只股票

加载预训练模型...
微调阶段: ['predictor']

开始微调 (498 只股票)...
Epoch 1/3, loss=3.2145
Epoch 2/3, loss=2.8753
Epoch 3/3, loss=2.6412
微调完成: 45.3s

模型已保存:
  Tokenizer: outputs/kronos/tokenizer
  Predictor: outputs/kronos/predictor
```

**两阶段微调（先 tokenizer 后 predictor）：**

```bash
# 方式一：一次性两阶段
python kronos_finetune.py --recipe aggressive --stage both --save-model

# 方式二：分开执行
# 先微调 tokenizer
python kronos_finetune.py --recipe aggressive --stage tokenizer --save-model --output ./outputs/stage1

# 再用微调过的 tokenizer 微调 predictor
python kronos_finetune.py --recipe aggressive --stage predictor \
    --finetuned-tokenizer ./outputs/stage1/tokenizer \
    --save-model --output ./outputs/stage2
```

**使用 Qlib 实时数据微调（日频场景）：**

```bash
# 用 2024-06-28 之前的数据做微调
python kronos_finetune.py \
    --recipe conservative \
    --use-qlib \
    --anchor-date 2024-06-28 \
    --save-model
```

| 参数 | 必填 | 默认值 | 说明 |
|------|------|--------|------|
| `--recipe` | 是 | — | 微调方案名称 |
| `--recipes-file` | 否 | `configs/kronos_recipes.yaml` | 方案配置文件 |
| `--data-path` | 否 | `data/kronos` | pkl 数据目录 |
| `--use-qlib` | 否 | 否 | 使用 Qlib 实时数据 |
| `--anchor-date` | `--use-qlib` 时必填 | — | 锚定日期 |
| `--stage` | 否 | `auto` | 微调阶段：`auto`/`tokenizer`/`predictor`/`both` |
| `--finetuned-tokenizer` | 否 | — | 已微调 Tokenizer 路径 |
| `--save-model` | 否 | 否 | 保存微调后的模型 |
| `--evaluate` | 否 | 否 | 在验证集上评估重建损失 |
| `--output` | 否 | `outputs/kronos` | 输出目录 |
| `--device` | 否 | `cuda` | 计算设备 |

---

### kronos_predict.py 信号预测

对全市场执行微调 + 推理一体化流程，产出 T+1 预测收益、T+5 预测收益和不确定性。

**单日预测：**

```bash
# 基本用法（使用 conservative 方案）
python kronos_predict.py --anchor-date 2024-06-28

# 使用指定方案
python kronos_predict.py --anchor-date 2024-06-28 --recipe aggressive

# 零样本推理（不微调，作为基线）
python kronos_predict.py --anchor-date 2024-06-28 --recipe zero_shot

# 查看 Top-30 股票
python kronos_predict.py --anchor-date 2024-06-28 --top-k 30

# 输出到 CSV
python kronos_predict.py --anchor-date 2024-06-28 --output signal_20240628.csv
```

输出示例：

```
使用方案: conservative (只微调 predictor 最后2层，安全稳健)
数据: 298 只股票
[2024-06-28] 微调开始 (recipe=conservative)
[2024-06-28] 微调完成 (42.5s)
[2024-06-28] 推理开始
[2024-06-28] 推理完成 (18.3s), 298 只股票

信号输出: 2024-06-28 (298 只股票)

Top-20 看多 (T+1 预测收益):
  SH600519: +0.0182  (不确定性: 0.0045)
  SZ000858: +0.0156  (不确定性: 0.0038)
  SH601318: +0.0143  (不确定性: 0.0052)
  ...

Bottom-5 看空:
  SZ002714: -0.0098
  ...

信号统计:
  均值: 0.0012
  标准差: 0.0085
  平均不确定性: 0.0061
```

**批量回测：**

```bash
# 对一段区间逐日产出信号
python kronos_predict.py --start 2024-01-01 --end 2024-06-30

# 使用指定方案 + 输出到 CSV
python kronos_predict.py --start 2024-01-01 --end 2024-06-30 \
    --recipe conservative --output signals_kronos.csv
```

| 参数 | 必填 | 默认值 | 说明 |
|------|------|--------|------|
| `--anchor-date` | 二选一 | — | 单日预测日期 |
| `--start` / `--end` | 二选一 | — | 批量回测区间 |
| `--recipe` | 否 | `conservative` | 微调方案名称 |
| `--top-k` | 否 | `20` | 显示 Top-K 股票 |
| `--output` | 否 | — | 信号输出 CSV 路径 |
| `--device` | 否 | `cuda` | 计算设备 |
| `--max-context` | 否 | `512` | 最大上下文长度 |
| `--market` | 否 | `csi300` | 股票池 |

**输出 CSV 格式：**

```
           return_1d  return_5d  uncertainty
SH600519    0.0182     0.0356      0.0045
SZ000858    0.0156     0.0298      0.0038
...
```

---

### kronos_experiment.py 方案对比

对比多个微调方案在同一评估区间上的表现，输出 IC、ICIR、多空夏普等指标。

**运行全部预设方案对比：**

```bash
python kronos_experiment.py --all
```

**运行指定方案：**

```bash
python kronos_experiment.py --recipes conservative,aggressive,zero_shot
```

**自定义评估区间：**

```bash
python kronos_experiment.py --all --start 2024-07-01 --end 2024-12-31
```

**保存结果：**

```bash
python kronos_experiment.py --all --output ./experiment_results
```

输出示例：

```
实验方案: ['conservative', 'aggressive', 'head_only', 'cumulative', 'recency_focus', 'zero_shot', 's1_heavy']
评估区间: 2024-07-01 ~ 2024-12-31

================================================================================
方案对比:
================================================================================
                 ic_mean   icir  ic_1d  long_short_sharpe  total_time_sec  ic_std  worst_week_ic
conservative      0.035   1.20  0.035              1.45           380.0   0.029         0.008
aggressive        0.041   0.95  0.041              1.60           720.0   0.043        -0.012
head_only         0.028   1.35  0.028              1.10           180.0   0.021         0.012
cumulative        0.033   0.80  0.033              1.20           200.0   0.041        -0.005
recency_focus     0.039   1.15  0.039              1.55           400.0   0.034         0.005
zero_shot         0.020   0.60  0.020              0.70            60.0   0.033        -0.008
s1_heavy          0.037   1.10  0.037              1.50           390.0   0.034         0.003

最优方案: head_only (ICIR 最高)
```

保存输出：

```
experiment_results/
├── recipe_comparison.csv      对比表
├── report_conservative.yaml   各方案详细报告
├── report_aggressive.yaml
└── ...
```

| 参数 | 必填 | 默认值 | 说明 |
|------|------|--------|------|
| `--all` | 二选一 | — | 运行全部预设方案 |
| `--recipes` | 二选一 | — | 指定方案名称，逗号分隔 |
| `--start` | 否 | `2024-07-01` | 评估起始日期 |
| `--end` | 否 | `2024-12-31` | 评估截止日期 |
| `--output` | 否 | — | 结果输出目录 |
| `--device` | 否 | `cuda` | 计算设备 |

---

### kronos_recipe_cli.py 方案管理

查看、创建、删除微调方案。所有方案保存在 `configs/kronos_recipes.yaml`。

**查看所有方案：**

```bash
python kronos_recipe_cli.py list
```

输出示例：

```
共 7 个方案:

名称                 Tok   Pred  策略          Epochs  LR         采样               描述
--------------------------------------------------------------------------------------------------------------
conservative         N     Y     last_2        3       2.0e-05    uniform            只微调 predictor 最后2层，安全稳健
aggressive           Y     Y     full          5       1.0e-05    uniform            全部解冻，充分微调
head_only            N     Y     head_only     10      5.0e-05    uniform            只微调 predictor 输出头，最轻量
cumulative           N     Y     last_2        1       5.0e-06    uniform            累积微调，不每天重置
recency_focus        N     Y     last_2        3       2.0e-05    recency_weighted   近期数据加权采样
zero_shot            N     N     none          3       2.0e-05    uniform            零样本基线，不微调
s1_heavy             N     Y     last_2        3       2.0e-05    uniform            加大 s1 粗粒度损失权重
```

**查看方案详情：**

```bash
python kronos_recipe_cli.py show conservative
```

输出示例：

```
方案: conservative
描述: 只微调 predictor 最后2层，安全稳健

  finetune_tokenizer         False
  finetune_predictor         True
  tokenizer_strategy         none
  predictor_strategy         last_n
  predictor_unfreeze_layers  2
  recon_pre_weight           1.0
  recon_full_weight          1.0
  bsq_weight                 1.0
  s1_loss_weight             1.0
  s2_loss_weight             1.0
  data_lookback              30
  sample_strategy            uniform
  epochs                     3
  learning_rate              2e-05
  batch_size                 64
  predict_horizon            5
  sample_count               10
  temperature                0.6
  top_p                      0.9
  reset_from_pretrained      True
```

**创建自定义方案：**

```bash
# 创建一个自定义方案
python kronos_recipe_cli.py create my_recipe \
    --description "自定义实验方案" \
    --predictor-strategy last_n \
    --unfreeze-layers 3 \
    --epochs 5 \
    --lr 1e-5 \
    --sample-strategy recency_weighted \
    --temperature 0.8

# 创建一个微调 tokenizer 的方案
python kronos_recipe_cli.py create tok_experiment \
    --description "测试 tokenizer 微调效果" \
    --finetune-tokenizer \
    --predictor-strategy full \
    --epochs 3
```

**删除方案：**

```bash
python kronos_recipe_cli.py delete my_recipe
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--recipes-file` | `configs/kronos_recipes.yaml` | 方案配置文件 |

---

## Python API 参考

### FinetuneRecipe 微调方案

```python
from quantlab.signal.signal_kronos import FinetuneRecipe

# 加载预设方案
recipe = FinetuneRecipe.load("configs/kronos_recipes.yaml", "conservative")

# 加载全部方案
all_recipes = FinetuneRecipe.load_all("configs/kronos_recipes.yaml")

# 创建自定义方案
custom = FinetuneRecipe(
    name="my_experiment",
    description="测试 s1 权重加大 + 近期加权",
    finetune_predictor=True,
    predictor_strategy="last_n",
    predictor_unfreeze_layers=3,
    s1_loss_weight=2.0,
    s2_loss_weight=0.5,
    sample_strategy="recency_weighted",
    recency_decay=0.93,
    epochs=5,
    learning_rate=1e-5,
)

# 保存方案
custom.save("configs/kronos_recipes.yaml")
```

**FinetuneRecipe 完整字段表：**

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| **模块选择** | | | |
| `finetune_tokenizer` | bool | `False` | 是否微调 tokenizer |
| `finetune_predictor` | bool | `True` | 是否微调 predictor |
| **冻结策略** | | | |
| `tokenizer_strategy` | str | `"none"` | `none`/`last_n`/`head_only`/`full` |
| `tokenizer_unfreeze_layers` | int | `0` | 解冻最后 N 层（`last_n` 时使用） |
| `predictor_strategy` | str | `"last_n"` | `none`/`last_n`/`head_only`/`full` |
| `predictor_unfreeze_layers` | int | `2` | 解冻最后 N 层 |
| **Tokenizer 损失** | | | |
| `recon_pre_weight` | float | `1.0` | s1 重建损失权重 |
| `recon_full_weight` | float | `1.0` | s1+s2 重建损失权重 |
| `bsq_weight` | float | `1.0` | BSQ 量化损失权重 |
| `bsq_beta` | float | `0.05` | commit loss 系数 |
| `bsq_gamma0` | float | `1.0` | 样本熵权重 |
| `bsq_gamma` | float | `1.1` | 码本熵权重 |
| **Predictor 损失** | | | |
| `s1_loss_weight` | float | `1.0` | s1（粗粒度）CE 权重 |
| `s2_loss_weight` | float | `1.0` | s2（细粒度）CE 权重 |
| **数据采样** | | | |
| `data_lookback` | int | `30` | 微调数据窗口天数 |
| `sample_strategy` | str | `"uniform"` | `uniform`/`recency_weighted`/`volatility_stratified` |
| `recency_decay` | float | `0.95` | 近期加权指数衰减率 |
| **训练超参** | | | |
| `epochs` | int | `3` | 微调轮数 |
| `learning_rate` | float | `2e-5` | 学习率 |
| `batch_size` | int | `64` | 批大小 |
| `weight_decay` | float | `0.1` | 权重衰减 |
| `accumulation_steps` | int | `1` | 梯度累积步数 |
| `warmup_ratio` | float | `0.1` | 学习率预热比例 |
| **推理超参** | | | |
| `predict_horizon` | int | `5` | 预测天数 |
| `sample_count` | int | `10` | 采样次数 |
| `temperature` | float | `0.6` | 采样温度 |
| `top_p` | float | `0.9` | nucleus sampling |
| `top_k` | int | `0` | top-k filtering |
| **每日重置** | | | |
| `reset_from_pretrained` | bool | `True` | 每天从预训练权重重新开始 |

---

### KronosFinetuner 微调执行器

```python
from model.kronos import KronosTokenizer, Kronos
from quantlab.signal.signal_kronos import KronosFinetuner, FinetuneRecipe

# 加载预训练模型
tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
model = Kronos.from_pretrained("NeoQuasar/Kronos-base")

# 创建微调器
finetuner = KronosFinetuner(tokenizer, model, device="cuda")

# 准备数据: {symbol: DataFrame[open,high,low,close,volume,amount]}
symbol_data = {"SH600519": df_600519, "SZ000858": df_000858, ...}

# 执行微调
recipe = FinetuneRecipe.load("configs/kronos_recipes.yaml", "conservative")
ft_tokenizer, ft_model = finetuner.finetune(symbol_data, recipe)
```

**关键行为：**

- `reset_from_pretrained=True`：每次微调从预训练权重重新开始（默认）
- `reset_from_pretrained=False`：在上一次微调结果上继续（累积模式）
- 预训练模型权重永远不会被修改（内部深拷贝）

---

### KronosInference 推理执行器

```python
from quantlab.signal.signal_kronos import KronosInference, FinetuneRecipe

inference = KronosInference(device="cuda", max_context=512)

output = inference.predict_all(
    tokenizer=ft_tokenizer,
    model=ft_model,
    symbol_data=symbol_data,
    recipe=recipe,
    anchor_date="2024-06-28",
)

# 输出
print(output.return_1d)      # Series: symbol → T+1 预测收益
print(output.return_5d)      # Series: symbol → T+1~T+5 预测收益
print(output.uncertainty)    # Series: symbol → 预测分歧度（多采样标准差）
print(output.pred_klines)    # Dict: symbol → 预测 K 线 DataFrame
```

**KronosOutput 字段：**

| 字段 | 类型 | 说明 |
|------|------|------|
| `return_1d` | `Series[symbol→float]` | T+1 预测收益率（多采样均值） |
| `return_5d` | `Series[symbol→float]` | T+1~T+5 预测收益率（多采样均值） |
| `uncertainty` | `Series[symbol→float]` | 预测分歧度（多采样标准差，越大越不确定） |
| `pred_klines` | `Dict[symbol, DataFrame]` | 完整预测 K 线（含 OHLCV，供可视化） |

**推理流程：**

1. 按序列长度将股票分组（`predict_batch` 要求同长度）
2. 每组执行 `sample_count` 次采样推理
3. 聚合：`return_1d = mean(第1天预测close / 当前close - 1)`，`uncertainty = std(5日预测收益)`

---

### KronosSignalPipeline 信号生产

日常回测/实盘的入口，封装微调 + 推理一体化流程。

```python
from quantlab.signal.signal_kronos import FinetuneRecipe, KronosSignalPipeline

recipe = FinetuneRecipe.load("configs/kronos_recipes.yaml", "conservative")
pipeline = KronosSignalPipeline(
    recipe=recipe,
    device="cuda",
    max_context=512,
)

# 单日运行：微调 + 推理
output = pipeline.daily_run(symbol_data, "2024-06-28", dm)
# output.return_1d → 传给 M5 融合

# 切换方案
new_recipe = FinetuneRecipe.load("configs/kronos_recipes.yaml", "aggressive")
pipeline.switch_recipe(new_recipe)
```

**daily_run 内部流程：**

```
1. 延迟加载预训练模型（首次调用时）
2. 调用 KronosFinetuner.finetune(symbol_data, recipe)
   - 如果 recipe 是 zero_shot，跳过微调
3. 调用 KronosInference.predict_all(tokenizer, model, symbol_data, recipe, anchor_date)
4. 返回 KronosOutput
```

---

### FinetuneExperiment 实验管理

```python
from quantlab.signal.signal_kronos import FinetuneRecipe, FinetuneExperiment

recipes = FinetuneRecipe.load_all("configs/kronos_recipes.yaml")
exp = FinetuneExperiment(
    data_manager=dm,
    recipes=recipes,
    eval_start="2024-07-01",
    eval_end="2024-12-31",
    device="cuda",
)

# 运行全部方案
exp.run_all()

# 查看对比表
print(exp.compare())

# 获取最优方案
best = exp.get_best_recipe()
print(f"最优方案: {best.name}")

# 保存结果
exp.save_results("./experiment_results")
```

**运行单个方案：**

```python
custom_recipe = FinetuneRecipe(
    name="custom_test",
    finetune_predictor=True,
    predictor_strategy="last_n",
    predictor_unfreeze_layers=3,
    s1_loss_weight=2.0,
)
report = exp.run_experiment(custom_recipe)
print(f"IC={report.ic_mean:.4f}, ICIR={report.icir:.2f}, Sharpe={report.long_short_sharpe:.2f}")
```

**RecipeReport 字段：**

| 字段 | 说明 |
|------|------|
| `ic_mean` | 全周期 Rank IC 均值 |
| `icir` | IC 信息比率 |
| `ic_1d` | T+1 日频 IC |
| `ic_5d` | T+5 日频 IC |
| `long_short_sharpe` | 多空组合年化夏普 |
| `long_only_return` | 纯多头年化收益 |
| `finetune_time_sec` | 单次微调平均耗时 |
| `predict_time_sec` | 单次推理平均耗时 |
| `total_time_sec` | 总平均耗时 |
| `ic_std` | IC 标准差（越小越稳定） |
| `worst_week_ic` | 最差一周的 IC 均值 |

---

## Kronos 模型原理简介

Kronos 是一个 Decoder-only Transformer 模型，专门用于 K 线数据预测。采用两阶段架构：

```
阶段1 Tokenizer: OHLCV K线 → Encoder → BSQ量化 → Decoder → 重建K线
   目的: 将连续的K线数据压缩为离散token
   输入: [open, high, low, close, volume, amount] × seq_len
   输出: (s1_indices, s2_indices) — 粗粒度+细粒度的离散token

阶段2 Predictor: token序列 → Transformer → 预测下一个token
   目的: 学习token序列的时间规律
   输入: (s1_ids, s2_ids) × seq_len
   输出: (s1_logits, s2_logits) — 下一个token的概率分布
```

**BSQ（Binary Spherical Quantization）：**

- 将连续向量量化为二进制码（`{-1, +1}^d`）
- 码本大小 = `2^s1_bits × 2^s2_bits`
- s1 是粗粒度（捕捉大方向），s2 是细粒度（捕捉细节）

**推理过程（自回归生成）：**

```
1. 历史K线 → Tokenizer.encode → token序列
2. for each step:
     token序列 → Predictor.decode_s1 → 采样 s1_token
     s1_token + context → Predictor.decode_s2 → 采样 s2_token
     将新token添加到序列
3. 全部token → Tokenizer.decode → 预测K线
```

**数据归一化：**

- 每只股票独立做 z-score 归一化（mean/std）
- 裁剪到 [-5, 5] 防止极端值
- 推理完成后反归一化得到真实价格

---

## 微调方案设计指南

### 7 个预设方案的适用场景

| 方案 | 适用场景 | 特点 |
|------|----------|------|
| **conservative** | 日常使用推荐 | 只调 predictor 最后2层，稳健不过拟合 |
| **aggressive** | 数据量充足时 | 全部解冻，更强拟合但风险过拟合 |
| **head_only** | 快速迭代/GPU 显存受限 | 最轻量，只调输出头 |
| **cumulative** | 连续运行场景 | 不每天重置，在昨天基础上继续 |
| **recency_focus** | 市场风格切换频繁时 | 近期数据权重更高 |
| **zero_shot** | 对照基线 | 不微调，直接用预训练模型推理 |
| **s1_heavy** | 偏好粗粒度方向性 | 加大 s1（大方向）损失权重 |

### 冻结策略选择

```
保守 ← ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ → 激进

none    head_only    last_n=2    last_n=4    full
(零样本)  (最轻量)    (推荐起点)   (中等)    (全解冻)
```

- 数据量少 → 保守（`head_only` 或 `last_n=1`）
- 数据量大 → 可以更激进（`last_n=4` 或 `full`）
- Tokenizer 一般不需要微调（除非数据分布与预训练差异很大）

### 数据采样策略

| 策略 | 原理 | 适用场景 |
|------|------|----------|
| `uniform` | 所有窗口等概率 | 默认选择 |
| `recency_weighted` | 近期窗口概率更高（`decay^天数`） | 市场风格变化快时 |
| `volatility_stratified` | 按波动率分层等量采样 | 防止高波动股主导训练 |

### 推理参数调优

| 参数 | 低值效果 | 高值效果 |
|------|----------|----------|
| `temperature` (0.1~2.0) | 更确定、多样性低 | 更随机、多样性高 |
| `sample_count` (1~20) | 快但方差大 | 慢但估计更稳 |
| `top_p` (0.5~1.0) | 只保留高概率token | 保留更多候选 |

推荐起点：`temperature=0.6, sample_count=10, top_p=0.9`

---

## 配置说明

**方案配置文件 `configs/kronos_recipes.yaml`：**

```yaml
recipes:
  - name: conservative
    description: "只微调 predictor 最后2层，安全稳健"
    finetune_tokenizer: false
    finetune_predictor: true
    tokenizer_strategy: "none"
    predictor_strategy: "last_n"
    predictor_unfreeze_layers: 2
    epochs: 3
    learning_rate: 2.0e-5
    reset_from_pretrained: true

  - name: zero_shot
    description: "零样本基线，不微调"
    finetune_tokenizer: false
    finetune_predictor: false
    # ...
```

**回测配置 `configs/backtest.yaml` 中与 M3 相关的字段：**

```yaml
# M3 Kronos
kronos_recipe_name: "conservative"   # 使用的微调方案
kronos_device: "cuda"                # 计算设备
```

**预训练模型路径：**

| 模型 | HuggingFace ID | 说明 |
|------|----------------|------|
| Tokenizer (base) | `NeoQuasar/Kronos-Tokenizer-base` | BSQ 量化编解码器 |
| Predictor (base) | `NeoQuasar/Kronos-base` | 自回归 Transformer |

---

## 运行测试

```bash
# 全部 M3 测试
python -m pytest quantlab/tests/test_signal_kronos.py -v

# 离线测试（不需要 Kronos 模型和 Qlib 数据）
python -m pytest quantlab/tests/test_signal_kronos.py::TestFinetuneRecipe -v
python -m pytest quantlab/tests/test_signal_kronos.py::TestKronosOutput -v
python -m pytest quantlab/tests/test_signal_kronos.py::TestRecipeReport -v

# 需要 Kronos 模型的测试
python -m pytest quantlab/tests/test_signal_kronos.py::TestKronosFinetuner -v
python -m pytest quantlab/tests/test_signal_kronos.py::TestKronosInference -v

# 端到端测试（需要 Kronos 模型 + Qlib 数据）
python -m pytest quantlab/tests/test_signal_kronos.py::TestKronosSignalPipeline -v
```

---

## 常见问题

### Q: 首次运行需要下载什么？

两个东西：
1. **Qlib 数据**（M1 已完成）
2. **Kronos 预训练模型**：`python kronos_data.py download-models`

模型约 170MB，首次下载后缓存到 `~/.cache/huggingface/hub/`。

### Q: 没有 GPU 能用吗？

可以，但非常慢。设置 `--device cpu`：

```bash
python kronos_predict.py --anchor-date 2024-06-28 --device cpu
```

推荐至少有一块 NVIDIA GPU（8GB+ 显存）。

### Q: 微调耗时多久？

取决于方案和数据量。CSI300（约 300 只股票）参考耗时：

| 方案 | GPU (A100) | GPU (RTX 3090) |
|------|------------|----------------|
| zero_shot | ~1分钟 | ~2分钟 |
| head_only | ~3分钟 | ~8分钟 |
| conservative | ~6分钟 | ~15分钟 |
| aggressive | ~12分钟 | ~30分钟 |

### Q: 如何选择微调方案？

建议流程：

1. 先跑 `python kronos_experiment.py --all` 在你的数据上做一次全面对比
2. 看 ICIR（信息比率）最高的方案
3. 如果 ICIR 差不多，选 `total_time_sec` 最短的（性价比）
4. 日常使用用 `conservative` 即可

### Q: `reset_from_pretrained` 设 True 还是 False？

- **True（默认推荐）**：每天从预训练权重重新微调。优点是稳定，不会因为某天数据异常导致模型偏移。
- **False（累积模式）**：在昨天微调结果上继续。优点是收敛更快（只需 1 epoch），缺点是可能漂移。

如果不确定，用 `True`。想省时间可以试 `cumulative` 方案。

### Q: 多采样推理的 `sample_count` 设多少？

- 快速预览：`sample_count=1`（结果方差大）
- 正常使用：`sample_count=10`（推荐）
- 高精度：`sample_count=20`（耗时翻倍但更稳）

`uncertainty` 值就是多次采样的标准差，`sample_count` 太低时 `uncertainty` 估计不准。

### Q: 不确定性信号有什么用？

`uncertainty` 代表模型对预测的置信程度：

- **低不确定性** → 模型对预测有信心 → 适合大仓位
- **高不确定性** → 模型不确定 → 应减仓或跳过

在 M5 信号融合时，`uncertainty` 可作为加权因子，对低不确定性的信号赋予更高权重。

### Q: 与 Kronos 原版 finetune/ 目录的区别？

| | 原版 `Kronos/finetune/` | QuantLab `kronos_finetune.py` |
|---|---|---|
| 训练方式 | DDP 多卡分布式 | 单卡 |
| 配置方式 | `config.py` 硬编码 | YAML recipe 可配置 |
| epoch 数 | 30 | 1-10（日频快速微调） |
| 适用场景 | 一次性大规模微调 | 每日滚动微调 |
| 冻结策略 | 手动改代码 | recipe 配置 |

如果需要大规模从头微调（30+ epoch），建议使用原版 `Kronos/finetune/` 目录的脚本。

### Q: 怎么把 Kronos 信号和 Alpha158 信号一起用？

这是 M5（信号融合）的工作。简单来说：

```python
# M2 信号
alpha_signal = alpha_pipeline.predict(date, dm)  # Series[symbol → score]

# M3 信号
kronos_output = kronos_pipeline.daily_run(symbol_data, date, dm)
kronos_signal = kronos_output.return_1d  # Series[symbol → return]

# M5 融合（简化示例）
fused = 0.6 * alpha_signal + 0.4 * kronos_signal
```

实际融合时会用 IC 加权、不确定性惩罚等更精细的策略。
