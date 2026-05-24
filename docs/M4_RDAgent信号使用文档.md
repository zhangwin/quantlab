# M4 RD-Agent 信号管线 使用文档

## 目录

- [模块简介](#模块简介)
- [环境准备](#环境准备)
- [快速开始](#快速开始)
- [脚本详细说明](#脚本详细说明)
  - [rdagent_data.py 数据准备](#rdagent_datapy-数据准备)
  - [rdagent_evolve.py 进化执行](#rdagent_evolvepy-进化执行)
  - [rdagent_compute.py 信号计算](#rdagent_computepy-信号计算)
  - [rdagent_registry_cli.py 注册表管理](#rdagent_registry_clipy-注册表管理)
- [Python API 参考](#python-api-参考)
  - [EvolutionConfig 进化配置](#evolutionconfig-进化配置)
  - [CodeFactorRegistry 因子注册表](#codefactorregistry-因子注册表)
  - [CodeFactorExecutor 沙箱执行器](#codefactorexecutor-沙箱执行器)
  - [EvolutionRunner 进化执行器](#evolutionrunner-进化执行器)
  - [RDAgentSignalPipeline 信号管线](#rdagentsignalpipeline-信号管线)
- [架构与原理](#架构与原理)
  - [进化循环](#进化循环)
  - [因子生命周期](#因子生命周期)
  - [沙箱安全机制](#沙箱安全机制)
  - [因子分流](#因子分流)
- [进化配置设计指南](#进化配置设计指南)
- [因子代码规范](#因子代码规范)
- [配置说明](#配置说明)
- [运行测试](#运行测试)
- [常见问题](#常见问题)

---

## 模块简介

M4 RD-Agent 信号管线是 QuantLab 三条信号管线之一（管线 C），负责：

1. **因子进化** — 调用 RD-Agent 的 LLM 进化循环，自动生成 Python 代码因子
2. **沙箱计算** — 在隔离环境中安全执行 LLM 生成的因子代码
3. **生命周期管理** — 跟踪每个因子的 IC 衰退，自动执行 active → probation → retired 状态切换
4. **信号合成** — 将多个因子的输出按 ICIR 加权合成为统一信号

所有脚本位于 `quantlab/signal/` 目录下。

**与其他模块的关系：**

```
                    RD-Agent 进化循环
                         ↓
                    因子代码候选
                    ↓           ↓
              Qlib 表达式    Python 代码因子
                ↓                   ↓
M1 DataManager ──→ M2 Alpha158  M4 RDAgent ──→ M5 信号融合
                                    ↓
                              signal（加权合成信号）
                              factor_signals（各因子独立信号）
                              factor_weights（ICIR 权重）
```

**与管线 A（M2）/ 管线 B（M3）的区别：**

| | M2 Alpha158 | M3 Kronos | M4 RD-Agent |
|---|---|---|---|
| 因子来源 | 人工定义 Qlib 表达式 | 端到端深度学习 | LLM 自动进化 |
| 因子类型 | 表达式因子 | K 线预测 | Python 代码因子 |
| 生成方式 | 手动或从文献 | 预训练模型微调 | 进化循环 (Propose→Code→Run→Evaluate) |
| 可扩展性 | 需人工干预 | 固定模型结构 | 全自动发现新因子 |
| 安全性 | 内置 Qlib 沙箱 | 模型推理，无安全风险 | 需 AST 静态检查 + 子进程沙箱 |
| 更新频率 | 定期人工更新 | 每日微调 | 每周末或按需进化 |

---

## 环境准备

M1 环境已装好的前提下，M4 核心模块**无额外依赖**（仅需 numpy、pandas、pyyaml）。

如需使用 RD-Agent 进化循环（非模板模式），还需：

```bash
cd RD-Agent
pip install -e .
```

并配置 `.env` 文件设置 LLM 后端（参考 RD-Agent 文档）。

> **注意：** 即使不安装 RD-Agent，M4 也可正常工作——进化执行器会退化为**模板因子生成模式**，提供 6 个内置因子模板。

确认 M1 数据已就绪：

```bash
cd quantlab/data
python data_status.py
```

---

## 快速开始

```bash
cd quantlab/signal

# 第一步：初始化因子目录和注册表
python rdagent_data.py init

# 第二步：查看可用的进化配置
python rdagent_registry_cli.py configs

# 第三步：执行因子进化（模板模式，无需 RD-Agent）
python rdagent_evolve.py --config mean_revert_focus --template-only

# 第四步：查看注册的因子
python rdagent_registry_cli.py list

# 第五步：查看某个因子详情（含代码）
python rdagent_registry_cli.py show rdagent_mean_rev_ma --show-code

# 第六步：单日信号计算
python rdagent_compute.py --anchor-date 2024-06-28

# 第七步：批量回测
python rdagent_compute.py --start 2024-01-01 --end 2024-06-30 --output signals.csv

# 第八步：衰退检查
python rdagent_compute.py --anchor-date 2024-06-28 --check-decay
```

**完整 RD-Agent 进化模式（需安装 RD-Agent + 配置 LLM）：**

```bash
# 使用 RD-Agent 进化循环生成因子
python rdagent_evolve.py --config broad_explore

# 限制轮数
python rdagent_evolve.py --config mean_revert_focus --max-rounds 5 --budget 10
```

---

## 脚本详细说明

### rdagent_data.py 数据准备

初始化因子目录、导入外部因子、检查系统状态。

**初始化：**

```bash
# 创建因子目录和空注册表
python rdagent_data.py init

# 指定自定义路径
python rdagent_data.py init \
    --code-dir ./data/rdagent/factors \
    --registry ./data/rdagent/registry.yaml
```

初始化后的目录结构：

```
data/rdagent/
├── factors/          因子代码文件目录（每个因子一个 .py 文件）
└── registry.yaml     因子注册表（元数据、生命周期状态）
```

**导入外部因子：**

```bash
# 从目录批量导入
python rdagent_data.py import-factors \
    --source-dir ./external_factors \
    --direction mean_revert

# 从 RD-Agent workspace 导入
python rdagent_data.py import-workspace \
    --workspace-path ./rdagent_output/workspace_001
```

导入时会自动执行静态检查，不合规的代码会被跳过。

**查看状态：**

```bash
python rdagent_data.py status
```

输出示例：

```
因子目录: data/rdagent/factors
注册表: data/rdagent/registry.yaml

因子总数: 6
  active: 4
  probation: 1
  retired: 1

方向分布:
  mean_revert: 2
  volatility_anomaly: 2
  liquidity_change: 1
  momentum_divergence: 1

活跃因子权重:
  最大: 0.3500
  最小: 0.1200
  总和: 1.0000

进化配置: 4 个
  mean_revert_focus: ['mean_revert', 'overreaction']
  volatility_anomaly: ['volatility_anomaly', 'regime_change']
  liquidity_focus: ['liquidity_change', 'market_microstructure']
  broad_explore: ['mean_revert', 'volatility_anomaly', ...]
```

**验证因子代码：**

```bash
# 对所有注册因子执行静态检查
python rdagent_data.py validate
```

**清理退役因子：**

```bash
# 查看退役因子
python rdagent_data.py cleanup

# 删除退役因子的代码文件（注册表记录保留）
python rdagent_data.py cleanup --remove-retired
```

| 子命令 | 说明 |
|--------|------|
| `init` | 初始化因子目录和注册表 |
| `import-factors` | 从目录批量导入因子代码 |
| `import-workspace` | 从 RD-Agent workspace 导入因子 |
| `status` | 查看因子系统状态 |
| `validate` | 验证所有因子代码 |
| `cleanup` | 清理退役因子代码文件 |

---

### rdagent_evolve.py 进化执行

触发因子进化循环，产出经过验证的新因子并注册到因子池。

**模板模式（无需 RD-Agent）：**

```bash
# 生成均值回复方向的模板因子
python rdagent_evolve.py --config mean_revert_focus --template-only

# 生成广泛探索方向的模板因子
python rdagent_evolve.py --config broad_explore --template-only
```

输出示例：

```
进化配置: mean_revert_focus
  方向: ['mean_revert', 'overreaction']
  最大轮数: 20
  总预算: 50
  正交约束: corr_alpha<0.3, corr_pool<0.5

注册表: data/rdagent/registry.yaml
  已有因子: 0
  活跃因子: 0

模式: 模板因子生成（不调用 RD-Agent）

模板因子: 4
注册成功: 4
  + rdagent_mean_rev_ma (mean_revert)
  + rdagent_rsi_divergence (mean_revert)
  + rdagent_vol_spike (volatility_anomaly)
  + rdagent_range_vol (volatility_anomaly)
```

**RD-Agent 进化模式：**

```bash
# 完整进化循环
python rdagent_evolve.py --config mean_revert_focus

# 限制轮数和预算
python rdagent_evolve.py --config volatility_anomaly --max-rounds 5 --budget 10

# 带数据验证（沙箱试运行）
python rdagent_evolve.py --config mean_revert_focus \
    --data-dir ~/.qlib/qlib_data/cn_data \
    --market csi300
```

进化流程：

```
加载配置 → 初始化注册表
      ↓
尝试 RD-Agent FactorRDLoop
      ↓ (失败时退化)
模板因子生成
      ↓
对每个候选因子:
  1. 静态检查（AST）
  2. 沙箱试运行（如有数据）
  3. IC/ICIR 验证
  4. 正交性检查
  5. 分流（Qlib 表达式 → M2，Python 代码 → M4）
  6. 注册
      ↓
输出报告
```

| 参数 | 必填 | 默认值 | 说明 |
|------|------|--------|------|
| `--config` | 是 | — | 进化配置名称 |
| `--configs-file` | 否 | `configs/rdagent_evolution.yaml` | 配置文件路径 |
| `--registry` | 否 | `data/rdagent/registry.yaml` | 注册表路径 |
| `--code-dir` | 否 | `data/rdagent/factors` | 因子代码目录 |
| `--max-rounds` | 否 | 配置文件中的值 | 覆盖最大轮数 |
| `--budget` | 否 | 配置文件中的值 | 覆盖总预算 |
| `--template-only` | 否 | 否 | 只使用模板因子 |
| `--data-dir` | 否 | `~/.qlib/qlib_data/cn_data` | Qlib 数据目录 |
| `--market` | 否 | `csi300` | 股票池 |
| `--output` | 否 | `outputs/rdagent` | 输出目录 |

---

### rdagent_compute.py 信号计算

用注册表中的 active 因子计算加权合成信号，是**日常使用的主入口**。

**单日计算：**

```bash
# 基本用法
python rdagent_compute.py --anchor-date 2024-06-28

# 查看 Top-30
python rdagent_compute.py --anchor-date 2024-06-28 --top-k 30

# 输出到 CSV
python rdagent_compute.py --anchor-date 2024-06-28 --output signal_20240628.csv

# 同时执行衰退检查
python rdagent_compute.py --anchor-date 2024-06-28 --check-decay
```

输出示例：

```
注册表: data/rdagent/registry.yaml
活跃因子: 4
  - rdagent_mean_rev_ma (w=0.350, mean_revert)
  - rdagent_rsi_divergence (w=0.250, mean_revert)
  - rdagent_vol_spike (w=0.220, volatility_anomaly)
  - rdagent_range_vol (w=0.180, volatility_anomaly)
数据源: ~/.qlib/qlib_data/cn_data (csi300)

日期: 2024-06-28
--------------------------------------------------
  因子数: 4
  失败因子: 无
  覆盖股票: 298
  因子权重:
    rdagent_mean_rev_ma: 0.3500
    rdagent_rsi_divergence: 0.2500
    rdagent_vol_spike: 0.2200
    rdagent_range_vol: 0.1800

  Top-20 股票:
    SH600519: 0.8742
    SZ000858: 0.8531
    SH601318: 0.8216
    ...

  Bottom-5 股票:
    SZ002714: 0.1205
    ...
```

**批量回测：**

```bash
# 区间回测
python rdagent_compute.py --start 2024-01-01 --end 2024-06-30

# 输出信号矩阵（日期 × 股票）
python rdagent_compute.py --start 2024-01-01 --end 2024-06-30 \
    --output signals_rdagent.csv

# 区间回测 + 每日衰退检查
python rdagent_compute.py --start 2024-01-01 --end 2024-06-30 --check-decay
```

**衰退检查输出示例：**

```
  衰退检查:
    [✓] rdagent_mean_rev_ma: stable (30d IC: 0.035)
    [⚠] rdagent_vol_spike: declining (30d IC: 0.015)
        动作: warn - IC 下降 (30d: 0.015)
    [✗] rdagent_old_factor: collapsed (30d IC: -0.002)
        动作: probation - IC 崩塌 (-0.002)
    建议触发进化: Active 因子数不足 (2 < 3)
```

| 参数 | 必填 | 默认值 | 说明 |
|------|------|--------|------|
| `--anchor-date` | 二选一 | — | 单日计算日期 |
| `--start` / `--end` | 二选一 | — | 批量回测区间 |
| `--registry` | 否 | `data/rdagent/registry.yaml` | 注册表路径 |
| `--code-dir` | 否 | `data/rdagent/factors` | 因子代码目录 |
| `--data-dir` | 否 | `~/.qlib/qlib_data/cn_data` | Qlib 数据目录 |
| `--market` | 否 | `csi300` | 股票池 |
| `--window` | 否 | `60` | OHLCV 回看窗口天数 |
| `--sandbox` | 否 | `subprocess` | 沙箱模式：`subprocess`（安全）或 `inprocess`（快速） |
| `--timeout` | 否 | `30` | 因子执行超时（秒） |
| `--top-k` | 否 | `20` | 显示 Top-K 股票 |
| `--output` | 否 | — | 信号输出 CSV 路径 |
| `--check-decay` | 否 | 否 | 同时执行衰退检查 |

---

### rdagent_registry_cli.py 注册表管理

因子注册表的增删改查工具，支持查看因子详情、手动注册/退役、更新权重、导出等。

**列出因子：**

```bash
# 全部因子
python rdagent_registry_cli.py list

# 只看活跃因子
python rdagent_registry_cli.py list --status active

# 只看退役因子
python rdagent_registry_cli.py list --status retired
```

**查看因子详情：**

```bash
# 基本信息
python rdagent_registry_cli.py show rdagent_vol_spike

# 包含代码
python rdagent_registry_cli.py show rdagent_vol_spike --show-code
```

输出示例：

```
因子: rdagent_vol_spike
  状态: active
  方向: volatility_anomaly
  来源配置: mean_revert_focus
  来源轮次: 0
  创建日期: 2024-06-28
  描述: Volume spike relative to 20-day average
  权重: 0.2200
  创建时 IC: 0.0000
  创建时 ICIR: 0.0000
  与 Alpha 相关: 0.0000
  衰退警告: 0

代码 (rdagent_vol_spike.py):
------------------------------------------------------------
import numpy as np
import pandas as pd

def compute_factor(ohlcv):
    """Volume spike: current volume / 20-day mean volume."""
    result = {}
    for sym, df in ohlcv.items():
        if len(df) < 20:
            continue
        vol = df["volume"].values if "volume" in df.columns else df["vol"].values
        mean_vol = np.mean(vol[-20:])
        if mean_vol > 0:
            result[sym] = vol[-1] / mean_vol - 1
        else:
            result[sym] = 0.0
    return pd.Series(result)
------------------------------------------------------------
```

**手动注册因子：**

```bash
# 从文件注册
python rdagent_registry_cli.py register \
    --name my_factor \
    --code-file ./factors/my_factor.py \
    --direction mean_revert \
    --description "My custom factor"

# 跳过静态检查（不推荐）
python rdagent_registry_cli.py register \
    --name my_factor \
    --code-file ./factors/my_factor.py \
    --direction mean_revert \
    --force
```

**退役和恢复：**

```bash
# 退役因子
python rdagent_registry_cli.py retire rdagent_vol_spike --reason "IC 持续衰退"

# 恢复因子为 active
python rdagent_registry_cli.py activate rdagent_vol_spike
```

**更新权重：**

```bash
# 手动指定 ICIR 值，系统自动归一化
python rdagent_registry_cli.py update-weights \
    --icir "rdagent_mean_rev_ma:1.2,rdagent_vol_spike:0.8,rdagent_rsi_divergence:0.5"
```

**导出：**

```bash
# 导出为 JSON
python rdagent_registry_cli.py export --format json --output registry.json

# 导出为 JSON（含代码）
python rdagent_registry_cli.py export --format json --include-code --output registry_full.json

# 导出为 CSV（概览信息）
python rdagent_registry_cli.py export --format csv --output registry.csv
```

**查看进化配置：**

```bash
# 列出全部配置
python rdagent_registry_cli.py configs

# 查看某个配置详情
python rdagent_registry_cli.py configs --name mean_revert_focus
```

| 子命令 | 说明 |
|--------|------|
| `list` | 列出因子（可按状态过滤） |
| `show <name>` | 查看因子详情 |
| `register` | 手动注册因子 |
| `retire <name>` | 退役因子 |
| `activate <name>` | 恢复因子为 active |
| `update-weights` | 更新权重 |
| `export` | 导出注册表 |
| `configs` | 查看进化配置 |

---

## Python API 参考

### EvolutionConfig 进化配置

完整描述"如何驱动 RD-Agent 进化"的配置。

```python
from quantlab.signals.signal_rdagent import EvolutionConfig

# 从 YAML 加载
config = EvolutionConfig.load("configs/rdagent_evolution.yaml", "mean_revert_focus")

# 查看字段
print(config.name)                  # "mean_revert_focus"
print(config.target_directions)     # ["mean_revert", "overreaction"]
print(config.max_rounds)            # 20
print(config.total_budget)          # 50
print(config.min_ic)                # 0.02
print(config.min_icir)              # 0.5
print(config.max_corr_with_alpha)   # 0.3

# 加载全部配置
all_configs = EvolutionConfig.load_all("configs/rdagent_evolution.yaml")

# 程序化创建
config = EvolutionConfig(
    name="custom",
    target_directions=["mean_revert"],
    max_rounds=10,
    total_budget=20,
    min_ic=0.03,
)

# 保存到 YAML
config.save("path/to/config.yaml")
```

**字段说明：**

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `name` | str | `"default"` | 配置名称 |
| `target_directions` | List[str] | `["mean_revert", "volatility_anomaly"]` | 目标进化方向 |
| `direction_prompt` | str | `""` | 方向提示词（注入 LLM） |
| `forbidden_patterns` | List[str] | `["future_data", ...]` | 禁止模式 |
| `max_corr_with_alpha` | float | `0.3` | 与 M2 因子最大相关性 |
| `max_corr_within_pool` | float | `0.5` | 因子池内最大相关性 |
| `min_ic` | float | `0.02` | 最低 IC 门槛 |
| `min_icir` | float | `0.5` | 最低 ICIR 门槛 |
| `max_overfit_gap` | float | `0.05` | 最大过拟合差距 |
| `validation_split` | float | `0.3` | 验证集比例 |
| `max_rounds` | int | `20` | 最大进化轮数 |
| `max_factors_per_round` | int | `5` | 每轮最大因子数 |
| `total_budget` | int | `50` | 总候选因子预算 |
| `timeout_hours` | float | `4.0` | 进化超时（小时） |
| `execution_env` | str | `"subprocess"` | 执行环境 |
| `max_code_lines` | int | `100` | 因子代码最大行数 |

---

### CodeFactorRegistry 因子注册表

管理 RD-Agent 产出的 Python 因子代码，提供 CRUD 操作和 ICIR 权重计算。

```python
from quantlab.signals.signal_rdagent import CodeFactorRegistry

# 初始化
registry = CodeFactorRegistry(
    factor_code_dir="data/rdagent/factors",
    registry_path="data/rdagent/registry.yaml",
)

# 注册因子
entry = registry.register(
    name="my_factor",
    code=factor_code_string,
    direction="mean_revert",
    description="My custom factor",
    ic=0.05,
    icir=1.2,
)

# 查询
all_entries = registry.get_all()           # 全部因子
active = registry.get_active()             # active + enabled
entry = registry.get("my_factor")          # 指定因子
code = registry.load_code("my_factor")     # 读取代码

# 生命周期操作
registry.set_probation("my_factor")        # → probation
registry.retire("my_factor", "IC 衰退")    # → retired

# 更新 IC 历史
registry.update_ic("my_factor", "2024-06-28", 0.035)

# 按 ICIR 重新计算权重
registry.update_weights({"my_factor": 1.2, "other_factor": 0.8})

# 因子概览
df = registry.summary()    # DataFrame: name, status, direction, weight, ...

# 持久化
registry.save()
```

**注册表存储格式（registry.yaml）：**

```yaml
factors:
  - name: rdagent_vol_spike
    code_path: rdagent_vol_spike.py
    source_round: 0
    source_config: mean_revert_focus
    direction: volatility_anomaly
    status: active
    enabled: true
    weight: 0.22
    ic_at_creation: 0.05
    icir_at_creation: 1.2
    ic_history:
      - ["2024-06-28", 0.035]
      - ["2024-07-01", 0.032]
    decay_warnings: 0
```

**权重计算逻辑：**

```
weight_i = max(0, ICIR_i) / Σ max(0, ICIR_j)

当所有 ICIR ≤ 0 时，退化为等权: weight_i = 1/N
```

---

### CodeFactorExecutor 沙箱执行器

在隔离环境中安全执行 LLM 生成的 Python 因子代码。

```python
from quantlab.signals.signal_rdagent import CodeFactorExecutor

# 初始化（subprocess 模式）
executor = CodeFactorExecutor(sandbox_mode="subprocess", timeout_sec=30)

# 静态检查
ok, err = executor.static_check(code, max_lines=100)
if not ok:
    print(f"检查失败: {err}")

# 执行单个因子
result = executor.execute_factor(code, ohlcv_data)
if result.success:
    print(result.values)       # pd.Series，index=股票代码
    print(result.elapsed_ms)   # 耗时（毫秒）
else:
    print(result.error)

# 批量执行
results = executor.execute_batch(entries, ohlcv_data, registry)
for name, result in results.items():
    print(f"{name}: {'OK' if result.success else result.error}")
```

**沙箱模式：**

| 模式 | 安全性 | 速度 | 说明 |
|------|--------|------|------|
| `subprocess` | 高 | 中 | **默认**。独立子进程，pickle 传输数据 |
| `inprocess` | 低 | 快 | 当前进程直接 exec，仅用于测试 |
| `docker` | 最高 | 慢 | 预留，当前退化为 subprocess |

**静态检查规则：**

| 检查项 | 说明 |
|--------|------|
| 行数限制 | 默认 100 行 |
| AST 解析 | 代码必须无语法错误 |
| 禁止 import | `os`, `sys`, `subprocess`, `socket`, `requests`, `urllib`, `shutil`, `pathlib`, `ctypes`, `multiprocessing`, `threading`, `signal`, `io`, `pickle`, `shelve`, `webbrowser`, `ftplib`, `smtplib`, `telnetlib`, `xmlrpc` |
| 禁止内置函数 | `exec`, `eval`, `compile`, `__import__`, `breakpoint` |
| 函数签名 | 必须定义 `compute_factor()` 函数 |

**允许的 import：** `numpy`, `pandas`, `scipy`, `sklearn`, `math`, `collections`, `functools`, `itertools`, `operator`, `statistics` 等纯计算库。

---

### EvolutionRunner 进化执行器

包装 RD-Agent 进化循环，离线运行，产出经过验证的因子代码。

```python
from quantlab.signals.signal_rdagent import (
    EvolutionConfig, CodeFactorRegistry, EvolutionRunner,
)

config = EvolutionConfig.load("configs/rdagent_evolution.yaml", "mean_revert_focus")
registry = CodeFactorRegistry("data/rdagent/factors", "data/rdagent/registry.yaml")

runner = EvolutionRunner(
    config=config,
    code_registry=registry,
    alpha_registry=None,     # 可选：M2 FactorRegistry，用于正交性检查
    validator=None,          # 可选：M2 FactorValidator
)

# 方式一：完整进化循环（尝试 RD-Agent，失败退化为模板）
report = runner.run_evolution(data_manager=dm)
print(f"候选: {report.total_candidates}, 注册: {report.registered}")

# 方式二：只生成模板因子
raw_factors = runner._generate_template_factors()
registered = runner.validate_and_register(raw_factors, data_manager=dm)

# 方式三：从 workspace 导入
raw_factors = runner.extract_factors("/path/to/workspace")
registered = runner.validate_and_register(raw_factors)
```

**验证注册流程：**

```
候选因子
  ↓
1. 静态检查（AST）────── 不通过 → 拒绝
  ↓
2. 沙箱试运行 ─────── 执行失败 → 拒绝
  ↓
3. IC/ICIR 验证 ────── 低于阈值 → 拒绝
  ↓
4. 正交性检查 ─────── 与已有因子高度相关 → 拒绝
  ↓
5. 分流判断
  ├── Qlib 表达式 → 注入 M2 FactorRegistry
  └── Python 代码 → 注册到 M4 CodeFactorRegistry
```

---

### RDAgentSignalPipeline 信号管线

日常回测/实盘入口，用注册表中的 active 因子计算加权合成信号。

```python
from quantlab.signals.signal_rdagent import (
    CodeFactorRegistry, CodeFactorExecutor, RDAgentSignalPipeline,
)

registry = CodeFactorRegistry("data/rdagent/factors", "data/rdagent/registry.yaml")
executor = CodeFactorExecutor(sandbox_mode="subprocess", timeout_sec=30)

pipeline = RDAgentSignalPipeline(
    code_registry=registry,
    executor=executor,
    window=60,    # OHLCV 回看窗口
)

# 计算信号
output = pipeline.compute("2024-06-28", data_manager)
print(output.signal)           # pd.Series，index=股票代码
print(output.factor_count)     # 成功执行的因子数
print(output.factor_weights)   # {name: weight}
print(output.factor_signals)   # {name: pd.Series} 各因子独立信号
print(output.failed_factors)   # 执行失败的因子名列表

# 衰退检查
decay_report = pipeline.check_decay("2024-06-28", data_manager)
for fr in decay_report.factor_reports:
    print(f"{fr.name}: {fr.ic_trend} → {fr.action}")
if decay_report.suggest_evolution:
    print(f"建议触发进化: {decay_report.suggest_reason}")

# 触发再进化
config = EvolutionConfig.load("configs/rdagent_evolution.yaml", "broad_explore")
evo_report = pipeline.trigger_evolution(config, data_manager)

# 因子状态概览
df = pipeline.get_factor_status()
```

**信号合成流程：**

```
活跃因子列表
  ↓
获取 OHLCV 数据 (window 天)
  ↓
沙箱批量执行 → {factor_name: pd.Series}
  ↓
Rank 归一化 (percentile, [0, 1])
  ↓
权重重归一化 (失败因子的权重分配给成功因子)
  ↓
加权合成: signal = Σ w_i × rank_i
  ↓
RDAgentOutput
```

> **NaN 处理：** 如果某只股票在某个因子中缺失值，使用中性值 0.5（rank 中位数）填充。

> **连续失败自动处理：** 某个因子连续执行失败 3 次，自动进入 probation 状态。

---

## 架构与原理

### 进化循环

M4 的核心思想是利用 RD-Agent 的 LLM 进化循环自动发现有效的量化因子：

```
                    ┌──────────────────────────────┐
                    │        EvolutionConfig         │
                    │  方向约束 + 正交约束 + 资源限制  │
                    └──────────────┬───────────────┘
                                   ↓
                    ┌──────────────────────────────┐
                    │       EvolutionRunner          │
                    └──────────────┬───────────────┘
                                   ↓
         ┌────────────────────────────────────────────────┐
         │              RD-Agent FactorRDLoop              │
         │                                                │
         │  Propose → Code → Run → Evaluate → Evolve     │
         │     ↑                                  │       │
         │     └──────── feedback ────────────────┘       │
         └────────────────────────┬───────────────────────┘
                                  ↓ (RD-Agent 不可用时)
                    ┌──────────────────────────────┐
                    │       模板因子生成（退化模式）   │
                    │      6 个内置金融因子模板       │
                    └──────────────┬───────────────┘
                                   ↓
                    ┌──────────────────────────────┐
                    │     validate_and_register      │
                    │  静态检查 → 沙箱 → IC → 正交    │
                    └──────────────┬───────────────┘
                                   ↓
                    ┌──────────────────────────────┐
                    │      CodeFactorRegistry        │
                    │    持久化的因子代码 + 元数据      │
                    └──────────────────────────────┘
```

**6 个内置因子模板：**

| 名称 | 方向 | 逻辑 |
|------|------|------|
| `rdagent_vol_spike` | volatility_anomaly | 成交量/20日均量 - 1 |
| `rdagent_mean_rev_ma` | mean_revert | (收盘价 - 20日均价) / 20日标准差 |
| `rdagent_amihud_illiq` | liquidity_change | Amihud 非流动性：mean(\|收益率\|/成交量) |
| `rdagent_rsi_divergence` | mean_revert | RSI(14) 偏离中位数 |
| `rdagent_range_vol` | volatility_anomaly | 日内波幅标准差：std((high-low)/close) |
| `rdagent_vol_price_corr` | momentum_divergence | 20日量价相关系数 |

### 因子生命周期

每个因子从创建到退役经历三个状态：

```
             注册
              ↓
         ┌─────────┐     IC 稳定
         │  active  │◄────────────┐
         └────┬─────┘             │
              │ 衰退警告 ≥3       │ IC 回升
              │ 或 IC 崩塌        │
              ↓                   │
         ┌─────────┐             │
         │probation │─────────────┘
         └────┬─────┘
              │ IC 持续崩塌
              ↓
         ┌─────────┐
         │ retired  │  权重=0, enabled=false
         └─────────┘
```

**衰退判定规则：**

| 指标 | stable | declining | collapsed |
|------|--------|-----------|-----------|
| 30日 IC | > 0.02 | > 0 但低于 90日的 50% | ≤ 0 |
| 触发动作 | 无 | 警告（累计 3 次→probation） | probation 或 retire |

**自动进化触发条件：**

- Active 因子数 < 3 时，`check_decay` 报告中 `suggest_evolution = True`
- 由上层调度器（M5 或手动）决定是否执行 `trigger_evolution`

### 沙箱安全机制

LLM 生成的代码可能包含危险操作，因此执行前后有两层防护：

**第一层：静态 AST 检查（执行前）**

- 解析 AST，遍历所有 `import` 和 `from...import` 语句
- 检查是否调用 `exec()`、`eval()` 等危险内置函数
- 验证 `compute_factor()` 函数存在
- 检查代码行数不超过限制

**第二层：子进程隔离（执行时）**

```
主进程                                子进程
  │                                    │
  ├─ pickle.dump(ohlcv_data)──────→ 读取数据
  │                                    │
  │                                    ├─ exec(factor_code)
  │                                    ├─ result = compute_factor(ohlcv)
  │                                    ├─ 校验 result 类型和值
  │                                    │
  ├─ pickle.load(result) ◄────────── 保存结果
  │                                    │
  ├─ timeout 强制 kill ──────────→ 超时终止
```

### 因子分流

RD-Agent 可能产出两种类型的因子：

1. **Qlib 表达式因子** — 简单表达式（如 `Mean($close, 20) / Std($close, 20)`），注入 M2 FactorRegistry
2. **Python 代码因子** — 复杂逻辑（需 for 循环、pandas 操作等），注册到 M4 CodeFactorRegistry

判断逻辑：如果代码中包含 Qlib 算子（`Mean`, `Std`, `Ref` 等）且不包含 pandas 操作（`DataFrame`, `groupby`, `rolling`），视为 Qlib 表达式。

---

## 进化配置设计指南

### 方向聚焦 vs 广泛探索

| 场景 | 推荐配置 | 说明 |
|------|---------|------|
| 想针对特定市场现象挖因子 | 方向聚焦 | `target_directions` 设 1-2 个方向 |
| 想寻找多样化的因子 | 广泛探索 | `target_directions` 设 5+ 个方向 |
| 首次使用 | 广泛探索 | 先广撒网，再针对有效方向深挖 |
| 因子池质量不高 | 降低门槛 | 降低 `min_ic` 和 `min_icir` |
| 因子池冗余严重 | 加强正交 | 降低 `max_corr_within_pool` |

### 预置配置说明

**mean_revert_focus（均值回复方向）：**

```yaml
target_directions: ["mean_revert", "overreaction"]
direction_prompt: |
  Focus on mean-reversion signals: price deviation from moving averages,
  volume-price divergence, RSI extremes, Bollinger Band breakouts.
max_corr_with_alpha: 0.3
max_rounds: 20
total_budget: 50
```

适合：相信市场存在短期过度反应，想挖掘反转信号。

**volatility_anomaly（波动率异常）：**

```yaml
target_directions: ["volatility_anomaly", "regime_change"]
direction_prompt: |
  Focus on volatility regime changes: GARCH residuals, realized vs implied
  vol spread, intraday range anomalies, volume spike detection.
max_rounds: 15
total_budget: 30
```

适合：关注波动率状态切换，捕捉异常波动事件。

**liquidity_focus（流动性因子）：**

```yaml
target_directions: ["liquidity_change", "market_microstructure"]
direction_prompt: |
  Focus on liquidity signals: Amihud illiquidity, turnover rate changes,
  bid-ask proxy from OHLC, Kyle's lambda estimation.
max_rounds: 15
total_budget: 30
```

适合：关注流动性变化对价格的影响。

**broad_explore（多方向探索）：**

```yaml
target_directions: ["mean_revert", "volatility_anomaly", "liquidity_change",
                     "momentum_divergence", "cross_section_anomaly"]
max_corr_with_alpha: 0.25
max_corr_within_pool: 0.4
max_rounds: 30
total_budget: 100
```

适合：首次使用或需要大量多样化因子。正交约束更严格以避免冗余。

### 自定义配置

可以在 `quantlab/configs/rdagent_evolution.yaml` 中添加新配置：

```yaml
configs:
  - name: my_custom
    target_directions: ["cross_section_anomaly"]
    direction_prompt: |
      Focus on cross-sectional anomalies: industry-relative value,
      size-momentum interaction, earnings surprise decay.
    max_corr_with_alpha: 0.25
    max_corr_within_pool: 0.35
    min_ic: 0.03
    min_icir: 0.6
    max_rounds: 15
    total_budget: 40
```

---

## 因子代码规范

所有 Python 代码因子必须遵守以下接口规范：

### 函数签名

```python
def compute_factor(ohlcv: Dict[str, pd.DataFrame]) -> pd.Series:
    """因子描述（推荐写 docstring）。"""
    ...
```

### 输入格式

`ohlcv` 是一个字典，key 为股票代码，value 为 DataFrame：

```python
{
    "SH600519": pd.DataFrame({
        "open":   [10.0, 10.5, ...],
        "high":   [10.8, 11.0, ...],
        "low":    [9.8, 10.2, ...],
        "close":  [10.5, 10.8, ...],
        "volume": [100000, 120000, ...],
        "amount": [1050000, 1296000, ...],
    }),
    "SZ000858": ...,
}
```

### 输出格式

返回 `pd.Series`，index 为股票代码，值为因子值：

```python
pd.Series({
    "SH600519": 1.25,
    "SZ000858": -0.87,
    ...
})
```

### 限制规则

- 代码不超过 100 行
- 禁止访问文件系统、网络、子进程
- 禁止调用 `exec`、`eval`、`compile`
- 只能 import 纯计算库（numpy, pandas, scipy 等）
- 不能返回全 NaN 或包含 inf 的 Series
- 执行超时 30 秒

### 示例：一个合规的因子

```python
import numpy as np
import pandas as pd

def compute_factor(ohlcv):
    """20 日动量因子：最近 20 天累计收益率。"""
    result = {}
    for sym, df in ohlcv.items():
        if len(df) < 21:
            continue
        close = df["close"].values
        ret_20d = close[-1] / close[-21] - 1
        result[sym] = ret_20d
    return pd.Series(result)
```

---

## 配置说明

### 配置文件位置

| 文件 | 路径 | 说明 |
|------|------|------|
| 进化配置 | `quantlab/configs/rdagent_evolution.yaml` | 进化方向、资源限制、门控阈值 |
| 因子注册表 | `data/rdagent/registry.yaml` | 因子元数据和生命周期状态 |
| 因子代码 | `data/rdagent/factors/*.py` | 每个因子一个 Python 文件 |

### 数据目录结构

```
data/rdagent/
├── factors/                      因子代码文件
│   ├── rdagent_vol_spike.py
│   ├── rdagent_mean_rev_ma.py
│   ├── rdagent_amihud_illiq.py
│   └── ...
└── registry.yaml                 因子注册表

outputs/rdagent/
├── factor_summary.csv            因子概览
└── evolve_record.json            进化执行记录

quantlab/configs/
└── rdagent_evolution.yaml        进化配置（4 个预置配置）
```

---

## 运行测试

```bash
cd quantlab/tests

# 运行全部 M4 测试
pytest test_signal_rdagent.py -v

# 运行指定测试类
pytest test_signal_rdagent.py::TestCodeFactorExecutor -v
pytest test_signal_rdagent.py::TestCodeFactorRegistry -v
pytest test_signal_rdagent.py::TestEvolutionConfig -v
pytest test_signal_rdagent.py::TestEvolutionRunner -v

# 运行指定测试
pytest test_signal_rdagent.py::TestCodeFactorExecutor::test_forbidden_import_os -v
```

测试覆盖内容（全部离线，不依赖 RD-Agent/Qlib/GPU）：

| 测试类 | 数量 | 覆盖内容 |
|--------|------|---------|
| `TestEvolutionConfig` | 8 | 默认值、参数验证、序列化、加载/保存 |
| `TestCodeFactorEntry` | 2 | 默认值、自定义字段 |
| `TestCodeFactorRegistry` | 12 | 注册/查询/退役/权重/持久化/概览 |
| `TestCodeFactorExecutor` | 14 | 静态检查（禁止import/builtin/签名/行数）、inprocess 执行、批量执行 |
| `TestDataStructures` | 6 | RDAgentOutput/DecayReport/EvolutionReport/FactorResult/RawFactor |
| `TestRDAgentSignalPipeline` | 5 | 空因子计算、状态概览、衰退建议 |
| `TestEvolutionRunner` | 5 | 模板生成、验证注册、坏代码拒绝、RD-Agent 退化 |
| `TestPresetConfigs` | 3 | 预置配置文件加载 |

---

## 常见问题

### 1. 没有安装 RD-Agent，能使用 M4 吗？

可以。`rdagent_evolve.py --template-only` 会使用 6 个内置因子模板，无需 RD-Agent。即使不加 `--template-only`，当 RD-Agent import 失败时也会自动退化为模板模式。

### 2. 因子执行报 "禁止 import: os"，怎么办？

这是安全机制。因子代码不允许访问文件系统、网络等。如果确实需要某个被禁止的功能，需要重写因子逻辑，只使用纯计算库（numpy, pandas, scipy 等）。

### 3. 因子执行超时怎么办？

默认超时 30 秒。可能原因：
- 因子逻辑过于复杂，考虑简化算法
- 数据量太大，考虑减少 `--window` 窗口
- 通过 `--timeout 60` 增大超时

### 4. subprocess 沙箱模式和 inprocess 有什么区别？

| | subprocess | inprocess |
|---|---|---|
| 安全性 | 高（独立进程隔离） | 低（当前进程直接 exec） |
| 速度 | 中（序列化+fork 开销） | 快 |
| 推荐场景 | 日常使用、生产环境 | 开发调试、单元测试 |

### 5. 因子进入 probation 后会自动恢复吗？

会。`check_decay` 检查时，如果 probation 因子的 30日 IC 回升到 > 0.02 且趋势稳定，会自动恢复为 active。

### 6. 如何手动恢复一个退役因子？

```bash
python rdagent_registry_cli.py activate <factor_name>
```

这会将因子状态从 retired 改为 active，并清零衰退警告。

### 7. 信号输出值的范围是什么？

信号值在 [0, 1] 区间，因为底层使用 rank percentile 归一化。0.8 以上表示强烈看多，0.2 以下表示强烈看空，0.5 附近为中性。

### 8. 权重为什么全是 0？

新注册的因子默认权重为 0。需要通过以下方式之一设置权重：

```bash
# 方式一：手动指定 ICIR
python rdagent_registry_cli.py update-weights --icir "factor1:1.0,factor2:0.8"

# 方式二：运行回测后自动计算（在 pipeline.compute 过程中积累 IC 历史后）
```

如果所有因子权重为 0，`compute` 会使用等权作为退化方案。

### 9. 如何查看某个因子的代码？

```bash
python rdagent_registry_cli.py show <name> --show-code
```

或直接查看文件：`data/rdagent/factors/<name>.py`

### 10. 进化配置中的 direction_prompt 有什么用？

`direction_prompt` 是注入 RD-Agent LLM 的方向提示词，告诉 LLM 应该关注哪类市场现象。模板模式下不使用此字段，但 `target_directions` 列表会用于过滤模板因子。

### 11. 因子分流到 M2 后在哪里管理？

分流到 M2 的 Qlib 表达式因子由 M2 的 `FactorRegistry` 管理，可通过 M2 的工具查看：

```bash
cd quantlab/signal
python alpha_registry_cli.py list    # M2 注册表管理
```

### 12. 如何在 M5 信号融合中使用 M4 信号？

M4 的 `RDAgentSignalPipeline.compute()` 返回 `RDAgentOutput`，其 `signal` 字段是 `pd.Series`，可直接传给 M5 信号融合模块：

```python
from quantlab.signals.signal_rdagent import (
    CodeFactorRegistry, CodeFactorExecutor, RDAgentSignalPipeline,
)

registry = CodeFactorRegistry("data/rdagent/factors", "data/rdagent/registry.yaml")
executor = CodeFactorExecutor()
pipeline = RDAgentSignalPipeline(code_registry=registry, executor=executor)

output = pipeline.compute("2024-06-28", data_manager)
m4_signal = output.signal    # → M5
```
