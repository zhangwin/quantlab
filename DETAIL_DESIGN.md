# 详细设计文档

基于概要设计（DESIGN.md），对 pipeline/ 下每个模块逐一说明功能、接口、依赖和测试方案。

---

## 模块总览

```
pipeline/
├── data_manager.py        M1   数据管理（更新 + 访问）
├── data_viewer.py         M1.5 数据检视（导出 + K线可视化）
├── signal_alpha.py        M2   Alpha158 信号管线
├── signal_kronos.py       M3   Kronos 信号管线
├── signal_rdagent.py      M4   RD-Agent 信号管线
├── signal_ensemble.py     M5   信号融合
├── execution.py           M6   交易执行
├── risk_control.py        M7   风控
├── evaluation.py          M8   评估与诊断
main.py                    M9   主循环（回测调度器）
```

**模块依赖关系：**

```
M1 ──→ M1.5             数据检视依赖数据管理
M1 ──→ M2, M3, M4       数据层为三条管线供数据
M2, M3, M4 ──→ M5       三条管线的信号汇入融合层
M5 ──→ M6               融合信号驱动交易执行
M6 ←──→ M7              执行前后风控介入
M6, M7 ──→ M8           交易记录和净值流入评估
M3 ──→ M1.5             Kronos 预测结果叠加到 K 线图
M9 调度以上全部模块
```

---

## M1 数据管理（data_manager.py）

### 职责

两项核心职责：

1. **数据更新** — 每日收盘后增量拉取最新行情，写入 Qlib 本地数据目录
2. **数据访问** — 为所有下游模块提供时间隔离的数据接口，任何模块不直接调用 Qlib D 对象

### 数据结构

```
RollingWindow
    finetune_lookback: int = 30
    predict_horizon: int = 5
    alpha_train_years: int = 3
    alpha_retrain_interval: int = 20
    ic_lookback: int = 60
    backtest_start: str
    backtest_end: str
```

### 数据存储

Qlib 本地数据目录 `~/.qlib/qlib_data/cn_data/`，结构：

```
cn_data/
├── calendars/
│   └── day.txt                每行一个交易日 "YYYY-MM-DD"
├── instruments/
│   ├── all.txt                全部股票及起止日期
│   └── csi300.txt             CSI300 成分股及变更日期
└── features/
    └── {symbol}/              每只股票一个目录
        ├── open.day.bin       float32 小端二进制，按日历顺序存储
        ├── high.day.bin
        ├── low.day.bin
        ├── close.day.bin
        ├── volume.day.bin
        └── factor.day.bin     复权因子
```

每个 `.day.bin` 文件是 NumPy float32 数组，按交易日历顺序排列。增量更新时在文件末尾追加新数据（append 模式），不重写历史。

### 数据更新方案

#### 运行时机

| 场景 | 触发方式 | 说明 |
|------|---------|------|
| 回测 | 回测开始前运行一次 `ensure_data_updated()` | 将数据补齐到回测结束日期 |
| 实盘 | 每个交易日 15:30 后自动触发（cron / Windows 任务计划） | 增量追加当日数据 |

#### 更新流程

```
ensure_data_updated(end_date)
    │
    ▼
读取 calendars/day.txt 最后一行 → last_date
    │
    ▼
last_date >= end_date ?
    ├─ 是 → 跳过，数据已是最新
    └─ 否 ↓
         ▼
    调用 Qlib Yahoo Collector 增量更新
    ┌───────────────────────────────────────┐
    │  1. 从 Yahoo Finance 下载             │
    │     last_date+1 ~ end_date 的行情     │
    │     （支持 A 股 .SS/.SZ 后缀）         │
    │                                       │
    │  2. 归一化（复权对齐）                  │
    │                                       │
    │  3. DumpDataUpdate 增量追加            │
    │     · 在 .bin 文件末尾 append          │
    │     · 更新 calendars/day.txt           │
    │     · 更新 instruments 日期范围         │
    └───────────────────────────────────────┘
         │
         ▼
    更新 CSI300 成分股列表
    （从中证指数官网拉取最新变更）
         │
         ▼
    重新初始化 Qlib（使 D 对象感知新数据）
```

#### 对应的 Qlib 命令

```bash
# 首次初始化（一次性）
python qlib/scripts/get_data.py qlib_data \
    --target_dir ~/.qlib/qlib_data/cn_data --region cn

# 每日增量更新
python qlib/scripts/data_collector/yahoo/collector.py \
    update_data_to_bin \
    --qlib_data_1d_dir ~/.qlib/qlib_data/cn_data \
    --end_date 2026-03-11

# 更新 CSI300 成分股
python qlib/scripts/data_collector/cn_index/collector.py \
    --index_name CSI300 \
    --qlib_dir ~/.qlib/qlib_data/cn_data \
    --method parse_instruments
```

#### 备选数据源

Yahoo Finance 对 A 股覆盖有延迟或缺失时，可切换：

| 数据源 | 覆盖 | 频率 | 说明 |
|--------|------|------|------|
| Yahoo Finance | A股/港股/美股 | 日频 | 默认，Qlib 内置支持 |
| Baostock | A股 | 日频/5分钟 | 免费，无限制，`baostock_5min/collector.py` |
| Tushare Pro | A股 | 日频/分钟 | 需 token，数据全面 |

不同数据源的 collector 共用 `DumpDataUpdate` 写入同一套 bin 文件，格式一致。

### 数据访问接口

| 方法 | 输入 | 输出 | 说明 |
|------|------|------|------|
| `__init__(provider_uri, market)` | 数据目录, 股票池 | — | |
| `init_qlib()` | — | — | 初始化 Qlib，连接数据目录 |
| `ensure_data_updated(end_date)` | 目标日期 str | bool | 检查并增量更新数据到 end_date，返回是否有更新 |
| `get_latest_date()` | — | Timestamp | 读取 calendars/day.txt 最后一行，返回数据库中的最新交易日 |
| `get_trading_calendar(start, end)` | 起止日期 str | List[Timestamp] | 交易日序列 |
| `get_ohlcv_before(anchor_date, lookback_days)` | 锚定日, 回看天数 | Dict[symbol, DataFrame] | 返回 ≤ anchor_date 的 OHLCV，按股票分组。列: open/high/low/close/volume/amount |
| `get_alpha158_features(anchor_date)` | 锚定日 | DataFrame | 返回 anchor_date 当天的 Alpha158 截面特征。MultiIndex(symbol, datetime) |
| `get_next_day_open(anchor_date)` | 锚定日 | Series[symbol→price] | T+1 开盘价。**仅 M6 成交判定调用** |
| `get_close_prices(anchor_date)` | 锚定日 | Series[symbol→price] | T 日收盘价 |
| `get_industry_map()` | — | Series[symbol→行业代码] | 申万一级行业 |
| `get_limit_prices(anchor_date)` | 锚定日 | (Series, Series) | T+1 涨停价和跌停价（基于 T 日收盘价 ±10%） |

### 时间隔离保证

- `get_ohlcv_before` 和 `get_alpha158_features` 的查询终点严格为 anchor_date
- `get_next_day_open` 和 `get_limit_prices` 仅由 M6 的成交判定环节调用，不流入信号计算
- 即使本地数据已更新到未来日期（回测场景），数据访问层仍只返回 ≤ anchor_date 的数据

### 测试

| 测试项 | 方法 | 验证点 |
|--------|------|--------|
| 时间隔离 | 数据已更新到 2025-03-11，但 `get_ohlcv_before(anchor="2025-03-01", lookback=30)` | 返回数据的最大日期 ≤ 2025-03-01，不含 03-02 ~ 03-11 |
| 数据新鲜度 | 调用 `get_latest_date()` | 返回值 = calendars/day.txt 的最后一行日期 |
| 增量更新幂等 | 连续两次 `ensure_data_updated("2025-03-11")` | 第一次更新，第二次跳过；数据一致 |
| 更新完整性 | 更新前后对比 day.txt 行数 | 新增行数 = 新增交易日数 |
| bin 文件追加 | 更新后检查 close.day.bin 文件大小 | 增长 = 新增天数 × 4 bytes |
| 数据完整性 | 对 CSI300 调用，统计返回股票数 | ≥ 280（允许少量停牌） |
| 空数据处理 | anchor_date 设为数据起始日之前 | 返回空 dict，不抛异常 |
| 日历正确性 | 检查返回日历不含周末和节假日 | 与交易所公布日历一致 |
| 开盘价隔离 | 在信号生成流程中 mock `get_next_day_open`，确认其未被 M2/M3/M4/M5 调用 | 信号生成全流程无 T+1 数据访问 |
| 网络失败回退 | 断网时调用 `ensure_data_updated` | 抛出明确异常，不损坏已有数据 |

---

## M1.5 数据检视（data_viewer.py）

### 职责

提供两项能力：

1. **数据导出** — 将 Qlib bin 格式转为 CSV，供人工检查和外部工具使用
2. **K线可视化** — 交互式蜡烛图，可叠加 Kronos 预测、买卖标记、技术指标

### 为什么需要这个模块

Qlib 的 `.bin` 文件是 float32 二进制，为计算性能优化，不可直接阅读。日常需要：

- 检查某只股票的数据是否完整、是否有异常值
- 看 K 线形态，直观理解 Kronos 预测质量
- 回测后复盘：在 K 线上标注实际买卖点
- 导出 CSV 给团队成员或其他分析工具

### 数据格式说明

```
bin 文件（Qlib 内部）             CSV 文件（人可读导出）
─────────────────────           ──────────────────────────────────────
float32 紧密二进制数组            date,open,high,low,close,volume,amount
按 calendars/day.txt             2025-03-01,10.5,10.8,10.3,10.6,1000000,10500000
的顺序一一对应                    2025-03-02,10.6,10.9,10.5,10.7,950000,10100000
无表头，无索引                    ...
每文件一个字段
每目录一只股票
```

### 接口

| 方法 | 输入 | 输出 | 说明 |
|------|------|------|------|
| `__init__(data_manager)` | M1 实例 | — | |
| `export_csv(symbols, start, end, output_dir)` | 股票列表, 起止日期, 输出目录 | List[Path] | 导出每只股票一个 CSV 文件 |
| `export_portfolio_csv(positions, start, end, output_path)` | 持仓字典, 起止日期, 输出路径 | Path | 当前持仓所有股票导出到一个合并 CSV |
| `plot_kline(symbol, start, end)` | 股票代码, 起止日期 | Figure | 单只股票交互式 K 线图（Plotly） |
| `plot_kline_with_prediction(symbol, hist_data, pred_data)` | 股票代码, 历史DataFrame, 预测DataFrame | Figure | K线 + Kronos 预测叠加 |
| `plot_kline_with_trades(symbol, start, end, trade_records)` | 股票代码, 起止日期, 交易记录 | Figure | K线 + 买卖标记 |
| `plot_portfolio_overview(positions, current_prices, industry_map)` | 持仓, 价格, 行业 | Figure | 持仓概览：行业分布饼图 + 个股盈亏柱状图 |
| `show(figure)` | Plotly Figure | — | 自动判断环境（Notebook / 浏览器 / 保存PNG） |

### plot_kline 图表内容

```
┌────────────────────────────────────────────────┐
│  蜡烛图（Plotly Candlestick）                    │
│                                                  │
│  · 上涨蜡烛：红色（A股习惯）                      │
│  · 下跌蜡烛：绿色                                │
│  · 可鼠标缩放、拖拽、悬停查看 OHLCV 数值          │
│                                                  │
├────────────────────────────────────────────────┤
│  成交量柱状图（与蜡烛图同步缩放）                  │
│                                                  │
│  · 上涨日红色，下跌日绿色                         │
└────────────────────────────────────────────────┘
  可选叠加层：
  · MA5 / MA10 / MA20 均线
  · 布林带
  · Kronos 预测区间（半透明色块）
  · 买入标记（▲ 绿色箭头）/ 卖出标记（▼ 红色箭头）
```

### plot_kline_with_prediction 说明

```
历史区间                   预测区间
│← hist_data (30天) →│← pred_data (5天) →│
│                     │                    │
│  实线蜡烛图          │  虚线/半透明蜡烛图   │
│  （实际行情）        │  （Kronos 预测）     │
│                     │                    │
│                     │  如有多次采样，       │
│                     │  显示预测区间阴影     │
```

利用 Kronos `webui/app.py` 中已有的 `create_prediction_chart()` 逻辑，复用其 Plotly 蜡烛图渲染代码。

### plot_kline_with_trades 说明

在标准 K 线图上叠加交易标记：

```
标记类型：
  · 买入成交    ▲ 实心绿色箭头，标注价格
  · 买入失败    △ 空心灰色箭头，标注原因（涨停/超目标价）
  · 卖出成交    ▼ 实心红色箭头，标注价格
  · 卖出失败    ▽ 空心灰色箭头，标注原因（跌停）
  · 止损触发    ✕ 黄色标记
```

### export_csv 输出格式

每只股票一个文件 `{output_dir}/{symbol}.csv`：

```csv
date,open,high,low,close,volume,amount
2025-03-01,10.50,10.80,10.30,10.60,1000000,10500000
2025-03-02,10.60,10.90,10.50,10.70,950000,10100000
```

- 日期格式：YYYY-MM-DD
- 价格：保留原始精度（不做四舍五入）
- 编码：UTF-8
- 无 index 列

### 使用场景

| 场景 | 调用方式 |
|------|---------|
| 数据更新后快速检查 | `viewer.plot_kline("SH600519", "2025-01-01", "2025-03-11")` |
| 检查 Kronos 预测效果 | `viewer.plot_kline_with_prediction("SH600519", hist, pred)` |
| 回测复盘单只票 | `viewer.plot_kline_with_trades("SH600519", start, end, trades)` |
| 导出给团队 | `viewer.export_csv(["SH600519","SZ000001"], "2024-01-01", "2025-03-11", "./export/")` |
| 持仓检视 | `viewer.plot_portfolio_overview(positions, prices, industries)` |

### 测试

| 测试项 | 方法 | 验证点 |
|--------|------|--------|
| CSV 导出正确性 | 导出后用 pandas 读回，与 `get_ohlcv_before` 返回值对比 | 数值完全一致 |
| CSV 日期连续 | 检查导出 CSV 的日期列 | 与交易日历一致，无遗漏无重复 |
| K线图渲染 | 调用 `plot_kline`，检查返回 Figure 的 trace 类型 | 包含 Candlestick trace + Bar trace（成交量） |
| 预测叠加 | 调用 `plot_kline_with_prediction`，检查 trace 数量 | 历史蜡烛 + 预测蜡烛 + 预测区间阴影 = 至少 3 个 trace |
| 交易标记 | 传入 3 笔买入 2 笔卖出的 trade_records | 图上出现 5 个 Scatter 标记 |
| 空数据 | 不存在的股票代码 | 返回空图或明确提示，不报错 |
| 大批量导出 | 导出 CSI300 全部股票 | 生成 ~300 个 CSV 文件，总耗时合理（< 2 分钟） |

---

## M2 因子信号管线（signal_alpha.py）

### 职责

三项核心职责：

1. **因子管理** — 维护一个因子池（Alpha158 为基础集 + 可动态增删的自定义因子）
2. **因子挖掘与验证** — 提供因子质量评估流程，支持快速试验新因子
3. **信号生产** — 用验证通过的因子集 + LightGBM 滚动训练，产出日频预测信号

### 设计思路

Qlib 的因子是**字符串表达式**，由基础字段（`$close`, `$volume` 等）和算子（`Mean`, `Std`, `Corr` 等 50+）自由组合。这意味着：

- 新增一个因子 = 写一行表达式字符串，不需要写 Python 代码
- 因子的计算、缓存、对齐全部由 Qlib 引擎完成
- 非常适合做动态增删和批量验证

```
因子表达式示例：
  "Mean($close, 5)/$close"                          → 5日均线偏离
  "Std($close, 20)/$close"                          → 20日波动率
  "Corr($close/Ref($close,1), Log($volume+1), 30)"  → 量价相关性
  "Slope($close, 10)/$close"                         → 10日线性趋势
  "Rsquare($close, 20)"                              → 趋势拟合度
```

### 子模块划分

```
signal_alpha.py
├── FactorRegistry        因子注册表（增删查改）
├── FactorValidator       因子验证器（IC / 相关性 / 稳定性）
├── FactorMiner           因子挖掘器（半自动探索）
└── AlphaSignalPipeline   信号生产（训练 + 预测）
```

---

### 子模块一：FactorRegistry（因子注册表）

#### 职责

管理因子池。Alpha158 作为基础集自动加载，用户可随时添加、删除、启用、禁用因子。因子定义持久化到 YAML 文件，跨会话保留。

#### 数据结构

```
FactorDef（单个因子定义）
    name: str                    唯一标识，如 "VOL_SKEW_20"
    expression: str              Qlib 表达式，如 "Skew($volume, 20)"
    category: str                分类标签："momentum" | "mean_revert" | "volatility" | "liquidity" | "custom"
    source: str                  来源："alpha158" | "manual" | "rdagent" | "miner"
    enabled: bool                是否启用（参与训练和预测）
    added_date: str              添加日期
    validation: dict | None      最近一次验证结果（IC, ICIR 等）
```

#### 持久化

```yaml
# configs/factors.yaml
factors:
  # Alpha158 基础因子（自动生成，一般不手动编辑）
  - name: MA5
    expression: "Mean($close, 5)/$close"
    category: momentum
    source: alpha158
    enabled: true

  - name: STD20
    expression: "Std($close, 20)/$close"
    category: volatility
    source: alpha158
    enabled: true

  # ... 158 个基础因子 ...

  # 用户手动添加的因子
  - name: VOL_SKEW_20
    expression: "Skew($volume, 20)"
    category: liquidity
    source: manual
    enabled: true
    added_date: "2025-03-11"
    validation:
      ic_mean: 0.032
      icir: 1.15
      max_corr_with_existing: 0.28

  # RD-Agent 进化产出的因子
  - name: RDAGENT_FACTOR_001
    expression: "Corr(Std($close,5), Mean($volume,10), 20)"
    category: mean_revert
    source: rdagent
    enabled: true
    added_date: "2025-03-08"
```

#### 接口

| 方法 | 输入 | 输出 | 说明 |
|------|------|------|------|
| `__init__(config_path)` | YAML 路径 | — | 加载因子定义。首次运行自动生成 Alpha158 基础集 |
| `add(name, expression, category, source)` | 因子定义字段 | FactorDef | 添加新因子。name 重复则报错 |
| `remove(name)` | 因子名 | — | 删除因子（alpha158 来源的只能 disable 不能删除） |
| `enable(name)` / `disable(name)` | 因子名 | — | 启用/禁用 |
| `get_enabled()` | — | List[FactorDef] | 返回所有 enabled=True 的因子 |
| `get_expressions()` | — | (List[str], List[str]) | 返回启用因子的 (表达式列表, 名称列表)，直接传给 QlibDataLoader |
| `list(category=None, source=None)` | 可选筛选条件 | List[FactorDef] | 查询因子 |
| `update_validation(name, result)` | 因子名, 验证结果 | — | 写入最近验证结果 |
| `save()` | — | — | 持久化到 YAML |
| `summary()` | — | DataFrame | 全因子概览表：name / category / source / enabled / ic_mean / icir |

#### 与 M4 的关系

M4（RD-Agent 管线）进化出新因子后，调用 `registry.add(source="rdagent")` 注册。注册后自动进入验证流程。

---

### 子模块二：FactorValidator（因子验证器）

#### 职责

评估单个因子或一组因子的质量，决定是否值得纳入因子池。

#### 接口

| 方法 | 输入 | 输出 | 说明 |
|------|------|------|------|
| `__init__(data_manager)` | M1 实例 | — | |
| `validate_single(expression, start, end)` | 表达式, 验证区间 | FactorReport | 全面评估单个因子 |
| `validate_incremental(expression, existing_expressions)` | 新因子, 现有因子列表 | IncrementalReport | 评估新因子相对于现有因子的增量贡献 |
| `validate_batch(expressions, start, end)` | 表达式列表, 区间 | List[FactorReport] | 批量验证 |
| `compare(expressions, start, end)` | 多个表达式, 区间 | DataFrame | 横向对比多个因子 |

#### FactorReport 内容

```
FactorReport
    expression: str
    ic_mean: float               每日 Rank IC 均值
    icir: float                  IC 均值 / IC 标准差（越高越稳定）
    ic_series: Series            每日 IC 时间序列
    ic_positive_ratio: float     IC > 0 的天数占比
    ic_decay: Dict[int, float]   IC 在 T+1 ~ T+10 的衰减
    turnover: float              因子日均换手率（Top/Bottom 组的变化率）
    auto_corr: float             因子自相关性（T 和 T-1 截面相关）
    sharpe_long_short: float     多空组合的年化夏普（Top20% 做多 - Bottom20% 做空）
    max_corr_with_existing: float  与现有因子的最大截面相关系数
    verdict: str                 "accept" | "reject" | "review"
```

#### 验证标准（自动判定）

```
accept（自动纳入）:
    IC 均值 > 0.02
    且 ICIR > 0.5
    且 与现有因子最大相关 < 0.7
    且 IC 正比 > 55%

reject（自动拒绝）:
    IC 均值 < 0.01
    或 ICIR < 0.2
    或 与现有因子最大相关 > 0.9（冗余因子）

review（需人工判断）:
    介于 accept 和 reject 之间
```

#### IncrementalReport 额外内容

```
IncrementalReport extends FactorReport
    marginal_ic: float           加入新因子后模型 IC 的提升
    feature_importance: float    LightGBM 中新因子的特征重要性排名
    correlation_matrix: DataFrame  新因子与前10大因子的相关性矩阵
```

#### validate_single 内部流程

```
1. 用 QlibDataLoader 计算因子值
   · 表达式 → Qlib 引擎计算 → 截面 DataFrame
2. 计算 label
   · Ref($close, -2) / Ref($close, -1) - 1
3. 逐日 Rank IC
   · 每天: spearmanr(因子截面排序, 收益截面排序)
4. IC 衰减分析
   · 因子值 vs T+1收益, T+2收益, ..., T+10收益 的 IC
5. 多空组合回测
   · 每天按因子值排序，Top20% 做多, Bottom20% 做空
   · 计算多空组合的累计收益和夏普
6. 相关性检查
   · 与 registry 中所有 enabled 因子计算截面相关
7. 输出 FactorReport + 自动判定 verdict
```

---

### 子模块三：FactorMiner（因子挖掘器）

#### 职责

半自动化因子探索。通过组合算子和参数生成候选因子，批量验证后推荐优质因子。

#### 接口

| 方法 | 输入 | 输出 | 说明 |
|------|------|------|------|
| `__init__(validator, registry)` | 验证器, 注册表 | — | |
| `explore_variants(base_expression, param_grid)` | 基础表达式, 参数网格 | List[FactorReport] | 对一个因子做参数搜索 |
| `explore_combinations(fields, operators, windows)` | 字段列表, 算子列表, 窗口列表 | List[FactorReport] | 笛卡尔积组合生成候选因子 |
| `explore_from_template(template, variables)` | 模板字符串, 变量字典 | List[FactorReport] | 按模板批量生成 |
| `auto_mine(budget)` | 最大候选数 | List[FactorReport] | 全自动挖掘（组合 + 验证 + 排序） |
| `accept_and_register(reports)` | 验证报告列表 | int | 将 verdict=accept 的因子注册到 registry，返回新增数 |

#### explore_variants 使用方式

```
# 已知 "Std($close, 20)/$close" 是个好因子
# 搜索不同窗口是否更好

miner.explore_variants(
    base_expression="Std($close, {window})/$close",
    param_grid={"window": [5, 10, 20, 30, 60, 120]}
)

# 返回 6 个 FactorReport，按 ICIR 排序
```

#### explore_combinations 使用方式

```
# 探索量价相关性因子
miner.explore_combinations(
    fields=["$close/Ref($close,1)", "$volume", "Log($volume+1)"],
    operators=["Corr", "Cov"],
    windows=[10, 20, 30, 60]
)

# 生成: Corr($close/Ref($close,1), $volume, 10)
#        Corr($close/Ref($close,1), $volume, 20)
#        Corr($close/Ref($close,1), Log($volume+1), 10)
#        ... 等 24 个候选因子
```

#### explore_from_template 使用方式

```
# 探索一系列均值回复因子
miner.explore_from_template(
    template="({field} - Mean({field}, {window})) / (Std({field}, {window}) + 1e-12)",
    variables={
        "field": ["$close", "$volume", "$high-$low"],
        "window": [5, 10, 20, 60]
    }
)

# 生成 3×4 = 12 个 Bollinger Band 变种因子
```

#### auto_mine 内部流程

```
1. 定义搜索空间
   · 基础字段: $open, $close, $high, $low, $volume, $vwap
   · 派生字段: $close/Ref($close,1), $high-$low, $close-$open, Log($volume+1)
   · 单元算子: Mean, Std, Skew, Kurt, Slope, Rsquare, Rank, Max, Min, Sum, Quantile
   · 二元算子: Corr, Cov
   · 窗口: 5, 10, 20, 30, 60

2. 生成候选因子
   · 单元: operator(field, window) → ~330 个
   · 二元: operator(field1, field2, window) → ~1500 个
   · 从中随机采样 budget 个（避免全量计算耗时过长）

3. 批量验证
   · validator.validate_batch(candidates)
   · 并行计算加速

4. 排序与去冗余
   · 按 ICIR 降序排序
   · 贪心去冗余：依次选入，跳过与已选因子相关 > 0.7 的

5. 返回 Top-K 报告
```

---

### 子模块四：AlphaSignalPipeline（信号生产）

#### 职责

用 FactorRegistry 中启用的因子集 + LightGBM 滚动训练，产出日频信号。

#### 接口

| 方法 | 输入 | 输出 | 说明 |
|------|------|------|------|
| `__init__(window, registry, market)` | 配置, 因子注册表, 股票池 | — | |
| `should_retrain(anchor_date)` | 锚定日 | bool | 每 retrain_interval 天重训一次，或因子池发生变更时强制重训 |
| `train(anchor_date, data_manager)` | 锚定日, M1实例 | — | 滚动训练 |
| `predict(anchor_date, data_manager)` | 锚定日, M1实例 | Series[symbol→score] | 当日截面预测 |
| `get_feature_importance()` | — | Series[factor_name→importance] | 当前模型中各因子的特征重要性 |

#### 内部状态

- `model`: 当前 LightGBM 模型实例
- `last_train_date`: 上次训练日期
- `train_count`: 累计预测次数
- `last_factor_hash`: 上次训练时的因子池 hash（检测因子变更）

#### 训练流程

```
1. 从 registry.get_expressions() 获取当前启用因子列表
2. 构建 QlibDataLoader（因子表达式 → Qlib 引擎计算）
3. 构建 DatasetH：
   · 训练集: [anchor - 3年, anchor - 2天]
   · 标签: Ref($close, -2) / Ref($close, -1) - 1
   · 处理器: DropnaLabel → ZScoreNorm → Fillna
4. 训练 LightGBM
5. 记录 feature_importance
```

#### 因子池变更触发重训

```
当用户通过 registry 添加/删除/启用/禁用因子后：
  · last_factor_hash 与当前因子池 hash 不一致
  · should_retrain() 返回 True
  · 下次 predict() 自动触发重训
```

#### 训练标签说明

```
训练数据的最后一天 = anchor_date - 2 个交易日
原因：标签 = Ref($close, -2) / Ref($close, -1) - 1
      需要 T+1 和 T+2 的 close
      anchor-2 日的标签需要 anchor 日的 close（刚好可用）
      anchor-1 日的标签需要 anchor+1 日的 close（不可用）
```

---

### M2 整体使用流程

```
场景1: 日常回测/实盘（信号生产）
  registry = FactorRegistry("configs/factors.yaml")  # 加载因子池
  pipeline = AlphaSignalPipeline(window, registry, "csi300")
  signal = pipeline.predict(today, dm)                # 直接产出信号

场景2: 手动添加新因子
  registry.add("MY_FACTOR", "Corr($close, $volume, 20)", "liquidity", "manual")
  validator = FactorValidator(dm)
  report = validator.validate_single("Corr($close, $volume, 20)", "2020-01-01", "2024-12-31")
  print(report.verdict)   # "accept" → 因子自动启用
  print(report.ic_mean)   # 0.035
  print(report.icir)      # 1.2
  registry.save()         # 持久化

场景3: 参数搜索
  miner = FactorMiner(validator, registry)
  reports = miner.explore_variants(
      "Std($close, {w})/$close",
      {"w": [5, 10, 20, 30, 60]}
  )
  miner.accept_and_register(reports)  # 自动注册通过验证的

场景4: 全自动挖掘
  reports = miner.auto_mine(budget=500)  # 生成500个候选，验证，去冗余
  miner.accept_and_register(reports)     # 注册优质因子
  # 下次 predict() 自动包含新因子

场景5: RD-Agent 进化因子接入
  # M4 进化出因子表达式后调用
  registry.add("RDAGENT_001", evolved_expression, "mean_revert", "rdagent")
  report = validator.validate_incremental(evolved_expression, registry.get_expressions()[0])
  if report.verdict == "accept":
      registry.enable("RDAGENT_001")
```

---

### 测试

#### FactorRegistry 测试

| 测试项 | 方法 | 验证点 |
|--------|------|--------|
| 初始加载 | 首次创建 registry | 自动生成 158 个 alpha158 因子 |
| 添加因子 | `add("TEST", "Mean($close,5)", ...)` | 因子池数量 +1 |
| 重名拒绝 | 添加已存在的 name | 抛出异常 |
| 删除保护 | 尝试 `remove` alpha158 来源的因子 | 拒绝删除，建议 disable |
| 启用禁用 | `disable("MA5")` 后 `get_enabled()` | MA5 不在返回列表中 |
| 持久化 | `save()` 后重新 `__init__` 加载 | 因子列表一致（含自定义因子） |
| get_expressions | 检查返回值 | 两个列表等长，表达式语法合法 |

#### FactorValidator 测试

| 测试项 | 方法 | 验证点 |
|--------|------|--------|
| 有效因子 | 验证 `"Mean($close, 20)/$close"` | IC > 0, ICIR > 0, verdict ≠ "reject" |
| 无效因子 | 验证 `"Ref($close, 0)"` （常数因子） | IC ≈ 0, verdict = "reject" |
| 冗余检测 | 验证 `"Mean($close, 21)/$close"` （与 MA20 高度相关） | max_corr_with_existing > 0.9 |
| IC 衰减 | 检查 ic_decay 字典 | T+1 > T+5 > T+10（单调递减） |
| 批量验证 | 传入 10 个表达式 | 返回 10 个 FactorReport |
| 表达式语法错误 | 传入 `"InvalidOp($close)"` | 返回错误信息，不崩溃 |

#### FactorMiner 测试

| 测试项 | 方法 | 验证点 |
|--------|------|--------|
| 参数搜索 | explore_variants 窗口 [5,10,20] | 返回 3 个 report，ICIR 不全相同 |
| 组合生成 | explore_combinations 2字段×2算子×2窗口 | 返回 8 个 report |
| 去冗余 | auto_mine 中相关 > 0.7 的因子 | 最终结果中任意两因子相关 < 0.7 |
| 自动注册 | accept_and_register 3 个 accept + 2 个 reject | registry 新增 3 个因子 |
| 空搜索 | budget=0 | 返回空列表，不报错 |

#### AlphaSignalPipeline 测试

| 测试项 | 方法 | 验证点 |
|--------|------|--------|
| 重训间隔 | 连续 predict 25 次，因子池不变 | train 在第1次和第21次被调用 |
| 因子变更触发 | predict → registry.add → predict | 第二次 predict 触发重训 |
| 标签无泄露 | 训练后检查 dataset 最大日期 | ≤ anchor_date - 2 交易日 |
| 输出格式 | predict 返回值 | Series, index 为股票代码, 无 NaN |
| 特征重要性 | train 后调用 get_feature_importance() | 返回 Series, index 为因子名, 值为 importance |
| 因子缺失容忍 | 某因子对部分股票算出 NaN | Fillna 处理器填充，不影响预测 |

---

## M3 Kronos 信号管线（signal_kronos.py）

### 职责

三项核心职责：

1. **微调实验管理** — 支持定义和对比多种微调方案（冻结策略、损失配置、数据采样）
2. **每日微调与推理** — 按选定方案执行滚动微调 + 批量推理
3. **信号输出** — 产出预测收益 + 不确定性度量

### 设计思路

Kronos 的训练分两个阶段，各有不同的监督信号：

```
阶段1 Tokenizer: 输入 OHLCV → 编码 → 量化 → 解码 → 重建 OHLCV
   监督信号: MSE 重建损失 + BSQ 量化损失(commit + 熵正则)
   目的: 学习将 K 线压缩为离散 token

阶段2 Predictor: 输入 token 序列 → Transformer → 预测下一个 token
   监督信号: Cross-entropy (s1 粗粒度 + s2 细粒度各 50%)
   目的: 学习 token 序列的时间规律
```

可实验的维度：

| 维度 | 选项 |
|------|------|
| 微调哪个模块 | tokenizer only / predictor only / 两者都调 / 都不调(零样本) |
| 冻结策略 | 全部解冻 / 最后N层 / 只调输出头 / LoRA 适配 |
| 损失权重 | s1/s2 权重比、重建损失中 pre/full 权重比、BSQ 各项权重 |
| 数据采样 | 均匀采样 / 近期加权 / 按行业分层 / 按波动率分层 |
| 训练强度 | epoch 数、学习率、batch_size、梯度累积步数 |
| 预测策略 | 采样温度 T、top_k、top_p、采样次数 |

### 子模块划分

```
signal_kronos.py
├── FinetuneRecipe          微调方案定义
├── FinetuneExperiment      实验管理（对比多方案）
├── KronosFinetuner         微调执行器
├── KronosPredictor         推理执行器
└── KronosSignalPipeline    信号生产（日常使用入口）
```

---

### 子模块一：FinetuneRecipe（微调方案定义）

#### 职责

一个 FinetuneRecipe 完整描述"如何微调 Kronos"。可定义多个 recipe 进行对比实验。

#### 数据结构

```
FinetuneRecipe
    name: str                        方案名称，如 "last2_predictor_only"
    description: str                 方案描述

    # === 模块选择 ===
    finetune_tokenizer: bool         是否微调 tokenizer（默认 False）
    finetune_predictor: bool         是否微调 predictor（默认 True）

    # === 冻结策略 ===
    tokenizer_strategy: str          "none" | "last_n" | "head_only" | "full"
    tokenizer_unfreeze_layers: int   解冻最后 N 层（strategy=last_n 时使用）
    predictor_strategy: str          "none" | "last_n" | "head_only" | "full"
    predictor_unfreeze_layers: int   解冻最后 N 层

    # === 损失配置 ===
    # Tokenizer 损失
    recon_pre_weight: float          s1-only 重建损失权重（默认 1.0）
    recon_full_weight: float         s1+s2 重建损失权重（默认 1.0）
    bsq_weight: float               BSQ 量化损失权重（默认 1.0）
    bsq_beta: float                  commit loss 系数（默认 0.05）
    bsq_gamma0: float                样本熵权重（默认 1.0）
    bsq_gamma: float                 码本熵权重（默认 1.1）

    # Predictor 损失
    s1_loss_weight: float            s1(粗粒度) CE 权重（默认 1.0）
    s2_loss_weight: float            s2(细粒度) CE 权重（默认 1.0）

    # === 数据采样 ===
    data_lookback: int               微调数据窗口天数（默认 30）
    sample_strategy: str             "uniform" | "recency_weighted" | "volatility_stratified"
    recency_decay: float             近期加权的指数衰减率（strategy=recency_weighted 时使用，默认 0.95）

    # === 训练超参 ===
    epochs: int                      微调轮数（默认 3）
    learning_rate: float             学习率（默认 2e-5）
    batch_size: int                  批大小（默认 64）
    weight_decay: float              权重衰减（默认 0.1）
    accumulation_steps: int          梯度累积步数（默认 1）
    warmup_ratio: float              学习率预热比例（默认 0.1）

    # === 推理超参 ===
    predict_horizon: int             预测天数（默认 5）
    sample_count: int                采样次数（默认 10）
    temperature: float               采样温度（默认 0.6）
    top_p: float                     nucleus sampling（默认 0.9）
    top_k: int                       top-k filtering（默认 0）

    # === 每日重置 ===
    reset_from_pretrained: bool      每天从预训练权重重新开始（默认 True）
                                     设为 False 则在前一天微调结果上继续（累积微调）
```

#### 预设方案

```yaml
# configs/kronos_recipes.yaml

recipes:
  # 方案A：保守策略 — 只调 predictor 最后2层
  - name: conservative
    finetune_tokenizer: false
    finetune_predictor: true
    predictor_strategy: last_n
    predictor_unfreeze_layers: 2
    epochs: 3
    learning_rate: 2e-5
    reset_from_pretrained: true

  # 方案B：激进策略 — tokenizer + predictor 全解冻
  - name: aggressive
    finetune_tokenizer: true
    finetune_predictor: true
    tokenizer_strategy: full
    predictor_strategy: full
    epochs: 5
    learning_rate: 1e-5
    reset_from_pretrained: true

  # 方案C：仅输出头
  - name: head_only
    finetune_tokenizer: false
    finetune_predictor: true
    predictor_strategy: head_only
    predictor_unfreeze_layers: 0
    epochs: 10
    learning_rate: 5e-5

  # 方案D：累积微调 — 每天在前一天基础上继续调
  - name: cumulative
    finetune_tokenizer: false
    finetune_predictor: true
    predictor_strategy: last_n
    predictor_unfreeze_layers: 2
    epochs: 1
    learning_rate: 5e-6
    reset_from_pretrained: false

  # 方案E：近期加权 — 最近的数据样本权重更高
  - name: recency_focus
    finetune_tokenizer: false
    finetune_predictor: true
    predictor_strategy: last_n
    predictor_unfreeze_layers: 2
    sample_strategy: recency_weighted
    recency_decay: 0.93
    epochs: 3

  # 方案F：零样本 — 不微调，直接用预训练模型推理（作为对照基线）
  - name: zero_shot
    finetune_tokenizer: false
    finetune_predictor: false

  # 方案G：侧重粗粒度 — s1 权重加大
  - name: s1_heavy
    finetune_predictor: true
    predictor_strategy: last_n
    predictor_unfreeze_layers: 2
    s1_loss_weight: 2.0
    s2_loss_weight: 0.5
```

---

### 子模块二：FinetuneExperiment（实验管理）

#### 职责

对比多个 FinetuneRecipe 的效果，找出最优方案。

#### 接口

| 方法 | 输入 | 输出 | 说明 |
|------|------|------|------|
| `__init__(data_manager, recipes, eval_period)` | M1实例, 方案列表, 评估区间 | — | |
| `run_experiment(recipe)` | 单个方案 | RecipeReport | 在 eval_period 上运行一个方案，返回评估报告 |
| `run_all()` | — | List[RecipeReport] | 逐个运行全部方案 |
| `compare()` | — | DataFrame | 横向对比全部方案的核心指标 |
| `get_best_recipe()` | — | FinetuneRecipe | 返回综合指标最优的方案 |
| `save_results(output_dir)` | 输出目录 | — | 保存对比表 + 各方案详细报告 |

#### RecipeReport 内容

```
RecipeReport
    recipe_name: str
    # 预测质量
    ic_mean: float                   全周期 Rank IC 均值
    icir: float                      IC 信息比率
    ic_1d: float                     T+1 的 IC（日频信号质量）
    ic_5d: float                     T+5 的 IC（5日信号质量）

    # 回测绩效（用信号单独跑简化回测）
    long_short_sharpe: float         多空组合夏普
    long_only_return: float          纯多头年化收益

    # 计算效率
    finetune_time_sec: float         单次微调平均耗时
    predict_time_sec: float          单次推理平均耗时
    total_time_sec: float            微调 + 推理总耗时

    # 稳定性
    ic_std: float                    IC 标准差（越小越稳定）
    worst_week_ic: float             最差一周的 IC 均值
```

#### compare 输出示例

```
DataFrame:
                 ic_mean  icir  ic_1d  long_short_sharpe  total_time_sec
conservative      0.035  1.20  0.038          1.45            380
aggressive        0.041  0.95  0.042          1.60            720
head_only         0.028  1.35  0.030          1.10            180
cumulative        0.033  0.80  0.035          1.20            200
recency_focus     0.039  1.15  0.041          1.55            400
zero_shot         0.020  0.60  0.022          0.70              60
s1_heavy          0.037  1.10  0.039          1.50            390
```

#### run_experiment 内部流程

```
1. 在 eval_period 的每个交易日：
   a. 用当天的 recipe 配置执行微调（或跳过微调）
   b. 批量推理全市场
   c. 记录信号 + 计时

2. 汇总全周期信号，计算：
   · IC / ICIR（与次日实际收益的 Rank 相关）
   · 多空组合收益
   · 耗时统计

3. 输出 RecipeReport
```

---

### 子模块三：KronosFinetuner（微调执行器）

#### 职责

按给定 FinetuneRecipe 对 Kronos 模型执行一次微调。

#### 接口

| 方法 | 输入 | 输出 | 说明 |
|------|------|------|------|
| `__init__(base_tokenizer, base_model, device)` | 预训练权重, 设备 | — | 加载并冻结预训练模型 |
| `finetune(symbol_data, recipe)` | 全市场数据, 微调方案 | (tokenizer, model) | 执行微调，返回微调后的模型 |

#### finetune 内部流程

```
1. 准备模型
   · if recipe.reset_from_pretrained:
       深拷贝预训练权重
     else:
       使用上一次微调后的权重（累积模式）

2. 应用冻结策略
   · 遍历 tokenizer 和 predictor 的参数
   · 根据 recipe 的 strategy 和 unfreeze_layers 设置 requires_grad

3. 构建数据集
   · 从 symbol_data 取最近 data_lookback 天
   · 根据 sample_strategy 构建采样器：
     - uniform: 所有窗口等概率
     - recency_weighted: 近期窗口概率 = decay^(距今天数)
     - volatility_stratified: 按波动率分层，各层等量采样

4. 配置优化器
   · AdamW(可训练参数, lr, weight_decay)
   · 学习率调度：warmup + cosine decay

5. 训练循环
   · for epoch in range(recipe.epochs):
       for batch in dataloader:
         if recipe.finetune_tokenizer:
           tokenizer_loss = recon_loss × 权重 + bsq_loss × 权重
         if recipe.finetune_predictor:
           predictor_loss = s1_ce × s1_weight + s2_ce × s2_weight
         loss = tokenizer_loss + predictor_loss
         loss.backward()
         if step % accumulation_steps == 0:
           optimizer.step()

6. 返回微调后的 (tokenizer, model)
```

#### 冻结策略实现

```
strategy = "none":
    全部冻结（不微调该模块）

strategy = "head_only":
    只解冻输出头（DualHead 的 s1_head + s2_head）
    或 tokenizer 的 decoder head

strategy = "last_n":
    解冻最后 N 个 Transformer block + 输出头
    tokenizer: 最后 N 个 decoder block + head
    predictor: 最后 N 个 transformer block + DualHead + DependencyAwareLayer

strategy = "full":
    全部解冻
```

#### 数据采样策略

```
uniform:
    每只股票的每个可用滑窗等概率采样
    样本数 = 股票数 × 窗口数

recency_weighted:
    窗口 i (距今 d 天) 的采样权重 = decay^d
    例: decay=0.95, 今天权重=1.0, 10天前=0.60, 30天前=0.21
    效果: 近期数据被更频繁采样，但远期数据不完全丢弃

volatility_stratified:
    按过去20天波动率将股票分为3组 (高/中/低波动)
    每组等量采样
    目的: 防止高波动股票主导训练（它们的损失天然更大）
```

---

### 子模块四：KronosInference（推理执行器）

#### 接口

| 方法 | 输入 | 输出 | 说明 |
|------|------|------|------|
| `__init__(device, max_context)` | 设备, 最大上下文 | — | |
| `predict_all(tokenizer, model, symbol_data, recipe, data_manager)` | 模型, 数据, 方案, M1 | KronosOutput | 批量推理 |

#### KronosOutput 输出

```
KronosOutput
    return_1d: Series[symbol→float]     T+1 预测收益率（多采样均值）
    return_5d: Series[symbol→float]     T+1~T+5 预测收益率（多采样均值）
    uncertainty: Series[symbol→float]   预测分歧度（多采样标准差）
    pred_klines: Dict[symbol, DataFrame]  可选：完整预测 K 线（供 M1.5 可视化）
```

#### 推理流程

```
1. 准备批量输入
   · 每只股票取最近 lookback 天的 OHLCV
   · 构造 df_list, x_timestamp_list, y_timestamp_list

2. 多次采样推理
   · for s in range(recipe.sample_count):
       preds[s] = predictor.predict_batch(
           df_list, x_ts_list, y_ts_list,
           pred_len=recipe.predict_horizon,
           T=recipe.temperature,
           top_p=recipe.top_p,
           top_k=recipe.top_k,
           sample_count=1
       )

3. 聚合
   · 对每只股票的 sample_count 次预测:
     return_1d = mean(第1天预测 close / 当前 close - 1)
     return_5d = mean(5天预测 close 均值 / 当前 close - 1)
     uncertainty = std(5天预测收益)

4. 可选：保留 pred_klines 用于可视化
```

---

### 子模块五：KronosSignalPipeline（日常使用入口）

#### 职责

日常回测/实盘的入口。使用选定的 recipe 执行微调 + 推理。

#### 接口

| 方法 | 输入 | 输出 | 说明 |
|------|------|------|------|
| `__init__(window, recipe, tokenizer_path, model_path, device)` | 配置, 选定方案, 模型路径, 设备 | — | |
| `daily_run(symbol_data, anchor_date, data_manager)` | 全市场数据, 锚定日, M1 | KronosOutput | 执行微调 + 推理，返回信号 |
| `switch_recipe(recipe)` | 新方案 | — | 切换微调方案（实验对比后决定用哪个） |

---

### M3 整体使用流程

```
场景1: 日常回测/实盘（信号生产）
  recipe = FinetuneRecipe.load("configs/kronos_recipes.yaml", "conservative")
  pipeline = KronosSignalPipeline(window, recipe, tok_path, model_path)
  output = pipeline.daily_run(symbol_data, today, dm)
  # output.return_1d → 传给 M5 融合

场景2: 对比实验 — 找最优微调方案
  recipes = FinetuneRecipe.load_all("configs/kronos_recipes.yaml")
  exp = FinetuneExperiment(dm, recipes, eval_period=("2024-07-01", "2024-12-31"))
  exp.run_all()
  print(exp.compare())
  #                  ic_mean  icir   total_time_sec
  # conservative      0.035  1.20           380
  # aggressive        0.041  0.95           720
  # head_only         0.028  1.35           180
  # zero_shot         0.020  0.60            60
  best = exp.get_best_recipe()
  pipeline.switch_recipe(best)

场景3: 自定义实验 — 测试新的损失配比
  custom = FinetuneRecipe(
      name="s1_heavy_recency",
      finetune_predictor=True,
      predictor_strategy="last_n",
      predictor_unfreeze_layers=3,
      s1_loss_weight=2.0,
      s2_loss_weight=0.5,
      sample_strategy="recency_weighted",
      recency_decay=0.93,
  )
  report = exp.run_experiment(custom)
  print(report.ic_mean, report.long_short_sharpe)

场景4: 累积微调 vs 每日重置
  # 直接在 recipe 中切换 reset_from_pretrained
  # 运行实验对比两种模式的稳定性

场景5: 可视化预测效果
  output = pipeline.daily_run(symbol_data, today, dm)
  viewer.plot_kline_with_prediction("SH600519",
      symbol_data["SH600519"], output.pred_klines["SH600519"])
```

---

### 测试

#### FinetuneRecipe 测试

| 测试项 | 方法 | 验证点 |
|--------|------|--------|
| YAML 加载 | 加载预设方案文件 | 7 个方案全部正确解析 |
| 默认值 | 创建空 recipe | 所有字段有合理默认值 |
| 参数校验 | epochs=-1 | 抛出 ValueError |
| 序列化 | save → load 往返 | 字段完全一致 |

#### KronosFinetuner 测试

| 测试项 | 方法 | 验证点 |
|--------|------|--------|
| 预训练权重不变 | 微调后对比 base_model 参数 | 预训练权重未被修改 |
| 每日重置 | reset=True，连续两天微调 | 两天的起点参数相同 |
| 累积微调 | reset=False，连续两天微调 | 第二天的起点 = 第一天的终点 |
| 冻结策略 none | finetune_predictor=False | 微调前后参数完全一致 |
| 冻结策略 last_n | unfreeze_layers=2 | 只有最后2层参数变化，其余不变 |
| 冻结策略 head_only | predictor_strategy="head_only" | 只有 DualHead 参数变化 |
| 冻结策略 full | predictor_strategy="full" | 所有层参数都有变化 |
| 损失权重生效 | s1_weight=0, s2_weight=1 | 训练 loss 中 s1 分量为 0 |
| 近期加权采样 | decay=0.5, 检查 dataloader 的采样分布 | 最近一天的采样频率 ≈ 30天前的 2^6 ≈ 64 倍 |
| 波动率分层 | 检查各层样本数 | 高/中/低波动各约 1/3 |

#### KronosInference 测试

| 测试项 | 方法 | 验证点 |
|--------|------|--------|
| 输出维度 | return_1d, return_5d, uncertainty 的 index | 三者一致，长度 = 输入股票数 |
| 不确定性合理 | uncertainty 值 | 全部 ≥ 0，非 NaN |
| 温度影响 | T=0.1 vs T=2.0 | T=0.1 的 uncertainty 显著低于 T=2.0 |
| 批量一致 | predict_batch vs 逐只 predict（相同 seed） | 结果一致 |
| 推理延迟 | 10 只股票计时 | 可线性推算全市场耗时 |

#### FinetuneExperiment 测试

| 测试项 | 方法 | 验证点 |
|--------|------|--------|
| 多方案对比 | 运行 zero_shot + conservative | 返回 2 行 DataFrame |
| 最优选择 | get_best_recipe | 返回 ICIR 最高的方案 |
| zero_shot 基线 | zero_shot 方案 | IC > 0（预训练模型有基础能力）但 < 微调方案 |
| 报告完整 | 检查 RecipeReport 字段 | 无 NaN、无缺失字段 |

#### KronosSignalPipeline 测试

| 测试项 | 方法 | 验证点 |
|--------|------|--------|
| 端到端 | daily_run 返回 KronosOutput | return_1d 非空，值在合理范围 |
| 方案切换 | switch_recipe 后 daily_run | 使用新方案的超参（检查模型解冻层数） |
| 退化测试 | 全为相同 K 线的输入 | return ≈ 0, uncertainty 低 |

---

## M4 RD-Agent 信号管线（signal_rdagent.py）

### 职责

三项核心职责：

1. **进化管理** — 配置并驱动 RD-Agent 离线进化循环，产出 Python 因子代码
2. **因子生命周期** — 管理 LLM 生成的因子代码（注册、验证、衰退检测、退役）
3. **安全在线计算** — 在沙箱中执行进化因子代码，按 IC 加权合成信号

### 与 M2 的边界

```
M2 管理 Qlib 表达式因子（Alpha158 + 手动添加 + 自动挖掘）
M4 管理 Python 代码因子（RD-Agent LLM 进化产出）

交叉点：
  · RD-Agent 进化出 Qlib 表达式 → 注入 M2 FactorRegistry（调用 registry.add）
  · RD-Agent 进化出 Python 代码 → M4 CodeFactorRegistry 管理
  · 两者共用 M2 FactorValidator 做验证（IC/ICIR/相关性）
  · 正交性约束：M4 的因子与 M2 因子池的截面相关 < 0.3
```

### 设计思路

RD-Agent 的核心是进化循环：假设生成 → LLM 编码 → QlibFBWorkspace 执行 → MLflow 指标反馈 → 下一轮迭代。
本模块需要：

1. **适配 RD-Agent 真实架构**（QlibFBWorkspace、QlibFactorRunner、Docker/Conda 环境），而非自行实现简化版
2. **约束进化方向**（通过 prompt 注入和验证门控），避免产出与 M2 冗余或过拟合的因子
3. **管理因子全生命周期**（不只是产出，还有衰退检测和退役）
4. **沙箱执行**（LLM 生成的代码不可信，在线计算必须隔离）

### 子模块划分

```
signal_rdagent.py
├── EvolutionConfig          进化循环配置（方向约束、资源限制）
├── CodeFactorRegistry       Python 代码因子注册表
├── EvolutionRunner          进化执行器（包装 RD-Agent 进化循环）
├── CodeFactorExecutor       沙箱因子计算引擎
└── RDAgentSignalPipeline    信号生产（日常使用入口）
```

---

### 子模块一：EvolutionConfig（进化配置）

#### 职责

完整描述"如何驱动 RD-Agent 进化"的配置。控制进化方向、资源限制和验证门控。

#### 数据结构

```
EvolutionConfig
    name: str                          配置名称，如 "mean_revert_focus"

    # === 进化方向约束 ===
    target_directions: List[str]       目标因子方向
                                        例: ["mean_revert", "volatility_anomaly",
                                             "liquidity_change", "momentum_divergence"]
    direction_prompt: str              注入 RD-Agent 的方向提示词
                                        描述期望的因子特征和禁止的模式
    forbidden_patterns: List[str]      禁止的代码模式
                                        例: ["future_data", "hardcoded_date", "internet_access"]

    # === 正交性约束 ===
    max_corr_with_alpha: float         与 M2 因子池最大截面相关（默认 0.3）
    max_corr_within_pool: float        M4 因子间最大截面相关（默认 0.5）

    # === 验证门控 ===
    min_ic: float                      最低 IC 阈值（默认 0.02）
    min_icir: float                    最低 ICIR 阈值（默认 0.5）
    max_overfit_gap: float             train IC - test IC 最大差值（默认 0.05）
    validation_split: float            验证集比例（默认 0.3）

    # === 进化资源 ===
    max_rounds: int                    每次进化最大轮数（默认 20）
    max_factors_per_round: int         每轮最多产出因子数（默认 5）
    total_budget: int                  单次进化总因子数上限（默认 50）
    timeout_hours: float               单次进化超时（默认 4.0）

    # === RD-Agent 执行环境 ===
    execution_env: str                 "docker" | "conda"（默认 "docker"）
    docker_image: str                  Docker 镜像名（默认 "rdagent-qlib:latest"）
    conda_env: str                     Conda 环境名（默认 "rdagent"）

    # === 因子代码约定 ===
    factor_interface: str              因子函数签名约定
                                        默认: "compute_factor(ohlcv: Dict[str, DataFrame]) -> Series"
    required_docstring: bool           是否要求因子代码包含 docstring（默认 True）
    max_code_lines: int                单个因子代码最大行数（默认 100）
```

#### 预设配置

```yaml
# configs/rdagent_evolution.yaml

configs:
  # 配置A：均值回复方向
  - name: mean_revert_focus
    target_directions: ["mean_revert", "overreaction"]
    direction_prompt: |
      Focus on mean-reversion signals: price deviation from moving averages,
      volume-price divergence, RSI extremes, Bollinger Band breakouts.
      Avoid trend-following or momentum signals.
    max_corr_with_alpha: 0.3
    max_rounds: 20
    total_budget: 50

  # 配置B：波动率异常方向
  - name: volatility_anomaly
    target_directions: ["volatility_anomaly", "regime_change"]
    direction_prompt: |
      Focus on volatility regime changes: GARCH residuals, realized vs implied
      vol spread, intraday range anomalies, volume spike detection.
    max_rounds: 15
    total_budget: 30

  # 配置C：流动性因子方向
  - name: liquidity_focus
    target_directions: ["liquidity_change", "market_microstructure"]
    direction_prompt: |
      Focus on liquidity signals: Amihud illiquidity, turnover rate changes,
      bid-ask proxy from OHLC, Kyle's lambda estimation.
    max_rounds: 15
    total_budget: 30

  # 配置D：多方向探索（宽泛搜索）
  - name: broad_explore
    target_directions: ["mean_revert", "volatility_anomaly", "liquidity_change",
                         "momentum_divergence", "cross_section_anomaly"]
    direction_prompt: |
      Explore diverse alpha signals. Prioritize novelty over signal strength.
      Each factor should capture a distinct market phenomenon.
    max_corr_with_alpha: 0.25
    max_corr_within_pool: 0.4
    max_rounds: 30
    total_budget: 100
```

---

### 子模块二：CodeFactorRegistry（代码因子注册表）

#### 职责

管理 RD-Agent 进化产出的 Python 因子代码，记录元数据和生命周期状态。
与 M2 FactorRegistry 并行，但管理的是 Python 代码而非 Qlib 表达式。

#### 数据结构

```
CodeFactorEntry
    name: str                      因子名称，如 "rdagent_vol_spike_001"
    code_path: str                 代码文件路径（相对于 factor_code_dir）
    source_round: int              产出该因子的进化轮次
    source_config: str             产出该因子的 EvolutionConfig 名称
    direction: str                 因子方向标签
    description: str               因子描述（从 docstring 提取）
    created_date: str              创建日期
    status: str                    "active" | "probation" | "retired"
    enabled: bool                  是否参与在线计算

    # === 验证指标（注册时快照） ===
    ic_at_creation: float          注册时的 IC
    icir_at_creation: float        注册时的 ICIR
    corr_with_alpha: float         注册时与 M2 因子池的最大截面相关

    # === 生命周期跟踪 ===
    ic_history: List[Tuple[str, float]]   (日期, 滚动IC) 序列
    last_decay_check: str          上次衰退检查日期
    decay_warnings: int            连续衰退警告次数
    weight: float                  当前合成权重（基于 ICIR）
```

#### 接口

| 方法 | 输入 | 输出 | 说明 |
|------|------|------|------|
| `__init__(factor_code_dir, registry_path)` | 因子代码目录, 注册表文件路径 | — | 加载或初始化注册表 |
| `register(name, code, direction, source_round, config_name)` | 因子信息 | CodeFactorEntry | 保存代码文件 + 注册元数据 |
| `get_active()` | — | List[CodeFactorEntry] | 返回 status="active" 且 enabled=True 的因子 |
| `get_all()` | — | List[CodeFactorEntry] | 返回全部因子（含 retired） |
| `retire(name, reason)` | 因子名, 退役原因 | — | 标记为 retired，禁用 |
| `set_probation(name)` | 因子名 | — | 标记为 probation（观察期） |
| `update_ic(name, date, ic_value)` | 因子名, 日期, IC值 | — | 追加滚动 IC 记录 |
| `update_weights(factor_icir_map)` | {name: icir} | — | 按 ICIR 重新计算权重 |
| `save()` | — | — | 持久化到 YAML |
| `load_code(name)` | 因子名 | str | 读取因子代码内容 |

#### 持久化格式

```yaml
# rdagent_factors/registry.yaml
factors:
  - name: rdagent_vol_spike_001
    code_path: rdagent_vol_spike_001.py
    source_round: 3
    source_config: volatility_anomaly
    direction: volatility_anomaly
    description: "Detects abnormal volume spikes relative to 20-day average"
    created_date: "2025-03-01"
    status: active
    enabled: true
    ic_at_creation: 0.035
    icir_at_creation: 0.85
    corr_with_alpha: 0.18
    decay_warnings: 0
    weight: 0.32
  - name: rdagent_mean_rev_002
    code_path: rdagent_mean_rev_002.py
    source_round: 7
    source_config: mean_revert_focus
    direction: mean_revert
    description: "Price deviation from adaptive moving average with volume confirmation"
    created_date: "2025-02-15"
    status: active
    enabled: true
    ic_at_creation: 0.028
    icir_at_creation: 0.62
    corr_with_alpha: 0.22
    decay_warnings: 1
    weight: 0.24
```

#### 权重计算

```
update_weights 流程:
1. 收集所有 active 因子的近期 ICIR
2. 截断负 ICIR 为 0
3. weight_i = max(0, icir_i) / sum(max(0, icir_j) for j in active)
4. 若所有 ICIR ≤ 0，退化为等权: weight_i = 1/N
```

---

### 子模块三：EvolutionRunner（进化执行器）

#### 职责

包装 RD-Agent 的进化循环，产出经过验证的因子代码。
离线运行（每周末或手动触发），不参与日常在线信号计算。

#### 接口

| 方法 | 输入 | 输出 | 说明 |
|------|------|------|------|
| `__init__(config, code_registry, alpha_registry, validator)` | 进化配置, M4注册表, M2注册表, M2验证器 | — | |
| `run_evolution(data_manager)` | M1实例 | EvolutionReport | 执行一轮完整进化循环 |
| `extract_factors(workspace_path)` | QlibFBWorkspace 路径 | List[RawFactor] | 从进化产出中提取因子代码 |
| `validate_and_register(raw_factors, data_manager)` | 候选因子, M1实例 | List[CodeFactorEntry] | 验证 + 去冗余 + 注册 |

#### RawFactor 中间结构

```
RawFactor
    code: str                  因子 Python 代码
    name: str                  LLM 给出的因子名
    description: str           LLM 给出的描述
    round_idx: int             来自第几轮进化
    mlflow_ic: float           RD-Agent 内部报告的 IC（参考值，不作为最终判据）
```

#### EvolutionReport 输出

```
EvolutionReport
    config_name: str           使用的配置名称
    total_rounds: int          实际运行轮数
    total_candidates: int      产出候选因子数
    passed_validation: int     通过验证的因子数
    registered: int            最终注册的因子数（去冗余后）
    rejected_reasons: Dict[str, int]   拒绝原因统计
                                {"low_ic": 5, "overfit": 3, "high_corr": 8, "code_error": 2}
    elapsed_hours: float       总耗时
    round_details: List[RoundDetail]   每轮详情
```

#### run_evolution 内部流程

```
1. 构建进化环境
   · 根据 config.execution_env 选择 Docker 或 Conda
   · 初始化 QlibFBWorkspace
   · 注入方向约束 prompt:
     - config.direction_prompt → 添加到 hypothesis generation prompt
     - 追加正交性要求: "生成的因子与以下已有因子相关系数须 < {max_corr}:
       {alpha_registry.get_expressions() 的摘要}"
     - 追加禁止模式: config.forbidden_patterns

2. 配置 RD-Agent 进化参数
   · 通过 FactorBasePropSetting / QuantBasePropSetting 设置:
     - scen: "qlib"
     - hypothesis_gen: 注入方向约束的 prompt
     - max_loop: config.max_rounds
     - evolving_n: config.max_factors_per_round
   · 设置超时: config.timeout_hours

3. 运行进化循环
   · 调用 RD-Agent 的 QuantRDLoop / FactorRDLoop
   · 监控进度（每轮回调记录状态）
   · 超时或达到 total_budget 时提前终止

4. 提取产出
   · extract_factors: 从 workspace 目录扫描产出的因子代码
   · 解析 MLflow 日志获取 RD-Agent 内部评估指标

5. 验证与注册
   · validate_and_register: 对每个候选因子:
     a. 静态检查（语法、行数、禁止模式）
     b. 沙箱试运行（能否正常计算出结果）
     c. M2 FactorValidator 验证（IC/ICIR/相关性）
     d. 正交性检查（与 M2 因子池 + M4 已有因子）
     e. 过拟合检查（train IC - test IC < max_overfit_gap）
     f. 通过全部检查 → 注册到 CodeFactorRegistry

6. 返回 EvolutionReport
```

#### 因子分流逻辑

```
RD-Agent 产出的因子有两种形态:

形态1: Qlib 表达式（如 "Corr($close, $volume, 20) / Std($close, 10)"）
  · 检测方式: 代码中仅包含 Qlib 算子调用，无自定义逻辑
  · 处理: 提取表达式 → alpha_registry.add(name, expr, direction, "rdagent")
  · 后续由 M2 管理

形态2: Python 代码（含自定义计算逻辑）
  · 检测方式: 包含 pandas/numpy 操作、循环、自定义函数
  · 处理: 注册到 M4 CodeFactorRegistry
  · 后续由 M4 管理

分流判断:
  · 解析因子代码 AST
  · 若仅包含 Qlib 表达式字符串赋值 → 形态1
  · 否则 → 形态2
```

---

### 子模块四：CodeFactorExecutor（沙箱因子计算引擎）

#### 职责

在隔离环境中执行 LLM 生成的 Python 因子代码，防止恶意代码影响主进程。

#### 接口

| 方法 | 输入 | 输出 | 说明 |
|------|------|------|------|
| `__init__(sandbox_mode, timeout_sec)` | 沙箱模式, 超时 | — | |
| `execute_factor(code, ohlcv_data)` | 因子代码, OHLCV数据 | FactorResult | 在沙箱中执行单个因子 |
| `execute_batch(entries, ohlcv_data)` | 因子列表, OHLCV数据 | Dict[name, FactorResult] | 批量执行 |

#### FactorResult 输出

```
FactorResult
    success: bool              是否执行成功
    values: Series             symbol→float（成功时）
    error: str                 错误信息（失败时）
    elapsed_ms: int            执行耗时（毫秒）
```

#### 沙箱策略

```
sandbox_mode = "subprocess" (默认):
    · 在独立子进程中执行因子代码
    · 限制: 禁止 import os/sys/subprocess/socket/http
    · 允许: import numpy, pandas, scipy.stats, math
    · 超时: 默认 30 秒
    · 内存: 限制 1GB
    · 实现: subprocess + restricted globals

sandbox_mode = "docker":
    · 在 Docker 容器中执行
    · 更严格的隔离（无网络、只读文件系统）
    · 适合生产环境
    · 数据通过 volume mount 传入，结果通过 stdout 返回

两种模式提供相同接口，通过配置切换
```

#### 静态代码检查

```
在执行前进行静态检查:

1. AST 解析 — 代码必须是合法 Python
2. 禁止 import 检查:
   forbidden_imports = {"os", "sys", "subprocess", "socket", "http",
                         "requests", "urllib", "shutil", "pathlib",
                         "ctypes", "multiprocessing", "threading"}
3. 禁止内置函数: exec, eval, compile, __import__, open (写模式)
4. 函数签名检查: 必须定义 compute_factor(ohlcv) 函数
5. 代码行数 < config.max_code_lines
6. 静态检查通过后才进入沙箱执行
```

#### execute_batch 流程

```
1. 对每个 active 因子:
   a. 加载代码: registry.load_code(name)
   b. 静态检查
   c. 沙箱执行: execute_factor(code, ohlcv_data)
   d. 结果校验: 无 inf, 无全 NaN, 长度匹配
   e. 记录成功/失败

2. 错误处理:
   · 单个因子失败不影响其他因子
   · 连续失败 3 次的因子自动标记 probation
   · 失败因子的权重在本次合成中设为 0

3. 返回 {name: FactorResult}
```

---

### 子模块五：RDAgentSignalPipeline（日常使用入口）

#### 职责

日常回测/实盘入口。用注册表中的 active 因子计算信号，管理因子生命周期。

#### 接口

| 方法 | 输入 | 输出 | 说明 |
|------|------|------|------|
| `__init__(window, code_registry, executor, alpha_registry, validator)` | 配置 | — | |
| `compute(anchor_date, data_manager)` | 锚定日, M1实例 | RDAgentOutput | 执行因子计算 + 加权合成 |
| `check_decay(anchor_date, data_manager)` | 锚定日, M1实例 | DecayReport | 检查因子衰退 |
| `trigger_evolution(config, data_manager)` | 进化配置, M1 | EvolutionReport | 触发新一轮进化 |
| `get_factor_status()` | — | DataFrame | 全部因子状态概览 |

#### RDAgentOutput 输出

```
RDAgentOutput
    signal: Series[symbol→float]       加权合成信号
    factor_count: int                   参与合成的因子数
    factor_signals: Dict[name, Series]  各因子独立信号（供诊断）
    factor_weights: Dict[name, float]   各因子权重
    failed_factors: List[str]           本次执行失败的因子名
```

#### DecayReport 输出

```
DecayReport
    checked_date: str
    factor_reports: List[FactorDecayStatus]

FactorDecayStatus
    name: str
    rolling_ic_30d: float           近30天滚动 IC
    rolling_ic_90d: float           近90天滚动 IC
    ic_trend: str                   "stable" | "declining" | "collapsed"
    action: str                     "none" | "warn" | "probation" | "retire"
    reason: str
```

#### compute 内部流程

```
1. 获取 active 因子列表
   · entries = registry.get_active()
   · 若无 active 因子 → 返回空信号

2. 获取 OHLCV 数据
   · data = data_manager.get_ohlcv_before(anchor_date, lookback=window)

3. 沙箱批量计算
   · results = executor.execute_batch(entries, data)

4. 加权合成
   · 对成功的因子:
     - Rank 归一化: rank_i = rank(values_i) / N
     - 加权合成: signal = Σ(weight_i × rank_i) for successful factors
     - 权重重归一化: 失败因子的权重按比例分配给成功因子

5. 返回 RDAgentOutput
```

#### check_decay 内部流程

```
每个交易日收盘后调用:

1. 对每个 active + probation 因子:
   a. 计算近30天滚动 IC（因子值 vs 次日收益的 Rank 相关）
   b. 计算近90天滚动 IC
   c. 判断趋势:
      · ic_30d > 0.02 且 ic_30d > ic_90d × 0.5 → "stable"
      · ic_30d > 0 且 ic_30d < ic_90d × 0.5   → "declining"
      · ic_30d ≤ 0                              → "collapsed"

2. 生命周期动作:
   · stable + active: 无动作
   · stable + probation: 恢复为 active
   · declining + active: decay_warnings += 1
     - warnings ≥ 3 → 设为 probation
   · declining + probation: 无额外动作（继续观察）
   · collapsed + active: 直接设为 probation
   · collapsed + probation: 退役 (retire)

3. 更新权重
   · validator 重新计算近期 ICIR
   · registry.update_weights(新的 ICIR map)

4. 检查是否需要触发再进化
   · active 因子数 < 3 → 建议触发进化
   · 返回 DecayReport
```

#### 再进化触发条件

```
以下情况建议触发新一轮进化:

1. active 因子数 < 3（因子储备不足）
2. 平均 ICIR 连续 5 天 < 0.3（整体信号衰退）
3. 距上次进化 > 30 天（定期补充）
4. 手动触发（用户判断）

trigger_evolution 流程:
  · 创建 EvolutionRunner
  · 调用 run_evolution
  · 自动注册通过验证的新因子
  · 返回 EvolutionReport
```

---

### M4 整体使用流程

```
场景1: 日常回测/实盘（信号生产）
  registry = CodeFactorRegistry("rdagent_factors/", "rdagent_factors/registry.yaml")
  executor = CodeFactorExecutor(sandbox_mode="subprocess", timeout_sec=30)
  pipeline = RDAgentSignalPipeline(window, registry, executor, alpha_registry, validator)
  output = pipeline.compute(today, dm)
  # output.signal → 传给 M5 融合

场景2: 离线进化 — 产出新因子
  config = EvolutionConfig.load("configs/rdagent_evolution.yaml", "mean_revert_focus")
  report = pipeline.trigger_evolution(config, dm)
  print(f"产出 {report.registered} 个新因子，拒绝 {report.total_candidates - report.registered} 个")
  print(report.rejected_reasons)
  # {'low_ic': 5, 'overfit': 3, 'high_corr': 8, 'code_error': 2}

场景3: 因子衰退管理
  decay = pipeline.check_decay(today, dm)
  for f in decay.factor_reports:
      if f.action != "none":
          print(f"{f.name}: {f.action} — {f.reason}")
  # rdagent_vol_spike_001: warn — IC declining (30d: 0.015, 90d: 0.035)
  # rdagent_old_factor_003: retire — IC collapsed (30d: -0.01)

场景4: 因子状态概览
  status = pipeline.get_factor_status()
  print(status)
  #                          status  weight  ic_30d  decay_warnings
  # rdagent_vol_spike_001   active    0.32   0.028        1
  # rdagent_mean_rev_002    active    0.24   0.022        0
  # rdagent_liq_003         probation 0.00   0.005        3
  # rdagent_old_004         retired   0.00  -0.010        5

场景5: 多方向进化对比
  for config_name in ["mean_revert_focus", "volatility_anomaly", "liquidity_focus"]:
      config = EvolutionConfig.load("configs/rdagent_evolution.yaml", config_name)
      report = pipeline.trigger_evolution(config, dm)
      print(f"{config_name}: 注册 {report.registered}, 耗时 {report.elapsed_hours:.1f}h")

场景6: RD-Agent 进化出 Qlib 表达式 → 自动注入 M2
  # EvolutionRunner 内部自动分流:
  # 若产出的是 Qlib 表达式 → alpha_registry.add(name, expr, direction, "rdagent")
  # 若产出的是 Python 代码 → code_registry.register(name, code, ...)
  # 两种形态统一经过 FactorValidator 验证
```

---

### 测试

#### EvolutionConfig 测试

| 测试项 | 方法 | 验证点 |
|--------|------|--------|
| YAML 加载 | 加载预设配置文件 | 4 个配置全部正确解析 |
| 默认值 | 创建空 config | 所有字段有合理默认值 |
| 参数校验 | max_rounds=-1 | 抛出 ValueError |
| 方向约束 | 检查 direction_prompt | 非空，包含目标方向关键词 |
| 序列化 | save → load 往返 | 字段完全一致 |

#### CodeFactorRegistry 测试

| 测试项 | 方法 | 验证点 |
|--------|------|--------|
| 注册因子 | register + get_active | 新因子出现在 active 列表 |
| 退役因子 | retire(name) 后 get_active | 退役因子不在 active 列表 |
| 持久化 | save → 重新 __init__ | 因子列表一致，元数据完整 |
| 权重计算 | update_weights({a: 1.0, b: 2.0, c: 0.5}) | 权重比 ≈ 2:4:1，总和=1.0 |
| 全负 ICIR | update_weights({a: -0.1, b: -0.2}) | 退化为等权 0.5, 0.5 |
| 观察期 | set_probation(name) | status="probation", enabled=True, weight=0 |
| IC 追踪 | update_ic 多次后检查 ic_history | 日期有序，值正确 |
| 代码加载 | load_code(name) | 返回注册时的完整代码内容 |

#### EvolutionRunner 测试

| 测试项 | 方法 | 验证点 |
|--------|------|--------|
| 端到端进化 | Mock RD-Agent 进化循环，产出3个因子 | EvolutionReport.registered ≤ 3 |
| 方向约束注入 | 检查传给 RD-Agent 的 prompt | 包含 direction_prompt 内容 |
| 正交性过滤 | 候选因子与 M2 相关 > 0.3 | 被拒绝，reason="high_corr" |
| 过拟合过滤 | train IC=0.08, test IC=0.02 | 被拒绝，reason="overfit" |
| 代码错误处理 | 产出语法错误的代码 | 被拒绝，reason="code_error"，不崩溃 |
| 因子分流 | 产出含 Qlib 表达式和 Python 代码 | 表达式 → M2 registry, 代码 → M4 registry |
| 超时终止 | timeout_hours=0.001 | 进化提前终止，返回已有结果 |
| 空产出 | 进化无有效因子 | registered=0，不报错 |

#### CodeFactorExecutor 测试

| 测试项 | 方法 | 验证点 |
|--------|------|--------|
| 正常因子 | 执行合法因子代码 | success=True, values 为有效 Series |
| 语法错误 | 执行语法错误代码 | success=False, error 包含语法信息 |
| 超时 | 执行含 `while True` 的代码 | success=False, error 包含 "timeout" |
| 禁止 import | 代码中 `import os` | 静态检查拒绝，不执行 |
| 禁止网络 | 代码中 `import requests` | 静态检查拒绝 |
| 禁止 exec | 代码中含 `exec()` | 静态检查拒绝 |
| 输出校验 | 因子返回含 inf 的 Series | success=False, error 包含 "inf" |
| 批量执行 | 3个因子(2成功+1失败) | 返回3个结果，失败的不影响成功的 |
| 内存限制 | 因子代码分配超大数组 | success=False, error 包含 "memory" |

#### RDAgentSignalPipeline 测试

| 测试项 | 方法 | 验证点 |
|--------|------|--------|
| 空注册表 | 无 active 因子 | compute 返回空信号，不报错 |
| 加权合成 | 3个因子 weight=[0.5, 0.3, 0.2] | 信号为加权合成，非等权 |
| 失败重分配 | 3个因子中1个执行失败 | 2个成功因子权重按比例放大，总和=1.0 |
| 衰退检测 | 模拟 IC 持续下降 | check_decay 报告 "declining" |
| 衰退→退役 | collapsed 状态的 probation 因子 | 自动退役 |
| 恢复机制 | probation 因子 IC 回升 | 恢复为 active |
| 再进化触发 | active 因子数 < 3 | check_decay 建议触发进化 |
| 输出格式 | compute 返回值 | signal 为 Series, index 为股票代码, 无 NaN |
| 因子诊断 | 检查 factor_signals 和 factor_weights | 每个因子独立信号可追溯 |

---

## M5 信号融合（signal_ensemble.py）

### 职责

三项核心职责：

1. **Horizon 对齐** — 将三条管线不同频率/视野的信号统一到日频
2. **动态加权** — 基于滚动 IC 自适应调整各管线权重，含冷启动、权重钳制、相关性感知
3. **信号质量监控** — 跟踪各管线贡献度变化，检测信号退化

### 设计思路

三条管线的信号特性差异大：

```
管线A (Alpha158): 日频截面信号，值域不固定，信号间相关性高
管线B (Kronos):   5日预测拆为日频 + 不确定性度量，信号分布偏态
管线C (RD-Agent): 日频截面信号，可能为空（进化因子不足时）

融合需要解决：
  · 量纲差异 → Rank 归一化
  · 权重分配 → 滚动 IC 加权（非固定权重）
  · 冗余信号 → 相关性感知降权
  · 预测不确定 → Kronos uncertainty 惩罚
  · 冷启动 → 历史不足时的降级策略
  · 管线缺失 → 优雅降级（C管线可空）
```

### 子模块划分

```
signal_ensemble.py
├── SignalNormalizer        信号预处理与归一化
├── ICWeightEngine          IC 动态加权引擎
├── EnsembleMonitor         融合质量监控
└── SignalEnsemblePipeline  融合入口
```

---

### 子模块一：SignalNormalizer（信号预处理）

#### 职责

将各管线原始信号统一到可比较的尺度。

#### 接口

| 方法 | 输入 | 输出 | 说明 |
|------|------|------|------|
| `rank_normalize(signal)` | Series[symbol→float] | Series[symbol→float] | 截面 rank 归一化到 [0, 1] |
| `winsorize(signal, lower, upper)` | Series, 分位数边界 | Series | 缩尾处理（默认 1%/99%） |
| `align_index(signals)` | Dict[name, Series] | Dict[name, Series] | 对齐股票 index，缺失填 NaN |

#### rank_normalize 细节

```
1. signal.rank(pct=True, na_option="keep")
2. NaN 保持为 NaN（不参与排序也不被赋值）
3. 返回值域 (0, 1]，均匀分布
```

#### 缩尾处理时机

```
原始信号 → winsorize（去极值） → rank_normalize（归一化）
缩尾在 rank 之前，防止极端离群值扭曲排名分布
仅对 alpha 和 rdagent 管线启用（Kronos 输出已经是收益率预测，分布相对稳定）
```

---

### 子模块二：ICWeightEngine（IC 动态加权引擎）

#### 职责

基于各管线历史预测准确度动态分配权重。核心是滚动 Rank IC。

#### 接口

| 方法 | 输入 | 输出 | 说明 |
|------|------|------|------|
| `__init__(ic_lookback, min_weight, max_weight)` | IC窗口, 权重下限, 权重上限 | — | |
| `update(date, signals, actual_return)` | 日期, 各管线信号, 当日实际收益 | — | 追加历史记录 |
| `compute_ic(signal_name)` | 信号源名 | float | 近 ic_lookback 天的平均 Rank IC |
| `compute_weights()` | — | Dict[name, float] | 计算各管线当前权重 |
| `get_ic_history()` | — | DataFrame | 各管线逐日 IC 时序（供 M8 分析） |

#### 数据结构

```
ICWeightEngine 内部状态:
    ic_lookback: int                 IC 窗口长度（默认 60 天）
    min_weight: float                单管线最低权重（默认 0.1）
    max_weight: float                单管线最高权重（默认 0.6）
    history: Deque[DayRecord]        滚动历史（定长队列，自动淘汰旧记录）
    correlation_penalty: float       信号相关惩罚系数（默认 0.3）

DayRecord:
    date: str
    signals: Dict[name, Series]      各管线当日信号快照
    actual_return: Series            当日实际收益（T-1 信号 vs T 收益）
```

#### compute_weights 流程

```
1. 计算各管线滚动 IC
   · ic_i = mean(daily_rank_ic(signal_i, actual_return)) for recent ic_lookback days
   · 若历史不足 ic_lookback 天 → 冷启动模式

2. 冷启动处理
   · 历史 < 20 天 → 纯等权: weight_i = 1/N
   · 历史 20~ic_lookback 天 → 混合: w = α × ic_weight + (1-α) × equal_weight
     其中 α = (历史天数 - 20) / (ic_lookback - 20)
   · 历史 ≥ ic_lookback 天 → 纯 IC 加权

3. IC 加权（非冷启动）
   · 截断负 IC: effective_ic_i = max(0, ic_i)
   · 若全部 effective_ic ≤ 0 → 退化为等权
   · raw_weight_i = effective_ic_i / Σ effective_ic_j

4. 相关性感知调整
   · 计算管线间信号的平均截面 Spearman 相关
   · 若 corr(A, B) > 0.5:
     - 保留 IC 更高的管线权重不变
     - IC 较低的管线权重 × (1 - correlation_penalty × corr)
   · 重新归一化

5. 权重钳制
   · weight_i = clip(weight_i, min_weight, max_weight)
   · 重新归一化使 Σ weight = 1.0
   · 目的: 防止单管线独占权重（极端 IC 差异时）

6. 返回 {name: weight}
```

#### 权重钳制的意义

```
即使某管线 IC 暂时最高，也不能超过 max_weight (60%)
即使某管线 IC 暂时为零，也保留 min_weight (10%)

原因:
  · IC 本身有噪声，短窗口 IC 不代表真实信号质量
  · 防止某管线偶然表现好时过度集中
  · 保留多样化收益来源
  · min_weight 保证即使某管线低迷也不完全丢弃其信息
```

---

### 子模块三：EnsembleMonitor（融合质量监控）

#### 职责

跟踪融合信号和各管线的贡献度变化，检测异常。

#### 接口

| 方法 | 输入 | 输出 | 说明 |
|------|------|------|------|
| `__init__()` | — | — | |
| `record(date, weights, combined_signal, actual_return)` | 日期, 权重, 融合信号, 次日收益 | — | 每日记录 |
| `get_contribution_report(lookback)` | 天数 | ContributionReport | 各管线贡献度分析 |
| `check_anomaly()` | — | List[str] | 返回告警消息列表 |

#### ContributionReport 输出

```
ContributionReport
    period: str                       分析区间
    pipeline_stats: Dict[name, PipelineStat]

PipelineStat
    name: str                         管线名称
    avg_weight: float                 平均权重
    avg_ic: float                     平均 IC
    marginal_ic: float                边际贡献 IC（去掉该管线后融合 IC 的下降量）
    weight_trend: str                 "increasing" | "stable" | "decreasing"
```

#### check_anomaly 规则

```
告警条件（返回对应告警消息）:

1. 融合信号 IC 连续 10 天 < 0
   → "WARN: Combined signal IC negative for 10 consecutive days"

2. 某管线权重连续 20 天 = min_weight
   → "WARN: Pipeline {name} at minimum weight for 20 days, consider disabling"

3. 所有管线 IC 同时下降
   → "WARN: All pipelines IC declining, possible regime change"

4. 融合信号与单一管线相关 > 0.95
   → "WARN: Combined signal dominated by {name}, diversification lost"
```

---

### 子模块四：SignalEnsemblePipeline（融合入口）

#### 职责

日常回测/实盘入口。接收三条管线信号，输出融合后的选股信号。

#### 接口

| 方法 | 输入 | 输出 | 说明 |
|------|------|------|------|
| `__init__(ic_lookback, uncertainty_penalty, min_weight, max_weight)` | 配置参数 | — | |
| `combine(signals, anchor_date)` | 信号字典, 锚定日 | EnsembleOutput | 融合输出 |
| `update_history(date, signals, actual_return)` | 日期, 信号, 实际收益 | — | 追加历史（评估前一日信号质量） |
| `get_weights()` | — | Dict[name, float] | 当前各管线权重 |
| `get_monitor()` | — | EnsembleMonitor | 获取监控器实例 |

#### combine 输入格式

```
signals = {
    "alpha":              Series[symbol→float],   # 管线A 信号
    "kronos":             Series[symbol→float],   # 管线B return_1d
    "kronos_uncertainty": Series[symbol→float],   # 管线B uncertainty
    "rdagent":            Series[symbol→float],   # 管线C（可为 None）
}
```

#### EnsembleOutput 输出

```
EnsembleOutput
    signal: Series[symbol→float]         最终融合信号
    weights: Dict[name, float]           本次使用的权重
    pipeline_ranks: Dict[name, Series]   各管线归一化后的 rank 信号（供诊断）
    uncertainty_adjusted: bool           是否应用了不确定性惩罚
    mode: str                            "ic_weighted" | "cold_start" | "blending"
```

#### combine 内部流程

```
1. 过滤空管线
   · 移除 None 值的管线（如 rdagent 无 active 因子时）
   · 记录实际参与融合的管线列表

2. 预处理
   · alpha: winsorize(1%, 99%) → rank_normalize
   · kronos: rank_normalize（不缩尾）
   · rdagent: winsorize(1%, 99%) → rank_normalize

3. 对齐股票 index
   · normalizer.align_index(signals)
   · 某管线缺少某股票 → 该股票在该管线的 rank 填 0.5（中性值）

4. 计算权重
   · weights = ic_engine.compute_weights()
   · 记录 mode: "cold_start" / "blending" / "ic_weighted"

5. 加权融合
   · raw_score = Σ(weight_i × rank_i) for each stock

6. 不确定性惩罚
   · 若 kronos_uncertainty 非空:
     - unc_rank = rank_normalize(kronos_uncertainty)
     - penalty = unc_rank × uncertainty_penalty（默认 0.1）
     - raw_score -= penalty
   · 目的: 对 Kronos 多采样方差大的票降权

7. 最终归一化
   · signal = rank_normalize(raw_score)
   · 保证输出在 (0, 1] 区间，下游 M6 可直接排序选股

8. 记录到 monitor
   · monitor.record(anchor_date, weights, signal, None)  # actual_return 在下一日补填

9. 返回 EnsembleOutput
```

#### 不确定性惩罚的设计考量

```
为什么不直接降低 kronos 权重，而是用 uncertainty 惩罚？

· 权重是管线级别的，uncertainty 是个股级别的
· Kronos 对某些股票预测置信度高、对另一些低
· 高置信度的预测应保持 kronos 权重
· 低置信度的预测应降权，但不影响其他股票

惩罚幅度 0.1 的含义:
  · uncertainty rank 最高的股票（最不确定），融合得分下调 0.1
  · uncertainty rank 最低的股票（最确定），几乎不影响
  · 0.1 约等于 rank 信号标准差的 1/3，是温和的调整
```

---

### M5 整体使用流程

```
场景1: 日常回测/实盘
  ensemble = SignalEnsemblePipeline(
      ic_lookback=60, uncertainty_penalty=0.1,
      min_weight=0.1, max_weight=0.6
  )
  output = ensemble.combine(signals, today)
  # output.signal → 传给 M6 选股
  # output.weights → 记录到日志

场景2: 冷启动阶段（回测前20天）
  output = ensemble.combine(signals, day_5)
  print(output.mode)      # "cold_start"
  print(output.weights)   # {"alpha": 0.333, "kronos": 0.333, "rdagent": 0.333}

场景3: IC 加权生效后
  # 假设 alpha IC=0.05, kronos IC=0.03, rdagent IC=0.01
  output = ensemble.combine(signals, day_80)
  print(output.mode)      # "ic_weighted"
  print(output.weights)   # {"alpha": 0.50, "kronos": 0.33, "rdagent": 0.17}
  # 受 min_weight/max_weight 钳制

场景4: 管线C 缺失
  signals["rdagent"] = None
  output = ensemble.combine(signals, today)
  print(output.weights)   # {"alpha": 0.55, "kronos": 0.45}  只用两管线

场景5: 监控告警
  alerts = ensemble.get_monitor().check_anomaly()
  for a in alerts:
      print(a)
  # "WARN: Pipeline rdagent at minimum weight for 20 days"

场景6: 贡献度分析
  report = ensemble.get_monitor().get_contribution_report(lookback=60)
  for name, stat in report.pipeline_stats.items():
      print(f"{name}: avg_weight={stat.avg_weight:.2f}, marginal_ic={stat.marginal_ic:.3f}")
  # alpha:   avg_weight=0.45, marginal_ic=0.012
  # kronos:  avg_weight=0.35, marginal_ic=0.008
  # rdagent: avg_weight=0.20, marginal_ic=0.003
```

---

### 测试

#### SignalNormalizer 测试

| 测试项 | 方法 | 验证点 |
|--------|------|--------|
| Rank 归一化 | 输入 [10, 30, 20, NaN] | 输出 [0.33, 1.0, 0.67, NaN] |
| 量纲无关 | 输入 [-100, 0, 100] vs [0.1, 0.2, 0.3] | rank 结果相同 |
| 缩尾 | 输入含极端值 [1,2,3,...,100] winsorize(5%,95%) | 第1值=5, 第100值=95 |
| Index 对齐 | A有[a,b,c], B有[b,c,d] | 对齐后都有[a,b,c,d]，缺失为NaN |

#### ICWeightEngine 测试

| 测试项 | 方法 | 验证点 |
|--------|------|--------|
| 冷启动等权 | 历史 < 20天 | 各管线权重相等（1/N） |
| 混合过渡 | 历史 40 天（ic_lookback=60） | 权重介于等权和 IC 加权之间 |
| IC 加权方向 | 构造 alpha IC=0.05, kronos=0.02, rdagent=-0.01 | alpha 权重最高, rdagent 权重 = min_weight |
| 全负 IC | 所有管线 IC < 0 | 退化为等权 |
| 权重钳制 | 某管线 IC 远高于其他 | 权重不超过 max_weight |
| 相关性惩罚 | 构造 alpha 和 kronos 信号相关 > 0.7 | IC 较低的管线权重被下调 |
| 无前视偏差 | T 日 compute_weights | 仅使用 T-1 及之前的 actual_return |
| 历史定长 | 调用 update 200次 | history 长度 = ic_lookback（旧记录淘汰） |

#### EnsembleMonitor 测试

| 测试项 | 方法 | 验证点 |
|--------|------|--------|
| 贡献度 | record 60 天后 get_contribution_report | marginal_ic 合理（去掉管线后 IC 下降） |
| IC 告警 | 融合 IC 连续 10 天 < 0 | check_anomaly 返回对应告警 |
| 权重集中告警 | 融合信号与单管线相关 > 0.95 | check_anomaly 返回 "dominated" 告警 |

#### SignalEnsemblePipeline 测试

| 测试项 | 方法 | 验证点 |
|--------|------|--------|
| 端到端 | 三管线信号 → combine | 返回 EnsembleOutput, signal 为有效 Series |
| 不确定性惩罚 | uncertainty 全为极大值 | 融合得分显著低于不带惩罚版本 |
| 管线缺失 | rdagent=None | 两管线融合，权重和=1.0 |
| 缺失股票填充 | alpha 含股票A, kronos 不含 | 股票A 在 kronos rank 中为 0.5 |
| 输出值域 | 检查 signal | 值在 (0, 1]，无 NaN |
| 确定性 | 相同输入两次 combine | 结果完全一致 |

---

## M6 交易执行（execution.py）

### 职责

三项核心职责：

1. **订单生成** — 根据融合信号 + 当前持仓决定买卖标的、数量和目标价
2. **成交判定** — 以 T+1 开盘价模拟真实成交，处理涨跌停、价格偏离、资金不足
3. **账户管理** — 维护现金、持仓、交易记录，确保资金守恒

### 设计思路

```
A股交易约束清单（必须全部体现在执行逻辑中）:

成本项:
  · 佣金: 买卖双向收取，万分之 2.5，最低 5 元
  · 印花税: 仅卖出收取，万分之 5
  · 滑点: 模拟冲击成本，千分之 1

交易限制:
  · T+1: 当日买入的股票次日才能卖出
  · 涨停板: 开盘一字涨停 → 买不到
  · 跌停板: 开盘一字跌停 → 卖不掉
  · 最小交易单位: 100 股（1 手）
  · 资金分配: 卖出回款同日可用于买入

仓位规则:
  · 单票仓位上限: max_single_weight × 总资产
  · 最大持仓数: max_positions
  · 行业集中度: M7 约束，此处配合执行
```

### 子模块划分

```
execution.py
├── PortfolioAccount         账户与持仓管理
├── OrderGenerator           订单生成器
├── OrderExecutor            成交判定引擎
└── ExecutionPipeline        执行入口
```

---

### 子模块一：PortfolioAccount（账户管理）

#### 数据结构

```
Position
    symbol: str
    shares: int                     持仓股数
    cost_price: float               买入均价（含佣金摊入）
    entry_date: str                 建仓日期
    highest_price: float            持仓期间最高价（M7 止损用）
    last_buy_date: str              最近一次买入日期（T+1 限制用）

常量
    COMMISSION_RATE = 0.00025       佣金费率（万 2.5）
    STAMP_TAX_RATE  = 0.0005        印花税率（万 5）
    MIN_COMMISSION  = 5.0           最低佣金（元）
    SLIPPAGE_RATE   = 0.001         滑点（千 1）
```

#### 接口

| 方法 | 输入 | 输出 | 说明 |
|------|------|------|------|
| `__init__(initial_cash)` | 初始资金 | — | |
| `get_cash()` | — | float | 当前可用现金 |
| `get_holdings()` | — | Dict[symbol, Position] | 当前持仓 |
| `get_portfolio_value(current_prices)` | 当前价格 | float | 现金 + 持仓市值 |
| `get_holding_weight(symbol, current_prices)` | 股票, 价格 | float | 该股票占总资产比例 |
| `update_highest_price(current_prices)` | 当前价格 | — | 更新所有持仓的 highest_price |
| `apply_buy(symbol, shares, price, cost, date)` | 成交信息 | — | 扣现金、更新持仓 |
| `apply_sell(symbol, shares, price, proceeds, date)` | 成交信息 | — | 加现金、减持仓 |
| `can_sell(symbol, exec_date)` | 股票, 执行日 | bool | 检查 T+1 限制（last_buy_date < exec_date） |
| `get_trade_history()` | — | List[TradeRecord] | 全部历史交易记录 |

#### apply_buy 细节

```
若已持有 symbol:
  · 加权平均更新 cost_price
  · shares 累加
  · last_buy_date 更新
若未持有:
  · 新建 Position
```

---

### 子模块二：OrderGenerator（订单生成器）

#### 数据结构

```
TradeOrder
    symbol: str
    direction: "buy" | "sell"
    target_price: float              买入上限 / 卖出下限
    shares: int                      目标股数
    signal_score: float              融合信号得分
    reason: str                      "signal" | "stop_loss" | "industry_limit" | "circuit_breaker"
    priority: int                    执行优先级（低值优先）
                                      1: circuit_breaker 清仓
                                      2: stop_loss 止损
                                      3: industry_limit 行业减仓
                                      4: signal 信号卖出
                                      5: signal 信号买入
```

#### 接口

| 方法 | 输入 | 输出 | 说明 |
|------|------|------|------|
| `__init__(max_positions, max_single_weight, target_buy_count, target_sell_count)` | 配置 | — | |
| `generate_sell_orders(signal, holdings, current_prices, exec_date)` | 信号, 持仓, 价格, 执行日 | List[TradeOrder] | 信号驱动的卖出订单 |
| `generate_buy_orders(signal, holdings, current_prices, industry_map, total_value, available_cash)` | 信号等 | List[TradeOrder] | 信号驱动的买入订单 |
| `generate_force_sell_orders(symbols, current_prices, holdings, reason)` | 强卖列表, 价格, 持仓, 原因 | List[TradeOrder] | 风控强卖订单 |
| `generate_liquidation_orders(holdings, current_prices)` | 持仓, 价格 | List[TradeOrder] | 熔断全部清仓订单 |

#### generate_sell_orders 逻辑

```
1. 遍历持仓，筛选可卖标的
   · can_sell(symbol, exec_date) = True（T+1 限制）

2. 卖出候选判定
   · 持仓股票的 signal_score < 全市场中位数 → 信号衰弱，加入候选
   · 持仓天数 > 60 天且 signal_score < 全市场 60 分位 → 长期低迷，加入候选

3. 按 signal_score 升序排列（最差的优先卖）

4. 取前 target_sell_count 只

5. 生成订单:
   · shares = 全部持仓（全仓卖出该票）
   · target_price = 收盘价 × 0.95（跌停附近放弃）
   · priority = 4
```

#### generate_buy_orders 逻辑

```
1. 候选筛选
   · 取融合得分最高的非持仓股
   · 排除已持仓 + 当日卖出的票
   · 排除行业已满的票（每行业 ≤ max_industry_count）

2. 行业约束
   · 统计当前各行业持仓数
   · max_industry_count = max(1, max_positions // 5)
   · 某行业已达上限 → 跳过该行业新票

3. 仓位约束
   · 单票分配资金 = min(available_cash / target_buy_count, total_value × max_single_weight)

4. 生成订单（逐只）:
   · target_price = 收盘价 × 1.02（允许 2% 溢价）
   · shares = floor(分配资金 / target_price / 100) × 100
   · 若 shares < 100 → 跳过该票（买不起一手）
   · priority = 5

5. 取前 target_buy_count 只
```

#### 订单优先级意义

```
执行时按 priority 排序:
  1. circuit_breaker: 熔断清仓最优先
  2. stop_loss: 止损卖出
  3. industry_limit: 行业减仓
  4. signal sell: 信号卖出
  5. signal buy: 信号买入最后

卖出优先于买入的原因:
  · 卖出释放现金，供后续买入使用
  · 止损/风控订单不应被资金不足阻塞
  · 保证风控指令一定能执行
```

---

### 子模块三：OrderExecutor（成交判定引擎）

#### 数据结构

```
TradeRecord
    date: str                        成交日期
    symbol: str
    direction: "buy" | "sell"
    status: str                      见下方状态枚举
    order_price: float               目标价
    exec_price: float                实际成交价（含滑点）
    shares: int                      成交股数
    amount: float                    成交金额
    commission: float                佣金
    stamp_tax: float                 印花税（买入为 0）
    total_cost: float                佣金 + 印花税
    reason: str                      订单来源

成交状态枚举:
    "filled"                         成交
    "failed_limit_up"                涨停买不到
    "failed_limit_down"              跌停卖不掉
    "failed_price_deviation"         开盘价超出目标价
    "failed_no_cash"                 资金不足
    "failed_t1_restriction"          T+1 限制（当日买入不可卖出）
    "partial_filled"                 部分成交（资金仅够买部分）
```

#### 接口

| 方法 | 输入 | 输出 | 说明 |
|------|------|------|------|
| `execute(orders, open_prices, limit_up, limit_down, exec_date, account)` | 订单列表, 开盘价, 涨跌停价, 日期, 账户 | List[TradeRecord] | 执行全部订单 |

#### execute 内部流程

```
1. 按 priority 排序（低值优先）

2. 逐个订单判定:

   卖出订单:
     a. T+1 检查: account.can_sell(symbol, exec_date)
        · 不可卖 → status="failed_t1_restriction"
     b. 跌停检查: open_price ≤ limit_down
        · 跌停 → status="failed_limit_down"
     c. 价格检查: open_price < target_price（低于卖出下限）
        · 信号卖出 (reason="signal") → 仍然成交（优先清仓）
        · 风控卖出 → 仍然成交（风控优先）
     d. 成交:
        · exec_price = open_price × (1 - SLIPPAGE_RATE)
        · amount = exec_price × shares
        · commission = max(amount × COMMISSION_RATE, MIN_COMMISSION)
        · stamp_tax = amount × STAMP_TAX_RATE
        · proceeds = amount - commission - stamp_tax
        · account.apply_sell(...)

   买入订单:
     a. 涨停检查: open_price ≥ limit_up
        · 涨停 → status="failed_limit_up"
     b. 价格检查: open_price > target_price
        · 超出目标价 → status="failed_price_deviation"
     c. 资金检查:
        · exec_price = open_price × (1 + SLIPPAGE_RATE)
        · needed = exec_price × shares + max(exec_price × shares × COMMISSION_RATE, MIN_COMMISSION)
        · if needed > account.get_cash():
            - 尝试减少股数: shares = floor(cash / exec_price / 100) × 100
            - shares < 100 → status="failed_no_cash"
            - 否则 → status="partial_filled"
     d. 成交:
        · amount = exec_price × shares
        · commission = max(amount × COMMISSION_RATE, MIN_COMMISSION)
        · total = amount + commission
        · account.apply_buy(...)

3. 返回全部 TradeRecord
```

---

### 子模块四：ExecutionPipeline（执行入口）

#### 接口

| 方法 | 输入 | 输出 | 说明 |
|------|------|------|------|
| `__init__(initial_cash, max_positions, max_single_weight, target_buy_count, target_sell_count)` | 配置 | — | |
| `generate_orders(signal, current_prices, industry_map, anchor_date)` | 融合信号, T日收盘价, 行业, 日期 | List[TradeOrder] | 生成信号驱动的买卖订单 |
| `add_force_sell_orders(orders, symbols, current_prices, reason)` | 现有订单, 强卖列表, 价格, 原因 | List[TradeOrder] | 追加风控强卖订单 |
| `add_liquidation_orders(orders, current_prices)` | 现有订单, 价格 | List[TradeOrder] | 追加熔断清仓订单 |
| `execute_orders(orders, open_prices, limit_up, limit_down, exec_date)` | 订单, 开盘价, 涨跌停, 日期 | List[TradeRecord] | 成交判定 |
| `get_portfolio_value(current_prices)` | 当前价格 | float | 总资产 |
| `get_holdings()` | — | Dict[symbol, Position] | 持仓快照 |
| `get_daily_summary(current_prices, exec_date)` | 价格, 日期 | DailySummary | 当日账户摘要 |

#### DailySummary 输出

```
DailySummary
    date: str
    total_value: float               总资产
    cash: float                      现金
    market_value: float              持仓市值
    position_count: int              持仓股票数
    daily_pnl: float                 当日盈亏
    daily_return: float              当日收益率
    buy_count: int                   当日买入成交笔数
    sell_count: int                  当日卖出成交笔数
    total_commission: float          当日佣金
    total_stamp_tax: float           当日印花税
```

---

### M6 整体使用流程

```
场景1: 日常交易（M9 主循环中调用）
  # T 日收盘后
  orders = execution.generate_orders(signal, close_prices, industry_map, T)
  # 追加风控强卖
  execution.add_force_sell_orders(orders, stop_loss_list, close_prices, "stop_loss")
  # T+1 执行
  records = execution.execute_orders(orders, open_prices, limit_up, limit_down, T_next)

场景2: 熔断清仓
  execution.add_liquidation_orders(orders, close_prices)
  records = execution.execute_orders(orders, open_prices, limit_up, limit_down, T_next)
  # 全部持仓生成 priority=1 的卖出订单

场景3: 检查账户状态
  summary = execution.get_daily_summary(current_prices, today)
  print(f"总资产: {summary.total_value:.0f}, 持仓: {summary.position_count}, "
        f"当日PnL: {summary.daily_pnl:+.0f}")
```

---

### 测试

#### PortfolioAccount 测试

| 测试项 | 方法 | 验证点 |
|--------|------|--------|
| 初始状态 | 初始资金 100万 | cash=100万, holdings 为空 |
| 买入 | apply_buy 10元×1000股 | cash 减少, holdings 新增 |
| 卖出 | apply_sell 已有持仓 | cash 增加, 持仓消失 |
| 加仓均价 | 10元买500股 → 12元买500股 | cost_price = 11元 |
| T+1 限制 | 今日买入, 同日 can_sell | 返回 False |
| T+1 次日 | 今日买入, 次日 can_sell | 返回 True |
| 资金守恒 | 多笔买卖后 | cash + 持仓市值 + 累计成本 = 初始资金 ± 盈亏 |

#### OrderGenerator 测试

| 测试项 | 方法 | 验证点 |
|--------|------|--------|
| 卖出候选 | 持仓 signal_score < 中位数 | 出现在卖出订单中 |
| 买入行业约束 | 某行业已满 | 该行业新票不在买入订单中 |
| 买入仓位约束 | max_single_weight=0.1, 总资产100万 | 单票最多 10万 |
| 100股整数倍 | 可用资金只够 150 股 | 订单为 100 股 |
| 不足一手 | 可用资金只够 80 股 | 该票被跳过 |
| 优先级 | 同时有 stop_loss 和 signal 卖出 | stop_loss priority < signal |
| 清仓订单 | generate_liquidation_orders | 每个持仓一个 priority=1 卖出订单 |

#### OrderExecutor 测试

| 测试项 | 方法 | 验证点 |
|--------|------|--------|
| 涨停不买 | open_price = limit_up | status="failed_limit_up" |
| 跌停不卖 | open_price = limit_down | status="failed_limit_down" |
| 价格偏离 | 开盘价 > 目标价 | status="failed_price_deviation" |
| 资金不足 | 余额只够 200 股，订单 500 股 | status="partial_filled", shares=200 |
| 完全不够 | 余额不够 100 股 | status="failed_no_cash" |
| 买入成本 | 10元×1000股 | 佣金 = max(10000×0.00025, 5) = 5元 |
| 卖出成本 | 10元×10000股 | 佣金25元 + 印花税50元 = 75元 |
| 印花税仅卖出 | 买入记录 | stamp_tax = 0 |
| 卖出先执行 | 卖A + 买B | 卖A回款后用于买B |
| 空订单 | execute([]) | 返回空列表 |
| 滑点 | 买入开盘10元 | exec_price = 10.01 |

#### ExecutionPipeline 测试

| 测试项 | 方法 | 验证点 |
|--------|------|--------|
| 端到端 | 信号 → 生成订单 → 执行 | 持仓和现金更新正确 |
| 多日连续 | 连续执行5个交易日 | daily_summary 每日更新, total_value 连续 |
| 风控+信号混合 | stop_loss 和 signal 同时存在 | stop_loss 先执行 |
| 全空仓 | 信号全为0 | 不生成买入订单 |

---

## M7 风控（risk_control/risk_control.py）

### 职责

三项核心职责：

1. **个股风控** — 单票止损（最高价回撤）+ 持仓时限管理
2. **结构风控** — 行业暴露控制 + 个股集中度限制
3. **组合风控** — 净值回撤熔断 + 暂停交易 + 恢复策略

### 设计思路

```
三级递进式风控，从微观到宏观:

Level 1 — 个股级（StopLossChecker）:
  · 止损: 持仓从最高价回撤 ≥ 8% → 强制卖出（force_sell）
  · 过期: 持仓超过 max_hold_days 且亏损 → 建议卖出（suggest_sell）
  · 盈利票不限时: 超期但盈利 → 不触发

Level 2 — 结构级（ExposureChecker）:
  · 行业暴露: 单一行业持仓市值 ≤ 总资产 30%
  · 个股集中: 单票市值 ≤ 总资产 20%（与 M6 的 max_single_weight 配合）
  · 超限减持: 从行业内市值最小的票开始减仓

Level 3 — 组合级（CircuitBreaker）:
  · 熔断: 净值从高水位回撤 ≥ 10% → 全部清仓
  · 暂停: 熔断后暂停交易 pause_days 个交易日（默认 5）
  · 恢复: 暂停期满后 recovery_days 个交易日内半仓运作（默认 3 天）
  · 防抖: 恢复期内不重复触发熔断，避免连续熔断陷阱
```

### 与 M6 的协作关系

```
M7 daily_check() 产出 RiskCheckResult
    │
    ├── circuit_breaker_triggered = True
    │     → M6.add_liquidation_orders(): 清除所有买入, 全持仓生成 priority=1 卖单
    │
    ├── force_sell_symbols = ["A", "B"]
    │     → M6.add_force_sell_orders(): 生成 priority=2 (stop_loss) / 3 (industry) 卖单
    │     → 与信号卖单去重: 同 symbol 取更高优先级
    │
    ├── position_limit = 0.0 / 0.5 / 1.0
    │     → M6.generate_buy_orders(): 按比例缩减买入规模
    │
    └── events → 汇入 M8 评估归因
```

**执行优先级（低值优先）：**

| priority | 来源 | 说明 |
|----------|------|------|
| 1 | circuit_breaker 清仓 | 最高优先，系统性风险 |
| 2 | stop_loss 止损 | 个股黑天鹅 |
| 3 | industry_limit 行业减仓 | 结构调整 |
| 4 | signal 信号卖出 | 正常轮换 |
| 5 | signal 信号买入 | 最后执行 |

### 子模块划分

```
quantlab/risk_control/
├── __init__.py              导出全部公开类
└── risk_control.py          核心实现
    ├── StopLossChecker      个股止损检查
    ├── ExposureChecker      结构暴露检查
    ├── CircuitBreaker       组合熔断控制
    ├── RiskEventLog         风控事件记录
    └── RiskController       风控入口（daily_check）
```

### 配置参数

来自 `quantlab/configs/backtest.yaml`：

| 参数 | 默认值 | 说明 | 对应构造参数 |
|------|--------|------|-------------|
| `stop_loss_pct` | 0.08 | 个股止损回撤阈值（8%） | StopLossChecker |
| `max_hold_days` | 60 | 亏损票最大持仓天数 | StopLossChecker |
| `max_industry_pct` | 0.30 | 单一行业持仓上限（30%） | ExposureChecker |
| `max_single_pct` | 0.20 | 单票持仓上限（20%） | ExposureChecker |
| `circuit_breaker_pct` | 0.10 | 组合熔断回撤阈值（10%） | CircuitBreaker |
| `pause_days` | 5 | 熔断暂停交易日数 | CircuitBreaker |
| `recovery_position_limit` | 0.5 | 恢复期仓位系数（50%） | CircuitBreaker |
| `recovery_days` | 3 | 恢复期交易日数 | CircuitBreaker |

---

### 子模块一：StopLossChecker（个股止损）

#### 接口

| 方法 | 输入 | 输出 | 说明 |
|------|------|------|------|
| `__init__(stop_loss_pct, max_hold_days)` | 止损比例, 最大持仓天数 | — | 默认 0.08, 60 |
| `check(positions, current_prices, current_date)` | Dict[str, Position], Dict[str, float], str | List[StopLossAction] | 逐票检查 |

#### StopLossAction 输出

```
StopLossAction
    symbol: str
    action: str                "force_sell" | "suggest_sell"
    reason: str                "drawdown_10.0%" | "hold_expired_loss"
    drawdown: float            当前回撤百分比（0~1）
    highest_price: float       持仓期间最高价
    current_price: float       当前价
    hold_days: int             自然日持仓天数
```

#### check 逻辑

```
对每个持仓:
  0. 防御: price 为 None 或 ≤ 0 → 跳过
     highest_price ≤ 0 → 退化为 cost_price

  1. 计算回撤:
     drawdown = (highest_price - current_price) / highest_price

  2. 计算持仓天数:
     hold_days = (current_date - entry_date).days    # 自然日

  3. 止损判定（优先级高于过期判定，用 continue 跳过）:
     · drawdown ≥ stop_loss_pct
       → action="force_sell", reason=f"drawdown_{drawdown*100:.1f}%"

  4. 持仓过期判定:
     · hold_days > max_hold_days 且 current_price < cost_price
       → action="suggest_sell", reason="hold_expired_loss"
     · hold_days > max_hold_days 且 current_price ≥ cost_price
       → 不触发（盈利票不限时）

注意: highest_price 的更新由 M6 的 PortfolioAccount.update_highest_price() 负责，
      M7 只读取 Position.highest_price 字段。
```

---

### 子模块二：ExposureChecker（结构暴露检查）

#### 接口

| 方法 | 输入 | 输出 | 说明 |
|------|------|------|------|
| `__init__(max_industry_pct, max_single_pct)` | 行业上限, 单票上限 | — | 默认 0.30, 0.20 |
| `check_industry(positions, current_prices, industry_map, total_value)` | 持仓, 价格, 行业映射, 总资产 | List[ExposureAction] | 行业暴露检查 |
| `check_concentration(positions, current_prices, total_value)` | 持仓, 价格, 总资产 | List[ExposureAction] | 个股集中度检查 |

#### ExposureAction 输出

```
ExposureAction
    symbol: str
    action: str = "reduce"
    reason: str                "industry_over_30%: 银行 35.2%" | "single_over_20%: 22.1%"
    current_pct: float         当前占比
    limit_pct: float           上限
    excess_value: float        需减持金额
```

#### check_industry 逻辑

```
前置: total_value ≤ 0 或 positions 为空 → 返回 []

1. 按行业汇总:
   industry_values = {行业: [(symbol, market_value), ...]}
   · market_value = shares × current_price
   · 未知行业归入 "unknown"

2. 逐行业检查:
   for ind, holdings in industry_values:
     ind_pct = Σmarket_value / total_value
     if ind_pct ≤ max_industry_pct: continue

     excess = Σmarket_value - total_value × max_industry_pct

3. 减持策略（从小到大）:
   · 该行业内按市值升序排列
   · 逐只减持，每只减持额 = min(该票市值, 剩余 excess)
   · 直到累计减持 ≥ excess

设计选择: 优先减最小仓位，对组合冲击最小，避免砍掉核心重仓。
```

#### check_concentration 逻辑

```
前置: total_value ≤ 0 或 positions 为空 → 返回 []

逐票检查:
  pct = shares × price / total_value
  if pct > max_single_pct:
    excess = market_value - total_value × max_single_pct
    → 生成 ExposureAction
```

---

### 子模块三：CircuitBreaker（组合熔断）

#### 接口

| 方法 | 输入 | 输出 | 说明 |
|------|------|------|------|
| `__init__(drawdown_pct, pause_days, recovery_position_limit, recovery_days)` | 阈值, 暂停天数, 恢复仓位比例, 恢复天数 | — | 默认 0.10, 5, 0.5, 3 |
| `update_high_watermark(value)` | 总资产 | — | 更新高水位（暂停期 pause_until 非 None 时跳过） |
| `check(current_value)` | 当前总资产 | bool | 是否触发熔断 |
| `trigger(current_date, calendar)` | 当前日期, 交易日历 | — | 计算暂停和恢复截止日 |
| `is_paused(current_date)` | 日期 | bool | 是否处于暂停期（含触发当日） |
| `is_recovery_mode(current_date)` | 日期 | bool | 是否处于恢复期 |
| `get_position_limit(current_date)` | 日期 | float | 0.0 / 0.5 / 1.0 |

#### 状态

```
CircuitBreaker 内部状态:
    high_watermark: float = 0.0               净值高水位
    pause_until: Optional[str] = None         暂停截止日期（含，交易日）
    recovery_until: Optional[str] = None      恢复期截止日期（含，交易日）
    trigger_count: int = 0                    历史熔断次数
    trigger_history: List[Tuple[str, float]]  (触发日期, 当时高水位)
```

#### 状态机

```
                 check()=True
    Normal ─────────────────→ Triggered
      ↑                          │
      │                     trigger()
      │                          │
      │                          ▼
      │                    ┌──────────┐
      │                    │  Pause   │  pause_days 个交易日
      │                    │  pos=0.0 │  禁止任何交易
      │                    └────┬─────┘
      │                         │ current_date > pause_until
      │                         ▼
      │                    ┌──────────┐
      │                    │ Recovery │  recovery_days 个交易日
      │                    │  pos=0.5 │  半仓限制，不触发熔断
      │                    └────┬─────┘
      │                         │ current_date > recovery_until
      └─────────────────────────┘
          清理 pause_until / recovery_until
```

#### trigger 截止日计算

```
trigger(current_date, calendar):
  1. 在 calendar 中定位 current_date 的 index
     · 若 current_date 不在 calendar 中，查找 ≥ current_date 的最近日
  2. pause_until  = calendar[index + pause_days]
  3. recovery_until = calendar[index + pause_days + recovery_days]
     · 若超出日历范围，取 calendar[-1]

注意: 日期比较使用字符串 "YYYY-MM-DD" 格式的字典序。
```

#### 恢复模式详细说明

```
三阶段完整生命周期:

阶段1 — 暂停期（current_date ≤ pause_until）:
  · position_limit = 0.0
  · daily_check 直接返回，跳过所有风控检查
  · high_watermark 不更新（防止暂停期间被拉高）

阶段2 — 恢复期（pause_until < current_date ≤ recovery_until）:
  · position_limit = recovery_position_limit（默认 0.5）
  · 正常执行三级风控检查
  · 关键: 恢复期内不重复触发熔断（即使净值仍低于高水位 10%）
  · 目的: 避免暂停期满后立即满仓又触发连续熔断

阶段3 — 正常模式（current_date > recovery_until）:
  · position_limit = 1.0
  · 清理 pause_until 和 recovery_until 为 None
  · high_watermark 恢复正常更新
```

#### get_position_limit 逻辑

```
if is_paused(date):
    return 0.0                      暂停期，禁止交易
elif is_recovery_mode(date):
    return recovery_position_limit   恢复期，半仓
else:
    # 恢复期结束后清理状态
    if recovery_until is not None and date > recovery_until:
        pause_until = None
        recovery_until = None
    return 1.0                      正常全仓
```

---

### 子模块四：RiskEventLog（风控事件记录）

#### 接口

| 方法 | 输入 | 输出 | 说明 |
|------|------|------|------|
| `log(date, event_type, symbol?, details?)` | 日期, 类型, 股票(可选), 详情(可选) | RiskEvent | 记录事件并返回 |
| `get_events(event_type?, start_date?, end_date?)` | 筛选条件（均可选） | List[RiskEvent] | 查询事件 |
| `summary()` | — | Dict[str, int] | 按类型统计事件数量 |
| `events` (property) | — | List[RiskEvent] | 全部事件的只读副本 |

#### RiskEvent 数据结构

```
RiskEvent
    date: str
    event_type: str            事件类型（见下表）
    symbol: Optional[str]      关联股票（组合级事件为 None）
    details: Dict              事件详情（默认空 dict）
```

#### 事件类型与 details 规范

| event_type | symbol | details 字段 |
|------------|--------|-------------|
| `stop_loss` | 股票代码 | drawdown, highest, current, hold_days |
| `industry_reduce` | 被减持股票 | current_pct, limit_pct, excess_value |
| `concentration_reduce` | 被减持股票 | current_pct, limit_pct, excess_value |
| `circuit_breaker` | None | high_watermark, current_value, drawdown_pct, pause_until, recovery_until |
| `recovery_start` | None | （空） |

---

### 子模块五：RiskController（风控入口）

#### 接口

| 方法 | 输入 | 输出 | 说明 |
|------|------|------|------|
| `__init__(stop_loss_pct, max_hold_days, max_industry_pct, max_single_pct, circuit_breaker_pct, pause_days, recovery_position_limit, recovery_days)` | 全部风控参数 | — | 默认 0.08, 60, 0.30, 0.20, 0.10, 5, 0.5, 3 |
| `daily_check(positions, current_prices, industry_map, total_value, current_date, calendar)` | 持仓, 价格, 行业, 总资产, 日期, 日历 | RiskCheckResult | 全量风控检查 |
| `is_paused(current_date)` | 日期 | bool | 委托 CircuitBreaker |
| `get_position_limit(current_date)` | 日期 | float | 委托 CircuitBreaker |
| `get_event_log()` | — | RiskEventLog | 获取事件记录 |

#### RiskCheckResult 输出

```
RiskCheckResult
    force_sell_symbols: List[str]      需强制卖出的股票（去重）
    suggest_sell_symbols: List[str]    建议卖出的股票（仅供参考，不生成订单）
    force_sell_reasons: Dict[str, str] 每只股票的卖出原因
    circuit_breaker_triggered: bool    是否触发熔断
    position_limit: float              当前仓位限制比例（0.0 / 0.5 / 1.0）
    events: List[RiskEvent]            本次检查产生的事件
```

#### daily_check 流程

```
1. 更新高水位
   · circuit_breaker.update_high_watermark(total_value)
   · 暂停期 pause_until 非 None 时跳过

2. 暂停期快速返回
   · if circuit_breaker.is_paused(current_date):
       return RiskCheckResult(position_limit=0.0)

3. 恢复期事件
   · if circuit_breaker.is_recovery_mode(current_date):
       首次进入恢复期时记录 "recovery_start" 事件

4. Level 1 — 个股止损
   · actions = stop_loss_checker.check(positions, prices, current_date)
   · action="force_sell" → 加入 force_sell + 记录 "stop_loss" 事件
   · action="suggest_sell" → 加入 suggest_sell（不记录事件）

5. Level 2 — 结构控制
   · industry_actions = exposure_checker.check_industry(...)
     → 加入 force_sell + 记录 "industry_reduce" 事件
   · conc_actions = exposure_checker.check_concentration(...)
     → 加入 force_sell + 记录 "concentration_reduce" 事件
   · force_sell 去重: 同一 symbol 只加入一次

6. Level 3 — 组合熔断
   · 恢复期内跳过熔断检查（防抖机制）
   · if not in_recovery and circuit_breaker.check(total_value):
       circuit_breaker.trigger(current_date, calendar)
       circuit_breaker_triggered = True
       记录 "circuit_breaker" 事件

7. 获取仓位限制
   · position_limit = circuit_breaker.get_position_limit(current_date)

8. 返回 RiskCheckResult
   · force_sell_symbols 用 set 去重后转 list
```

---

### M7 整体使用流程

```
场景1: 日常风控（M9 主循环中调用）

  # 初始化（回测开始时创建一次）
  risk = RiskController(
      stop_loss_pct=0.08, max_hold_days=60,
      max_industry_pct=0.30, max_single_pct=0.20,
      circuit_breaker_pct=0.10, pause_days=5,
      recovery_position_limit=0.5, recovery_days=3,
  )

  # T 日收盘后
  result = risk.daily_check(positions, prices, industry_map, total_value, T, calendar)

  # 根据结果调整 M6 订单
  if result.circuit_breaker_triggered:
      orders = execution.add_liquidation_orders(orders, prices)
  elif result.force_sell_symbols:
      for sym in result.force_sell_symbols:
          reason = result.force_sell_reasons.get(sym, "stop_loss")
          orders = execution.add_force_sell_orders(orders, [sym], prices, reason)


场景2: 熔断完整生命周期

  Day 0: 触发熔断（NAV 从高水位跌 12%）
    → circuit_breaker_triggered = True
    → M6 生成全清仓订单（priority=1）
    → position_limit = 0.0（但触发当日已生成清仓单）

  Day 1~5: 暂停期
    → daily_check 快速返回 position_limit=0.0
    → M9 跳过订单生成和执行

  Day 6~8: 恢复期
    → position_limit = 0.5
    → M6 买入规模减半
    → 期间不触发二次熔断

  Day 9+: 正常恢复
    → position_limit = 1.0
    → 清理熔断状态


场景3: 事件回顾（供 M8 使用）

  log = risk.get_event_log()
  stops = log.get_events("stop_loss", "2024-01-01", "2024-12-31")
  print(f"全年止损 {len(stops)} 次")
  print(log.summary())
  # {"stop_loss": 23, "industry_reduce": 5, "concentration_reduce": 2,
  #  "circuit_breaker": 1, "recovery_start": 1}
```

---

### 实现细节与边界处理

#### 数据防御

```
· current_prices 中缺少某股票 → 跳过该票的止损检查
· highest_price ≤ 0 → 退化使用 cost_price
· cost_price 也 ≤ 0 → 跳过该票
· total_value ≤ 0 → ExposureChecker 返回空列表
· positions 为空 → 所有 checker 返回空列表
· current_date 不在 calendar 中 → 查找最近的未来交易日
```

#### 数据依赖关系

```
M7 只读取 M6 的 Position 数据，不修改:
  · Position.highest_price — M6 在 execute_orders() 中调用 update_highest_price() 更新
  · Position.entry_date — 建仓日期，M6 在 apply_buy() 中设置
  · Position.cost_price — 加权平均成本，M6 在 apply_buy() 中更新

M7 的输出通过 M6 的接口执行:
  · force_sell_symbols → M6.add_force_sell_orders()
  · circuit_breaker_triggered → M6.add_liquidation_orders()
  · position_limit → M6 在 generate_buy_orders() 中按比例缩减

M7 的事件流向 M8:
  · RiskEventLog.events → M8 评估风控有效性
```

---

### 测试

#### StopLossChecker 测试

| 测试项 | 方法 | 验证点 |
|--------|------|--------|
| 阈值内不触发 | 最高价10元，当前9.5元（回撤5%） | 返回空列表 |
| 回撤触发强卖 | 最高价10元，当前9.0元（回撤10%） | action="force_sell", drawdown≈0.10 |
| 精确阈值触发 | 最高价100元，当前92元（回撤8%） | action="force_sell" |
| 过期亏损建议卖 | 持仓65天，当前价<成本价 | action="suggest_sell" |
| 过期盈利不触发 | 持仓65天，当前价>成本价 | 返回空列表 |
| 多票批量检查 | A回撤10%, B回撤5%, C回撤13% | A和C触发, B不触发 |
| 缺失价格跳过 | current_prices 为空 | 返回空列表 |

#### ExposureChecker 测试

| 测试项 | 方法 | 验证点 |
|--------|------|--------|
| 行业未超限 | 两行业各50%占比 ≤ 总资产30% | 返回空列表 |
| 行业超限减持 | 银行业80%（2只票） | 返回银行业中市值较小的票 |
| 个股集中未超限 | 单票占比10% | 返回空列表 |
| 个股集中超限 | 单票占比30% > 20% | 返回该票, excess=超出金额 |
| 空持仓安全 | positions={} | 返回空列表 |

#### CircuitBreaker 测试

| 测试项 | 方法 | 验证点 |
|--------|------|--------|
| 阈值内不触发 | HWM=100万, 当前95万 | check=False |
| 阈值处触发 | HWM=100万, 当前90万 | check=True |
| 超阈值触发 | HWM=100万, 当前85万 | check=True |
| 暂停+恢复生命周期 | trigger后逐日检查 | 暂停→恢复→正常 |
| 仓位限制数值 | 各阶段 get_position_limit | 0.0→0.5→1.0 |
| 暂停期HWM冻结 | 暂停期 update_high_watermark | 高水位不变 |
| 熔断计数 | 两次 trigger | trigger_count=2, history长度=2 |

#### RiskEventLog 测试

| 测试项 | 方法 | 验证点 |
|--------|------|--------|
| 记录与查询 | log 3条, 按类型/日期筛选 | 数量正确 |
| 统计摘要 | 多类型事件 | summary 各类型计数正确 |

#### RiskController 集成测试

| 测试项 | 方法 | 验证点 |
|--------|------|--------|
| 无风险正常 | 正常持仓, 无回撤 | force_sell=[], position_limit=1.0 |
| 止损强卖 | 持仓回撤10% | symbol 在 force_sell 中 |
| 熔断完整流程 | 建HWM → 回撤12% → 暂停 → 恢复 → 正常 | 各阶段 position_limit 正确 |
| 行业减持 | 银行业占比50% | force_sell 含银行股 |
| 集中度减持 | 单票占比30% | force_sell 含该票 |
| 暂停委托 | 熔断后 is_paused | 返回 True |
| 事件记录 | 触发止损后查 event_log | summary 含 stop_loss |
| 级联测试 | 止损+熔断同时 | force_sell 含止损票, circuit_breaker_triggered=True |

---

## M8 评估与诊断（evaluation/evaluation.py）

### 职责

四项核心职责：

1. **绩效计算** — 年化收益、夏普、最大回撤、Calmar、超额收益等标准指标
2. **交易分析** — 成交率、换手率、持仓天数、成本拖累等交易质量指标
3. **信号归因** — 各管线 IC 衰减、边际贡献、信号相关性分析
4. **分环境诊断** — 牛/熊/震荡市分别的表现，找出策略弱项

### 设计思路

```
评估不只是计算几个数字，还需要回答关键问题:

Q1: 策略赚不赚钱？ → PerformanceCalculator.summary()
Q2: 交易执行好不好？ → TradeAnalyzer.summary() + cost_breakdown()
Q3: 哪条管线在贡献？ → SignalAttributor.ic_decay_analysis() + marginal_contribution()
Q4: 什么环境下表现差？ → RegimeAnalyzer.regime_performance()
Q5: 信号还有效吗？ → SignalAttributor.rolling_ic() + PerformanceCalculator.rolling_sharpe()
Q6: 风控起作用了吗？ → RegimeAnalyzer.risk_impact()
```

### 数据依赖

```
M8 是纯读取模块，不修改任何上游状态:

来自 M6:
  · List[TradeRecord] — 全部交易记录（含 filled / failed）
  · pd.Series (daily_nav) — 日净值序列（date→total_value）

来自 M7:
  · List[RiskEvent] — 风控事件列表（止损/熔断等）

来自 M5:
  · signal_history: {pipeline_name: [(date, pd.Series), ...]}
  · return_history: {date: pd.Series[symbol→return]}

来自 M1:
  · pd.Series (benchmark_nav) — 基准日净值

常量:
  · TRADING_DAYS_PER_YEAR = 252

外部依赖:
  · scipy.stats.spearmanr — Rank IC 计算
```

### 子模块划分

```
quantlab/evaluation/
├── __init__.py              导出全部公开类
└── evaluation.py            核心实现
    ├── PerformanceCalculator    绩效指标计算
    ├── TradeAnalyzer            交易统计分析
    ├── SignalAttributor         信号归因分析
    ├── RegimeAnalyzer           分环境归因 + 风控有效性
    └── EvaluationPipeline       评估入口
```

---

### 子模块一：PerformanceCalculator（绩效指标）

#### 接口

| 方法 | 输入 | 输出 | 说明 |
|------|------|------|------|
| `__init__(daily_nav, benchmark_nav?, risk_free_rate?)` | pd.Series, pd.Series(可选), float(默认0.02) | — | 构造时自动计算日收益序列 |
| `summary()` | — | PerformanceSummary | 核心绩效指标 |
| `monthly_returns()` | — | DataFrame | 月度收益矩阵 |
| `drawdown_series()` | — | pd.Series | 逐日回撤序列（≤ 0） |
| `rolling_sharpe(window=60)` | 窗口天数 | pd.Series | 滚动夏普比率 |

#### PerformanceSummary 输出

```
PerformanceSummary
    # === 收益 ===
    total_return: float = 0.0              总收益率
    annualized_return: float = 0.0         年化收益率
    excess_annual_return: float = 0.0      超额年化收益（相对基准）

    # === 风险 ===
    annualized_volatility: float = 0.0     年化波动率
    max_drawdown: float = 0.0              最大回撤
    max_drawdown_duration: int = 0         最大回撤持续天数
    downside_volatility: float = 0.0       下行波动率

    # === 风险调整 ===
    sharpe_ratio: float = 0.0              夏普比率
    sortino_ratio: float = 0.0             索提诺比率
    calmar_ratio: float = 0.0              Calmar 比率
    information_ratio: float = 0.0         信息比率（超额收益/跟踪误差）

    # === 胜率 ===
    win_rate_daily: float = 0.0            日胜率
    profit_loss_ratio: float = 0.0         盈亏比（平均盈利/平均亏损）

    # === 成本（由 EvaluationPipeline 从 CostBreakdown 回填）===
    total_transaction_cost: float = 0.0    累计交易成本
    cost_drag_annual: float = 0.0          年化成本拖累
```

#### 计算公式

```
常量: N = 日收益天数, 252 = 年交易日数, rf = risk_free_rate

annualized_return = (nav_end / nav_start) ^ (252 / N) - 1
annualized_volatility = std(daily_returns) × sqrt(252)
downside_volatility = std(daily_returns[ret < 0]) × sqrt(252)
sharpe_ratio = (annualized_return - rf) / annualized_volatility
sortino_ratio = (annualized_return - rf) / downside_volatility
calmar_ratio = annualized_return / max_drawdown
information_ratio = mean(excess_daily) / std(excess_daily) × sqrt(252)
    其中 excess_daily = daily_return - benchmark_daily_return
profit_loss_ratio = mean(positive_returns) / abs(mean(negative_returns))
win_rate_daily = count(ret > 0) / N
```

#### 最大回撤计算

```
_calc_max_drawdown(nav) → (mdd: float, duration: int)

1. peak = nav.cummax()          # 逐日历史最高净值
2. dd = (nav - peak) / peak     # 逐日回撤序列（≤ 0）
3. mdd = abs(min(dd))           # 最大回撤幅度

持续天数:
  · 遍历 dd 序列，统计连续 < 0 的最长段
  · 反映从开始回撤到完全恢复的最长时间
```

#### 边界处理

```
· 空净值（len=0）→ 返回全零 PerformanceSummary
· 单日净值 → total_return=0, 其余为 0
· 零波动（净值恒定）→ sharpe=0, sortino=0, ann_vol=0
· 无基准 → excess_annual_return=0, information_ratio=0
· 日收益全正 → downside_volatility=0, sortino=0
```

#### monthly_returns 输出格式

```
DataFrame:
          1月    2月    3月   ...   12月    全年
2023    2.1%  -1.3%   3.5%  ...   1.8%   15.2%
2024    0.5%   4.2%  -2.1%  ...   2.3%   18.7%

计算: 月度收益 = Π(1 + daily_return) - 1, 按年-月分组
      全年收益 = Π(1 + daily_return) - 1, 按年分组
```

---

### 子模块二：TradeAnalyzer（交易统计）

#### 接口

| 方法 | 输入 | 输出 | 说明 |
|------|------|------|------|
| `__init__(trade_records, daily_nav?)` | List[TradeRecord], pd.Series(可选) | — | daily_nav 用于换手率计算 |
| `summary()` | — | TradeSummary | 交易统计 |
| `cost_breakdown()` | — | CostBreakdown | 成本分解 |
| `holding_analysis()` | — | HoldingAnalysis | 持仓分析 |

#### TradeSummary 输出

```
TradeSummary
    total_trades: int = 0                总交易笔数（含失败）
    buy_trades: int = 0                  买入笔数
    sell_trades: int = 0                 卖出笔数
    filled_rate_buy: float = 0.0         买入成交率（filled + partial_filled）
    filled_rate_sell: float = 0.0        卖出成交率（仅 filled）
    failed_reasons: Dict[str, int]       失败原因统计
                                          {"limit_up": 12, "price_deviation": 8, "no_cash": 3}
    avg_daily_turnover: float = 0.0      日均换手率（日均成交额 / 日均净值）
    avg_holding_days: float = 0.0        平均持仓天数（自然日）
    median_holding_days: float = 0.0     持仓天数中位数
```

#### 失败原因提取

```
status 字段前缀 "failed_" 被剥离:
  "failed_limit_up"       → "limit_up"
  "failed_limit_down"     → "limit_down"
  "failed_price_deviation"→ "price_deviation"
  "failed_no_cash"        → "no_cash"
  "failed_t1_restriction" → "t1_restriction"
```

#### 持仓天数计算

```
_calc_holding_days() → (avg: float, median: float)

基于买卖配对（FIFO 简化版）:
  1. 按日期排序所有 filled 记录
  2. 维护 buy_dates: {symbol: last_buy_date}
  3. 遇到 buy → 记录该 symbol 的买入日期
  4. 遇到 sell → 计算 (sell_date - buy_date).days
  5. 对所有配对取 mean 和 median
```

#### CostBreakdown 输出

```
CostBreakdown
    total_commission: float = 0.0          总佣金
    total_stamp_tax: float = 0.0           总印花税
    total_slippage: float = 0.0            总滑点成本（估算: |exec_price - order_price| × shares）
    total_cost: float = 0.0                总成本 = commission + stamp_tax + slippage
    cost_per_trade: float = 0.0            每笔成交平均成本
    cost_as_pct_of_nav: float = 0.0        成本占初始净值百分比
    annual_cost_drag: float = 0.0          年化成本拖累 = cost_pct × (252 / 交易天数)
```

#### HoldingAnalysis 输出

```
HoldingAnalysis
    avg_position_count: float = 0.0        日均持仓数（预留字段）
    max_position_count: int = 0            最大持仓数（预留字段）
    avg_concentration: float = 0.0         日均 Top1 持仓占比（预留字段）
    industry_distribution: Dict[str, float]  行业分布（预留字段）
    win_rate_per_trade: float = 0.0        单笔交易胜率（卖出价 > 买入价）
    best_trade_pnl: float = 0.0            最佳单笔交易收益率
    worst_trade_pnl: float = 0.0           最差单笔交易收益率
```

**单笔交易胜率计算：**

```
1. 构建买入成本表: buy_prices = {symbol: 最近一次买入 exec_price}
2. 遍历 filled 卖出记录:
   pnl_pct = (sell_exec_price - buy_exec_price) / buy_exec_price
   pnl > 0 → 计入 wins
3. win_rate = wins / total_sell_with_matching_buy
4. best/worst = max/min(pnl_pct)
```

#### 边界处理

```
· 空记录（len=0）→ 返回全零 TradeSummary / CostBreakdown / HoldingAnalysis
· daily_nav 为 None → 换手率 = 0, 年化成本拖累 = 0
· 无配对卖出 → avg_holding_days = 0, win_rate = 0
```

---

### 子模块三：SignalAttributor（信号归因）

#### 接口

| 方法 | 输入 | 输出 | 说明 |
|------|------|------|------|
| `__init__(signal_history, return_history)` | 各管线信号历史, 收益历史 | — | 自动过滤空管线 |
| `ic_decay_analysis(horizons?)` | List[int] (默认 [1,2,3,5,10]) | DataFrame | 各管线 IC 衰减 |
| `marginal_contribution()` | — | DataFrame | 各管线边际贡献 |
| `signal_correlation()` | — | DataFrame | 管线间信号相关矩阵 |
| `rolling_ic(pipeline_name, window=60)` | 管线名, 窗口 | pd.Series | 某管线滚动 IC |

#### signal_history 输入格式

```
signal_history: Dict[str, List[Tuple[str, pd.Series]]]

signal_history = {
    "alpha":    [(date1, Series[symbol→score]), (date2, Series), ...],
    "kronos":   [(date1, Series), (date2, Series), ...],
    "rdagent":  [(date1, Series), (date2, Series), ...],
    "combined": [(date1, Series), (date2, Series), ...],  # 融合信号
}

return_history: Dict[str, pd.Series]

return_history = {
    date1: Series[symbol→daily_return],
    date2: Series[symbol→daily_return],
    ...
}

约定: date 格式为 "YYYY-MM-DD" 字符串
```

#### ic_decay_analysis 计算流程

```
ic_decay_analysis(horizons=[1, 2, 3, 5, 10]) → DataFrame

1. 构建日期有序索引: all_dates = sorted(return_history.keys())

2. 对每条管线、每个 horizon h:
   对每个 (date, signal) in signal_history[pipeline]:
     a. 在 all_dates 中查找 date 的 index
     b. future_date = all_dates[index + h]
     c. future_ret = return_history[future_date]
     d. common = signal.index ∩ future_ret.index
     e. 若 len(common) < 5 → 跳过（样本太少）
     f. ic = spearmanr(signal[common], future_ret[common])
     g. 若 ic 非 NaN → 加入 ics 列表

3. ic_h = mean(ics)

输出 DataFrame: 行=管线名, 列=T+h
```

#### marginal_contribution 计算流程

```
marginal_contribution() → DataFrame

前置: 需要 signal_history 包含 "combined" 键

1. full_ic = combined 信号的 T+1 Rank IC 均值

2. 对每条管线 P（排除 "combined"）:
   a. remaining = 除 P 和 combined 之外的其他管线
   b. 对每个日期, without_signal = mean(remaining 管线的信号)
   c. without_ic = without_signal 的 T+1 Rank IC 均值
   d. marginal_ic = full_ic - without_ic

输出 DataFrame: index=管线名, columns=[full_ic, without_ic, marginal_ic]
```

#### signal_correlation 计算流程

```
signal_correlation() → DataFrame

排除 "combined" 管线，仅计算基础管线间的相关性

1. 找所有管线的共同日期
2. 对每个共同日期:
   a. 找所有管线的公共 symbol（需 ≥ 5 个）
   b. 对每对管线 (i, j): spearmanr(signal_i, signal_j)
3. corr_matrix = 各日期相关系数的平均值
4. 对角线 = 1.0

输出 DataFrame: 行=管线名, 列=管线名
  低相关 → 管线间信息互补，融合有意义
  高相关 → 信号冗余，应检查管线独立性
```

#### rolling_ic 逻辑

```
rolling_ic(pipeline_name, window=60) → pd.Series

1. 对每个 (date, signal) 计算与 T+1 收益的 Rank IC → 得到逐日 IC 序列
2. 滚动平均: ic_series.rolling(window, min_periods=window//3).mean()

输出: pd.Series(index=date, values=rolling_mean_ic)
```

#### 边界处理

```
· 空 signal_history → 返回空 DataFrame
· 某管线历史为空 → 自动跳过
· 无 "combined" 键 → marginal_contribution 返回空 DataFrame
· 单管线 → signal_correlation 返回空 DataFrame（需 ≥ 2 条管线）
· 公共 symbol < 5 → 跳过该日期的 IC 计算
· spearmanr 返回 NaN → 跳过
```

---

### 子模块四：RegimeAnalyzer（分环境归因）

#### 接口

| 方法 | 输入 | 输出 | 说明 |
|------|------|------|------|
| `__init__(daily_nav, benchmark_nav, risk_events?, return_history?)` | 日净值, 基准净值, 风控事件(可选), 收益历史(可选) | — | |
| `classify_regimes()` | — | pd.Series[date→str] | 每日环境标签 |
| `regime_performance()` | — | DataFrame | 分环境绩效 |
| `risk_impact()` | — | RiskImpactReport | 风控事件的收益影响 |

#### 环境分类逻辑

```
classify_regimes() → pd.Series

基于 benchmark 净值的 20 日简单移动均线:

ma20 = benchmark_nav.rolling(20, min_periods=20).mean()

if benchmark > ma20 × 1.02:
    regime = "bull"      牛市（价格显著高于均线）
elif benchmark < ma20 × 0.98:
    regime = "bear"      熊市（价格显著低于均线）
else:
    regime = "sideways"  震荡（围绕均线波动）

前 19 天 MA20 不可用 → 默认归为 "sideways"
benchmark 不足 20 天 → 全部归为 "sideways"
```

#### regime_performance 输出

```
DataFrame (index=regime):
              trading_days  annual_return  sharpe  max_drawdown  win_rate  excess_return
bull                 120      0.352       2.10      0.053        0.62       0.121
bear                  80     -0.085      -0.40      0.152        0.38       0.083
sideways             100      0.128       1.20      0.071        0.55       0.065

分环境计算逻辑:
  · 按 regime 标签分组 daily_returns
  · 每组独立计算: 年化收益、夏普（rf=0.02）、最大回撤、日胜率
  · excess_return = 策略期间总收益 - 基准期间总收益
  · 交易天数 ≤ 10 时不年化（直接用总收益）

关键解读:
  · 牛市表现好是基本要求
  · 熊市 excess_return > 0 → 有防御能力
  · 震荡市正收益 → 策略不依赖趋势
```

#### 风控有效性分析

```
risk_impact() → RiskImpactReport

RiskImpactReport
    stop_loss_count: int = 0             止损次数
    stop_loss_saved_pct: float = 0.0     止损避免的平均后续亏损
    stop_loss_missed_pct: float = 0.0    止损后股票反弹的平均幅度
    circuit_breaker_count: int = 0       熔断次数
    circuit_breaker_impact: float = 0.0  熔断期间市场累计涨跌
```

#### 止损有效性计算

```
对每个 stop_loss 事件:
  1. 从 return_history 中取止损日后 5 个交易日该股票的日收益
  2. future_ret = Σ(daily_return[t+1] ... daily_return[t+5])  # 简单累加
  3. if future_ret < 0:
       saved_pcts.append(|future_ret|)     # 止损有效，避免了后续亏损
     else:
       missed_pcts.append(future_ret)       # 止损过早，错过了反弹

  stop_loss_saved_pct = mean(saved_pcts)    # 平均避损幅度
  stop_loss_missed_pct = mean(missed_pcts)  # 平均过度止损幅度
```

#### 熔断有效性计算

```
对每个 circuit_breaker 事件:
  1. 从 event.details 取 pause_until 日期
  2. 取暂停期间 benchmark 的日收益
  3. cum_ret = Π(1 + bench_daily_ret) - 1   # 暂停期间市场累计涨跌

  circuit_breaker_impact = mean(cum_ret)
    · 负值 → 市场下跌，熔断有效（避开了下跌）
    · 正值 → 市场上涨，熔断过度保守（错过了反弹）
```

---

### 子模块五：EvaluationPipeline（评估入口）

#### 接口

| 方法 | 输入 | 输出 | 说明 |
|------|------|------|------|
| `__init__(trade_records, daily_nav, benchmark_nav?, signal_history?, return_history?, risk_events?, risk_free_rate?)` | 全部数据 | — | 所有参数除 trade_records 和 daily_nav 外均可选 |
| `generate_report()` | — | EvaluationReport | 生成完整评估报告 |
| `print_summary(report?)` | EvaluationReport(可选) | str | 打印摘要到控制台，返回摘要文本 |

#### generate_report 流程

```
generate_report() → EvaluationReport

1. 绩效计算
   perf = PerformanceCalculator(daily_nav, benchmark_nav, risk_free_rate)
   performance = perf.summary()
   monthly = perf.monthly_returns()
   drawdown = perf.drawdown_series()
   rolling_s = perf.rolling_sharpe()

2. 交易分析
   analyzer = TradeAnalyzer(trade_records, daily_nav)
   trade = analyzer.summary()
   cost = analyzer.cost_breakdown()
   holding = analyzer.holding_analysis()

3. 成本回填
   performance.total_transaction_cost = cost.total_cost
   performance.cost_drag_annual = cost.annual_cost_drag

4. 信号归因（若 signal_history 非空）
   attr = SignalAttributor(signal_history, return_history)
   ic_decay = attr.ic_decay_analysis()
   marginal = attr.marginal_contribution()
   sig_corr = attr.signal_correlation()

5. 分环境归因（若 benchmark_nav 存在）
   regime = RegimeAnalyzer(daily_nav, benchmark_nav, risk_events, return_history)
   regime_df = regime.regime_performance()
   risk_impact = regime.risk_impact()

6. 组装 EvaluationReport
```

#### EvaluationReport 输出

```
EvaluationReport
    performance: PerformanceSummary        核心绩效指标
    trade: TradeSummary                    交易统计
    cost: CostBreakdown                    成本分解
    holding: HoldingAnalysis               持仓分析
    ic_decay: Optional[DataFrame]          IC 衰减矩阵（行=管线, 列=T+h）
    marginal_contribution: Optional[DataFrame]  边际贡献（行=管线）
    signal_correlation: Optional[DataFrame]     信号相关矩阵
    regime: Optional[DataFrame]            分环境绩效（行=regime）
    risk_impact: RiskImpactReport          风控有效性
    monthly: Optional[DataFrame]           月度收益矩阵
    rolling_sharpe: Optional[Series]       滚动夏普序列
    drawdown_series: Optional[Series]      逐日回撤序列
```

#### print_summary 输出格式

```
========== 回测评估报告 ==========
【绩效】
  年化收益: 25.3%    夏普: 1.85    最大回撤: 12.1%    Calmar: 2.09
  超额收益: 15.8%    信息比率: 1.42
  日胜率: 56.2%      盈亏比: 1.35

【交易】
  总交易: 1245 笔   买入成交率: 82.3%   日均换手: 8.5%
  平均持仓: 12.3 天   单笔胜率: 53.1%
  总成本: ¥28,500 (年化 0.95%)

【信号质量】                          ← 仅在 signal_history 非空时输出
  管线        IC(T+1)  边际贡献
  alpha         0.042     0.016
  kronos        0.031     0.010
  rdagent       0.025     0.004
  combined      0.048       —

【分环境】                            ← 仅在 benchmark_nav 存在时输出
  bull    : +35.2% (超额+12.1%)
  bear    : -8.5%  (超额+8.3%)
  sideways: +12.8% (超额+6.5%)

【风控】
  止损 23 次: 平均避损 2.3%，过度止损率 30.0%
  熔断 1 次: 暂停期间市场下跌 3.5%
===================================

print_summary 同时返回完整摘要文本（str），方便写入文件或日志。
若未传入 report 参数，内部自动调用 generate_report()。
```

---

### M8 整体使用流程

```
场景1: 回测结束后完整评估（M9 调用）

  evaluator = EvaluationPipeline(
      trade_records=all_trade_records,
      daily_nav=nav_series,
      benchmark_nav=benchmark_series,
      signal_history=signal_hist,
      return_history=return_hist,
      risk_events=risk_event_list,
      risk_free_rate=0.02,
  )
  report = evaluator.generate_report()
  evaluator.print_summary(report)


场景2: 信号质量诊断（独立使用）

  attr = SignalAttributor(signal_history, return_history)
  print(attr.ic_decay_analysis())           # 哪条管线信号衰减快？
  print(attr.marginal_contribution())       # 哪条管线贡献最大？
  print(attr.signal_correlation())          # 管线间是否冗余？
  alpha_ic = attr.rolling_ic("alpha", 60)   # alpha 管线滚动 IC


场景3: 滚动绩效监控（独立使用）

  perf = PerformanceCalculator(daily_nav, benchmark_nav)
  rolling = perf.rolling_sharpe(window=60)
  # 画图观察策略是否在特定时期失效


场景4: 风控复盘

  regime = RegimeAnalyzer(daily_nav, benchmark_nav, risk_events, return_history)
  impact = regime.risk_impact()
  if impact.stop_loss_saved_pct > 0:
      effective_rate = impact.stop_loss_count / (impact.stop_loss_count + ...)
      print(f"止损有效: 平均避损 {impact.stop_loss_saved_pct:.1%}")
  print(f"熔断 {impact.circuit_breaker_count} 次, 暂停期市场: {impact.circuit_breaker_impact:+.1%}")


场景5: 成本优化

  analyzer = TradeAnalyzer(trade_records, daily_nav)
  cost = analyzer.cost_breakdown()
  print(f"年化成本: {cost.annual_cost_drag:.2%}")
  print(f"滑点占比: {cost.total_slippage / cost.total_cost:.0%}")
  # 若成本过高 → 降低换手率（调整 target_sell_count）
```

---

### 测试

#### PerformanceCalculator 测试

| 测试项 | 方法 | 验证点 |
|--------|------|--------|
| 基本指标 | 252 天随机净值 | 年化波动率 > 0, 日胜率 ∈ [0,1] |
| 正收益序列 | 每日 +0.2% | total_return > 0, sharpe > 0, max_drawdown ≈ 0 |
| 已知回撤 | [100, 110, 95, 105, 90] | max_drawdown = 20/110 ≈ 18.18% |
| Calmar 一致 | 计算值 | calmar = annualized_return / max_drawdown |
| 超额收益 | 策略+基准 | excess 为两者年化之差 |
| 空净值 | pd.Series(dtype=float) | 全零，不报错 |
| 单日净值 | 仅 1 天 | total_return=0，不报错 |
| 月度矩阵 | 252 天 | 含 "全年" 列 |
| 回撤序列 | 任意净值 | 长度一致，值 ≤ 0 |
| 滚动夏普 | 100 天, window=20 | 输出非空 |
| 零波动 | 恒定净值 | sharpe=0, ann_vol=0 |

#### TradeAnalyzer 测试

| 测试项 | 方法 | 验证点 |
|--------|------|--------|
| 基本统计 | 2 买 1 卖, 1 失败 | total=3, filled_rate_buy=50% |
| 失败原因 | limit_up 失败 | failed_reasons 含 "limit_up" |
| 成本分解 | 已知 commission + stamp_tax | total_cost 正确 |
| 空记录 | [] | 全零，不报错 |
| 持仓天数 | 1/10 买, 1/20 卖 | avg=10, median=10 |
| 单笔胜率 | 1 盈利 + 1 亏损 | win_rate=50% |

#### SignalAttributor 测试

| 测试项 | 方法 | 验证点 |
|--------|------|--------|
| IC 衰减 | 与 T+1 高相关的信号 | T+1 IC > T+10 IC |
| IC 非空 | 30 天 × 20 只 | DataFrame 行=管线, 列=T+h |
| 边际贡献 | 去掉 alpha 后 IC 下降 | marginal_ic > 0 |
| 信号相关 | 两管线 | 对角线 = 1.0 |
| 滚动 IC | alpha, window=10 | 输出非空 |
| 空信号 | {} | 返回空 DataFrame |
| 单管线 | 仅 alpha | IC 衰减正常, 相关矩阵为空 |

#### RegimeAnalyzer 测试

| 测试项 | 方法 | 验证点 |
|--------|------|--------|
| 全牛市 | benchmark 单调上涨 | MA20 后大部分为 "bull" |
| 全熊市 | benchmark 单调下跌 | MA20 后大部分为 "bear" |
| 分环境绩效 | 混合行情 | DataFrame 非空, 含 annual_return 列 |
| 止损有效 | 止损后标的继续跌 | stop_loss_saved_pct > 0 |
| 无事件 | risk_events=[] | count=0, pct=0 |
| 短基准 | < 20 天 | 全部 "sideways" |

#### EvaluationPipeline 测试

| 测试项 | 方法 | 验证点 |
|--------|------|--------|
| 完整报告 | generate_report | performance / trade / cost 非 None |
| 控制台输出 | print_summary | 不报错, 返回含 "回测评估报告" 的字符串 |
| 空回测 | 无交易记录 | trade.total_trades=0, cost.total_cost=0 |
| 含信号历史 | signal_history 非空 | ic_decay 非 None 且非空 |
| 含风控事件 | stop_loss 事件 | risk_impact.stop_loss_count=1 |

---

## M9 主循环（main.py）

### 职责

三项核心职责：

1. **回测调度** — 逐交易日驱动 M1-M8 的完整流程
2. **配置管理** — 统一管理所有模块的参数、模式切换
3. **状态与检查点** — 进度跟踪、断点续跑、结果持久化

### 设计思路

```
M9 是"胶水层"，自身不包含业务逻辑，只负责:
  · 按正确顺序调用各模块
  · 传递数据（上一模块的输出 → 下一模块的输入）
  · 管理交易日历迭代
  · 收集中间结果（净值、信号历史、交易记录）
  · 提供检查点功能（长回测可断点续跑）
  · 各管线信号异常不中断回测（try/except 降级）
```

### 模块协作全景

```
                    BacktestConfig
                         │
                    BacktestRunner
                    ┌────┴────┐
              _init_modules()  run() / resume()
                    │              │
    M1 DataManager ─┤         主循环: for T in calendar
    M2 AlphaPipeline┤              │
    M3 KronosPipeline              DailyStep.execute(T, T_next, ...)
    M4 RDAgentPipeline             │
    M5 EnsemblePipeline       ┌────┼────────────────┐
    M6 ExecutionPipeline      │    │                 │
    M7 RiskController         │  BacktestState    _evaluate()
                              │    · daily_nav       │
                              │    · trade_records  M8 EvaluationPipeline
                              │    · signal_history  │
                              │    · risk_events   BacktestResult
                              │
                         DailyResult
```

### 数据依赖

```
M9 汇聚所有上游模块的输出:

→ M1 DataManager:
  · get_trading_calendar(start, end) → 交易日历
  · get_close_prices(T) → T 日收盘价
  · get_open_prices(T_next) → T+1 日开盘价
  · get_limit_prices(T_next) → T+1 日涨跌停价
  · get_daily_returns(T) → T 日收益率
  · get_industry_map() → 行业映射
  · get_ohlcv_before(T, 60) → OHLCV 回看（M3 用）
  · get_benchmark_nav(benchmark) → 基准净值（M8 用）

→ M2 AlphaSignalPipeline:
  · predict(T, dm) → pd.Series[symbol→score]

→ M3 KronosSignalPipeline:
  · daily_run(ohlcv, T, dm) → KronosOutput.return_1d

→ M4 RDAgentSignalPipeline:
  · compute(T, dm) → RDAgentOutput.signal

→ M5 SignalEnsemblePipeline:
  · combine(signals, T) → EnsembleOutput.signal
  · update_history(T, signals, actual_returns)

→ M6 ExecutionPipeline:
  · generate_orders(signal, prices, industry, T) → orders
  · add_force_sell_orders(orders, symbols, prices, reason)
  · add_liquidation_orders(orders, prices)
  · execute_orders(orders, open_prices, lup, ldown, T_next) → trades
  · get_portfolio_value(prices) → float
  · get_holdings() → Dict[symbol, Position]

→ M7 RiskController:
  · daily_check(positions, prices, industry, value, T, calendar) → RiskCheckResult
  · is_paused(T) → bool

→ M8 EvaluationPipeline:
  · generate_report() → EvaluationReport
  · print_summary(report) → str

外部依赖:
  · yaml — 配置文件解析
  · pickle — 检查点序列化
  · numpy — 随机种子
```

### 子模块划分

```
quantlab/
├── main.py                      M9 核心实现
│   ├── BacktestConfig               配置加载 + 校验
│   ├── BacktestState                运行时状态 + 断点
│   ├── DailyStep                    单日编排（Phase 0–8）
│   ├── DailyResult                  单日结果
│   ├── BacktestResult               回测结果
│   ├── BacktestRunner               主入口
│   ├── _apply_position_limit()      恢复期仓位限制
│   └── main()                       CLI 入口
└── configs/
    └── backtest.yaml                默认回测配置
```

---

### 子模块一：BacktestConfig（回测配置）

#### 数据结构

```
BacktestConfig
    # === 时间 ===
    start_date: str = "2023-01-01"     回测起始日期
    end_date: str = "2025-03-01"       回测结束日期
    warmup_days: int = 20              预热天数（前 N 天不产生信号，仅记录 NAV）

    # === 数据 ===
    market: str = "csi300"             股票池（"csi300" | "csi500" | "csi800" | "all"）
    qlib_data_dir: str                 Qlib 数据目录（默认 ~/.qlib/qlib_data/cn_data）

    # === 管线开关 ===
    enable_alpha: bool = True          启用管线 A（Alpha158 + LightGBM）
    enable_kronos: bool = True         启用管线 B（Kronos 预测）
    enable_rdagent: bool = False       启用管线 C（RD-Agent 进化因子）

    # === M2 配置 ===
    alpha_retrain_interval: int = 20   重训间隔天数
    alpha_train_years: int = 3         滚动训练窗口年数

    # === M3 配置 ===
    kronos_recipe_name: str = "conservative"   微调方案名
    kronos_device: str = "cuda"                推理设备

    # === M5 配置 ===
    ensemble_ic_lookback: int = 60             IC 回看窗口
    ensemble_uncertainty_penalty: float = 0.1  不确定性惩罚系数
    ensemble_min_weight: float = 0.1           最低管线权重
    ensemble_max_weight: float = 0.6           最高管线权重

    # === M6 配置 ===
    initial_cash: float = 1_000_000.0  初始资金
    max_positions: int = 10            最大持仓数
    max_single_weight: float = 0.20    单票最大仓位
    target_buy_count: int = 3          每日目标买入数
    target_sell_count: int = 3         每日目标卖出数

    # === M7 配置 ===
    stop_loss_pct: float = 0.08        止损比例
    max_hold_days: int = 60            最大持仓天数
    max_industry_pct: float = 0.30     行业上限
    circuit_breaker_pct: float = 0.10  熔断阈值
    pause_days: int = 5                暂停天数

    # === 运行控制 ===
    random_seed: int = 42              随机种子
    checkpoint_interval: int = 50      检查点间隔天数（0=不保存）
    checkpoint_dir: str = "checkpoints"检查点目录
    log_level: str = "INFO"            日志级别
    benchmark: str = "SH000300"        基准指数
    risk_free_rate: float = 0.02       无风险利率（传给 M8）
```

#### 配置文件格式

```yaml
# configs/backtest.yaml
start_date: "2023-01-01"
end_date: "2025-03-01"
market: "csi300"
initial_cash: 1000000

enable_alpha: true
enable_kronos: true
enable_rdagent: false

alpha_retrain_interval: 20
alpha_train_years: 3
kronos_recipe_name: "conservative"
kronos_device: "cuda"

ensemble_ic_lookback: 60
ensemble_uncertainty_penalty: 0.1

max_positions: 10
max_single_weight: 0.20
target_buy_count: 3
target_sell_count: 3

stop_loss_pct: 0.08
max_hold_days: 60
max_industry_pct: 0.30
circuit_breaker_pct: 0.10
pause_days: 5

random_seed: 42
checkpoint_interval: 50
benchmark: "SH000300"
log_level: "INFO"
```

#### 接口

| 方法 | 输入 | 输出 | 说明 |
|------|------|------|------|
| `load(yaml_path)` | 配置文件路径 | BacktestConfig | 类方法，从 YAML 加载，自动过滤未知字段 |
| `validate()` | — | — | 校验: start<end, cash>0, warmup≥0, 0<stop_loss<1, 0<circuit_breaker<1 |
| `to_dict()` | — | Dict | 序列化所有 dataclass 字段为字典 |

#### 校验规则

```
validate() 检查项:
  · start_date < end_date                → ValueError
  · initial_cash > 0                     → ValueError
  · warmup_days >= 0                     → ValueError
  · 0 < stop_loss_pct < 1               → ValueError
  · 0 < circuit_breaker_pct < 1         → ValueError

YAML 加载时:
  · 未知字段自动忽略（不报错）
  · 缺失字段使用 dataclass 默认值
  · load() 结尾自动调用 validate()
```

---

### 子模块二：BacktestState（运行时状态）

#### 职责

维护回测运行过程中的全部累积数据，支持 pickle 序列化/反序列化实现断点续跑。

#### 数据结构

```
BacktestState
    # === 进度 ===
    current_idx: int = 0               当前交易日索引
    total_days: int = 0                总交易日数（calendar 长度 - 1）

    # === 累积数据 ===
    daily_nav: List[Tuple[str, float]]         逐日净值 [(date, nav), ...]
    trade_records: list                         全部交易记录 List[TradeRecord]
    signal_history: Dict[str, List[Tuple]]      各管线信号历史
                                                 {"alpha": [(date, Series), ...], ...}
    return_history: Dict[str, Any]              逐日收益历史
                                                 {date: Series[symbol→return]}
    risk_events: list                           风控事件列表 List[RiskEvent]
```

#### 接口

| 方法 | 输入 | 输出 | 说明 |
|------|------|------|------|
| `append_nav(date, value)` | 日期, 净值 | — | 追加一天净值 |
| `append_trades(records)` | 交易记录列表 | — | extend 追加 |
| `append_signals(date, signals)` | 日期, {管线名: Series} | — | 按管线分别追加，跳过 None 值 |
| `append_return(date, returns)` | 日期, Series | — | 追加当日收益 |
| `get_nav_series()` | — | pd.Series | 转换为 DatetimeIndex 的 Series |
| `save_checkpoint(path)` | 文件路径 | — | pickle.dump，自动创建父目录 |
| `load_checkpoint(path)` | 文件路径 | BacktestState | 静态方法，pickle.load |

#### 序列化格式

```
使用 Python pickle 协议:
  · save_checkpoint → pickle.dump(self, file)
  · load_checkpoint → pickle.load(file)

断点文件存储位置: {checkpoint_dir}/checkpoint_{date}.pkl
最终状态文件: {checkpoint_dir}/final.pkl
```

---

### 子模块三：DailyStep（单日编排）

#### 职责

封装单个交易日的完整处理逻辑（9 个阶段），作为静态方法供主循环调用。

#### 接口

| 方法 | 输入 | 输出 | 说明 |
|------|------|------|------|
| `execute(T, T_next, dm, alpha, kronos, rdagent, ensemble, execution, risk, config, calendar_strs, day_idx)` | 当日, 次日, 各模块实例, 配置, 日历, 日索引 | DailyResult | 静态方法 |

#### DailyResult 输出

```
DailyResult
    date: str = ""                     交易日期
    portfolio_value: float = 0.0       当日净值
    daily_return: float = 0.0          当日收益率
    trade_records: list = []           当日成交记录
    signals: Dict[str, Any] = {}       当日各管线信号（含 "combined"）
    risk_result: Any = None            风控检查结果 (RiskCheckResult)
    ensemble_output: Any = None        融合输出 (EnsembleOutput)
    skipped: bool = False              是否因暂停跳过
```

#### execute 内部流程（9 阶段）

```
Phase 0 — 风控暂停检查
    if risk.is_paused(T):
        # 暂停期: 仅获取收盘价计算净值，立即返回
        close_dict = dm.get_close_prices(T).to_dict()
        nav = execution.get_portfolio_value(close_dict)
        return DailyResult(date=T, portfolio_value=nav, skipped=True)

Phase 1 — 数据准备
    close_prices = dm.get_close_prices(T)       # T 日收盘价
    close_dict = close_prices.to_dict()
    holdings = execution.get_holdings()          # 当前持仓
    total_value = execution.get_portfolio_value(close_dict)
    industry_map = dm.get_industry_map()
    execution.account.update_highest_price(close_dict)  # 更新最高价

Phase 2 — 风控检查
    risk_result = risk.daily_check(
        positions=holdings, current_prices=close_dict,
        industry_map=industry_dict, total_value=total_value,
        current_date=T, calendar=calendar_strs
    )

Phase 3 — 信号生成（warmup 期或熔断触发时跳过）
    is_warmup = day_idx < config.warmup_days

    if not is_warmup and not risk_result.circuit_breaker_triggered:
        # 各管线独立 try/except，失败不影响其他管线
        if enable_alpha:   signals["alpha"]   = alpha.predict(T, dm)
        if enable_kronos:  signals["kronos"]  = kronos.daily_run(ohlcv, T, dm).return_1d
        if enable_rdagent: signals["rdagent"] = rdagent.compute(T, dm).signal  # 仅 factor_count > 0

Phase 4 — 信号融合
    if signals and ensemble is not None:
        ensemble_input = {"alpha": ..., "kronos": ..., "rdagent": ...}
        ensemble_output = ensemble.combine(ensemble_input, T)
        combined_signal = ensemble_output.signal
        signals["combined"] = combined_signal    # 记录融合信号

Phase 5 — 订单生成
    if circuit_breaker_triggered:
        orders = execution.add_liquidation_orders([], close_dict)  # 熔断清仓
    else:
        orders = execution.generate_orders(combined_signal, close_dict, industry, T)
        if force_sell_symbols:
            orders = execution.add_force_sell_orders(orders, symbols, prices, "risk_control")

    # 恢复期仓位限制
    if position_limit < 1.0:
        _apply_position_limit(orders, position_limit, holdings, total_value, close_dict)

Phase 6 — T+1 执行
    if orders:
        open_prices = dm.get_open_prices(T_next)
        limit_up, limit_down = dm.get_limit_prices(T_next)
        trade_records = execution.execute_orders(orders, open_dict, lup, ldown, T_next)

Phase 7 — NAV 更新
    nav = execution.get_portfolio_value(close_dict)  # 用 T 日收盘价
    daily_return = (nav / total_value) - 1.0

Phase 8 — 融合历史更新
    yesterday_returns = dm.get_daily_returns(T)
    ensemble.update_history(T, signals, yesterday_returns)
```

#### 信号异常处理策略

```
各管线信号生成均有独立的 try/except:
  · alpha.predict 失败 → signals 中无 "alpha" 键，其他管线继续
  · kronos.daily_run 失败 → signals 中无 "kronos" 键
  · rdagent.compute 失败 → signals 中无 "rdagent" 键
  · ensemble.combine 失败 → combined_signal 为空 Series，不生成订单
  · execution.execute_orders 失败 → trade_records 为空

降级原则:
  · 任何单个管线失败不中断回测
  · 融合层自动处理缺失管线（M5 支持部分输入）
  · 没有有效信号时不生成订单（等价于空仓等待）
```

#### _apply_position_limit 辅助函数

```
_apply_position_limit(orders, position_limit, holdings, total_value, current_prices)

作用: 恢复期限制买入总规模

逻辑:
  · position_limit = 0.0 → 移除所有买入订单（暂停期）
  · position_limit = 0.5 → 计算:
      holding_value = Σ(pos.shares × price)    # 当前持仓市值
      max_holding = total_value × 0.5          # 最大允许持仓
      available = max(0, max_holding - holding_value)
      按顺序保留买入订单，累计金额不超过 available
  · 卖出订单始终保留
```

---

### 子模块四：BacktestRunner（回测入口）

#### 接口

| 方法 | 输入 | 输出 | 说明 |
|------|------|------|------|
| `__init__(config)` | BacktestConfig | — | 仅保存配置，不初始化模块 |
| `run()` | — | BacktestResult | 执行完整回测 |
| `resume(checkpoint_path)` | 检查点路径 | BacktestResult | 从检查点续跑 |

#### BacktestResult 输出

```
BacktestResult
    config: BacktestConfig               回测配置
    daily_nav: pd.Series                 逐日净值（DatetimeIndex）
    trade_records: list                  全部交易记录
    signal_history: Dict[str, List]      各管线信号历史
    evaluation: Any                      M8 EvaluationReport（可能为 None）
    risk_events: list                    风控事件列表
    elapsed_seconds: float               运行耗时（秒）
```

#### _init_modules 模块初始化

```
_init_modules() — 按需初始化 M1–M7

顺序:
  1. 设置日志级别（config.log_level）
  2. 设置随机种子（np.random.seed）
  3. M1 DataManager(provider_uri=qlib_data_dir, market=market)
  4. M2 AlphaSignalPipeline（若 enable_alpha）
       FactorRegistry() → AlphaSignalPipeline(registry, market, retrain_interval, train_years)
  5. M3 KronosSignalPipeline（若 enable_kronos）
       FinetuneRecipe(name=recipe_name) → KronosSignalPipeline(recipe, device=device)
  6. M4 RDAgentSignalPipeline（若 enable_rdagent）
       CodeFactorRegistry() + CodeFactorExecutor() → RDAgentSignalPipeline(registry, executor)
  7. M5 SignalEnsemblePipeline(ic_lookback, uncertainty_penalty, min_weight, max_weight)
  8. M6 ExecutionPipeline(initial_cash, max_positions, max_single_weight, ...)
  9. M7 RiskController(stop_loss_pct, max_hold_days, max_industry_pct, ...)

容错:
  · M2/M3/M4 初始化失败 → 对应管线设为 None，不中断
  · M5/M6/M7 是必要模块，初始化失败会抛出异常
```

#### run 内部流程

```
run() → BacktestResult

1. 初始化
   _init_modules()                          # 初始化 M1–M7

2. 获取交易日历
   calendar = dm.get_trading_calendar(start_date, end_date)
   calendar_strs = [d.strftime("%Y-%m-%d") for d in calendar]
   # 至少需要 2 个交易日（1 个 T + 1 个 T_next）

3. 初始化状态
   state = BacktestState(total_days=len(calendar) - 1)
   state.append_nav(calendar[0], initial_cash)   # 记录初始 NAV

4. 主循环
   for i in range(len(calendar) - 1):
       T = calendar[i]
       T_next = calendar[i + 1]
       state.current_idx = i

       daily_result = DailyStep.execute(T, T_next, dm, alpha, ..., day_idx=i)

       # 更新状态
       state.append_nav(T_next, daily_result.portfolio_value)
       state.append_trades(daily_result.trade_records)
       state.append_signals(T, daily_result.signals)
       state.risk_events.extend(daily_result.risk_result.events)

       # 记录当日市场收益（供 M8 用）
       state.append_return(T, dm.get_daily_returns(T))

       # 进度日志（每 20 天）
       if (i + 1) % 20 == 0:
           log("Progress: {i+1}/{total} ({pct}%) | NAV={nav} | Date={T}")

       # 检查点保存（每 checkpoint_interval 天）
       if checkpoint_interval > 0 and (i + 1) % checkpoint_interval == 0:
           state.save_checkpoint("checkpoints/checkpoint_{T}.pkl")

5. 评估
   nav_series = state.get_nav_series()
   evaluation = _evaluate(state, nav_series)

6. 保存最终状态
   state.save_checkpoint("checkpoints/final.pkl")

7. 返回 BacktestResult
```

#### _evaluate 评估流程

```
_evaluate(state, nav_series) → EvaluationReport | None

1. 获取基准净值（可选）
   benchmark_nav = dm.get_benchmark_nav(config.benchmark, start, end)

2. 构建 M8 EvaluationPipeline
   evaluator = EvaluationPipeline(
       trade_records=state.trade_records,
       daily_nav=nav_series,
       benchmark_nav=benchmark_nav,
       signal_history=state.signal_history,
       return_history=state.return_history,
       risk_events=state.risk_events,
       risk_free_rate=config.risk_free_rate,
   )

3. 生成并打印报告
   report = evaluator.generate_report()
   summary = evaluator.print_summary(report)
   logger.info(summary)

4. 若评估失败 → 返回 None，不中断主流程
```

#### resume 流程

```
resume(checkpoint_path) → BacktestResult

1. 初始化模块
   _init_modules()

2. 加载状态
   state = BacktestState.load_checkpoint(checkpoint_path)

3. 获取交易日历并定位续跑起点
   calendar = dm.get_trading_calendar(start_date, end_date)
   start_idx = state.current_idx + 1

4. 从 start_idx 继续主循环
   （逻辑与 run 相同）

注意:
  · resume 不恢复 M6/M7 的内部状态（账户/风控）
  · 适用于纯重放场景，完整状态恢复需要更精细的模块级序列化
  · 实际使用中建议 checkpoint_interval 设置较大值（50-100）
```

---

### CLI 入口

```
python -m quantlab.main --config configs/backtest.yaml
python -m quantlab.main --config configs/backtest.yaml --resume checkpoints/checkpoint_2024-06-15.pkl

参数:
  --config, -c    回测配置文件路径（默认: quantlab/configs/backtest.yaml）
  --resume, -r    断点文件路径（可选）

输出:
  Backtest finished in 1234.5s
  Final NAV: 1250000.00
  Total trades: 1245
  Risk events: 28
```

---

### M9 整体使用流程

```
场景1: 标准回测
  config = BacktestConfig.load("configs/backtest.yaml")
  runner = BacktestRunner(config)
  result = runner.run()
  # 自动打印评估摘要

场景2: 快速单管线测试
  config = BacktestConfig.load("configs/backtest.yaml")
  config.enable_kronos = False
  config.enable_rdagent = False
  runner = BacktestRunner(config)
  result = runner.run()
  # 仅用管线 A，快速验证基本流程

场景3: 断点续跑
  # 第一次运行（中途中断，第 50 天自动保存了检查点）
  runner.run()

  # 第二次续跑
  result = runner.resume("checkpoints/checkpoint_2024-03-15.pkl")

场景4: 参数扫描
  for stop_loss in [0.05, 0.08, 0.10, 0.15]:
      config.stop_loss_pct = stop_loss
      runner = BacktestRunner(config)
      result = runner.run()
      print(f"stop_loss={stop_loss}: sharpe={result.evaluation.performance.sharpe_ratio:.2f}")

场景5: 管线消融实验
  for pipelines in [("A",), ("B",), ("C",), ("A","B"), ("A","B","C")]:
      config.enable_alpha = "A" in pipelines
      config.enable_kronos = "B" in pipelines
      config.enable_rdagent = "C" in pipelines
      runner = BacktestRunner(config)
      result = runner.run()
      label = "+".join(pipelines)
      print(f"{label}: sharpe={result.evaluation.performance.sharpe_ratio:.2f}")

场景6: CLI 命令行
  python -m quantlab.main -c configs/backtest.yaml
  python -m quantlab.main -c configs/fast_test.yaml --resume checkpoints/checkpoint_2024-06-15.pkl
```

---

### 测试

#### BacktestConfig 测试

| 测试项 | 方法 | 验证点 |
|--------|------|--------|
| YAML 加载 | load 完整配置 | 所有字段正确解析（start_date, market, initial_cash 等） |
| 日期校验 | start_date > end_date | 抛出 ValueError("start_date ... must be before") |
| 负资金校验 | initial_cash = -100 | 抛出 ValueError("initial_cash must be positive") |
| 止损范围校验 | stop_loss_pct = 1.5 | 抛出 ValueError("stop_loss_pct must be in (0, 1)") |
| to_dict | 默认配置 | 字典包含 start_date, initial_cash, market 等键 |
| 未知字段忽略 | YAML 含 unknown_field: 999 | 正常加载，不报错，无 unknown_field 属性 |

#### BacktestState 测试

| 测试项 | 方法 | 验证点 |
|--------|------|--------|
| 追加净值 | append_nav 两次 | daily_nav 长度 = 2 |
| 转 Series | get_nav_series | 返回 pd.Series, len=2, iloc[1] 正确 |
| 空 Series | 未追加时 get_nav_series | 返回空 Series |
| 信号追加 | append_signals(date, {"alpha": sig, "kronos": None}) | alpha 记录, kronos 跳过 |
| 交易追加 | append_trades 两批 | trade_records 长度为两批之和 |
| 检查点存取 | save → load | current_idx, total_days, daily_nav 完全一致 |

#### DailyStep 测试

| 测试项 | 方法 | 验证点 |
|--------|------|--------|
| 正常日 | day_idx=2（超过 warmup=2） | 信号生成, ensemble 调用, 返回完整 DailyResult |
| 预热期跳过信号 | day_idx=0（< warmup=2） | alpha.predict 未调用, signals 为空 |
| 暂停期跳过一切 | risk.is_paused=True | skipped=True, daily_check 未调用, predict 未调用 |
| 熔断触发清仓 | circuit_breaker_triggered=True | add_liquidation_orders 调用, predict 未调用 |
| 风控强卖追加 | force_sell_symbols=["SH600000"] | add_force_sell_orders 调用 |
| 信号异常不崩溃 | alpha.predict 抛出 RuntimeError | signals 无 "alpha" 键, 回测继续 |

#### _apply_position_limit 测试

| 测试项 | 方法 | 验证点 |
|--------|------|--------|
| 零限制移除买入 | position_limit=0.0, 含 2 buy + 1 sell | 仅剩 1 个 sell 订单 |
| 部分限制通过 | position_limit=0.5, 买入金额在限额内 | 所有买入保留 |
| 限制截断买入 | position_limit=0.5, 持仓已接近上限 | 超限买入被剔除 |

#### BacktestRunner 测试

| 测试项 | 方法 | 验证点 |
|--------|------|--------|
| 构造函数 | BacktestRunner(config) | config 保存, _dm 为 None |
| 主流程 | mock 模块 + 5 天日历 | 返回 BacktestResult, daily_nav 长度=5, elapsed>0 |
| 断点加载 | save → load checkpoint | current_idx, daily_nav 一致 |

#### DailyResult 测试

| 测试项 | 方法 | 验证点 |
|--------|------|--------|
| 默认值 | DailyResult() | date="", portfolio_value=0, skipped=False, signals={} |

---

## 集成测试方案

单元测试覆盖各模块后，需要以下集成测试确保端到端正确性。

---

### IT-1 数据隔离端到端验证

```
方法：在 M1 的所有对外接口上加入 assert anchor_date 检查的 wrapper。
      运行完整回测，全程无断言失败。
验证：任何模块在 T 日不可能接触到 T+1 的数据。
严重性：数据泄露会导致回测失真，这是最高优先级的集成测试。
```

### IT-2 小规模全流程

```
方法：选取 10 只股票、20 个交易日运行完整回测。
验证：
  · 净值曲线与手工计算一致（逐日核对）
  · 每笔交易的成本扣除正确（佣金 + 印花税 + 滑点）
  · 止损和熔断在预期时点触发
  · 资金守恒：cash + 持仓市值 + 累计成本 = 初始资金 ± 盈亏
  · T+1 约束：当日买入股票当日不出现卖出记录
```

### IT-3 信号质量基准

```
方法：用随机信号替代三条管线输出，运行回测。
验证：
  · 随机信号的夏普 ≈ 0（扣除成本后略负）
  · 实际信号的夏普显著高于随机基准（至少高 0.5）
  · 衡量信号是否有效的最低标准
实现：固定 random_seed，生成 N(0,1) 截面随机信号替代各管线输出
```

### IT-4 成本敏感性

```
方法：分别以 0 成本、正常成本、2倍成本运行回测。
验证：
  · 0 成本 > 正常成本 > 2倍成本（收益严格递减）
  · 成本差异在合理范围（年化 1-3%）
  · 若成本拖累 > 3%，说明换手率过高，需调整 target_sell_count
```

### IT-5 分管线消融

```
方法：依次只开启一条管线运行回测:
  · A only / B only / C only / A+B / A+C / B+C / A+B+C
验证：
  · 各管线单独有正向贡献（夏普 > 0）
  · 融合效果 ≥ 任何单管线（夏普最高或回撤最低）
  · 如不满足则需调整融合权重或检查信号质量
输出：消融对比表
```

### IT-6 确定性复现

```
方法：相同配置（含相同 random_seed）运行两次完整回测。
验证：
  · 两次 daily_nav 完全一致（逐日逐值对比，允许浮点误差 1e-6）
  · 两次 trade_records 完全一致（笔数、方向、股数、价格）
目的：确保系统无隐式随机性（如 dict 遍历顺序、未固定的 Kronos 采样）
```

### IT-7 风控级联

```
方法：构造以下场景序列:
  1. 正常交易 10 天
  2. 某票暴跌触发止损
  3. 市场继续下跌触发组合熔断
  4. 暂停 5 天
  5. 恢复模式半仓 3 天
  6. 正常恢复
验证：
  · 止损卖出记录正确（reason="stop_loss"）
  · 熔断后全部清仓
  · 暂停期无交易记录
  · 恢复期买入仓位 ≤ 50%
  · 恢复后仓位限制解除
```

### IT-8 检查点一致性

```
方法：
  · 运行 A：一次性跑 100 天
  · 运行 B：跑 50 天 → 保存检查点 → 续跑 50 天
验证：
  · A 和 B 的 daily_nav 完全一致
  · A 和 B 的 trade_records 完全一致
  · A 和 B 的 evaluation 指标一致
```

### IT-9 边界条件

```
方法：测试各种极端情况:
  · 单日回测（仅2个交易日）
  · 全部股票涨停（无法买入）
  · 全部股票跌停（无法卖出）
  · 初始资金极小（仅够买1手最低价股票）
  · 所有管线都关闭（enable_alpha/kronos/rdagent 全 False）
验证：
  · 不崩溃、不死循环
  · 合理的默认行为（空仓、跳过等）
  · 错误信息清晰
```

### IT-10 性能基准

```
方法：记录各模块在标准场景（CSI300, 1年回测）下的耗时
验证：
  · M2 predict: < 1 秒/天
  · M3 finetune + predict: < 15 分钟/天（GPU）
  · M4 compute: < 5 秒/天
  · M5 combine: < 0.1 秒/天
  · M6 execute: < 0.1 秒/天
  · 全流程 1 年回测: < 24 小时（~250 天 × 15 分钟/天 ≈ 60 小时 理论值）
  · 不含 M3 的 1 年回测: < 1 小时
目的：建立性能基线，后续优化有参照
```
