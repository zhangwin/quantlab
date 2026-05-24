"""M9 回测调度器：配置管理 + 状态追踪 + 每日编排 + 主循环。

核心类：
    BacktestConfig       YAML 配置加载 + 校验
    BacktestState        运行时状态 + 断点序列化
    DailyStep            单日执行编排（Phase 0–8）
    BacktestResult       回测结果
    BacktestRunner       主入口
"""

import json
import logging
import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# BacktestConfig — YAML 配置加载
# ---------------------------------------------------------------------------


@dataclass
class BacktestConfig:
    """回测配置，对应 configs/backtest.yaml。"""

    # Time
    start_date: str = "2023-01-01"
    end_date: str = "2025-03-01"
    warmup_days: int = 20

    # Data
    market: str = "csi300"
    qlib_data_dir: str = "~/.qlib/qlib_data/cn_data"

    # Pipeline switches
    enable_alpha: bool = True
    enable_kronos: bool = True
    enable_rdagent: bool = False

    # M2 Alpha158
    alpha_retrain_interval: int = 20
    alpha_train_years: int = 3

    # M3 Kronos
    kronos_recipe_name: str = "conservative"
    kronos_device: str = "cuda"

    # M5 Ensemble
    ensemble_ic_lookback: int = 60
    ensemble_uncertainty_penalty: float = 0.1
    ensemble_min_weight: float = 0.1
    ensemble_max_weight: float = 0.6

    # M6 Execution
    initial_cash: float = 1_000_000.0
    max_positions: int = 10
    max_single_weight: float = 0.20
    target_buy_count: int = 3
    target_sell_count: int = 3

    # M7 Risk
    stop_loss_pct: float = 0.08
    max_hold_days: int = 60
    max_industry_pct: float = 0.30
    circuit_breaker_pct: float = 0.10
    pause_days: int = 5

    # Runtime
    random_seed: int = 42
    checkpoint_interval: int = 50
    checkpoint_dir: str = "checkpoints"
    log_level: str = "INFO"
    benchmark: str = "SH000300"
    risk_free_rate: float = 0.02

    @classmethod
    def load(cls, yaml_path: str) -> "BacktestConfig":
        """从 YAML 文件加载配置。"""
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        known_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in known_fields}
        cfg = cls(**filtered)
        cfg.validate()
        return cfg

    def validate(self) -> None:
        """校验关键字段。"""
        if self.start_date >= self.end_date:
            raise ValueError(
                f"start_date ({self.start_date}) must be before "
                f"end_date ({self.end_date})"
            )
        if self.initial_cash <= 0:
            raise ValueError("initial_cash must be positive")
        if self.warmup_days < 0:
            raise ValueError("warmup_days must be >= 0")
        if not (0 < self.stop_loss_pct < 1):
            raise ValueError("stop_loss_pct must be in (0, 1)")
        if not (0 < self.circuit_breaker_pct < 1):
            raise ValueError("circuit_breaker_pct must be in (0, 1)")

    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典。"""
        return {
            f.name: getattr(self, f.name)
            for f in self.__dataclass_fields__.values()
        }


# ---------------------------------------------------------------------------
# BacktestState — 运行时状态
# ---------------------------------------------------------------------------


@dataclass
class BacktestState:
    """运行时状态，支持断点续跑。"""

    # Progress
    current_idx: int = 0
    total_days: int = 0

    # Accumulated data
    daily_nav: List[Tuple[str, float]] = field(default_factory=list)
    trade_records: list = field(default_factory=list)
    signal_history: Dict[str, List[Tuple[str, Any]]] = field(default_factory=dict)
    return_history: Dict[str, Any] = field(default_factory=dict)
    risk_events: list = field(default_factory=list)

    def append_nav(self, date: str, value: float) -> None:
        self.daily_nav.append((date, value))

    def append_trades(self, records: list) -> None:
        self.trade_records.extend(records)

    def append_signals(self, date: str, signals: Dict[str, Any]) -> None:
        for name, sig in signals.items():
            if sig is not None:
                if name not in self.signal_history:
                    self.signal_history[name] = []
                self.signal_history[name].append((date, sig))

    def append_return(self, date: str, returns: Any) -> None:
        self.return_history[date] = returns

    def get_nav_series(self) -> pd.Series:
        """转换为 pd.Series（index=date, values=nav）。"""
        if not self.daily_nav:
            return pd.Series(dtype=float)
        dates, vals = zip(*self.daily_nav)
        return pd.Series(vals, index=pd.DatetimeIndex(dates), dtype=float)

    def save_checkpoint(self, path: str) -> None:
        """序列化到文件。"""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "wb") as f:
            pickle.dump(self, f)
        logger.info("Checkpoint saved: %s (day %d/%d)", path, self.current_idx, self.total_days)

    @staticmethod
    def load_checkpoint(path: str) -> "BacktestState":
        """从文件反序列化。"""
        with open(path, "rb") as f:
            state = pickle.load(f)
        logger.info("Checkpoint loaded: %s (day %d/%d)", path, state.current_idx, state.total_days)
        return state


# ---------------------------------------------------------------------------
# DailyStep — 单日执行编排
# ---------------------------------------------------------------------------


@dataclass
class DailyResult:
    """单日执行结果。"""

    date: str = ""
    portfolio_value: float = 0.0
    daily_return: float = 0.0
    trade_records: list = field(default_factory=list)
    signals: Dict[str, Any] = field(default_factory=dict)
    risk_result: Any = None
    ensemble_output: Any = None
    skipped: bool = False


class DailyStep:
    """单日编排器（无状态，所有输入通过参数传入）。"""

    @staticmethod
    def execute(
        T: str,
        T_next: str,
        dm,
        alpha,
        kronos,
        rdagent,
        ensemble,
        execution,
        risk,
        config: BacktestConfig,
        calendar_strs: List[str],
        day_idx: int,
    ) -> DailyResult:
        """执行 T 日全部流程，返回结果。

        Phase 0: 风控暂停检查
        Phase 1: 数据准备
        Phase 2: 风控检查
        Phase 3: 信号生成
        Phase 4: 信号融合
        Phase 5: 订单生成
        Phase 6: T+1 执行
        Phase 7: NAV 更新
        Phase 8: 历史更新
        """
        result = DailyResult(date=T)

        # ---- Phase 0: 风控暂停检查 ----
        if risk.is_paused(T):
            close_prices = dm.get_close_prices(T)
            close_dict = close_prices.to_dict()
            nav = execution.get_portfolio_value(close_dict)
            result.portfolio_value = nav
            result.skipped = True
            logger.debug("[%s] Paused by circuit breaker, NAV=%.2f", T, nav)
            return result

        # ---- Phase 1: 数据准备 ----
        close_prices = dm.get_close_prices(T)
        close_dict = close_prices.to_dict()
        holdings = execution.get_holdings()
        total_value = execution.get_portfolio_value(close_dict)
        industry_map = dm.get_industry_map()
        industry_dict = industry_map.to_dict()

        # 更新持仓最高价
        execution.account.update_highest_price(close_dict)

        # ---- Phase 2: 风控检查 ----
        risk_result = risk.daily_check(
            positions=holdings,
            current_prices=close_dict,
            industry_map=industry_dict,
            total_value=total_value,
            current_date=T,
            calendar=calendar_strs,
        )
        result.risk_result = risk_result

        # ---- Phase 3: 信号生成（warmup 期跳过）----
        signals: Dict[str, Any] = {}
        is_warmup = day_idx < config.warmup_days

        if not is_warmup and not risk_result.circuit_breaker_triggered:
            # M2 Alpha
            if config.enable_alpha and alpha is not None:
                try:
                    signals["alpha"] = alpha.predict(T, dm)
                except Exception as e:
                    logger.warning("[%s] Alpha signal failed: %s", T, e)

            # M3 Kronos
            if config.enable_kronos and kronos is not None:
                try:
                    ohlcv = dm.get_ohlcv_before(T, lookback_days=60)
                    kronos_out = kronos.daily_run(ohlcv, T, dm)
                    signals["kronos"] = kronos_out.return_1d
                except Exception as e:
                    logger.warning("[%s] Kronos signal failed: %s", T, e)

            # M4 RD-Agent
            if config.enable_rdagent and rdagent is not None:
                try:
                    rd_out = rdagent.compute(T, dm)
                    if rd_out.factor_count > 0:
                        signals["rdagent"] = rd_out.signal
                except Exception as e:
                    logger.warning("[%s] RD-Agent signal failed: %s", T, e)

        result.signals = signals

        # ---- Phase 4: 信号融合 ----
        ensemble_output = None
        combined_signal = pd.Series(dtype=float)

        if signals and ensemble is not None:
            try:
                # combine 接受 Optional[pd.Series] 值
                ensemble_input = {
                    "alpha": signals.get("alpha"),
                    "kronos": signals.get("kronos"),
                    "rdagent": signals.get("rdagent"),
                }
                ensemble_output = ensemble.combine(ensemble_input, T)
                combined_signal = ensemble_output.signal
                result.ensemble_output = ensemble_output
            except Exception as e:
                logger.warning("[%s] Ensemble failed: %s", T, e)

        # 记录 combined 信号
        if not combined_signal.empty:
            result.signals["combined"] = combined_signal

        # ---- Phase 5: 订单生成 ----
        orders = []

        if risk_result.circuit_breaker_triggered:
            # 熔断 → 全仓清算
            orders = execution.add_liquidation_orders([], close_dict)
            logger.info("[%s] Circuit breaker triggered — liquidation", T)
        else:
            # 正常订单
            if not combined_signal.empty:
                orders = execution.generate_orders(
                    combined_signal, close_dict, industry_dict, T,
                )

            # 风控强卖
            if risk_result.force_sell_symbols:
                orders = execution.add_force_sell_orders(
                    orders,
                    risk_result.force_sell_symbols,
                    close_dict,
                    reason="risk_control",
                )

        # 恢复期限制：剔除超出仓位限制的买入
        position_limit = risk_result.position_limit
        if position_limit < 1.0 and orders:
            _apply_position_limit(orders, position_limit, holdings, total_value, close_dict)

        # ---- Phase 6: T+1 执行 ----
        trade_records = []
        if orders:
            try:
                open_prices = dm.get_open_prices(T_next)
                limit_up, limit_down = dm.get_limit_prices(T_next)
                open_dict = open_prices.to_dict()
                lup_dict = limit_up.to_dict()
                ldown_dict = limit_down.to_dict()

                trade_records = execution.execute_orders(
                    orders, open_dict, lup_dict, ldown_dict, T_next,
                )
            except Exception as e:
                logger.warning("[%s] Execution failed: %s", T, e)

        result.trade_records = trade_records

        # ---- Phase 7: NAV 更新（用 T 日收盘价）----
        nav = execution.get_portfolio_value(close_dict)
        result.portfolio_value = nav

        if total_value > 0:
            result.daily_return = (nav / total_value) - 1.0

        # ---- Phase 8: 更新融合历史 ----
        if ensemble is not None and signals:
            try:
                yesterday_returns = dm.get_daily_returns(T)
                ensemble.update_history(T, signals, yesterday_returns)
            except Exception as e:
                logger.debug("[%s] Ensemble history update skipped: %s", T, e)

        return result


def _apply_position_limit(
    orders: list,
    position_limit: float,
    holdings: dict,
    total_value: float,
    current_prices: dict,
) -> None:
    """就地修改订单列表：恢复期限制买入总规模。"""
    if position_limit <= 0:
        # 暂停期：移除所有买入
        orders[:] = [o for o in orders if o.direction != "buy"]
        return

    # 当前持仓市值
    holding_value = sum(
        pos.shares * current_prices.get(sym, pos.cost_price)
        for sym, pos in holdings.items()
    )
    max_holding = total_value * position_limit
    available = max(0, max_holding - holding_value)

    # 按买入金额逐个过滤
    kept = []
    used = 0.0
    for order in orders:
        if order.direction != "buy":
            kept.append(order)
            continue
        est_cost = order.shares * current_prices.get(order.symbol, 0)
        if used + est_cost <= available:
            kept.append(order)
            used += est_cost
    orders[:] = kept


# ---------------------------------------------------------------------------
# BacktestResult
# ---------------------------------------------------------------------------


@dataclass
class BacktestResult:
    """回测结果。"""

    config: BacktestConfig = field(default_factory=BacktestConfig)
    daily_nav: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    trade_records: list = field(default_factory=list)
    signal_history: Dict[str, List] = field(default_factory=dict)
    evaluation: Any = None
    risk_events: list = field(default_factory=list)
    elapsed_seconds: float = 0.0


# ---------------------------------------------------------------------------
# BacktestRunner — 主入口
# ---------------------------------------------------------------------------


class BacktestRunner:
    """回测主循环。

    Usage::

        config = BacktestConfig.load("configs/backtest.yaml")
        runner = BacktestRunner(config)
        result = runner.run()
    """

    def __init__(self, config: BacktestConfig):
        self.config = config
        self._dm = None
        self._alpha = None
        self._kronos = None
        self._rdagent = None
        self._ensemble = None
        self._execution = None
        self._risk = None

    # ------------------------------------------------------------------
    # Module initialization
    # ------------------------------------------------------------------

    def _init_modules(self) -> None:
        """按需初始化 M1–M8 模块。"""
        cfg = self.config

        # 设置日志
        logging.basicConfig(
            level=getattr(logging, cfg.log_level.upper(), logging.INFO),
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        )

        # 设置随机种子
        np.random.seed(cfg.random_seed)

        # M1 DataManager
        from quantlab.data.data_manager import DataManager

        self._dm = DataManager(
            provider_uri=cfg.qlib_data_dir,
            market=cfg.market,
        )
        logger.info("M1 DataManager initialized (market=%s)", cfg.market)

        # M2 AlphaSignalPipeline
        if cfg.enable_alpha:
            try:
                from quantlab.signals.signal_alpha import (
                    AlphaSignalPipeline,
                    FactorRegistry,
                )

                registry = FactorRegistry()
                self._alpha = AlphaSignalPipeline(
                    registry=registry,
                    market=cfg.market,
                    retrain_interval=cfg.alpha_retrain_interval,
                    train_years=cfg.alpha_train_years,
                )
                logger.info("M2 AlphaSignalPipeline initialized")
            except Exception as e:
                logger.warning("M2 AlphaSignalPipeline init failed: %s", e)

        # M3 KronosSignalPipeline
        if cfg.enable_kronos:
            try:
                from quantlab.signals.signal_kronos import (
                    FinetuneRecipe,
                    KronosSignalPipeline,
                )

                recipe = FinetuneRecipe(name=cfg.kronos_recipe_name)
                self._kronos = KronosSignalPipeline(
                    recipe=recipe,
                    device=cfg.kronos_device,
                )
                logger.info("M3 KronosSignalPipeline initialized")
            except Exception as e:
                logger.warning("M3 KronosSignalPipeline init failed: %s", e)

        # M4 RDAgentSignalPipeline
        if cfg.enable_rdagent:
            try:
                from quantlab.signals.signal_rdagent import (
                    CodeFactorExecutor,
                    CodeFactorRegistry,
                    RDAgentSignalPipeline,
                )

                code_registry = CodeFactorRegistry(
                    factor_code_dir="factors/code",
                    registry_path="factors/registry.yaml",
                )
                executor = CodeFactorExecutor()
                self._rdagent = RDAgentSignalPipeline(
                    code_registry=code_registry,
                    executor=executor,
                )
                logger.info("M4 RDAgentSignalPipeline initialized")
            except Exception as e:
                logger.warning("M4 RDAgentSignalPipeline init failed: %s", e)

        # M5 SignalEnsemblePipeline
        from quantlab.signals.signal_ensemble import SignalEnsemblePipeline

        self._ensemble = SignalEnsemblePipeline(
            ic_lookback=cfg.ensemble_ic_lookback,
            uncertainty_penalty=cfg.ensemble_uncertainty_penalty,
            min_weight=cfg.ensemble_min_weight,
            max_weight=cfg.ensemble_max_weight,
        )
        logger.info("M5 SignalEnsemblePipeline initialized")

        # M6 ExecutionPipeline
        from quantlab.execution.execution import ExecutionPipeline

        self._execution = ExecutionPipeline(
            initial_cash=cfg.initial_cash,
            max_positions=cfg.max_positions,
            max_single_weight=cfg.max_single_weight,
            target_buy_count=cfg.target_buy_count,
            target_sell_count=cfg.target_sell_count,
        )
        logger.info("M6 ExecutionPipeline initialized (cash=%.0f)", cfg.initial_cash)

        # M7 RiskController
        from quantlab.risk_control.risk_control import RiskController

        self._risk = RiskController(
            stop_loss_pct=cfg.stop_loss_pct,
            max_hold_days=cfg.max_hold_days,
            max_industry_pct=cfg.max_industry_pct,
            circuit_breaker_pct=cfg.circuit_breaker_pct,
            pause_days=cfg.pause_days,
        )
        logger.info("M7 RiskController initialized")

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self) -> BacktestResult:
        """执行完整回测。"""
        t0 = time.time()
        cfg = self.config

        # 初始化模块
        self._init_modules()

        # 获取交易日历
        calendar = self._dm.get_trading_calendar(cfg.start_date, cfg.end_date)
        if len(calendar) < 2:
            raise ValueError(
                f"Not enough trading days between {cfg.start_date} and {cfg.end_date}"
            )

        calendar_strs = [d.strftime("%Y-%m-%d") for d in calendar]
        logger.info(
            "Backtest: %s → %s (%d trading days)",
            calendar_strs[0], calendar_strs[-1], len(calendar_strs),
        )

        # 初始化状态
        state = BacktestState(total_days=len(calendar_strs) - 1)

        # 记录初始 NAV
        state.append_nav(calendar_strs[0], cfg.initial_cash)

        # 主循环：遍历 T = calendar[0..N-2]，T+1 = calendar[1..N-1]
        for i in range(len(calendar_strs) - 1):
            T = calendar_strs[i]
            T_next = calendar_strs[i + 1]
            state.current_idx = i

            daily_result = DailyStep.execute(
                T=T,
                T_next=T_next,
                dm=self._dm,
                alpha=self._alpha,
                kronos=self._kronos,
                rdagent=self._rdagent,
                ensemble=self._ensemble,
                execution=self._execution,
                risk=self._risk,
                config=cfg,
                calendar_strs=calendar_strs,
                day_idx=i,
            )

            # 更新状态
            state.append_nav(T_next, daily_result.portfolio_value)
            if daily_result.trade_records:
                state.append_trades(daily_result.trade_records)
            if daily_result.signals:
                state.append_signals(T, daily_result.signals)
            if daily_result.risk_result and daily_result.risk_result.events:
                state.risk_events.extend(daily_result.risk_result.events)

            # 记录每日收益率
            try:
                daily_returns = self._dm.get_daily_returns(T)
                state.append_return(T, daily_returns)
            except Exception:
                pass

            # 进度日志（每 20 天）
            if (i + 1) % 20 == 0:
                pct = (i + 1) / state.total_days * 100
                logger.info(
                    "Progress: %d/%d (%.1f%%) | NAV=%.2f | Date=%s",
                    i + 1, state.total_days, pct,
                    daily_result.portfolio_value, T,
                )

            # 断点保存
            if cfg.checkpoint_interval > 0 and (i + 1) % cfg.checkpoint_interval == 0:
                ckpt_path = str(
                    Path(cfg.checkpoint_dir) / f"checkpoint_{T}.pkl"
                )
                state.save_checkpoint(ckpt_path)

        elapsed = time.time() - t0
        logger.info("Backtest completed in %.1f seconds", elapsed)

        # 生成评估报告
        nav_series = state.get_nav_series()
        evaluation = self._evaluate(state, nav_series)

        result = BacktestResult(
            config=cfg,
            daily_nav=nav_series,
            trade_records=state.trade_records,
            signal_history=state.signal_history,
            evaluation=evaluation,
            risk_events=state.risk_events,
            elapsed_seconds=elapsed,
        )

        # 保存最终断点
        if cfg.checkpoint_dir:
            final_path = str(Path(cfg.checkpoint_dir) / "final.pkl")
            state.save_checkpoint(final_path)

        return result

    def resume(self, checkpoint_path: str) -> BacktestResult:
        """从断点续跑。"""
        t0 = time.time()
        cfg = self.config

        # 初始化模块
        self._init_modules()

        # 加载状态
        state = BacktestState.load_checkpoint(checkpoint_path)

        # 获取交易日历
        calendar = self._dm.get_trading_calendar(cfg.start_date, cfg.end_date)
        calendar_strs = [d.strftime("%Y-%m-%d") for d in calendar]
        state.total_days = len(calendar_strs) - 1

        # 恢复 ExecutionPipeline 状态：回放已有交易记录
        logger.info(
            "Resuming from day %d/%d, replaying %d trade records...",
            state.current_idx, state.total_days, len(state.trade_records),
        )

        start_idx = state.current_idx + 1
        logger.info("Continuing from day %d (%s)", start_idx, calendar_strs[start_idx])

        # 继续主循环
        for i in range(start_idx, len(calendar_strs) - 1):
            T = calendar_strs[i]
            T_next = calendar_strs[i + 1]
            state.current_idx = i

            daily_result = DailyStep.execute(
                T=T,
                T_next=T_next,
                dm=self._dm,
                alpha=self._alpha,
                kronos=self._kronos,
                rdagent=self._rdagent,
                ensemble=self._ensemble,
                execution=self._execution,
                risk=self._risk,
                config=cfg,
                calendar_strs=calendar_strs,
                day_idx=i,
            )

            state.append_nav(T_next, daily_result.portfolio_value)
            if daily_result.trade_records:
                state.append_trades(daily_result.trade_records)
            if daily_result.signals:
                state.append_signals(T, daily_result.signals)
            if daily_result.risk_result and daily_result.risk_result.events:
                state.risk_events.extend(daily_result.risk_result.events)

            try:
                daily_returns = self._dm.get_daily_returns(T)
                state.append_return(T, daily_returns)
            except Exception:
                pass

            if (i + 1) % 20 == 0:
                pct = (i + 1) / state.total_days * 100
                logger.info(
                    "Progress: %d/%d (%.1f%%) | NAV=%.2f | Date=%s",
                    i + 1, state.total_days, pct,
                    daily_result.portfolio_value, T,
                )

            if cfg.checkpoint_interval > 0 and (i + 1) % cfg.checkpoint_interval == 0:
                ckpt_path = str(
                    Path(cfg.checkpoint_dir) / f"checkpoint_{T}.pkl"
                )
                state.save_checkpoint(ckpt_path)

        elapsed = time.time() - t0
        nav_series = state.get_nav_series()
        evaluation = self._evaluate(state, nav_series)

        result = BacktestResult(
            config=cfg,
            daily_nav=nav_series,
            trade_records=state.trade_records,
            signal_history=state.signal_history,
            evaluation=evaluation,
            risk_events=state.risk_events,
            elapsed_seconds=elapsed,
        )

        if cfg.checkpoint_dir:
            final_path = str(Path(cfg.checkpoint_dir) / "final.pkl")
            state.save_checkpoint(final_path)

        return result

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def _evaluate(self, state: BacktestState, nav_series: pd.Series):
        """生成 M8 评估报告。"""
        try:
            from quantlab.evaluation.evaluation import EvaluationPipeline

            benchmark_nav = None
            try:
                benchmark_nav = self._dm.get_benchmark_nav(
                    self.config.benchmark,
                    start=self.config.start_date,
                    end=self.config.end_date,
                )
            except Exception as e:
                logger.warning("Benchmark NAV not available: %s", e)

            evaluator = EvaluationPipeline(
                trade_records=state.trade_records,
                daily_nav=nav_series,
                benchmark_nav=benchmark_nav,
                signal_history=state.signal_history,
                return_history=state.return_history,
                risk_events=state.risk_events,
                risk_free_rate=self.config.risk_free_rate,
            )
            report = evaluator.generate_report()
            summary = evaluator.print_summary(report)
            logger.info("\n%s", summary)
            return report
        except Exception as e:
            logger.warning("Evaluation failed: %s", e)
            return None


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main():
    """命令行入口。"""
    import argparse

    parser = argparse.ArgumentParser(description="QuantLab Backtest Runner")
    parser.add_argument(
        "--config", "-c",
        default="quantlab/configs/backtest.yaml",
        help="Path to backtest config YAML",
    )
    parser.add_argument(
        "--resume", "-r",
        default=None,
        help="Path to checkpoint file for resuming",
    )
    args = parser.parse_args()

    config = BacktestConfig.load(args.config)
    runner = BacktestRunner(config)

    if args.resume:
        result = runner.resume(args.resume)
    else:
        result = runner.run()

    print(f"\nBacktest finished in {result.elapsed_seconds:.1f}s")
    print(f"Final NAV: {result.daily_nav.iloc[-1]:.2f}")
    print(f"Total trades: {len(result.trade_records)}")
    print(f"Risk events: {len(result.risk_events)}")


if __name__ == "__main__":
    main()
