"""M8 评估与诊断：绩效 + 交易分析 + 信号归因 + 分环境诊断。

子模块：
    PerformanceCalculator   绩效指标计算（夏普/回撤/Calmar 等）
    TradeAnalyzer           交易统计分析（成交率/成本/持仓天数）
    SignalAttributor        信号归因（IC 衰减/边际贡献/相关矩阵）
    RegimeAnalyzer          分环境归因（牛/熊/震荡 + 风控有效性）
    EvaluationPipeline      评估入口（generate_report / print_summary）
"""

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from quantlab.execution.execution import TradeRecord
from quantlab.risk_control.risk_control import RiskEvent

logger = logging.getLogger(__name__)

TRADING_DAYS_PER_YEAR = 252


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------


@dataclass
class PerformanceSummary:
    """核心绩效指标。"""
    # 收益
    total_return: float = 0.0
    annualized_return: float = 0.0
    excess_annual_return: float = 0.0
    # 风险
    annualized_volatility: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_duration: int = 0
    downside_volatility: float = 0.0
    # 风险调整
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    information_ratio: float = 0.0
    # 胜率
    win_rate_daily: float = 0.0
    profit_loss_ratio: float = 0.0
    # 成本
    total_transaction_cost: float = 0.0
    cost_drag_annual: float = 0.0


@dataclass
class TradeSummary:
    """交易统计。"""
    total_trades: int = 0
    buy_trades: int = 0
    sell_trades: int = 0
    filled_rate_buy: float = 0.0
    filled_rate_sell: float = 0.0
    failed_reasons: Dict[str, int] = field(default_factory=dict)
    avg_daily_turnover: float = 0.0
    avg_holding_days: float = 0.0
    median_holding_days: float = 0.0


@dataclass
class CostBreakdown:
    """成本分解。"""
    total_commission: float = 0.0
    total_stamp_tax: float = 0.0
    total_slippage: float = 0.0
    total_cost: float = 0.0
    cost_per_trade: float = 0.0
    cost_as_pct_of_nav: float = 0.0
    annual_cost_drag: float = 0.0


@dataclass
class HoldingAnalysis:
    """持仓分析。"""
    avg_position_count: float = 0.0
    max_position_count: int = 0
    avg_concentration: float = 0.0
    industry_distribution: Dict[str, float] = field(default_factory=dict)
    win_rate_per_trade: float = 0.0
    best_trade_pnl: float = 0.0
    worst_trade_pnl: float = 0.0


@dataclass
class RiskImpactReport:
    """风控有效性报告。"""
    stop_loss_count: int = 0
    stop_loss_saved_pct: float = 0.0
    stop_loss_missed_pct: float = 0.0
    circuit_breaker_count: int = 0
    circuit_breaker_impact: float = 0.0


@dataclass
class EvaluationReport:
    """完整评估报告。"""
    performance: PerformanceSummary = field(default_factory=PerformanceSummary)
    trade: TradeSummary = field(default_factory=TradeSummary)
    cost: CostBreakdown = field(default_factory=CostBreakdown)
    holding: HoldingAnalysis = field(default_factory=HoldingAnalysis)
    ic_decay: Optional[pd.DataFrame] = None
    marginal_contribution: Optional[pd.DataFrame] = None
    signal_correlation: Optional[pd.DataFrame] = None
    regime: Optional[pd.DataFrame] = None
    risk_impact: RiskImpactReport = field(default_factory=RiskImpactReport)
    monthly: Optional[pd.DataFrame] = None
    rolling_sharpe: Optional[pd.Series] = None
    drawdown_series: Optional[pd.Series] = None


# ---------------------------------------------------------------------------
# PerformanceCalculator — 绩效指标
# ---------------------------------------------------------------------------


class PerformanceCalculator:
    """绩效指标计算器。

    Args:
        daily_nav: 日净值序列 (index=date, values=nav)
        benchmark_nav: 基准日净值序列 (可选)
        risk_free_rate: 年化无风险利率 (默认 0.02)
    """

    def __init__(
        self,
        daily_nav: pd.Series,
        benchmark_nav: Optional[pd.Series] = None,
        risk_free_rate: float = 0.02,
    ):
        self.daily_nav = daily_nav.sort_index()
        self.benchmark_nav = benchmark_nav.sort_index() if benchmark_nav is not None else None
        self.risk_free_rate = risk_free_rate

        self._daily_returns = self.daily_nav.pct_change().dropna()
        if self.benchmark_nav is not None:
            self._bench_returns = self.benchmark_nav.pct_change().dropna()
        else:
            self._bench_returns = pd.Series(dtype=float)

    def summary(self) -> PerformanceSummary:
        """计算核心绩效指标。"""
        nav = self.daily_nav
        ret = self._daily_returns
        n_days = len(ret)

        if n_days == 0:
            return PerformanceSummary()

        # 收益
        total_return = nav.iloc[-1] / nav.iloc[0] - 1
        ann_factor = TRADING_DAYS_PER_YEAR / n_days
        annualized_return = (1 + total_return) ** ann_factor - 1

        # 基准超额
        excess_annual = 0.0
        if self.benchmark_nav is not None and len(self.benchmark_nav) >= 2:
            bench_total = self.benchmark_nav.iloc[-1] / self.benchmark_nav.iloc[0] - 1
            bench_ann = (1 + bench_total) ** ann_factor - 1
            excess_annual = annualized_return - bench_ann

        # 波动率
        ann_vol = ret.std() * math.sqrt(TRADING_DAYS_PER_YEAR) if n_days > 1 else 0.0
        neg_ret = ret[ret < 0]
        downside_vol = neg_ret.std() * math.sqrt(TRADING_DAYS_PER_YEAR) if len(neg_ret) > 1 else 0.0

        # 最大回撤
        mdd, mdd_dur = self._calc_max_drawdown(nav)

        # 风险调整
        sharpe = (annualized_return - self.risk_free_rate) / ann_vol if ann_vol > 0 else 0.0
        sortino = (annualized_return - self.risk_free_rate) / downside_vol if downside_vol > 0 else 0.0
        calmar = annualized_return / mdd if mdd > 0 else 0.0

        # 信息比率
        info_ratio = 0.0
        if len(self._bench_returns) > 1:
            common = ret.index.intersection(self._bench_returns.index)
            if len(common) > 1:
                excess = ret.loc[common] - self._bench_returns.loc[common]
                te = excess.std()
                if te > 0:
                    info_ratio = excess.mean() / te * math.sqrt(TRADING_DAYS_PER_YEAR)

        # 胜率
        win_rate = (ret > 0).sum() / n_days if n_days > 0 else 0.0
        pos_ret = ret[ret > 0]
        neg_ret_abs = ret[ret < 0].abs()
        pl_ratio = pos_ret.mean() / neg_ret_abs.mean() if len(neg_ret_abs) > 0 and neg_ret_abs.mean() > 0 else 0.0

        return PerformanceSummary(
            total_return=total_return,
            annualized_return=annualized_return,
            excess_annual_return=excess_annual,
            annualized_volatility=ann_vol,
            max_drawdown=mdd,
            max_drawdown_duration=mdd_dur,
            downside_volatility=downside_vol,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            calmar_ratio=calmar,
            information_ratio=info_ratio,
            win_rate_daily=win_rate,
            profit_loss_ratio=pl_ratio,
        )

    def monthly_returns(self) -> pd.DataFrame:
        """月度收益矩阵（行=年, 列=1-12月+全年）。"""
        if len(self._daily_returns) == 0:
            return pd.DataFrame()

        ret = self._daily_returns.copy()
        ret.index = pd.to_datetime(ret.index)

        monthly = ret.groupby([ret.index.year, ret.index.month]).apply(
            lambda x: (1 + x).prod() - 1
        )
        monthly.index.names = ["year", "month"]
        monthly = monthly.unstack(level="month")
        monthly.columns = [f"{m}月" for m in monthly.columns]

        # 全年收益
        yearly = ret.groupby(ret.index.year).apply(lambda x: (1 + x).prod() - 1)
        monthly["全年"] = yearly

        return monthly

    def drawdown_series(self) -> pd.Series:
        """逐日回撤序列。"""
        peak = self.daily_nav.cummax()
        dd = (self.daily_nav - peak) / peak
        return dd

    def rolling_sharpe(self, window: int = 60) -> pd.Series:
        """滚动夏普比率。"""
        if len(self._daily_returns) < window:
            return pd.Series(dtype=float)

        daily_rf = self.risk_free_rate / TRADING_DAYS_PER_YEAR
        excess = self._daily_returns - daily_rf
        rolling_mean = excess.rolling(window).mean()
        rolling_std = excess.rolling(window).std()

        sharpe = (rolling_mean / rolling_std * math.sqrt(TRADING_DAYS_PER_YEAR)).dropna()
        return sharpe

    @staticmethod
    def _calc_max_drawdown(nav: pd.Series) -> Tuple[float, int]:
        """计算最大回撤及持续天数。"""
        if len(nav) < 2:
            return 0.0, 0

        peak = nav.cummax()
        dd = (nav - peak) / peak
        mdd = abs(dd.min())

        # 持续天数
        max_dur = 0
        cur_dur = 0
        for v in dd.values:
            if v < 0:
                cur_dur += 1
                max_dur = max(max_dur, cur_dur)
            else:
                cur_dur = 0

        return mdd, max_dur


# ---------------------------------------------------------------------------
# TradeAnalyzer — 交易统计
# ---------------------------------------------------------------------------


class TradeAnalyzer:
    """交易统计分析器。

    Args:
        trade_records: M6 交易记录列表
        daily_nav: 日净值序列（用于换手率计算）
    """

    def __init__(
        self,
        trade_records: List[TradeRecord],
        daily_nav: Optional[pd.Series] = None,
    ):
        self.records = trade_records
        self.daily_nav = daily_nav

    def summary(self) -> TradeSummary:
        """交易统计摘要。"""
        if not self.records:
            return TradeSummary()

        buys = [r for r in self.records if r.direction == "buy"]
        sells = [r for r in self.records if r.direction == "sell"]

        filled_buys = [r for r in buys if r.status in ("filled", "partial_filled")]
        filled_sells = [r for r in sells if r.status == "filled"]

        filled_rate_buy = len(filled_buys) / len(buys) if buys else 0.0
        filled_rate_sell = len(filled_sells) / len(sells) if sells else 0.0

        # 失败原因统计
        failed = [r for r in self.records if r.status.startswith("failed")]
        failed_reasons: Dict[str, int] = {}
        for r in failed:
            key = r.status.replace("failed_", "")
            failed_reasons[key] = failed_reasons.get(key, 0) + 1

        # 换手率
        avg_turnover = self._calc_avg_turnover()

        # 持仓天数（基于买卖配对）
        avg_days, median_days = self._calc_holding_days()

        return TradeSummary(
            total_trades=len(self.records),
            buy_trades=len(buys),
            sell_trades=len(sells),
            filled_rate_buy=filled_rate_buy,
            filled_rate_sell=filled_rate_sell,
            failed_reasons=failed_reasons,
            avg_daily_turnover=avg_turnover,
            avg_holding_days=avg_days,
            median_holding_days=median_days,
        )

    def cost_breakdown(self) -> CostBreakdown:
        """成本分解。"""
        filled = [r for r in self.records if r.status in ("filled", "partial_filled")]
        if not filled:
            return CostBreakdown()

        total_commission = sum(r.commission for r in filled)
        total_stamp_tax = sum(r.stamp_tax for r in filled)

        # 滑点估算：成交价与目标价的差额
        total_slippage = 0.0
        for r in filled:
            if r.exec_price > 0 and r.order_price > 0:
                slip = abs(r.exec_price - r.order_price) * r.shares
                total_slippage += slip

        total_cost = total_commission + total_stamp_tax + total_slippage
        cost_per_trade = total_cost / len(filled)

        # 成本占比
        initial_nav = self.daily_nav.iloc[0] if self.daily_nav is not None and len(self.daily_nav) > 0 else 0
        cost_pct = total_cost / initial_nav if initial_nav > 0 else 0.0

        # 年化成本拖累
        n_days = len(self.daily_nav) if self.daily_nav is not None else 0
        if n_days > 0 and initial_nav > 0:
            annual_drag = total_cost / initial_nav * (TRADING_DAYS_PER_YEAR / n_days)
        else:
            annual_drag = 0.0

        return CostBreakdown(
            total_commission=total_commission,
            total_stamp_tax=total_stamp_tax,
            total_slippage=total_slippage,
            total_cost=total_cost,
            cost_per_trade=cost_per_trade,
            cost_as_pct_of_nav=cost_pct,
            annual_cost_drag=annual_drag,
        )

    def holding_analysis(self) -> HoldingAnalysis:
        """持仓分析。"""
        filled = [r for r in self.records if r.status in ("filled", "partial_filled")]
        if not filled:
            return HoldingAnalysis()

        # 单笔交易胜率（用买卖配对的盈亏）
        sell_records = [r for r in filled if r.direction == "sell"]
        buy_records = [r for r in filled if r.direction == "buy"]

        # 构建买入成本表（简化：用 symbol 最近一次买入价）
        buy_prices: Dict[str, float] = {}
        for r in buy_records:
            buy_prices[r.symbol] = r.exec_price

        wins = 0
        total_sell = 0
        best_pnl = 0.0
        worst_pnl = 0.0
        for r in sell_records:
            bp = buy_prices.get(r.symbol, 0)
            if bp > 0 and r.exec_price > 0:
                pnl_pct = (r.exec_price - bp) / bp
                if pnl_pct > 0:
                    wins += 1
                best_pnl = max(best_pnl, pnl_pct)
                worst_pnl = min(worst_pnl, pnl_pct)
                total_sell += 1

        win_rate = wins / total_sell if total_sell > 0 else 0.0

        return HoldingAnalysis(
            win_rate_per_trade=win_rate,
            best_trade_pnl=best_pnl,
            worst_trade_pnl=worst_pnl,
        )

    def _calc_avg_turnover(self) -> float:
        """日均换手率 = 日均成交额 / 日均净值。"""
        if self.daily_nav is None or len(self.daily_nav) == 0:
            return 0.0

        filled = [r for r in self.records if r.status in ("filled", "partial_filled")]
        if not filled:
            return 0.0

        # 按日汇总成交额
        daily_amounts: Dict[str, float] = {}
        for r in filled:
            daily_amounts[r.date] = daily_amounts.get(r.date, 0) + r.amount

        if not daily_amounts:
            return 0.0

        avg_amount = sum(daily_amounts.values()) / len(self.daily_nav)
        avg_nav = self.daily_nav.mean()
        return avg_amount / avg_nav if avg_nav > 0 else 0.0

    def _calc_holding_days(self) -> Tuple[float, float]:
        """估算平均和中位数持仓天数。"""
        filled = [r for r in self.records if r.status in ("filled", "partial_filled")]
        buy_dates: Dict[str, str] = {}
        holding_days: List[int] = []

        for r in sorted(filled, key=lambda x: x.date):
            if r.direction == "buy":
                buy_dates[r.symbol] = r.date
            elif r.direction == "sell" and r.symbol in buy_dates:
                days = (pd.Timestamp(r.date) - pd.Timestamp(buy_dates[r.symbol])).days
                if days >= 0:
                    holding_days.append(days)
                del buy_dates[r.symbol]

        if not holding_days:
            return 0.0, 0.0

        return float(np.mean(holding_days)), float(np.median(holding_days))


# ---------------------------------------------------------------------------
# SignalAttributor — 信号归因
# ---------------------------------------------------------------------------


class SignalAttributor:
    """信号归因分析器。

    Args:
        signal_history: {pipeline_name: [(date, pd.Series), ...]}
        return_history: {date: pd.Series[symbol→return]}
    """

    def __init__(
        self,
        signal_history: Dict[str, List[Tuple[str, pd.Series]]],
        return_history: Dict[str, pd.Series],
    ):
        self.signal_history = signal_history
        self.return_history = return_history
        self._pipeline_names = [k for k in signal_history.keys() if signal_history[k]]

    def ic_decay_analysis(self, horizons: Optional[List[int]] = None) -> pd.DataFrame:
        """各管线 IC 衰减分析。

        Returns:
            DataFrame: 行=管线名, 列=T+h (IC值)
        """
        if horizons is None:
            horizons = [1, 2, 3, 5, 10]

        # 构建有序日期列表
        all_dates = sorted(self.return_history.keys())
        date_to_idx = {d: i for i, d in enumerate(all_dates)}

        results = {}
        for name in self._pipeline_names:
            ic_by_h = {}
            for h in horizons:
                ics = []
                for date, signal in self.signal_history[name]:
                    if date not in date_to_idx:
                        continue
                    idx = date_to_idx[date]
                    future_idx = idx + h
                    if future_idx >= len(all_dates):
                        continue
                    future_date = all_dates[future_idx]
                    future_ret = self.return_history.get(future_date)
                    if future_ret is None:
                        continue

                    common = signal.index.intersection(future_ret.index)
                    if len(common) < 5:
                        continue

                    corr, _ = sp_stats.spearmanr(
                        signal.loc[common].values,
                        future_ret.loc[common].values,
                    )
                    if not np.isnan(corr):
                        ics.append(corr)

                ic_by_h[f"T+{h}"] = np.mean(ics) if ics else 0.0
            results[name] = ic_by_h

        return pd.DataFrame(results).T if results else pd.DataFrame()

    def marginal_contribution(self) -> pd.DataFrame:
        """各管线边际贡献。

        对每条管线计算：去掉该管线后融合信号的 IC 下降多少。
        需要 signal_history 中包含 "combined" 键。
        """
        if "combined" not in self.signal_history or not self.signal_history["combined"]:
            return pd.DataFrame()

        all_dates = sorted(self.return_history.keys())
        date_to_idx = {d: i for i, d in enumerate(all_dates)}

        # 计算 combined 的 T+1 IC
        full_ic = self._calc_avg_ic("combined", 1, all_dates, date_to_idx)

        results = []
        for name in self._pipeline_names:
            if name == "combined":
                continue
            # 简化：用去掉该管线后剩余管线的均值作为"without"融合信号
            without_ic = self._calc_without_ic(name, all_dates, date_to_idx)
            marginal = full_ic - without_ic
            results.append({
                "pipeline": name,
                "full_ic": full_ic,
                "without_ic": without_ic,
                "marginal_ic": marginal,
            })

        df = pd.DataFrame(results)
        if not df.empty:
            df = df.set_index("pipeline")
        return df

    def signal_correlation(self) -> pd.DataFrame:
        """管线间信号相关矩阵（截面 Rank 相关的时间均值）。"""
        names = [n for n in self._pipeline_names if n != "combined"]
        if len(names) < 2:
            return pd.DataFrame()

        # 收集所有共同日期
        common_dates = None
        for name in names:
            dates_set = {d for d, _ in self.signal_history[name]}
            common_dates = dates_set if common_dates is None else common_dates & dates_set

        if not common_dates:
            return pd.DataFrame()

        # 按日期计算截面相关，取平均
        corr_sums = np.zeros((len(names), len(names)))
        count = 0

        # 建立日期到信号的快速索引
        sig_by_date = {}
        for name in names:
            sig_by_date[name] = {d: s for d, s in self.signal_history[name]}

        for date in sorted(common_dates):
            signals = {}
            for name in names:
                signals[name] = sig_by_date[name][date]

            # 找公共 symbol
            common_syms = None
            for s in signals.values():
                idx = set(s.index)
                common_syms = idx if common_syms is None else common_syms & idx

            if not common_syms or len(common_syms) < 5:
                continue

            common_syms = sorted(common_syms)
            for i, n1 in enumerate(names):
                for j, n2 in enumerate(names):
                    if i == j:
                        corr_sums[i][j] += 1.0
                    elif i < j:
                        c, _ = sp_stats.spearmanr(
                            signals[n1].loc[common_syms].values,
                            signals[n2].loc[common_syms].values,
                        )
                        if not np.isnan(c):
                            corr_sums[i][j] += c
                            corr_sums[j][i] += c
            count += 1

        if count == 0:
            return pd.DataFrame()

        corr_matrix = corr_sums / count
        return pd.DataFrame(corr_matrix, index=names, columns=names)

    def rolling_ic(self, pipeline_name: str, window: int = 60) -> pd.Series:
        """某管线的滚动 Rank IC（T+1）。"""
        if pipeline_name not in self.signal_history:
            return pd.Series(dtype=float)

        all_dates = sorted(self.return_history.keys())
        date_to_idx = {d: i for i, d in enumerate(all_dates)}

        daily_ics = {}
        for date, signal in self.signal_history[pipeline_name]:
            if date not in date_to_idx:
                continue
            idx = date_to_idx[date]
            if idx + 1 >= len(all_dates):
                continue
            next_date = all_dates[idx + 1]
            ret = self.return_history.get(next_date)
            if ret is None:
                continue
            common = signal.index.intersection(ret.index)
            if len(common) < 5:
                continue
            c, _ = sp_stats.spearmanr(signal.loc[common].values, ret.loc[common].values)
            if not np.isnan(c):
                daily_ics[date] = c

        if not daily_ics:
            return pd.Series(dtype=float)

        ic_series = pd.Series(daily_ics).sort_index()
        return ic_series.rolling(window, min_periods=max(1, window // 3)).mean()

    def _calc_avg_ic(
        self, name: str, horizon: int, all_dates: List[str], date_to_idx: Dict[str, int],
    ) -> float:
        ics = []
        for date, signal in self.signal_history[name]:
            if date not in date_to_idx:
                continue
            idx = date_to_idx[date]
            future_idx = idx + horizon
            if future_idx >= len(all_dates):
                continue
            future_ret = self.return_history.get(all_dates[future_idx])
            if future_ret is None:
                continue
            common = signal.index.intersection(future_ret.index)
            if len(common) < 5:
                continue
            c, _ = sp_stats.spearmanr(signal.loc[common].values, future_ret.loc[common].values)
            if not np.isnan(c):
                ics.append(c)
        return float(np.mean(ics)) if ics else 0.0

    def _calc_without_ic(
        self, exclude_name: str, all_dates: List[str], date_to_idx: Dict[str, int],
    ) -> float:
        """去掉 exclude_name 后，用剩余管线均值作为信号计算 IC。"""
        remaining = [n for n in self._pipeline_names if n not in (exclude_name, "combined")]
        if not remaining:
            return 0.0

        # 建立日期索引
        sigs_by_date: Dict[str, List[pd.Series]] = {}
        for name in remaining:
            for date, signal in self.signal_history[name]:
                sigs_by_date.setdefault(date, []).append(signal)

        ics = []
        for date, sig_list in sigs_by_date.items():
            if date not in date_to_idx:
                continue
            idx = date_to_idx[date]
            if idx + 1 >= len(all_dates):
                continue
            future_ret = self.return_history.get(all_dates[idx + 1])
            if future_ret is None:
                continue

            # 合并信号：取均值
            combined = pd.concat(sig_list, axis=1).mean(axis=1)
            common = combined.index.intersection(future_ret.index)
            if len(common) < 5:
                continue
            c, _ = sp_stats.spearmanr(combined.loc[common].values, future_ret.loc[common].values)
            if not np.isnan(c):
                ics.append(c)

        return float(np.mean(ics)) if ics else 0.0


# ---------------------------------------------------------------------------
# RegimeAnalyzer — 分环境归因
# ---------------------------------------------------------------------------


class RegimeAnalyzer:
    """分环境归因分析器。

    使用 benchmark 的 20 日均线判断市场环境：
    - bull: benchmark > MA20 × 1.02
    - bear: benchmark < MA20 × 0.98
    - sideways: 其他

    Args:
        daily_nav: 策略日净值
        benchmark_nav: 基准日净值
        risk_events: 风控事件列表 (可选)
        return_history: {date: pd.Series} 个股收益 (用于止损有效性分析)
    """

    def __init__(
        self,
        daily_nav: pd.Series,
        benchmark_nav: pd.Series,
        risk_events: Optional[List[RiskEvent]] = None,
        return_history: Optional[Dict[str, pd.Series]] = None,
    ):
        self.daily_nav = daily_nav.sort_index()
        self.benchmark_nav = benchmark_nav.sort_index()
        self.risk_events = risk_events or []
        self.return_history = return_history or {}

    def classify_regimes(self) -> pd.Series:
        """每日市场环境分类。"""
        bench = self.benchmark_nav
        if len(bench) < 20:
            return pd.Series("sideways", index=bench.index)

        ma20 = bench.rolling(20, min_periods=20).mean()

        regimes = pd.Series("sideways", index=bench.index)
        valid = ma20.notna()
        regimes.loc[valid & (bench > ma20 * 1.02)] = "bull"
        regimes.loc[valid & (bench < ma20 * 0.98)] = "bear"

        return regimes

    def regime_performance(self) -> pd.DataFrame:
        """分环境绩效。"""
        regimes = self.classify_regimes()
        nav_ret = self.daily_nav.pct_change().dropna()
        bench_ret = self.benchmark_nav.pct_change().dropna()

        common = nav_ret.index.intersection(bench_ret.index).intersection(regimes.index)
        if len(common) == 0:
            return pd.DataFrame()

        nav_ret = nav_ret.loc[common]
        bench_ret = bench_ret.loc[common]
        regimes = regimes.loc[common]

        results = []
        for regime in ["bull", "bear", "sideways"]:
            mask = regimes == regime
            r = nav_ret[mask]
            b = bench_ret[mask]
            n = len(r)

            if n == 0:
                continue

            total_ret = (1 + r).prod() - 1
            ann_factor = TRADING_DAYS_PER_YEAR / n if n > 0 else 1
            ann_ret = (1 + total_ret) ** ann_factor - 1 if n > 10 else total_ret

            vol = r.std() * math.sqrt(TRADING_DAYS_PER_YEAR) if n > 1 else 0
            sharpe = (ann_ret - 0.02) / vol if vol > 0 else 0

            cum = (1 + r).cumprod()
            peak = cum.cummax()
            mdd = ((cum - peak) / peak).min()
            mdd = abs(mdd) if not np.isnan(mdd) else 0

            win_rate = (r > 0).sum() / n

            bench_total = (1 + b).prod() - 1
            excess = total_ret - bench_total

            results.append({
                "regime": regime,
                "trading_days": n,
                "annual_return": ann_ret,
                "sharpe": sharpe,
                "max_drawdown": mdd,
                "win_rate": win_rate,
                "excess_return": excess,
            })

        df = pd.DataFrame(results)
        if not df.empty:
            df = df.set_index("regime")
        return df

    def risk_impact(self) -> RiskImpactReport:
        """风控事件有效性分析。"""
        stop_losses = [e for e in self.risk_events if e.event_type == "stop_loss"]
        breakers = [e for e in self.risk_events if e.event_type == "circuit_breaker"]

        # 止损有效性
        all_dates = sorted(self.return_history.keys())
        date_to_idx = {d: i for i, d in enumerate(all_dates)}

        saved_pcts = []
        missed_pcts = []

        for event in stop_losses:
            sym = event.symbol
            if sym is None or event.date not in date_to_idx:
                continue
            idx = date_to_idx[event.date]

            # 止损后 5 日累计收益
            future_ret = 0.0
            for h in range(1, 6):
                fi = idx + h
                if fi >= len(all_dates):
                    break
                fd = all_dates[fi]
                ret_series = self.return_history.get(fd)
                if ret_series is not None and sym in ret_series.index:
                    future_ret += ret_series[sym]

            if future_ret < 0:
                saved_pcts.append(abs(future_ret))
            else:
                missed_pcts.append(future_ret)

        # 熔断有效性
        cb_impacts = []
        for event in breakers:
            pause_until = event.details.get("pause_until")
            if not pause_until:
                continue

            # 暂停期间 benchmark 涨跌
            bench_ret = self.benchmark_nav.pct_change()
            mask = (bench_ret.index > event.date) & (bench_ret.index <= pause_until)
            period_ret = bench_ret[mask]
            if len(period_ret) > 0:
                cum_ret = (1 + period_ret).prod() - 1
                cb_impacts.append(cum_ret)

        return RiskImpactReport(
            stop_loss_count=len(stop_losses),
            stop_loss_saved_pct=float(np.mean(saved_pcts)) if saved_pcts else 0.0,
            stop_loss_missed_pct=float(np.mean(missed_pcts)) if missed_pcts else 0.0,
            circuit_breaker_count=len(breakers),
            circuit_breaker_impact=float(np.mean(cb_impacts)) if cb_impacts else 0.0,
        )


# ---------------------------------------------------------------------------
# EvaluationPipeline — 评估入口
# ---------------------------------------------------------------------------


class EvaluationPipeline:
    """评估入口：汇总所有子模块，生成完整报告。

    Usage::

        evaluator = EvaluationPipeline(
            trade_records=records,
            daily_nav=nav_series,
            benchmark_nav=bench_series,
        )
        report = evaluator.generate_report()
        evaluator.print_summary()
    """

    def __init__(
        self,
        trade_records: List[TradeRecord],
        daily_nav: pd.Series,
        benchmark_nav: Optional[pd.Series] = None,
        signal_history: Optional[Dict[str, List[Tuple[str, pd.Series]]]] = None,
        return_history: Optional[Dict[str, pd.Series]] = None,
        risk_events: Optional[List[RiskEvent]] = None,
        risk_free_rate: float = 0.02,
    ):
        self.trade_records = trade_records
        self.daily_nav = daily_nav
        self.benchmark_nav = benchmark_nav
        self.signal_history = signal_history or {}
        self.return_history = return_history or {}
        self.risk_events = risk_events or []
        self.risk_free_rate = risk_free_rate

    def generate_report(self) -> EvaluationReport:
        """生成完整评估报告。"""
        # 绩效
        perf_calc = PerformanceCalculator(
            self.daily_nav, self.benchmark_nav, self.risk_free_rate,
        )
        performance = perf_calc.summary()
        monthly = perf_calc.monthly_returns()
        dd_series = perf_calc.drawdown_series()
        rolling_s = perf_calc.rolling_sharpe()

        # 交易分析
        trade_analyzer = TradeAnalyzer(self.trade_records, self.daily_nav)
        trade_summary = trade_analyzer.summary()
        cost = trade_analyzer.cost_breakdown()
        holding = trade_analyzer.holding_analysis()

        # 更新绩效中的成本字段
        performance.total_transaction_cost = cost.total_cost
        performance.cost_drag_annual = cost.annual_cost_drag

        # 信号归因
        ic_decay = None
        marginal = None
        sig_corr = None
        if self.signal_history:
            attr = SignalAttributor(self.signal_history, self.return_history)
            ic_decay = attr.ic_decay_analysis()
            marginal = attr.marginal_contribution()
            sig_corr = attr.signal_correlation()

        # 分环境归因
        regime_df = None
        risk_impact = RiskImpactReport()
        if self.benchmark_nav is not None and len(self.benchmark_nav) > 0:
            regime_analyzer = RegimeAnalyzer(
                self.daily_nav, self.benchmark_nav,
                self.risk_events, self.return_history,
            )
            regime_df = regime_analyzer.regime_performance()
            risk_impact = regime_analyzer.risk_impact()

        return EvaluationReport(
            performance=performance,
            trade=trade_summary,
            cost=cost,
            holding=holding,
            ic_decay=ic_decay,
            marginal_contribution=marginal,
            signal_correlation=sig_corr,
            regime=regime_df,
            risk_impact=risk_impact,
            monthly=monthly,
            rolling_sharpe=rolling_s,
            drawdown_series=dd_series,
        )

    def print_summary(self, report: Optional[EvaluationReport] = None):
        """打印摘要到控制台。"""
        if report is None:
            report = self.generate_report()

        p = report.performance
        t = report.trade
        c = report.cost
        ri = report.risk_impact

        lines = [
            "========== 回测评估报告 ==========",
            "【绩效】",
            f"  年化收益: {p.annualized_return:.1%}    夏普: {p.sharpe_ratio:.2f}"
            f"    最大回撤: {p.max_drawdown:.1%}    Calmar: {p.calmar_ratio:.2f}",
            f"  超额收益: {p.excess_annual_return:.1%}    信息比率: {p.information_ratio:.2f}",
            f"  日胜率: {p.win_rate_daily:.1%}      盈亏比: {p.profit_loss_ratio:.2f}",
            "",
            "【交易】",
            f"  总交易: {t.total_trades} 笔   买入成交率: {t.filled_rate_buy:.1%}"
            f"   日均换手: {t.avg_daily_turnover:.1%}",
            f"  平均持仓: {t.avg_holding_days:.1f} 天   单笔胜率: {report.holding.win_rate_per_trade:.1%}",
            f"  总成本: ¥{c.total_cost:,.0f} (年化 {c.annual_cost_drag:.2%})",
        ]

        # 信号质量
        if report.ic_decay is not None and not report.ic_decay.empty:
            lines.append("")
            lines.append("【信号质量】")
            lines.append("  管线        IC(T+1)  边际贡献")
            for name in report.ic_decay.index:
                ic1 = report.ic_decay.loc[name].get("T+1", 0)
                mc = 0.0
                if report.marginal_contribution is not None and name in report.marginal_contribution.index:
                    mc = report.marginal_contribution.loc[name].get("marginal_ic", 0)
                lines.append(f"  {name:<12} {ic1:>6.3f}    {mc:>6.3f}")

        # 分环境
        if report.regime is not None and not report.regime.empty:
            lines.append("")
            lines.append("【分环境】")
            for regime in report.regime.index:
                row = report.regime.loc[regime]
                lines.append(
                    f"  {regime:<8}: {row['annual_return']:>+.1%}"
                    f" (超额{row['excess_return']:>+.1%})"
                )

        # 风控
        lines.append("")
        lines.append("【风控】")
        if ri.stop_loss_count > 0:
            lines.append(
                f"  止损 {ri.stop_loss_count} 次: "
                f"平均避损 {ri.stop_loss_saved_pct:.1%}，"
                f"过度止损率 {ri.stop_loss_missed_pct:.1%}"
            )
        else:
            lines.append("  止损 0 次")
        if ri.circuit_breaker_count > 0:
            direction = "下跌" if ri.circuit_breaker_impact < 0 else "上涨"
            lines.append(
                f"  熔断 {ri.circuit_breaker_count} 次: "
                f"暂停期间市场{direction} {abs(ri.circuit_breaker_impact):.1%}"
            )
        else:
            lines.append("  熔断 0 次")

        lines.append("===================================")

        summary_text = "\n".join(lines)
        print(summary_text)
        return summary_text
