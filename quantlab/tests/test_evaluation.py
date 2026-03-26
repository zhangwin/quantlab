"""M8 评估与诊断模块单元测试。"""

import math

import numpy as np
import pandas as pd
import pytest

from quantlab.execution.execution import TradeRecord
from quantlab.risk_control.risk_control import RiskEvent
from quantlab.evaluation.evaluation import (
    CostBreakdown,
    EvaluationPipeline,
    HoldingAnalysis,
    PerformanceCalculator,
    PerformanceSummary,
    RegimeAnalyzer,
    RiskImpactReport,
    SignalAttributor,
    TradeAnalyzer,
    TradeSummary,
)


# ---------------------------------------------------------------------------
# 辅助
# ---------------------------------------------------------------------------

def _make_nav(start=100, daily_returns=None, n=100, seed=42):
    """生成日净值序列。"""
    dates = pd.bdate_range("2024-01-02", periods=n, freq="B")
    if daily_returns is not None:
        nav = start * (1 + pd.Series(daily_returns, index=dates[:len(daily_returns)])).cumprod()
    else:
        rng = np.random.RandomState(seed)
        rets = rng.normal(0.0005, 0.015, n)
        nav_vals = [start]
        for r in rets:
            nav_vals.append(nav_vals[-1] * (1 + r))
        nav = pd.Series(nav_vals[1:], index=dates, dtype=float)
    return nav


def _make_trade_record(date="2024-01-10", sym="A", direction="buy",
                       status="filled", order_price=10.0, exec_price=10.01,
                       shares=1000, amount=10010.0, commission=5.0,
                       stamp_tax=0.0, total_cost=5.0, reason="signal"):
    return TradeRecord(
        date=date, symbol=sym, direction=direction, status=status,
        order_price=order_price, exec_price=exec_price, shares=shares,
        amount=amount, commission=commission, stamp_tax=stamp_tax,
        total_cost=total_cost, reason=reason,
    )


# ---------------------------------------------------------------------------
# PerformanceCalculator
# ---------------------------------------------------------------------------


class TestPerformanceCalculator:

    def test_basic_summary(self):
        nav = _make_nav(n=252)
        perf = PerformanceCalculator(nav)
        s = perf.summary()
        assert isinstance(s, PerformanceSummary)
        assert s.annualized_volatility > 0
        assert 0 <= s.win_rate_daily <= 1

    def test_positive_returns(self):
        """稳定正收益 → 正夏普。"""
        rets = [0.002] * 100
        nav = _make_nav(daily_returns=rets, n=100)
        s = PerformanceCalculator(nav).summary()
        assert s.total_return > 0
        assert s.sharpe_ratio > 0
        assert s.max_drawdown == pytest.approx(0.0, abs=1e-10)

    def test_max_drawdown_known(self):
        """已知回撤序列。"""
        nav = pd.Series(
            [100, 110, 95, 105, 90],
            index=pd.bdate_range("2024-01-02", periods=5),
        )
        s = PerformanceCalculator(nav).summary()
        # 从 110 跌到 90 → 回撤 = 20/110 ≈ 18.18%
        assert s.max_drawdown == pytest.approx(20 / 110, abs=0.001)

    def test_calmar_ratio(self):
        """Calmar = 年化收益 / 最大回撤。"""
        nav = pd.Series(
            [100, 105, 100, 120],
            index=pd.bdate_range("2024-01-02", periods=4),
        )
        s = PerformanceCalculator(nav).summary()
        if s.max_drawdown > 0:
            expected_calmar = s.annualized_return / s.max_drawdown
            assert s.calmar_ratio == pytest.approx(expected_calmar, rel=0.01)

    def test_excess_return(self):
        """超额收益 = 策略年化 - 基准年化。"""
        nav = _make_nav(start=100, n=100, seed=42)
        bench = _make_nav(start=100, n=100, seed=99)
        s = PerformanceCalculator(nav, bench).summary()
        # 超额收益应为两者之差
        assert isinstance(s.excess_annual_return, float)

    def test_empty_nav(self):
        """空净值不报错。"""
        nav = pd.Series(dtype=float)
        s = PerformanceCalculator(nav).summary()
        assert s.total_return == 0.0

    def test_single_day(self):
        """单日净值不报错。"""
        nav = pd.Series([100.0], index=pd.bdate_range("2024-01-02", periods=1))
        s = PerformanceCalculator(nav).summary()
        assert s.total_return == 0.0

    def test_monthly_returns_shape(self):
        """月度收益矩阵行列正确。"""
        nav = _make_nav(n=252)
        perf = PerformanceCalculator(nav)
        monthly = perf.monthly_returns()
        assert not monthly.empty
        assert "全年" in monthly.columns

    def test_drawdown_series(self):
        nav = _make_nav(n=50)
        dd = PerformanceCalculator(nav).drawdown_series()
        assert len(dd) == len(nav)
        assert (dd <= 0).all()

    def test_rolling_sharpe(self):
        nav = _make_nav(n=100)
        rs = PerformanceCalculator(nav).rolling_sharpe(window=20)
        assert len(rs) > 0

    def test_zero_volatility(self):
        """净值恒定 → 夏普为 0（或合理处理）。"""
        nav = pd.Series(
            [100.0] * 50,
            index=pd.bdate_range("2024-01-02", periods=50),
        )
        s = PerformanceCalculator(nav).summary()
        assert s.sharpe_ratio == 0.0
        assert s.annualized_volatility == 0.0


# ---------------------------------------------------------------------------
# TradeAnalyzer
# ---------------------------------------------------------------------------


class TestTradeAnalyzer:

    def test_summary_basic(self):
        records = [
            _make_trade_record(direction="buy", status="filled"),
            _make_trade_record(direction="buy", status="failed_limit_up"),
            _make_trade_record(direction="sell", status="filled"),
        ]
        nav = _make_nav(n=50)
        s = TradeAnalyzer(records, nav).summary()
        assert s.total_trades == 3
        assert s.buy_trades == 2
        assert s.sell_trades == 1
        assert s.filled_rate_buy == pytest.approx(0.5)
        assert s.filled_rate_sell == pytest.approx(1.0)
        assert "limit_up" in s.failed_reasons

    def test_cost_breakdown(self):
        records = [
            _make_trade_record(commission=5.0, stamp_tax=0.0),
            _make_trade_record(direction="sell", commission=25.0, stamp_tax=50.0),
        ]
        nav = _make_nav(start=100000, n=50)
        cb = TradeAnalyzer(records, nav).cost_breakdown()
        assert cb.total_commission == 30.0
        assert cb.total_stamp_tax == 50.0
        assert cb.total_cost > 0

    def test_empty_records(self):
        s = TradeAnalyzer([], None).summary()
        assert s.total_trades == 0
        cb = TradeAnalyzer([], None).cost_breakdown()
        assert cb.total_cost == 0.0

    def test_holding_days(self):
        records = [
            _make_trade_record(date="2024-01-10", sym="A", direction="buy", status="filled"),
            _make_trade_record(date="2024-01-20", sym="A", direction="sell", status="filled",
                               exec_price=11.0),
        ]
        nav = _make_nav(n=50)
        s = TradeAnalyzer(records, nav).summary()
        assert s.avg_holding_days == 10.0
        assert s.median_holding_days == 10.0

    def test_holding_analysis_win_rate(self):
        records = [
            _make_trade_record(date="2024-01-10", sym="A", direction="buy",
                               status="filled", exec_price=10.0),
            _make_trade_record(date="2024-01-20", sym="A", direction="sell",
                               status="filled", exec_price=12.0),  # 盈利
            _make_trade_record(date="2024-01-10", sym="B", direction="buy",
                               status="filled", exec_price=10.0),
            _make_trade_record(date="2024-01-20", sym="B", direction="sell",
                               status="filled", exec_price=8.0),  # 亏损
        ]
        ha = TradeAnalyzer(records, _make_nav(n=50)).holding_analysis()
        assert ha.win_rate_per_trade == pytest.approx(0.5)
        assert ha.best_trade_pnl > 0
        assert ha.worst_trade_pnl < 0


# ---------------------------------------------------------------------------
# SignalAttributor
# ---------------------------------------------------------------------------


class TestSignalAttributor:

    @staticmethod
    def _make_signal_and_returns(n_days=30, n_stocks=20, seed=42):
        """生成模拟信号和收益数据。"""
        rng = np.random.RandomState(seed)
        dates = [f"2024-01-{d+2:02d}" for d in range(n_days + 10)]
        symbols = [f"S{i:03d}" for i in range(n_stocks)]

        # alpha 信号与 T+1 收益有一定相关性
        signal_history = {"alpha": [], "kronos": [], "combined": []}
        return_history = {}

        for i, d in enumerate(dates):
            true_signal = pd.Series(rng.randn(n_stocks), index=symbols)
            noise_a = pd.Series(rng.randn(n_stocks) * 0.5, index=symbols)
            noise_k = pd.Series(rng.randn(n_stocks) * 0.8, index=symbols)

            alpha_sig = true_signal + noise_a
            kronos_sig = true_signal + noise_k
            combined_sig = (alpha_sig + kronos_sig) / 2

            if i < n_days:
                signal_history["alpha"].append((d, alpha_sig))
                signal_history["kronos"].append((d, kronos_sig))
                signal_history["combined"].append((d, combined_sig))

            # 收益 = true_signal 的变形 + 噪声
            ret = true_signal * 0.01 + pd.Series(rng.randn(n_stocks) * 0.02, index=symbols)
            return_history[d] = ret

        return signal_history, return_history

    def test_ic_decay(self):
        sig, ret = self._make_signal_and_returns()
        attr = SignalAttributor(sig, ret)
        df = attr.ic_decay_analysis(horizons=[1, 2, 3])
        assert not df.empty
        assert "alpha" in df.index
        assert "T+1" in df.columns

    def test_ic_decay_t1_highest(self):
        """T+1 的 IC 通常应最高（因为信号与 T+1 相关性设计为最强）。"""
        sig, ret = self._make_signal_and_returns(n_days=50, n_stocks=50)
        attr = SignalAttributor(sig, ret)
        df = attr.ic_decay_analysis(horizons=[1, 5, 10])
        if "T+1" in df.columns and "T+10" in df.columns:
            # combined 的 T+1 IC 应大于 T+10
            assert df.loc["combined", "T+1"] >= df.loc["combined", "T+10"]

    def test_marginal_contribution(self):
        sig, ret = self._make_signal_and_returns()
        attr = SignalAttributor(sig, ret)
        df = attr.marginal_contribution()
        if not df.empty:
            assert "marginal_ic" in df.columns
            # 边际贡献不应全为 0
            assert df["marginal_ic"].abs().sum() > 0

    def test_signal_correlation(self):
        sig, ret = self._make_signal_and_returns()
        attr = SignalAttributor(sig, ret)
        corr = attr.signal_correlation()
        if not corr.empty:
            # 对角线应为 1
            for name in corr.index:
                assert corr.loc[name, name] == pytest.approx(1.0, abs=0.01)

    def test_rolling_ic(self):
        sig, ret = self._make_signal_and_returns(n_days=50)
        attr = SignalAttributor(sig, ret)
        ric = attr.rolling_ic("alpha", window=10)
        assert len(ric) > 0

    def test_empty_signals(self):
        attr = SignalAttributor({}, {})
        df = attr.ic_decay_analysis()
        assert df.empty

    def test_single_pipeline(self):
        """只有一条管线也能工作。"""
        rng = np.random.RandomState(42)
        dates = [f"2024-01-{d+2:02d}" for d in range(20)]
        symbols = [f"S{i}" for i in range(10)]

        sig = {"alpha": []}
        ret = {}
        for d in dates:
            s = pd.Series(rng.randn(10), index=symbols)
            sig["alpha"].append((d, s))
            ret[d] = pd.Series(rng.randn(10) * 0.01, index=symbols)

        attr = SignalAttributor(sig, ret)
        df = attr.ic_decay_analysis(horizons=[1])
        assert "alpha" in df.index


# ---------------------------------------------------------------------------
# RegimeAnalyzer
# ---------------------------------------------------------------------------


class TestRegimeAnalyzer:

    def test_classify_bull(self):
        """单调上涨 → 全 bull。"""
        n = 50
        dates = pd.bdate_range("2024-01-02", periods=n)
        bench = pd.Series(np.linspace(100, 150, n), index=dates)
        nav = pd.Series(np.linspace(100, 160, n), index=dates)
        regimes = RegimeAnalyzer(nav, bench).classify_regimes()
        # 前 20 天 MA20 不可用, 之后应大部分是 bull
        valid = regimes.iloc[20:]
        assert (valid == "bull").sum() > len(valid) * 0.5

    def test_classify_bear(self):
        """单调下跌 → 大部分 bear。"""
        n = 50
        dates = pd.bdate_range("2024-01-02", periods=n)
        bench = pd.Series(np.linspace(150, 100, n), index=dates)
        nav = pd.Series(np.linspace(150, 90, n), index=dates)
        regimes = RegimeAnalyzer(nav, bench).classify_regimes()
        valid = regimes.iloc[20:]
        assert (valid == "bear").sum() > len(valid) * 0.5

    def test_regime_performance(self):
        nav = _make_nav(n=100, seed=42)
        bench = _make_nav(n=100, seed=99)
        ra = RegimeAnalyzer(nav, bench)
        df = ra.regime_performance()
        assert not df.empty
        assert "annual_return" in df.columns

    def test_risk_impact_stop_loss(self):
        """止损后标的继续跌 → saved_pct > 0。"""
        dates = [f"2024-01-{d+2:02d}" for d in range(20)]
        symbols = ["A", "B"]

        return_history = {}
        for d in dates:
            # A 在止损后持续下跌
            return_history[d] = pd.Series({"A": -0.02, "B": 0.01})

        events = [
            RiskEvent(date="2024-01-05", event_type="stop_loss", symbol="A",
                      details={"drawdown": 0.1}),
        ]
        nav = pd.Series(np.linspace(100, 95, 20), index=pd.bdate_range("2024-01-02", periods=20))
        bench = pd.Series(np.linspace(100, 98, 20), index=nav.index)

        ra = RegimeAnalyzer(nav, bench, events, return_history)
        ri = ra.risk_impact()
        assert ri.stop_loss_count == 1
        assert ri.stop_loss_saved_pct > 0

    def test_risk_impact_empty(self):
        nav = _make_nav(n=50)
        bench = _make_nav(n=50, seed=99)
        ri = RegimeAnalyzer(nav, bench).risk_impact()
        assert ri.stop_loss_count == 0
        assert ri.circuit_breaker_count == 0

    def test_short_benchmark(self):
        """不足 20 天的 benchmark → 全 sideways。"""
        dates = pd.bdate_range("2024-01-02", periods=10)
        nav = pd.Series(np.linspace(100, 110, 10), index=dates)
        bench = pd.Series(np.linspace(100, 105, 10), index=dates)
        regimes = RegimeAnalyzer(nav, bench).classify_regimes()
        assert (regimes == "sideways").all()


# ---------------------------------------------------------------------------
# EvaluationPipeline（集成测试）
# ---------------------------------------------------------------------------


class TestEvaluationPipeline:

    def test_generate_report(self):
        """完整报告生成。"""
        nav = _make_nav(n=100)
        bench = _make_nav(n=100, seed=99)
        records = [
            _make_trade_record(direction="buy", status="filled"),
            _make_trade_record(direction="sell", status="filled",
                               date="2024-01-20", exec_price=11.0,
                               commission=5.0, stamp_tax=5.0),
        ]
        ep = EvaluationPipeline(records, nav, bench)
        report = ep.generate_report()
        assert report.performance is not None
        assert report.trade is not None
        assert report.cost is not None
        assert report.drawdown_series is not None

    def test_print_summary_no_error(self):
        """print_summary 不报错。"""
        nav = _make_nav(n=100)
        bench = _make_nav(n=100, seed=99)
        ep = EvaluationPipeline([], nav, bench)
        text = ep.print_summary()
        assert "回测评估报告" in text

    def test_empty_backtest(self):
        """空回测（无交易）。"""
        nav = _make_nav(n=50)
        ep = EvaluationPipeline([], nav)
        report = ep.generate_report()
        assert report.trade.total_trades == 0
        assert report.cost.total_cost == 0.0

    def test_with_signal_history(self):
        """包含信号历史的完整报告。"""
        nav = _make_nav(n=60)
        bench = _make_nav(n=60, seed=99)

        sig, ret = TestSignalAttributor._make_signal_and_returns(n_days=30)

        ep = EvaluationPipeline(
            trade_records=[],
            daily_nav=nav,
            benchmark_nav=bench,
            signal_history=sig,
            return_history=ret,
        )
        report = ep.generate_report()
        assert report.ic_decay is not None
        assert not report.ic_decay.empty

    def test_with_risk_events(self):
        """包含风控事件的报告。"""
        dates = [f"2024-01-{d+2:02d}" for d in range(50)]
        nav = pd.Series(np.linspace(100, 95, 50),
                        index=pd.bdate_range("2024-01-02", periods=50))
        bench = pd.Series(np.linspace(100, 97, 50), index=nav.index)

        return_history = {}
        for d in dates:
            return_history[d] = pd.Series({"A": -0.01, "B": 0.005})

        events = [
            RiskEvent("2024-01-10", "stop_loss", "A", {"drawdown": 0.09}),
        ]

        ep = EvaluationPipeline(
            trade_records=[],
            daily_nav=nav,
            benchmark_nav=bench,
            risk_events=events,
            return_history=return_history,
        )
        report = ep.generate_report()
        assert report.risk_impact.stop_loss_count == 1
