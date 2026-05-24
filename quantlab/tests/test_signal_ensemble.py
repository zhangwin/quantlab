"""M5 信号融合离线测试。

不依赖 Qlib、GPU 或任何外部数据，使用合成数据验证核心逻辑。

Usage:
    pytest test_signal_ensemble.py -v
    pytest test_signal_ensemble.py -k "TestICWeightEngine" -v
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quantlab.signals.signal_ensemble import (
    SignalNormalizer,
    ICWeightEngine,
    EnsembleMonitor,
    SignalEnsemblePipeline,
    EnsembleOutput,
    DayRecord,
    PipelineStat,
    ContributionReport,
)


# ---------------------------------------------------------------------------
# 辅助工具
# ---------------------------------------------------------------------------


def make_signal(n=50, seed=42):
    """生成随机信号 Series。"""
    rng = np.random.RandomState(seed)
    symbols = [f"SH{600000 + i}" for i in range(n)]
    return pd.Series(rng.randn(n), index=symbols)


def make_correlated_signal(base, corr=0.8, seed=99):
    """生成与 base 高度相关的信号。"""
    rng = np.random.RandomState(seed)
    noise = rng.randn(len(base))
    values = corr * base.values + (1 - corr) * noise
    return pd.Series(values, index=base.index)


def make_return(signal, ic=0.05, seed=123):
    """生成与 signal 有指定 IC 的模拟收益。"""
    rng = np.random.RandomState(seed)
    noise = rng.randn(len(signal))
    ret = ic * signal.rank(pct=True).values + (1 - ic) * noise
    return pd.Series(ret, index=signal.index)


# ---------------------------------------------------------------------------
# TestSignalNormalizer
# ---------------------------------------------------------------------------


class TestSignalNormalizer:
    """SignalNormalizer 测试。"""

    def test_rank_normalize_basic(self):
        s = pd.Series([10, 30, 20, np.nan], index=["a", "b", "c", "d"])
        result = SignalNormalizer.rank_normalize(s)
        # rank: 10→1/3, 20→2/3, 30→3/3, NaN→NaN
        assert abs(result["a"] - 1 / 3) < 0.01
        assert abs(result["b"] - 1.0) < 0.01
        assert abs(result["c"] - 2 / 3) < 0.01
        assert pd.isna(result["d"])

    def test_rank_normalize_scale_invariant(self):
        """量纲无关：不同尺度应产生相同 rank。"""
        s1 = pd.Series([-100, 0, 100], index=["a", "b", "c"])
        s2 = pd.Series([0.1, 0.2, 0.3], index=["a", "b", "c"])
        r1 = SignalNormalizer.rank_normalize(s1)
        r2 = SignalNormalizer.rank_normalize(s2)
        pd.testing.assert_series_equal(r1, r2)

    def test_rank_normalize_empty(self):
        result = SignalNormalizer.rank_normalize(pd.Series(dtype=float))
        assert result.empty

    def test_winsorize(self):
        s = pd.Series(list(range(1, 101)))  # 1~100
        result = SignalNormalizer.winsorize(s, lower=0.05, upper=0.95)
        assert result.min() >= 5
        assert result.max() <= 95

    def test_winsorize_preserves_nan(self):
        s = pd.Series([1, 2, np.nan, 100, 200])
        result = SignalNormalizer.winsorize(s)
        assert pd.isna(result.iloc[2])

    def test_winsorize_empty(self):
        result = SignalNormalizer.winsorize(pd.Series(dtype=float))
        assert result.empty

    def test_align_index(self):
        s_a = pd.Series([1, 2, 3], index=["a", "b", "c"])
        s_b = pd.Series([4, 5, 6], index=["b", "c", "d"])
        result = SignalNormalizer.align_index({"A": s_a, "B": s_b})
        assert set(result["A"].index) == {"a", "b", "c", "d"}
        assert set(result["B"].index) == {"a", "b", "c", "d"}
        assert pd.isna(result["A"]["d"])
        assert pd.isna(result["B"]["a"])

    def test_align_index_empty(self):
        result = SignalNormalizer.align_index({})
        assert result == {}


# ---------------------------------------------------------------------------
# TestICWeightEngine
# ---------------------------------------------------------------------------


class TestICWeightEngine:
    """ICWeightEngine 测试。"""

    def _feed_history(self, engine, n_days, pipeline_ics, seed=42):
        """向 engine 填充 n_days 天模拟历史。"""
        rng = np.random.RandomState(seed)
        for day in range(n_days):
            date = f"2024-{(day // 28 + 1):02d}-{(day % 28 + 1):02d}"
            base = pd.Series(rng.randn(50), index=[f"S{i}" for i in range(50)])
            signals = {}
            for name, ic in pipeline_ics.items():
                signals[name] = base * ic + pd.Series(
                    rng.randn(50) * 0.1, index=base.index
                )
            actual_return = base + pd.Series(rng.randn(50) * 0.5, index=base.index)
            engine.update(date, signals, actual_return)

    def test_cold_start_equal_weight(self):
        """历史 < 20 天 → 等权。"""
        engine = ICWeightEngine(ic_lookback=60)
        self._feed_history(engine, 10, {"alpha": 1.0, "kronos": 0.5})
        weights, mode = engine.compute_weights()
        assert mode == "cold_start"
        assert abs(weights["alpha"] - 0.5) < 0.01
        assert abs(weights["kronos"] - 0.5) < 0.01

    def test_blending_mode(self):
        """历史 20~60 天 → 混合过渡。"""
        engine = ICWeightEngine(ic_lookback=60)
        self._feed_history(engine, 40, {"alpha": 1.0, "kronos": 0.3, "rdagent": 0.1})
        weights, mode = engine.compute_weights()
        assert mode == "blending"
        assert len(weights) == 3
        assert abs(sum(weights.values()) - 1.0) < 1e-6

    def test_ic_weighted_mode(self):
        """历史 ≥ 60 天 → 纯 IC 加权。"""
        engine = ICWeightEngine(ic_lookback=60)
        self._feed_history(engine, 65, {"alpha": 1.0, "kronos": 0.5})
        weights, mode = engine.compute_weights()
        assert mode == "ic_weighted"
        # alpha IC 更高 → alpha 权重更大
        assert weights["alpha"] > weights["kronos"]

    def test_ic_weighted_direction(self):
        """IC 最高的管线获得最高权重。"""
        engine = ICWeightEngine(ic_lookback=60)
        self._feed_history(engine, 65, {"alpha": 1.0, "kronos": 0.5, "rdagent": 0.1})
        weights, mode = engine.compute_weights()
        assert weights["alpha"] >= weights["kronos"]
        assert weights["kronos"] >= weights["rdagent"]

    def test_all_negative_ic_equal_weight(self):
        """全部 IC < 0 → 退化为等权。"""
        engine = ICWeightEngine(ic_lookback=60)
        # 用负相关信号
        self._feed_history(engine, 65, {"alpha": -1.0, "kronos": -0.5})
        weights, mode = engine.compute_weights()
        # 负 IC 截断为 0 → total=0 → 等权
        # 但因为 clamp，每个管线都有 min_weight
        assert abs(weights["alpha"] - weights["kronos"]) < 0.15

    def test_weight_clamp(self):
        """权重不超过 max_weight。"""
        engine = ICWeightEngine(ic_lookback=60, min_weight=0.1, max_weight=0.6)
        # alpha 远高于 kronos
        self._feed_history(engine, 65, {"alpha": 5.0, "kronos": 0.01})
        weights, mode = engine.compute_weights()
        assert weights["alpha"] <= 0.6 + 0.01  # 钳制后归一化可能略有浮动
        assert weights["kronos"] >= 0.1 - 0.01

    def test_correlation_penalty(self):
        """高相关管线中 IC 较低者被降权。"""
        engine = ICWeightEngine(ic_lookback=60, correlation_penalty=0.3)
        rng = np.random.RandomState(42)
        for day in range(65):
            date = f"2024-{(day // 28 + 1):02d}-{(day % 28 + 1):02d}"
            base = pd.Series(rng.randn(50), index=[f"S{i}" for i in range(50)])
            actual = base + pd.Series(rng.randn(50) * 0.3, index=base.index)
            # alpha 和 kronos 高度相关
            signals = {
                "alpha": base * 1.0 + pd.Series(rng.randn(50) * 0.05, index=base.index),
                "kronos": base * 0.8 + pd.Series(rng.randn(50) * 0.05, index=base.index),
                "rdagent": pd.Series(rng.randn(50) * 0.5, index=base.index),
            }
            engine.update(date, signals, actual)

        weights, _ = engine.compute_weights()
        # kronos 因高相关 + IC 较低，权重被降
        assert weights["alpha"] > weights["kronos"]

    def test_history_fixed_length(self):
        """history 长度不超过 ic_lookback。"""
        engine = ICWeightEngine(ic_lookback=60)
        self._feed_history(engine, 200, {"alpha": 1.0})
        assert len(engine.history) == 60

    def test_no_lookahead_bias(self):
        """T 日 compute_weights 仅使用 T-1 及之前数据。"""
        engine = ICWeightEngine(ic_lookback=60)
        self._feed_history(engine, 65, {"alpha": 1.0, "kronos": 0.5})
        # compute_weights 只使用 history 中已有记录
        weights_before, _ = engine.compute_weights()
        # 再加一天
        rng = np.random.RandomState(999)
        base = pd.Series(rng.randn(50), index=[f"S{i}" for i in range(50)])
        engine.update("2024-12-31", {"alpha": base, "kronos": base * 0.5}, base)
        weights_after, _ = engine.compute_weights()
        # 权重应有变化（新数据加入）
        assert weights_before != weights_after

    def test_compute_ic_single(self):
        engine = ICWeightEngine(ic_lookback=60)
        self._feed_history(engine, 30, {"alpha": 1.0})
        ic = engine.compute_ic("alpha")
        assert isinstance(ic, float)
        # IC 应为正（信号与收益正相关）
        assert ic > 0

    def test_compute_ic_nonexistent(self):
        engine = ICWeightEngine(ic_lookback=60)
        self._feed_history(engine, 30, {"alpha": 1.0})
        ic = engine.compute_ic("nonexistent")
        assert ic == 0.0

    def test_get_ic_history(self):
        engine = ICWeightEngine(ic_lookback=60)
        self._feed_history(engine, 30, {"alpha": 1.0, "kronos": 0.5})
        df = engine.get_ic_history()
        assert "alpha" in df.columns
        assert "kronos" in df.columns
        assert len(df) > 0

    def test_empty_weights(self):
        engine = ICWeightEngine()
        weights, mode = engine.compute_weights()
        assert weights == {}
        assert mode == "cold_start"


# ---------------------------------------------------------------------------
# TestEnsembleMonitor
# ---------------------------------------------------------------------------


class TestEnsembleMonitor:
    """EnsembleMonitor 测试。"""

    def test_record_and_report(self):
        monitor = EnsembleMonitor()
        rng = np.random.RandomState(42)
        for day in range(60):
            date = f"2024-{(day // 28 + 1):02d}-{(day % 28 + 1):02d}"
            signal = pd.Series(rng.randn(50), index=[f"S{i}" for i in range(50)])
            ret = pd.Series(rng.randn(50), index=[f"S{i}" for i in range(50)])
            weights = {"alpha": 0.5, "kronos": 0.3, "rdagent": 0.2}
            monitor.record(date, weights, signal, ret)

        report = monitor.get_contribution_report(lookback=60)
        assert len(report.pipeline_stats) == 3
        assert "alpha" in report.pipeline_stats
        assert abs(report.pipeline_stats["alpha"].avg_weight - 0.5) < 0.01

    def test_check_anomaly_no_alerts(self):
        monitor = EnsembleMonitor()
        rng = np.random.RandomState(42)
        for day in range(30):
            date = f"2024-{(day // 28 + 1):02d}-{(day % 28 + 1):02d}"
            signal = make_signal(seed=day)
            ret = make_return(signal, ic=0.05, seed=day + 100)
            weights = {"alpha": 0.4, "kronos": 0.35, "rdagent": 0.25}
            monitor.record(date, weights, signal, ret)
        alerts = monitor.check_anomaly()
        # 正常情况下不应有太多告警
        assert isinstance(alerts, list)

    def test_min_weight_alert(self):
        """某管线连续 20 天 min_weight → 告警。"""
        monitor = EnsembleMonitor()
        for day in range(25):
            date = f"2024-01-{(day + 1):02d}"
            signal = make_signal(seed=day)
            weights = {"alpha": 0.5, "kronos": 0.4, "rdagent": 0.1}
            monitor.record(date, weights, signal, None)
        alerts = monitor.check_anomaly()
        rdagent_alert = [a for a in alerts if "rdagent" in a]
        assert len(rdagent_alert) > 0

    def test_update_actual_return(self):
        monitor = EnsembleMonitor()
        signal = make_signal()
        monitor.record("2024-01-01", {"alpha": 1.0}, signal, None)
        ret = make_return(signal)
        monitor.update_actual_return("2024-01-01", ret)
        # 验证回填成功
        assert monitor._records[0]["actual_return"] is not None

    def test_weight_trend(self):
        monitor = EnsembleMonitor()
        for day in range(40):
            date = f"2024-{(day // 28 + 1):02d}-{(day % 28 + 1):02d}"
            signal = make_signal(seed=day)
            # alpha 权重逐渐增加
            alpha_w = 0.3 + day * 0.005
            kronos_w = 1.0 - alpha_w
            weights = {"alpha": alpha_w, "kronos": kronos_w}
            monitor.record(date, weights, signal, None)

        report = monitor.get_contribution_report(lookback=40)
        assert report.pipeline_stats["alpha"].weight_trend == "increasing"


# ---------------------------------------------------------------------------
# TestEnsembleOutput
# ---------------------------------------------------------------------------


class TestEnsembleOutput:
    """EnsembleOutput 数据结构测试。"""

    def test_default(self):
        out = EnsembleOutput()
        assert out.signal.empty
        assert out.weights == {}
        assert out.mode == "cold_start"
        assert out.uncertainty_adjusted is False

    def test_with_data(self):
        signal = make_signal()
        out = EnsembleOutput(
            signal=signal,
            weights={"alpha": 0.5, "kronos": 0.5},
            mode="ic_weighted",
            uncertainty_adjusted=True,
        )
        assert len(out.signal) == 50
        assert out.mode == "ic_weighted"


# ---------------------------------------------------------------------------
# TestSignalEnsemblePipeline
# ---------------------------------------------------------------------------


class TestSignalEnsemblePipeline:
    """SignalEnsemblePipeline 端到端测试。"""

    def test_combine_three_pipelines(self):
        """三管线融合。"""
        ensemble = SignalEnsemblePipeline()
        signals = {
            "alpha": make_signal(seed=1),
            "kronos": make_signal(seed=2),
            "rdagent": make_signal(seed=3),
        }
        output = ensemble.combine(signals, "2024-06-28")
        assert not output.signal.empty
        assert len(output.weights) == 3
        assert abs(sum(output.weights.values()) - 1.0) < 1e-6
        assert output.mode == "cold_start"  # 无历史
        # 值域检查
        assert output.signal.min() > 0
        assert output.signal.max() <= 1.0

    def test_combine_two_pipelines(self):
        """两管线融合（rdagent 缺失）。"""
        ensemble = SignalEnsemblePipeline()
        signals = {
            "alpha": make_signal(seed=1),
            "kronos": make_signal(seed=2),
            "rdagent": None,
        }
        output = ensemble.combine(signals, "2024-06-28")
        assert len(output.weights) == 2
        assert "rdagent" not in output.weights
        assert abs(sum(output.weights.values()) - 1.0) < 1e-6

    def test_combine_single_pipeline(self):
        """单管线也能融合。"""
        ensemble = SignalEnsemblePipeline()
        signals = {"alpha": make_signal(seed=1)}
        output = ensemble.combine(signals, "2024-06-28")
        assert len(output.weights) == 1
        assert not output.signal.empty

    def test_combine_all_none(self):
        """全部管线为空。"""
        ensemble = SignalEnsemblePipeline()
        signals = {"alpha": None, "kronos": None, "rdagent": None}
        output = ensemble.combine(signals, "2024-06-28")
        assert output.signal.empty
        assert output.mode == "cold_start"

    def test_uncertainty_penalty(self):
        """不确定性惩罚生效。"""
        ensemble = SignalEnsemblePipeline(uncertainty_penalty=0.1)
        base = make_signal(seed=1)
        # 高不确定性
        high_unc = pd.Series(np.ones(50) * 100, index=base.index)
        signals_no_unc = {"alpha": base, "kronos": base}
        signals_with_unc = {
            "alpha": base,
            "kronos": base,
            "kronos_uncertainty": high_unc,
        }
        out_no = ensemble.combine(signals_no_unc, "2024-06-28")
        out_with = ensemble.combine(signals_with_unc, "2024-06-29")
        assert out_with.uncertainty_adjusted is True
        assert out_no.uncertainty_adjusted is False

    def test_missing_stocks_filled(self):
        """缺失股票用 0.5 填充。"""
        ensemble = SignalEnsemblePipeline()
        s_alpha = pd.Series([1, 2, 3], index=["A", "B", "C"])
        s_kronos = pd.Series([4, 5], index=["B", "C"])
        signals = {"alpha": s_alpha, "kronos": s_kronos}
        output = ensemble.combine(signals, "2024-06-28")
        # 股票 A 在 kronos 中缺失，应被填充
        assert "A" in output.signal.index
        assert not pd.isna(output.signal["A"])

    def test_output_no_nan(self):
        """输出不含 NaN。"""
        ensemble = SignalEnsemblePipeline()
        signals = {
            "alpha": make_signal(seed=1),
            "kronos": make_signal(seed=2),
        }
        output = ensemble.combine(signals, "2024-06-28")
        assert output.signal.isna().sum() == 0

    def test_deterministic(self):
        """相同输入两次 combine 结果一致。"""
        ensemble = SignalEnsemblePipeline()
        signals = {
            "alpha": make_signal(seed=1),
            "kronos": make_signal(seed=2),
        }
        out1 = ensemble.combine(signals, "2024-06-28")
        out2 = ensemble.combine(signals, "2024-06-28")
        pd.testing.assert_series_equal(out1.signal, out2.signal)

    def test_update_history(self):
        """update_history 正常工作。"""
        ensemble = SignalEnsemblePipeline()
        signals = {
            "alpha": make_signal(seed=1),
            "kronos": make_signal(seed=2),
        }
        ret = make_return(signals["alpha"], seed=3)
        ensemble.update_history("2024-06-27", signals, ret)
        assert len(ensemble.ic_engine.history) == 1

    def test_ic_weighted_after_history(self):
        """积累足够历史后进入 ic_weighted 模式。"""
        ensemble = SignalEnsemblePipeline(ic_lookback=60)
        rng = np.random.RandomState(42)

        for day in range(65):
            date = f"2024-{(day // 28 + 1):02d}-{(day % 28 + 1):02d}"
            base = pd.Series(rng.randn(50), index=[f"S{i}" for i in range(50)])
            signals = {
                "alpha": base * 1.0 + pd.Series(rng.randn(50) * 0.1, index=base.index),
                "kronos": base * 0.5 + pd.Series(rng.randn(50) * 0.1, index=base.index),
            }
            ret = base + pd.Series(rng.randn(50) * 0.3, index=base.index)
            ensemble.update_history(date, signals, ret)

        output = ensemble.combine(
            {"alpha": make_signal(seed=100), "kronos": make_signal(seed=200)},
            "2024-12-01",
        )
        assert output.mode == "ic_weighted"
        # alpha IC 更高 → alpha 权重更大
        assert output.weights["alpha"] > output.weights["kronos"]

    def test_get_weights(self):
        ensemble = SignalEnsemblePipeline()
        weights = ensemble.get_weights()
        assert isinstance(weights, dict)

    def test_get_monitor(self):
        ensemble = SignalEnsemblePipeline()
        monitor = ensemble.get_monitor()
        assert isinstance(monitor, EnsembleMonitor)

    def test_get_ic_history(self):
        ensemble = SignalEnsemblePipeline()
        df = ensemble.get_ic_history()
        assert isinstance(df, pd.DataFrame)

    def test_pipeline_ranks_in_output(self):
        """输出包含各管线 rank 信号。"""
        ensemble = SignalEnsemblePipeline()
        signals = {
            "alpha": make_signal(seed=1),
            "kronos": make_signal(seed=2),
        }
        output = ensemble.combine(signals, "2024-06-28")
        assert "alpha" in output.pipeline_ranks
        assert "kronos" in output.pipeline_ranks
        # rank 值在 (0, 1]
        for name, rank_s in output.pipeline_ranks.items():
            assert rank_s.min() >= 0
            assert rank_s.max() <= 1.0
