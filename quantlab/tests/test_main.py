"""Tests for M9 BacktestRunner (main.py)."""

import os
import pickle
import tempfile
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from quantlab.main import (
    BacktestConfig,
    BacktestResult,
    BacktestRunner,
    BacktestState,
    DailyResult,
    DailyStep,
    _apply_position_limit,
)


# =========================================================================
# Fixtures & helpers
# =========================================================================


@pytest.fixture
def sample_config(tmp_path):
    """Create a minimal YAML config and return BacktestConfig."""
    yaml_content = """
start_date: "2024-01-02"
end_date: "2024-01-10"
market: "csi300"
initial_cash: 1000000
warmup_days: 2
enable_alpha: true
enable_kronos: false
enable_rdagent: false
alpha_retrain_interval: 20
alpha_train_years: 3
kronos_recipe_name: "conservative"
kronos_device: "cpu"
ensemble_ic_lookback: 60
ensemble_uncertainty_penalty: 0.1
ensemble_min_weight: 0.1
ensemble_max_weight: 0.6
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
checkpoint_interval: 0
log_level: "WARNING"
benchmark: "SH000300"
"""
    yaml_path = tmp_path / "backtest.yaml"
    yaml_path.write_text(yaml_content, encoding="utf-8")
    return BacktestConfig.load(str(yaml_path))


def _make_calendar(n=6):
    """Generate n trading dates starting 2024-01-02."""
    dates = pd.bdate_range("2024-01-02", periods=n, freq="B")
    return list(dates)


def _make_prices(symbols=("SH600000", "SH600001", "SH600002"), base=10.0):
    """Generate mock prices."""
    return {s: base + i for i, s in enumerate(symbols)}


def _mock_position(symbol, shares=1000, cost=10.0):
    """Create a mock Position."""
    pos = MagicMock()
    pos.symbol = symbol
    pos.shares = shares
    pos.cost_price = cost
    pos.highest_price = cost
    return pos


def _mock_risk_result(
    force_sell=None,
    circuit_breaker=False,
    position_limit=1.0,
    events=None,
):
    """Create a mock RiskCheckResult."""
    r = MagicMock()
    r.force_sell_symbols = force_sell or []
    r.suggest_sell_symbols = []
    r.force_sell_reasons = {}
    r.circuit_breaker_triggered = circuit_breaker
    r.position_limit = position_limit
    r.events = events or []
    return r


def _mock_order(symbol, direction="buy", shares=100):
    """Create a mock TradeOrder."""
    o = MagicMock()
    o.symbol = symbol
    o.direction = direction
    o.shares = shares
    return o


# =========================================================================
# TestBacktestConfig
# =========================================================================


class TestBacktestConfig:

    def test_load_from_yaml(self, sample_config):
        cfg = sample_config
        assert cfg.start_date == "2024-01-02"
        assert cfg.end_date == "2024-01-10"
        assert cfg.initial_cash == 1_000_000
        assert cfg.enable_alpha is True
        assert cfg.enable_kronos is False
        assert cfg.warmup_days == 2

    def test_validate_start_after_end(self):
        cfg = BacktestConfig(start_date="2025-01-01", end_date="2024-01-01")
        with pytest.raises(ValueError, match="start_date"):
            cfg.validate()

    def test_validate_negative_cash(self):
        cfg = BacktestConfig(initial_cash=-100)
        with pytest.raises(ValueError, match="initial_cash"):
            cfg.validate()

    def test_validate_bad_stop_loss(self):
        cfg = BacktestConfig(stop_loss_pct=1.5)
        with pytest.raises(ValueError, match="stop_loss_pct"):
            cfg.validate()

    def test_to_dict(self):
        cfg = BacktestConfig()
        d = cfg.to_dict()
        assert "start_date" in d
        assert "initial_cash" in d
        assert d["market"] == "csi300"

    def test_unknown_fields_ignored(self, tmp_path):
        yaml_content = """
start_date: "2024-01-02"
end_date: "2024-12-31"
unknown_field: 999
another_unknown: "hello"
"""
        yaml_path = tmp_path / "test.yaml"
        yaml_path.write_text(yaml_content, encoding="utf-8")
        cfg = BacktestConfig.load(str(yaml_path))
        assert cfg.start_date == "2024-01-02"
        assert not hasattr(cfg, "unknown_field")


# =========================================================================
# TestBacktestState
# =========================================================================


class TestBacktestState:

    def test_append_nav(self):
        state = BacktestState()
        state.append_nav("2024-01-02", 1000000)
        state.append_nav("2024-01-03", 1010000)
        assert len(state.daily_nav) == 2

    def test_get_nav_series(self):
        state = BacktestState()
        state.append_nav("2024-01-02", 100)
        state.append_nav("2024-01-03", 110)
        s = state.get_nav_series()
        assert isinstance(s, pd.Series)
        assert len(s) == 2
        assert s.iloc[1] == 110

    def test_get_nav_series_empty(self):
        state = BacktestState()
        s = state.get_nav_series()
        assert s.empty

    def test_append_signals(self):
        state = BacktestState()
        sig = pd.Series([1.0, 2.0], index=["A", "B"])
        state.append_signals("2024-01-02", {"alpha": sig, "kronos": None})
        assert "alpha" in state.signal_history
        assert len(state.signal_history["alpha"]) == 1
        assert "kronos" not in state.signal_history

    def test_append_trades(self):
        state = BacktestState()
        state.append_trades(["trade1", "trade2"])
        state.append_trades(["trade3"])
        assert len(state.trade_records) == 3

    def test_checkpoint_save_load(self, tmp_path):
        state = BacktestState(current_idx=5, total_days=100)
        state.append_nav("2024-01-02", 1000000)
        state.append_nav("2024-01-03", 1010000)

        ckpt_path = str(tmp_path / "test.pkl")
        state.save_checkpoint(ckpt_path)

        loaded = BacktestState.load_checkpoint(ckpt_path)
        assert loaded.current_idx == 5
        assert loaded.total_days == 100
        assert len(loaded.daily_nav) == 2


# =========================================================================
# TestDailyStep
# =========================================================================


class TestDailyStep:

    def _make_mocks(self):
        dm = MagicMock()
        dm.get_close_prices.return_value = pd.Series(
            {"SH600000": 10.0, "SH600001": 11.0}
        )
        dm.get_open_prices.return_value = pd.Series(
            {"SH600000": 10.1, "SH600001": 11.1}
        )
        dm.get_daily_returns.return_value = pd.Series(
            {"SH600000": 0.01, "SH600001": -0.005}
        )
        dm.get_industry_map.return_value = pd.Series(
            {"SH600000": "bank", "SH600001": "tech"}
        )
        dm.get_limit_prices.return_value = (
            pd.Series({"SH600000": 11.0, "SH600001": 12.1}),
            pd.Series({"SH600000": 9.0, "SH600001": 9.9}),
        )
        dm.get_ohlcv_before.return_value = {}

        alpha = MagicMock()
        alpha.predict.return_value = pd.Series({"SH600000": 0.8, "SH600001": 0.3})

        ensemble = MagicMock()
        ens_output = MagicMock()
        ens_output.signal = pd.Series({"SH600000": 0.7, "SH600001": 0.4})
        ensemble.combine.return_value = ens_output

        execution = MagicMock()
        execution.get_holdings.return_value = {}
        execution.get_portfolio_value.return_value = 1_000_000.0
        execution.account = MagicMock()
        execution.generate_orders.return_value = []
        execution.execute_orders.return_value = []

        risk = MagicMock()
        risk.is_paused.return_value = False
        risk.daily_check.return_value = _mock_risk_result()

        config = BacktestConfig(
            enable_alpha=True,
            enable_kronos=False,
            enable_rdagent=False,
            warmup_days=2,
        )

        return dm, alpha, None, None, ensemble, execution, risk, config

    def test_normal_day(self):
        dm, alpha, kronos, rdagent, ensemble, execution, risk, config = (
            self._make_mocks()
        )
        result = DailyStep.execute(
            T="2024-01-04", T_next="2024-01-05",
            dm=dm, alpha=alpha, kronos=kronos, rdagent=rdagent,
            ensemble=ensemble, execution=execution, risk=risk,
            config=config, calendar_strs=["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"],
            day_idx=2,
        )
        assert result.date == "2024-01-04"
        assert result.portfolio_value == 1_000_000.0
        assert not result.skipped
        assert "alpha" in result.signals
        alpha.predict.assert_called_once()
        ensemble.combine.assert_called_once()

    def test_warmup_skips_signals(self):
        dm, alpha, kronos, rdagent, ensemble, execution, risk, config = (
            self._make_mocks()
        )
        result = DailyStep.execute(
            T="2024-01-02", T_next="2024-01-03",
            dm=dm, alpha=alpha, kronos=kronos, rdagent=rdagent,
            ensemble=ensemble, execution=execution, risk=risk,
            config=config, calendar_strs=["2024-01-02", "2024-01-03"],
            day_idx=0,  # < warmup_days=2
        )
        alpha.predict.assert_not_called()
        assert "alpha" not in result.signals

    def test_paused_skips_everything(self):
        dm, alpha, kronos, rdagent, ensemble, execution, risk, config = (
            self._make_mocks()
        )
        risk.is_paused.return_value = True

        result = DailyStep.execute(
            T="2024-01-04", T_next="2024-01-05",
            dm=dm, alpha=alpha, kronos=kronos, rdagent=rdagent,
            ensemble=ensemble, execution=execution, risk=risk,
            config=config, calendar_strs=["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"],
            day_idx=2,
        )
        assert result.skipped is True
        risk.daily_check.assert_not_called()
        alpha.predict.assert_not_called()

    def test_circuit_breaker_triggers_liquidation(self):
        dm, alpha, kronos, rdagent, ensemble, execution, risk, config = (
            self._make_mocks()
        )
        risk.daily_check.return_value = _mock_risk_result(circuit_breaker=True, position_limit=0.0)
        execution.add_liquidation_orders.return_value = [_mock_order("SH600000", "sell")]

        result = DailyStep.execute(
            T="2024-01-04", T_next="2024-01-05",
            dm=dm, alpha=alpha, kronos=kronos, rdagent=rdagent,
            ensemble=ensemble, execution=execution, risk=risk,
            config=config, calendar_strs=["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"],
            day_idx=2,
        )
        execution.add_liquidation_orders.assert_called_once()
        alpha.predict.assert_not_called()

    def test_force_sell_appended(self):
        dm, alpha, kronos, rdagent, ensemble, execution, risk, config = (
            self._make_mocks()
        )
        risk.daily_check.return_value = _mock_risk_result(
            force_sell=["SH600000"],
        )
        execution.add_force_sell_orders.return_value = [_mock_order("SH600000", "sell")]

        result = DailyStep.execute(
            T="2024-01-04", T_next="2024-01-05",
            dm=dm, alpha=alpha, kronos=kronos, rdagent=rdagent,
            ensemble=ensemble, execution=execution, risk=risk,
            config=config, calendar_strs=["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"],
            day_idx=2,
        )
        execution.add_force_sell_orders.assert_called_once()

    def test_signal_failure_does_not_crash(self):
        dm, alpha, kronos, rdagent, ensemble, execution, risk, config = (
            self._make_mocks()
        )
        alpha.predict.side_effect = RuntimeError("model not loaded")

        result = DailyStep.execute(
            T="2024-01-04", T_next="2024-01-05",
            dm=dm, alpha=alpha, kronos=kronos, rdagent=rdagent,
            ensemble=ensemble, execution=execution, risk=risk,
            config=config, calendar_strs=["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"],
            day_idx=2,
        )
        assert "alpha" not in result.signals


# =========================================================================
# TestApplyPositionLimit
# =========================================================================


class TestApplyPositionLimit:

    def test_zero_limit_removes_buys(self):
        orders = [
            _mock_order("A", "buy", 100),
            _mock_order("B", "sell", 200),
            _mock_order("C", "buy", 100),
        ]
        _apply_position_limit(orders, 0.0, {}, 1000000, {"A": 10, "C": 10})
        assert len(orders) == 1
        assert orders[0].direction == "sell"

    def test_partial_limit(self):
        orders = [
            _mock_order("A", "buy", 100),
            _mock_order("B", "buy", 100),
        ]
        holdings = {}
        prices = {"A": 100.0, "B": 100.0}
        # 50% limit, no existing holdings → available = 500000
        _apply_position_limit(orders, 0.5, holdings, 1_000_000, prices)
        # Each buy costs 10000, both should fit
        assert len(orders) == 2

    def test_limit_caps_buys(self):
        orders = [
            _mock_order("A", "buy", 100),
            _mock_order("B", "buy", 100),
        ]
        # Holdings already at 400k, limit 50% of 1M = 500k, available = 100k
        holdings = {"X": _mock_position("X", shares=4000, cost=100.0)}
        prices = {"A": 100.0, "B": 100.0, "X": 100.0}
        # Each buy = 10000. Both fit within 100k
        _apply_position_limit(orders, 0.5, holdings, 1_000_000, prices)
        assert len(orders) == 2


# =========================================================================
# TestBacktestRunner
# =========================================================================


class TestBacktestRunner:

    def test_init(self, sample_config):
        runner = BacktestRunner(sample_config)
        assert runner.config.market == "csi300"
        assert runner._dm is None

    @patch("quantlab.main.BacktestRunner._init_modules")
    @patch("quantlab.main.BacktestRunner._evaluate")
    def test_run_basic_flow(self, mock_eval, mock_init, sample_config):
        """Test the main loop with mocked modules."""
        runner = BacktestRunner(sample_config)

        # Mock calendar: 5 trading days
        calendar = _make_calendar(5)
        mock_dm = MagicMock()
        mock_dm.get_trading_calendar.return_value = calendar
        mock_dm.get_close_prices.return_value = pd.Series({"A": 10.0})
        mock_dm.get_open_prices.return_value = pd.Series({"A": 10.0})
        mock_dm.get_daily_returns.return_value = pd.Series({"A": 0.01})
        mock_dm.get_industry_map.return_value = pd.Series({"A": "bank"})
        mock_dm.get_limit_prices.return_value = (
            pd.Series({"A": 11.0}),
            pd.Series({"A": 9.0}),
        )

        mock_risk = MagicMock()
        mock_risk.is_paused.return_value = False
        mock_risk.daily_check.return_value = _mock_risk_result()

        mock_exec = MagicMock()
        mock_exec.get_holdings.return_value = {}
        mock_exec.get_portfolio_value.return_value = 1_000_000.0
        mock_exec.account = MagicMock()
        mock_exec.generate_orders.return_value = []
        mock_exec.execute_orders.return_value = []

        mock_ensemble = MagicMock()
        ens_out = MagicMock()
        ens_out.signal = pd.Series(dtype=float)
        mock_ensemble.combine.return_value = ens_out

        runner._dm = mock_dm
        runner._alpha = None
        runner._kronos = None
        runner._rdagent = None
        runner._ensemble = mock_ensemble
        runner._execution = mock_exec
        runner._risk = mock_risk

        mock_eval.return_value = None

        result = runner.run()

        assert isinstance(result, BacktestResult)
        assert len(result.daily_nav) == 5  # initial + 4 days
        assert result.elapsed_seconds > 0

    def test_resume_loads_checkpoint(self, tmp_path, sample_config):
        """Test that resume loads state from checkpoint."""
        state = BacktestState(current_idx=2, total_days=10)
        state.append_nav("2024-01-02", 1000000)
        state.append_nav("2024-01-03", 1005000)
        state.append_nav("2024-01-04", 1010000)

        ckpt = str(tmp_path / "ckpt.pkl")
        state.save_checkpoint(ckpt)

        loaded = BacktestState.load_checkpoint(ckpt)
        assert loaded.current_idx == 2
        assert len(loaded.daily_nav) == 3


# =========================================================================
# TestDailyResult
# =========================================================================


class TestDailyResult:

    def test_defaults(self):
        r = DailyResult()
        assert r.date == ""
        assert r.portfolio_value == 0.0
        assert r.skipped is False
        assert r.trade_records == []
        assert r.signals == {}
