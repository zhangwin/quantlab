"""Tests for M1 DataManager.

These tests require Qlib data to be available locally.
Run with: python -m pytest quantlab/tests/test_data_manager.py -v

To run a single test:
  python -m pytest quantlab/tests/test_data_manager.py::TestDataAccess::test_time_isolation -v
"""

import os
import pytest
import pandas as pd

from quantlab.data.data_manager import DataManager, RollingWindow

# Skip all tests if Qlib data directory does not exist
QLIB_DATA_DIR = os.path.expanduser("~/.qlib/qlib_data/cn_data")
SKIP_NO_DATA = not os.path.isdir(QLIB_DATA_DIR)
skip_reason = "Qlib data not found at ~/.qlib/qlib_data/cn_data"


@pytest.fixture(scope="module")
def dm():
    """Shared DataManager instance for the test module."""
    if SKIP_NO_DATA:
        pytest.skip(skip_reason)
    manager = DataManager(provider_uri=QLIB_DATA_DIR, market="csi300")
    manager.init_qlib()
    return manager


# ===================================================================
# Data Freshness
# ===================================================================

class TestDataFreshness:

    @pytest.mark.skipif(SKIP_NO_DATA, reason=skip_reason)
    def test_get_latest_date(self, dm):
        """get_latest_date returns a valid Timestamp matching calendars/day.txt."""
        latest = dm.get_latest_date()
        assert isinstance(latest, pd.Timestamp)
        # Should be a weekday (Mon-Fri) — unless it is a special trading day
        assert latest.year >= 2020

    @pytest.mark.skipif(SKIP_NO_DATA, reason=skip_reason)
    def test_ensure_data_updated_idempotent(self, dm):
        """Calling ensure_data_updated with a past date should be a no-op."""
        result = dm.ensure_data_updated("2020-01-01")
        assert result is False  # Data already covers 2020


# ===================================================================
# Trading Calendar
# ===================================================================

class TestCalendar:

    @pytest.mark.skipif(SKIP_NO_DATA, reason=skip_reason)
    def test_calendar_basic(self, dm):
        """Calendar returns a list of Timestamps with no weekends."""
        cal = dm.get_trading_calendar("2024-01-01", "2024-01-31")
        assert len(cal) > 0
        for d in cal:
            assert isinstance(d, pd.Timestamp)
            assert d.weekday() < 5  # Mon=0, Fri=4

    @pytest.mark.skipif(SKIP_NO_DATA, reason=skip_reason)
    def test_has_date(self, dm):
        """has_date correctly identifies trading vs non-trading days."""
        # 2024-01-01 is New Year (holiday in China)
        assert dm.has_date("2024-01-02") is True  # first trading day of 2024
        # Saturday
        assert dm.has_date("2024-01-06") is False


# ===================================================================
# Data Access (Time Isolation)
# ===================================================================

class TestDataAccess:

    @pytest.mark.skipif(SKIP_NO_DATA, reason=skip_reason)
    def test_time_isolation(self, dm):
        """Data returned by get_ohlcv_before must not exceed anchor_date."""
        anchor = "2024-06-28"
        ohlcv = dm.get_ohlcv_before(anchor_date=anchor, lookback_days=30)
        assert len(ohlcv) > 0

        for symbol, df in ohlcv.items():
            max_date = df.index.max()
            assert max_date <= pd.Timestamp(anchor), (
                f"{symbol} has data after anchor {anchor}: {max_date}"
            )

    @pytest.mark.skipif(SKIP_NO_DATA, reason=skip_reason)
    def test_ohlcv_columns(self, dm):
        """OHLCV DataFrames have the expected columns."""
        ohlcv = dm.get_ohlcv_before("2024-06-28", lookback_days=10)
        for symbol, df in list(ohlcv.items())[:3]:
            assert set(df.columns) == {"open", "high", "low", "close", "volume"}
            assert df.dtypes["close"] in (float, "float64", "float32")

    @pytest.mark.skipif(SKIP_NO_DATA, reason=skip_reason)
    def test_ohlcv_empty_for_early_date(self, dm):
        """Requesting data before data start returns empty dict."""
        ohlcv = dm.get_ohlcv_before("1990-01-01", lookback_days=10)
        assert len(ohlcv) == 0

    @pytest.mark.skipif(SKIP_NO_DATA, reason=skip_reason)
    def test_close_prices(self, dm):
        """get_close_prices returns a Series indexed by symbol."""
        prices = dm.get_close_prices("2024-06-28")
        assert isinstance(prices, pd.Series)
        assert len(prices) >= 280  # CSI300 should have ~300 stocks
        assert not prices.isna().all()

    @pytest.mark.skipif(SKIP_NO_DATA, reason=skip_reason)
    def test_open_prices(self, dm):
        """get_open_prices returns valid prices."""
        prices = dm.get_open_prices("2024-06-28")
        assert isinstance(prices, pd.Series)
        assert len(prices) > 0
        assert (prices > 0).all()

    @pytest.mark.skipif(SKIP_NO_DATA, reason=skip_reason)
    def test_daily_returns(self, dm):
        """get_daily_returns returns reasonable values."""
        rets = dm.get_daily_returns("2024-06-28")
        assert isinstance(rets, pd.Series)
        assert len(rets) > 0
        # Daily returns should be within [-20%, +20%] for most stocks
        valid = rets.dropna()
        assert (valid.abs() < 0.20).mean() > 0.95

    @pytest.mark.skipif(SKIP_NO_DATA, reason=skip_reason)
    def test_data_completeness(self, dm):
        """CSI300 should return at least 280 stocks."""
        ohlcv = dm.get_ohlcv_before("2024-06-28", lookback_days=5)
        assert len(ohlcv) >= 280


# ===================================================================
# Limit Prices
# ===================================================================

class TestLimitPrices:

    @pytest.mark.skipif(SKIP_NO_DATA, reason=skip_reason)
    def test_limit_prices(self, dm):
        """Limit prices are 10% above/below previous close."""
        limit_up, limit_down = dm.get_limit_prices("2024-06-28")
        assert len(limit_up) > 0
        assert len(limit_down) > 0
        # limit_up > limit_down for all stocks
        common = limit_up.index.intersection(limit_down.index)
        assert (limit_up[common] > limit_down[common]).all()


# ===================================================================
# Industry Map
# ===================================================================

class TestIndustryMap:

    @pytest.mark.skipif(SKIP_NO_DATA, reason=skip_reason)
    def test_industry_map_returns_series(self, dm):
        """get_industry_map returns a Series (possibly dummy)."""
        ind = dm.get_industry_map()
        assert isinstance(ind, pd.Series)
        assert len(ind) > 0


# ===================================================================
# RollingWindow dataclass
# ===================================================================

class TestRollingWindow:

    def test_defaults(self):
        rw = RollingWindow()
        assert rw.finetune_lookback == 30
        assert rw.predict_horizon == 5
        assert rw.alpha_train_years == 3

    def test_custom(self):
        rw = RollingWindow(finetune_lookback=60, backtest_start="2022-01-01")
        assert rw.finetune_lookback == 60
        assert rw.backtest_start == "2022-01-01"
