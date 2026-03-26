"""Tests for M1.5 DataViewer.

Run with: python -m pytest quantlab/tests/test_data_viewer.py -v
"""

import os
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from quantlab.data.data_manager import DataManager
from quantlab.data.data_viewer import DataViewer

QLIB_DATA_DIR = os.path.expanduser("~/.qlib/qlib_data/cn_data")
SKIP_NO_DATA = not os.path.isdir(QLIB_DATA_DIR)
skip_reason = "Qlib data not found at ~/.qlib/qlib_data/cn_data"

try:
    import plotly
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False


@pytest.fixture(scope="module")
def viewer():
    if SKIP_NO_DATA:
        pytest.skip(skip_reason)
    dm = DataManager(provider_uri=QLIB_DATA_DIR, market="csi300")
    dm.init_qlib()
    return DataViewer(dm)


# ===================================================================
# CSV Export
# ===================================================================

class TestCSVExport:

    @pytest.mark.skipif(SKIP_NO_DATA, reason=skip_reason)
    def test_export_csv(self, viewer):
        """Export CSV for a single stock."""
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = viewer.export_csv(
                ["SH600519"], "2024-01-01", "2024-06-30", output_dir=tmpdir
            )
            assert len(paths) == 1
            assert paths[0].exists()
            df = pd.read_csv(paths[0], index_col=0, parse_dates=True)
            assert len(df) > 100
            assert "close" in df.columns

    @pytest.mark.skipif(SKIP_NO_DATA, reason=skip_reason)
    def test_export_csv_missing_symbol(self, viewer):
        """Exporting a non-existent symbol produces no file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = viewer.export_csv(
                ["NONEXIST000"], "2024-01-01", "2024-06-30", output_dir=tmpdir
            )
            assert len(paths) == 0

    @pytest.mark.skipif(SKIP_NO_DATA, reason=skip_reason)
    def test_export_portfolio_csv(self, viewer):
        """Export portfolio CSV merges multiple symbols."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = os.path.join(tmpdir, "portfolio.csv")
            viewer.export_portfolio_csv(
                ["SH600519", "SH601318"],
                "2024-01-01",
                "2024-03-31",
                output_path=out_path,
            )
            assert Path(out_path).exists()
            df = pd.read_csv(out_path, index_col=0, parse_dates=True)
            assert "symbol" in df.columns
            assert df["symbol"].nunique() >= 1


# ===================================================================
# K-line Charts
# ===================================================================

class TestKlineChart:

    @pytest.mark.skipif(
        SKIP_NO_DATA or not HAS_PLOTLY, reason="Needs data + plotly"
    )
    def test_plot_kline(self, viewer):
        """plot_kline returns a Plotly Figure."""
        fig = viewer.plot_kline("SH600519", "2024-01-01", "2024-06-30")
        assert fig is not None
        # Should have candlestick + volume + MA traces
        assert len(fig.data) >= 2

    @pytest.mark.skipif(
        SKIP_NO_DATA or not HAS_PLOTLY, reason="Needs data + plotly"
    )
    def test_plot_kline_with_trades(self, viewer):
        """plot_kline_with_trades renders buy/sell markers."""
        trades = [
            {
                "date": "2024-03-15",
                "symbol": "SH600519",
                "direction": "buy",
                "exec_price": 1700.0,
                "status": "filled",
            },
            {
                "date": "2024-05-20",
                "symbol": "SH600519",
                "direction": "sell",
                "exec_price": 1750.0,
                "status": "filled",
            },
        ]
        fig = viewer.plot_kline_with_trades(
            "SH600519", "2024-01-01", "2024-06-30", trades
        )
        assert fig is not None
        # Should have markers for buy and sell
        trace_names = [t.name for t in fig.data if t.name]
        assert "Buy" in trace_names
        assert "Sell" in trace_names

    @pytest.mark.skipif(
        SKIP_NO_DATA or not HAS_PLOTLY, reason="Needs data + plotly"
    )
    def test_plot_portfolio_overview_empty(self, viewer):
        """Empty portfolio shows annotation."""
        fig = viewer.plot_portfolio_overview(
            positions={},
            current_prices=pd.Series(dtype=float),
            industry_map=pd.Series(dtype=str),
        )
        assert fig is not None

    @pytest.mark.skipif(not HAS_PLOTLY, reason="Needs plotly")
    def test_plot_kline_with_prediction(self, viewer):
        """Prediction overlay renders without error."""
        dates = pd.date_range("2024-06-01", "2024-06-28", freq="B")
        hist = pd.DataFrame(
            {
                "open": [100 + i * 0.1 for i in range(len(dates))],
                "high": [101 + i * 0.1 for i in range(len(dates))],
                "low": [99 + i * 0.1 for i in range(len(dates))],
                "close": [100.5 + i * 0.1 for i in range(len(dates))],
                "volume": [1000000] * len(dates),
            },
            index=dates,
        )
        pred_dates = pd.date_range("2024-07-01", periods=5, freq="B")
        pred = pd.DataFrame(
            {
                "open": [103, 104, 103, 105, 106],
                "high": [104, 105, 104, 106, 107],
                "low": [102, 103, 102, 104, 105],
                "close": [103.5, 104.5, 103.5, 105.5, 106.5],
                "volume": [1000000] * 5,
            },
            index=pred_dates,
        )
        fig = viewer.plot_kline_with_prediction("TEST", hist, pred)
        assert fig is not None
        trace_names = [t.name for t in fig.data if t.name]
        assert "Prediction" in trace_names
