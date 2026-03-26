"""M1.5 Data Viewer — CSV export + interactive K-line visualization.

Converts Qlib binary data to human-readable formats and provides
Plotly-based candlestick charts with optional overlays (predictions,
trade markers, technical indicators).
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False
    logger.warning("plotly not installed — visualization functions unavailable")


class DataViewer:
    """Data inspection and visualization layer on top of DataManager.

    Parameters
    ----------
    data_manager : DataManager
        An initialized DataManager instance.
    """

    # A-share convention: red=up, green=down
    COLOR_UP = "#ef5350"
    COLOR_DOWN = "#26a69a"

    def __init__(self, data_manager):
        self.dm = data_manager

    # ------------------------------------------------------------------
    # CSV Export
    # ------------------------------------------------------------------

    def export_csv(
        self,
        symbols: List[str],
        start: str,
        end: str,
        output_dir: str = "./export",
    ) -> List[Path]:
        """Export OHLCV data for each symbol as individual CSV files.

        Parameters
        ----------
        symbols : list of str
            Stock codes, e.g. ``["SH600519", "SZ000001"]``.
        start, end : str
            Date range.
        output_dir : str
            Output directory (created if not exists).

        Returns
        -------
        list of Path
            Paths to the generated CSV files.
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        ohlcv = self.dm.get_ohlcv_before(anchor_date=end, lookback_days=9999)
        paths = []

        for symbol in symbols:
            if symbol not in ohlcv:
                logger.warning("Symbol %s not found in data, skipping", symbol)
                continue

            df = ohlcv[symbol]
            # Filter by start date
            df = df[df.index >= pd.Timestamp(start)]
            if df.empty:
                logger.warning("No data for %s in [%s, %s]", symbol, start, end)
                continue

            csv_path = out / f"{symbol}_{start}_{end}.csv"
            df.to_csv(csv_path, float_format="%.4f")
            paths.append(csv_path)
            logger.info("Exported %s (%d rows) → %s", symbol, len(df), csv_path)

        return paths

    def export_portfolio_csv(
        self,
        symbols: List[str],
        start: str,
        end: str,
        output_path: str = "./export/portfolio.csv",
    ) -> Path:
        """Export multiple symbols into a single CSV with a 'symbol' column.

        Returns
        -------
        Path
            Path to the merged CSV file.
        """
        ohlcv = self.dm.get_ohlcv_before(anchor_date=end, lookback_days=9999)
        frames = []

        for symbol in symbols:
            if symbol not in ohlcv:
                continue
            df = ohlcv[symbol].copy()
            df = df[df.index >= pd.Timestamp(start)]
            df.insert(0, "symbol", symbol)
            frames.append(df)

        if not frames:
            logger.warning("No data to export")
            return Path(output_path)

        merged = pd.concat(frames)
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        merged.to_csv(out, float_format="%.4f")
        logger.info("Portfolio CSV exported (%d rows) → %s", len(merged), out)
        return out

    # ------------------------------------------------------------------
    # K-line Visualization
    # ------------------------------------------------------------------

    def _require_plotly(self):
        if not HAS_PLOTLY:
            raise ImportError(
                "plotly is required for visualization. "
                "Install with: pip install plotly"
            )

    def plot_kline(
        self,
        symbol: str,
        start: str,
        end: str,
        ma_windows: Optional[List[int]] = None,
    ) -> "go.Figure":
        """Plot an interactive candlestick chart for a single stock.

        Parameters
        ----------
        symbol : str
            Stock code.
        start, end : str
            Date range.
        ma_windows : list of int, optional
            Moving average windows to overlay (default: [5, 10, 20]).

        Returns
        -------
        plotly.graph_objects.Figure
        """
        self._require_plotly()
        if ma_windows is None:
            ma_windows = [5, 10, 20]

        ohlcv = self.dm.get_ohlcv_before(anchor_date=end, lookback_days=9999)
        if symbol not in ohlcv:
            raise ValueError(f"Symbol {symbol} not found in data")

        df = ohlcv[symbol]
        df = df[df.index >= pd.Timestamp(start)]
        if df.empty:
            raise ValueError(f"No data for {symbol} in [{start}, {end}]")

        fig = self._create_kline_figure(df, symbol, ma_windows)
        return fig

    def plot_kline_with_prediction(
        self,
        symbol: str,
        hist_data: pd.DataFrame,
        pred_data: pd.DataFrame,
        ma_windows: Optional[List[int]] = None,
    ) -> "go.Figure":
        """Plot K-line with Kronos prediction overlay.

        Parameters
        ----------
        symbol : str
            Stock code (for title).
        hist_data : DataFrame
            Historical OHLCV with DatetimeIndex.
        pred_data : DataFrame
            Predicted OHLCV with DatetimeIndex (future dates).
            Columns: open, high, low, close, volume.

        Returns
        -------
        plotly.graph_objects.Figure
        """
        self._require_plotly()
        if ma_windows is None:
            ma_windows = [5, 10, 20]

        fig = self._create_kline_figure(hist_data, symbol, ma_windows)

        # Overlay predicted candles (semi-transparent)
        fig.add_trace(
            go.Candlestick(
                x=pred_data.index,
                open=pred_data["open"],
                high=pred_data["high"],
                low=pred_data["low"],
                close=pred_data["close"],
                increasing_line_color="rgba(239,83,80,0.4)",
                decreasing_line_color="rgba(38,166,154,0.4)",
                increasing_fillcolor="rgba(239,83,80,0.2)",
                decreasing_fillcolor="rgba(38,166,154,0.2)",
                name="Prediction",
            ),
            row=1,
            col=1,
        )

        # Prediction boundary line
        boundary = hist_data.index[-1]
        fig.add_vline(
            x=boundary,
            line_dash="dash",
            line_color="gray",
            annotation_text="Prediction →",
            row=1,
            col=1,
        )

        return fig

    def plot_kline_with_trades(
        self,
        symbol: str,
        start: str,
        end: str,
        trade_records: List,
        ma_windows: Optional[List[int]] = None,
    ) -> "go.Figure":
        """Plot K-line with buy/sell markers from trade records.

        Parameters
        ----------
        trade_records : list
            List of trade record objects/dicts with fields:
            ``date``, ``symbol``, ``direction`` ("buy"/"sell"),
            ``exec_price``, ``status``.
        """
        self._require_plotly()
        if ma_windows is None:
            ma_windows = [5, 10, 20]

        ohlcv = self.dm.get_ohlcv_before(anchor_date=end, lookback_days=9999)
        if symbol not in ohlcv:
            raise ValueError(f"Symbol {symbol} not found in data")

        df = ohlcv[symbol]
        df = df[df.index >= pd.Timestamp(start)]
        fig = self._create_kline_figure(df, symbol, ma_windows)

        # Filter trades for this symbol
        sym_trades = [
            t for t in trade_records
            if _get_field(t, "symbol") == symbol
            and _get_field(t, "status") == "filled"
        ]

        buy_dates, buy_prices = [], []
        sell_dates, sell_prices = [], []

        for t in sym_trades:
            date = pd.Timestamp(_get_field(t, "date"))
            price = _get_field(t, "exec_price")
            if _get_field(t, "direction") == "buy":
                buy_dates.append(date)
                buy_prices.append(price)
            else:
                sell_dates.append(date)
                sell_prices.append(price)

        if buy_dates:
            fig.add_trace(
                go.Scatter(
                    x=buy_dates,
                    y=buy_prices,
                    mode="markers",
                    marker=dict(
                        symbol="triangle-up",
                        size=12,
                        color=self.COLOR_UP,
                        line=dict(width=1, color="black"),
                    ),
                    name="Buy",
                ),
                row=1,
                col=1,
            )

        if sell_dates:
            fig.add_trace(
                go.Scatter(
                    x=sell_dates,
                    y=sell_prices,
                    mode="markers",
                    marker=dict(
                        symbol="triangle-down",
                        size=12,
                        color=self.COLOR_DOWN,
                        line=dict(width=1, color="black"),
                    ),
                    name="Sell",
                ),
                row=1,
                col=1,
            )

        return fig

    def plot_portfolio_overview(
        self,
        positions: Dict,
        current_prices: pd.Series,
        industry_map: pd.Series,
    ) -> "go.Figure":
        """Plot portfolio overview: industry pie + per-stock PnL bar.

        Parameters
        ----------
        positions : dict
            ``{symbol: Position}`` where Position has ``shares`` and ``cost_price``.
        current_prices : Series
            Current close prices indexed by symbol.
        industry_map : Series
            Industry classification indexed by symbol.

        Returns
        -------
        plotly.graph_objects.Figure
        """
        self._require_plotly()

        symbols = list(positions.keys())
        if not symbols:
            fig = go.Figure()
            fig.add_annotation(text="No positions", showarrow=False)
            return fig

        # Compute market values and PnL
        market_values = {}
        pnl = {}
        industries = {}

        for sym, pos in positions.items():
            shares = _get_field(pos, "shares")
            cost = _get_field(pos, "cost_price")
            price = current_prices.get(sym, cost)
            mv = shares * price
            market_values[sym] = mv
            pnl[sym] = (price - cost) / cost if cost > 0 else 0.0
            industries[sym] = industry_map.get(sym, "unknown")

        # Aggregate by industry
        industry_mv = {}
        for sym, mv in market_values.items():
            ind = industries[sym]
            industry_mv[ind] = industry_mv.get(ind, 0) + mv

        fig = make_subplots(
            rows=1, cols=2,
            specs=[[{"type": "pie"}, {"type": "bar"}]],
            subplot_titles=["Industry Distribution", "Position PnL (%)"],
        )

        # Pie: industry allocation
        fig.add_trace(
            go.Pie(
                labels=list(industry_mv.keys()),
                values=list(industry_mv.values()),
                hole=0.3,
            ),
            row=1, col=1,
        )

        # Bar: per-stock PnL
        colors = [
            self.COLOR_UP if v >= 0 else self.COLOR_DOWN
            for v in pnl.values()
        ]
        fig.add_trace(
            go.Bar(
                x=list(pnl.keys()),
                y=[v * 100 for v in pnl.values()],
                marker_color=colors,
                name="PnL %",
            ),
            row=1, col=2,
        )

        fig.update_layout(
            title="Portfolio Overview",
            height=400,
            showlegend=False,
        )
        return fig

    # ------------------------------------------------------------------
    # Display helper
    # ------------------------------------------------------------------

    def show(self, fig: "go.Figure") -> None:
        """Display figure — auto-detects environment.

        - Jupyter: inline display
        - Terminal: opens browser
        - If neither works: saves to ``./chart.html``
        """
        self._require_plotly()
        try:
            fig.show()
        except Exception:
            path = Path("./chart.html")
            fig.write_html(str(path))
            logger.info("Chart saved to %s", path)

    # ------------------------------------------------------------------
    # Internal: build candlestick figure
    # ------------------------------------------------------------------

    def _create_kline_figure(
        self,
        df: pd.DataFrame,
        title: str,
        ma_windows: List[int],
    ) -> "go.Figure":
        """Build a two-panel candlestick + volume chart."""
        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.75, 0.25],
        )

        # Candlestick
        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df["open"],
                high=df["high"],
                low=df["low"],
                close=df["close"],
                increasing_line_color=self.COLOR_UP,
                decreasing_line_color=self.COLOR_DOWN,
                increasing_fillcolor=self.COLOR_UP,
                decreasing_fillcolor=self.COLOR_DOWN,
                name="OHLC",
            ),
            row=1,
            col=1,
        )

        # Moving averages
        ma_colors = ["#FFA726", "#42A5F5", "#AB47BC", "#66BB6A", "#EF5350"]
        for i, w in enumerate(ma_windows):
            if len(df) >= w:
                ma = df["close"].rolling(w).mean()
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=ma,
                        mode="lines",
                        line=dict(width=1, color=ma_colors[i % len(ma_colors)]),
                        name=f"MA{w}",
                    ),
                    row=1,
                    col=1,
                )

        # Volume bars
        vol_colors = [
            self.COLOR_UP if c >= o else self.COLOR_DOWN
            for o, c in zip(df["open"], df["close"])
        ]
        fig.add_trace(
            go.Bar(
                x=df.index,
                y=df["volume"],
                marker_color=vol_colors,
                name="Volume",
                showlegend=False,
            ),
            row=2,
            col=1,
        )

        fig.update_layout(
            title=title,
            xaxis_rangeslider_visible=False,
            height=600,
            template="plotly_white",
        )
        fig.update_xaxes(type="category", row=1, col=1)
        fig.update_xaxes(type="category", row=2, col=1)

        return fig


# ---------------------------------------------------------------------------
# Utility: access dict or object field uniformly
# ---------------------------------------------------------------------------

def _get_field(obj, name):
    """Get a field from a dict or a dataclass/object."""
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)
