"""M1 Data Manager — data update + time-isolated data access.

Wraps Qlib's data infrastructure to provide:
  1. Incremental data update (Yahoo / Baostock / CSV → bin files)
  2. Time-isolated data access (all queries respect anchor_date)

Data update supports three sources:
  - yahoo  : Qlib built-in Yahoo Finance collector (default)
  - baostock: Free A-share data via baostock SDK (no rate limit)
  - csv     : Manual CSV files dumped to bin via DumpDataUpdate
"""

import logging
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class RollingWindow:
    """Rolling window parameters shared across pipeline modules."""
    finetune_lookback: int = 30
    predict_horizon: int = 5
    alpha_train_years: int = 3
    alpha_retrain_interval: int = 20
    ic_lookback: int = 60
    backtest_start: str = "2023-01-01"
    backtest_end: str = "2025-03-01"


# ---------------------------------------------------------------------------
# Data Manager
# ---------------------------------------------------------------------------

class DataManager:
    """Unified data layer wrapping Qlib local data.

    Parameters
    ----------
    provider_uri : str
        Path to Qlib data directory (e.g. ``~/.qlib/qlib_data/cn_data``).
    market : str
        Stock pool name (``"csi300"`` / ``"csi500"`` / ``"all"``).
    qlib_dir : str | None
        Path to the qlib source repo (for running collector scripts).
        Defaults to ``../qlib`` relative to this file's grandparent.
    """

    # A-share price limit ratio
    LIMIT_PCT = 0.10

    # Supported data sources for incremental update
    SOURCE_YAHOO = "yahoo"
    SOURCE_BAOSTOCK = "baostock"
    SOURCE_CSV = "csv"

    def __init__(
        self,
        provider_uri: str = "~/.qlib/qlib_data/cn_data",
        market: str = "csi300",
        qlib_dir: Optional[str] = None,
        update_source: str = "yahoo",
    ):
        self.provider_uri = str(Path(provider_uri).expanduser().resolve())
        self.market = market
        self._qlib_dir = qlib_dir  # kept for backward compat; used by _compat if set
        self.update_source = update_source
        self._initialized = False
        self._D = None  # lazy-loaded Qlib data accessor

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def init_qlib(self) -> None:
        """Initialize (or re-initialize) the Qlib runtime."""
        import qlib
        qlib.init(
            provider_uri=self.provider_uri,
            region_name="cn",
        )
        from qlib.data import D
        self._D = D
        self._initialized = True
        logger.info("Qlib initialized, provider_uri=%s", self.provider_uri)

    def _ensure_init(self) -> None:
        if not self._initialized:
            self.init_qlib()

    # ------------------------------------------------------------------
    # Data Update
    # ------------------------------------------------------------------

    def get_latest_date(self) -> pd.Timestamp:
        """Read the last line of ``calendars/day.txt`` to get the latest date in local data."""
        cal_path = Path(self.provider_uri) / "calendars" / "day.txt"
        if not cal_path.exists():
            raise FileNotFoundError(f"Calendar file not found: {cal_path}")
        with open(cal_path, "r") as f:
            lines = f.read().strip().splitlines()
        if not lines:
            raise ValueError("Calendar file is empty")
        return pd.Timestamp(lines[-1].strip())

    def ensure_data_updated(self, end_date: Optional[str] = None) -> bool:
        """Check local data freshness and run incremental update if needed.

        Parameters
        ----------
        end_date : str or None
            Target date string, e.g. ``"2025-03-11"``.
            If None, defaults to today.

        Returns
        -------
        bool
            True if an update was performed, False if data was already fresh.

        Raises
        ------
        RuntimeError
            If the update fails.
        """
        if end_date is None:
            end_date = pd.Timestamp.now().strftime("%Y-%m-%d")
        target = pd.Timestamp(end_date)

        try:
            latest = self.get_latest_date()
            if latest >= target:
                logger.info(
                    "Data already up-to-date (latest=%s, target=%s)",
                    latest.strftime("%Y-%m-%d"),
                    end_date,
                )
                return False
            logger.info(
                "Data outdated: latest=%s, target=%s, source=%s",
                latest.strftime("%Y-%m-%d"),
                end_date,
                self.update_source,
            )
        except FileNotFoundError:
            logger.warning("No existing data found, will run full download first")

        if self.update_source == self.SOURCE_YAHOO:
            self._update_via_yahoo(end_date)
        elif self.update_source == self.SOURCE_BAOSTOCK:
            self._update_via_baostock(end_date)
        elif self.update_source == self.SOURCE_CSV:
            self._update_via_csv()
        else:
            raise ValueError(f"Unknown update_source: {self.update_source}")

        # Verify update succeeded
        new_latest = self.get_latest_date()
        logger.info("Update done. Data now covers up to %s", new_latest.strftime("%Y-%m-%d"))

        # Re-initialize Qlib so D object picks up new data
        self.init_qlib()
        return True

    # ------------------------------------------------------------------
    # Update source: Yahoo Finance (Qlib built-in)
    # ------------------------------------------------------------------

    def _update_via_yahoo(self, end_date: str) -> None:
        """Run Qlib's Yahoo collector ``update_data_to_bin`` via subprocess.

        This is Qlib's built-in updater. It downloads from Yahoo Finance,
        normalizes, and appends to the local bin files.

        Pros: no extra dependency, updates index components (CSI300 etc.)
        Cons: Yahoo A-share data can be delayed or have gaps
        """
        collector_script = (
            Path(self._qlib_dir)
            / "scripts"
            / "data_collector"
            / "yahoo"
            / "collector.py"
        )
        if not collector_script.exists():
            raise FileNotFoundError(
                f"Yahoo collector not found at {collector_script}. "
                f"Set qlib_dir to the qlib repository root."
            )

        cmd = [
            sys.executable,
            str(collector_script),
            "update_data_to_bin",
            "--qlib_data_1d_dir", self.provider_uri,
            "--end_date", end_date,
        ]
        logger.info("Running Yahoo collector: %s", " ".join(cmd))
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=3600
        )

        if result.returncode != 0:
            logger.error("Yahoo collector stderr:\n%s", result.stderr[-2000:])
            raise RuntimeError(
                f"Yahoo update failed (exit code {result.returncode}). "
                f"Try switching to baostock: DataManager(update_source='baostock')"
            )
        logger.info("Yahoo update completed")

    # ------------------------------------------------------------------
    # Update source: Baostock (free, stable for A-shares)
    # ------------------------------------------------------------------

    def _update_via_baostock(self, end_date: str) -> None:
        """Download A-share daily data via baostock SDK and dump to bin.

        Baostock is free, no rate limit, and covers all A-share stocks.
        It does not require a token — just ``pip install baostock``.

        Flow: baostock API → CSV files → DumpDataUpdate → bin files
        """
        try:
            import baostock as bs
        except ImportError:
            raise ImportError(
                "baostock is required for this update source. "
                "Install with: pip install baostock"
            )

        latest = self.get_latest_date()
        start_date = (latest + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

        if pd.Timestamp(start_date) > pd.Timestamp(end_date):
            logger.info("No new dates to fetch")
            return

        logger.info("Baostock: fetching %s ~ %s", start_date, end_date)

        # Login
        login_result = bs.login()
        if login_result.error_code != "0":
            raise RuntimeError(f"Baostock login failed: {login_result.error_msg}")

        try:
            self._baostock_download_and_dump(bs, start_date, end_date)
        finally:
            bs.logout()

    def _baostock_download_and_dump(self, bs, start_date: str, end_date: str) -> None:
        """Core logic: query baostock → save CSV → dump bin."""
        import sys
        print(">>> 步骤1: 获取A股股票列表...", flush=True)
        sys.stdout.flush()

        csv_dir = Path(self.provider_uri).parent / "_baostock_tmp"
        csv_dir.mkdir(parents=True, exist_ok=True)

        stock_rs = bs.query_stock_industry()
        print(f">>> query_stock_industry 响应: error_code={stock_rs.error_code}, error_msg={stock_rs.error_msg}", flush=True)

        stock_list = []
        count = 0
        while stock_rs.error_code == "0" and stock_rs.next():
            row = stock_rs.get_row_data()
            code = row[1]
            stock_list.append(code)
            count += 1
            if count % 1000 == 0:
                print(f">>> 已读取 {count} 只股票...", flush=True)

        print(f">>> 共获取 {len(stock_list)} 只股票", flush=True)

        if not stock_list:
            print(">>> 股票列表为空，尝试使用 instruments 文件...", flush=True)
            inst_path = Path(self.provider_uri) / "instruments" / "all.txt"
            if inst_path.exists():
                with open(inst_path) as f:
                    for line in f:
                        parts = line.strip().split()
                        if parts:
                            sym = parts[0]
                            bs_code = sym[:2].lower() + "." + sym[2:]
                            stock_list.append(bs_code)
            print(f">>> 从 instruments 读取 {len(stock_list)} 只股票", flush=True)

        logger.info("Fetching %d stocks from baostock", len(stock_list))
        print(f">>> 步骤2: 开始下载数据, 共 {len(stock_list)} 只股票, 日期范围 {start_date} ~ {end_date}", flush=True)

        fields = "date,open,high,low,close,volume,amount"
        all_frames = []

        for i, code in enumerate(stock_list):
            if i == 0:
                print(f">>> 开始下载第1只股票: {code}", flush=True)
            elif i == 5:
                print(f">>> 已下载 {i} 只股票...", flush=True)
            elif i % 100 == 0:
                print(f">>> 已下载 {i} / {len(stock_list)} 只股票...", flush=True)

            rs = bs.query_history_k_data_plus(
                code, fields,
                start_date=start_date,
                end_date=end_date,
                frequency="d",
                adjustflag="2",
            )
            rows = []
            while rs.error_code == "0" and rs.next():
                rows.append(rs.get_row_data())

            if not rows:
                continue

            df = pd.DataFrame(rows, columns=fields.split(","))
            qlib_symbol = code[:2].upper() + code[3:]
            df.insert(0, "symbol", qlib_symbol)

            for col in ["open", "high", "low", "close", "volume", "amount"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")

            all_frames.append(df)

            if (i + 1) % 500 == 0:
                logger.info("  Progress: %d / %d stocks", i + 1, len(stock_list))

        print(f">>> 下载完成, 共 {len(all_frames)} 只股票有数据", flush=True)

        if not all_frames:
            logger.warning("No new data from baostock")
            return

        merged = pd.concat(all_frames, ignore_index=True)
        logger.info("Downloaded %d rows for %d stocks", len(merged), len(all_frames))

        # Save to CSV per symbol (DumpDataUpdate expects this)
        for symbol, group in merged.groupby("symbol"):
            csv_path = csv_dir / f"{symbol}.csv"
            group.to_csv(csv_path, index=False)

        # Dump CSV to bin using Qlib's DumpDataUpdate
        from quantlab._compat import import_dump_bin
        DumpDataUpdate = import_dump_bin()

        dumper = DumpDataUpdate(
            data_path=str(csv_dir),
            qlib_dir=self.provider_uri,
            exclude_fields="symbol,date",
            date_field_name="date",
            symbol_field_name="symbol",
        )
        dumper.dump()
        logger.info("Baostock data dumped to bin")

        # Update calendar
        self._update_calendar(merged["date"].unique())

        # Clean up temp CSVs
        import shutil
        shutil.rmtree(csv_dir, ignore_errors=True)

    def _update_calendar(self, new_dates) -> None:
        """Append new trading dates to calendars/day.txt."""
        cal_path = Path(self.provider_uri) / "calendars" / "day.txt"
        existing = set()
        if cal_path.exists():
            with open(cal_path) as f:
                existing = set(line.strip() for line in f if line.strip())

        new_dates_str = sorted(set(str(d) for d in new_dates) - existing)
        if new_dates_str:
            with open(cal_path, "a") as f:
                for d in new_dates_str:
                    f.write(d + "\n")
            logger.info("Appended %d new dates to calendar", len(new_dates_str))

    # ------------------------------------------------------------------
    # Update source: CSV (manual import)
    # ------------------------------------------------------------------

    def _update_via_csv(self) -> None:
        """Import CSV files from a staging directory and dump to bin.

        Place CSV files in ``{provider_uri}/../csv_import/`` with format:
        ``{SYMBOL}.csv`` containing columns: date,open,high,low,close,volume,amount
        """
        csv_dir = Path(self.provider_uri).parent / "csv_import"
        if not csv_dir.exists() or not list(csv_dir.glob("*.csv")):
            raise FileNotFoundError(
                f"No CSV files found in {csv_dir}. "
                f"Place CSV files there with columns: date,open,high,low,close,volume,amount"
            )

        from quantlab._compat import import_dump_bin
        DumpDataUpdate = import_dump_bin()

        dumper = DumpDataUpdate(
            data_path=str(csv_dir),
            qlib_dir=self.provider_uri,
            exclude_fields="symbol,date",
            date_field_name="date",
            symbol_field_name="symbol",
        )
        dumper.dump()
        logger.info("CSV import completed")

        # Update calendar from imported CSVs
        all_dates = set()
        for csv_file in csv_dir.glob("*.csv"):
            df = pd.read_csv(csv_file, usecols=["date"])
            all_dates.update(df["date"].tolist())
        self._update_calendar(all_dates)

    # ------------------------------------------------------------------
    # Trading Calendar
    # ------------------------------------------------------------------

    def get_trading_calendar(
        self,
        start: str,
        end: str,
    ) -> List[pd.Timestamp]:
        """Return list of trading dates in [start, end]."""
        self._ensure_init()
        return self._D.calendar(start_time=start, end_time=end, freq="day")

    def has_date(self, date: str) -> bool:
        """Check whether *date* exists in the trading calendar."""
        self._ensure_init()
        ts = pd.Timestamp(date)
        cal = self._D.calendar(start_time=date, end_time=date, freq="day")
        return len(cal) > 0 and cal[0] == ts

    # ------------------------------------------------------------------
    # Instrument helpers
    # ------------------------------------------------------------------

    def _get_instrument_list(
        self,
        anchor_date: str,
    ) -> List[str]:
        """Return stock list active on *anchor_date* for current market."""
        self._ensure_init()
        instruments = self._D.instruments(market=self.market)
        inst_list = self._D.list_instruments(
            instruments,
            start_time=anchor_date,
            end_time=anchor_date,
            as_list=True,
        )
        return inst_list

    # ------------------------------------------------------------------
    # OHLCV Access (time-isolated)
    # ------------------------------------------------------------------

    def get_ohlcv_before(
        self,
        anchor_date: str,
        lookback_days: int = 60,
    ) -> Dict[str, pd.DataFrame]:
        """Get OHLCV data for each stock, strictly ≤ anchor_date.

        Parameters
        ----------
        anchor_date : str
            The latest date allowed in the returned data.
        lookback_days : int
            Number of *trading days* to look back.

        Returns
        -------
        dict
            ``{symbol: DataFrame}`` with columns
            ``[open, high, low, close, volume]`` and DatetimeIndex.
        """
        self._ensure_init()

        # Determine start date from calendar
        cal = self._D.calendar(end_time=anchor_date, freq="day")
        if len(cal) == 0:
            return {}
        start_idx = max(0, len(cal) - lookback_days)
        start_date = cal[start_idx]

        instruments = self._D.instruments(market=self.market)
        fields = ["$open", "$high", "$low", "$close", "$volume"]
        df = self._D.features(
            instruments,
            fields,
            start_time=start_date,
            end_time=anchor_date,
            freq="day",
        )
        if df is None or df.empty:
            return {}

        # D.features returns MultiIndex (instrument, datetime)
        df.columns = ["open", "high", "low", "close", "volume"]
        result = {}
        for symbol in df.index.get_level_values(0).unique():
            sym_df = df.loc[symbol].copy()
            # Enforce time isolation: drop anything after anchor_date
            sym_df = sym_df[sym_df.index <= pd.Timestamp(anchor_date)]
            if not sym_df.empty:
                result[symbol] = sym_df

        return result

    def get_close_prices(self, anchor_date: str) -> pd.Series:
        """Get close prices on *anchor_date* for all stocks in the market.

        Returns
        -------
        Series
            Index = symbol, values = close price.
        """
        self._ensure_init()
        instruments = self._D.instruments(market=self.market)
        df = self._D.features(
            instruments,
            ["$close"],
            start_time=anchor_date,
            end_time=anchor_date,
            freq="day",
        )
        if df is None or df.empty:
            return pd.Series(dtype=float)

        df.columns = ["close"]
        # Extract cross-section: one value per symbol
        series = df["close"].droplevel("datetime")
        return series

    def get_open_prices(self, date: str) -> pd.Series:
        """Get open prices on *date*. Used by M6 for T+1 execution only."""
        self._ensure_init()
        instruments = self._D.instruments(market=self.market)
        df = self._D.features(
            instruments,
            ["$open"],
            start_time=date,
            end_time=date,
            freq="day",
        )
        if df is None or df.empty:
            return pd.Series(dtype=float)

        df.columns = ["open"]
        return df["open"].droplevel("datetime")

    def get_daily_returns(self, date: str) -> pd.Series:
        """Get daily returns (close-to-close) on *date* for all stocks.

        Returns ``(close_date / close_prev - 1)`` as a cross-sectional Series.
        """
        self._ensure_init()
        instruments = self._D.instruments(market=self.market)
        # Ref($close, 1) is *yesterday's* close in Qlib's convention
        df = self._D.features(
            instruments,
            ["$close/Ref($close, 1) - 1"],
            start_time=date,
            end_time=date,
            freq="day",
        )
        if df is None or df.empty:
            return pd.Series(dtype=float)

        df.columns = ["return"]
        return df["return"].droplevel("datetime")

    # ------------------------------------------------------------------
    # Limit prices (涨跌停)
    # ------------------------------------------------------------------

    def get_limit_prices(
        self,
        date: str,
    ) -> Tuple[pd.Series, pd.Series]:
        """Compute limit-up and limit-down prices for *date*.

        Uses the *previous* trading day's close × (1 ± 10%).

        Returns
        -------
        (limit_up, limit_down)
            Each is ``Series[symbol → price]``.
        """
        self._ensure_init()
        instruments = self._D.instruments(market=self.market)
        # Ref($close, 1) = previous day's close
        df = self._D.features(
            instruments,
            ["Ref($close, 1)"],
            start_time=date,
            end_time=date,
            freq="day",
        )
        if df is None or df.empty:
            empty = pd.Series(dtype=float)
            return empty, empty

        df.columns = ["prev_close"]
        prev_close = df["prev_close"].droplevel("datetime")

        limit_up = (prev_close * (1 + self.LIMIT_PCT)).round(2)
        limit_down = (prev_close * (1 - self.LIMIT_PCT)).round(2)
        return limit_up, limit_down

    # ------------------------------------------------------------------
    # Alpha158 features
    # ------------------------------------------------------------------

    def get_alpha158_features(self, anchor_date: str) -> pd.DataFrame:
        """Get Alpha158 cross-sectional features on *anchor_date*.

        Returns
        -------
        DataFrame
            MultiIndex (instrument, datetime), 158 feature columns.
        """
        self._ensure_init()
        from qlib.contrib.data.handler import Alpha158

        handler = Alpha158(
            instruments=self.market,
            start_time=anchor_date,
            end_time=anchor_date,
        )
        return handler.fetch()

    # ------------------------------------------------------------------
    # Industry mapping
    # ------------------------------------------------------------------

    def get_industry_map(self) -> pd.Series:
        """Return Shenwan Level-1 industry classification.

        Returns
        -------
        Series
            Index = symbol, values = industry name/code.

        Notes
        -----
        Qlib does not ship industry data by default. This method tries to
        load from a local CSV file at ``{provider_uri}/../industry_map.csv``.
        If not found, returns a dummy mapping (all stocks → "unknown").
        Users should prepare this file with columns: symbol, industry.
        """
        industry_path = Path(self.provider_uri).parent / "industry_map.csv"
        if industry_path.exists():
            df = pd.read_csv(industry_path, dtype=str)
            return df.set_index("symbol")["industry"]
        else:
            logger.warning(
                "Industry map not found at %s, using dummy mapping. "
                "Please prepare a CSV with columns [symbol, industry].",
                industry_path,
            )
            self._ensure_init()
            instruments = self._D.instruments(market=self.market)
            inst_list = self._D.list_instruments(instruments, as_list=True)
            return pd.Series("unknown", index=inst_list, name="industry")

    # ------------------------------------------------------------------
    # Convenience: benchmark NAV
    # ------------------------------------------------------------------

    def get_benchmark_nav(
        self,
        benchmark: str = "SH000300",
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> pd.Series:
        """Get benchmark index close price series.

        Parameters
        ----------
        benchmark : str
            Index symbol, e.g. ``"SH000300"`` for CSI300.
        """
        self._ensure_init()
        df = self._D.features(
            [benchmark],
            ["$close"],
            start_time=start,
            end_time=end,
            freq="day",
        )
        if df is None or df.empty:
            return pd.Series(dtype=float)

        df.columns = ["close"]
        return df["close"].droplevel("instrument")
