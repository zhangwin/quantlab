"""Utilities for sector fund-flow data.

Sector flow data is stored outside the Qlib provider directory so it can be
validated and versioned independently from OHLCV bars.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


DEFAULT_EXT_DIR = Path("quantlab/A_data_ext/sector_flow")


STANDARD_COLUMNS = [
    "date",
    "vendor",
    "sector_type",
    "sector_code",
    "sector_name",
    "pct_change",
    "close",
    "net_amount",
    "net_pct",
    "buy_elg_amount",
    "buy_elg_pct",
    "buy_lg_amount",
    "buy_lg_pct",
    "buy_md_amount",
    "buy_md_pct",
    "buy_sm_amount",
    "buy_sm_pct",
    "lead_stock",
    "company_num",
    "source_indicator",
    "source_update_time",
]


def read_calendar(data_dir: str | Path) -> list[str]:
    path = Path(data_dir) / "calendars" / "day.txt"
    if not path.exists():
        raise FileNotFoundError(path)
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def normalize_date(value) -> str | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        if len(text) == 8 and text.isdigit():
            return pd.Timestamp(text).strftime("%Y-%m-%d")
        return pd.Timestamp(text).strftime("%Y-%m-%d")
    except Exception:
        return None


def to_float(value) -> float:
    if value is None:
        return np.nan
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    text = str(value).strip().replace(",", "")
    if not text or text in {"-", "--", "None", "nan", "NaN"}:
        return np.nan
    multiplier = 1.0
    if text.endswith("%"):
        text = text[:-1]
    if text.endswith("亿"):
        multiplier = 1e8
        text = text[:-1]
    elif text.endswith("万"):
        multiplier = 1e4
        text = text[:-1]
    try:
        return float(text) * multiplier
    except Exception:
        return np.nan


def find_col(columns: Iterable[str], *patterns: str) -> str | None:
    cols = [str(c) for c in columns]
    for pattern in patterns:
        for col in cols:
            if pattern in col:
                return col
    return None


def ensure_standard_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in STANDARD_COLUMNS:
        if col not in out.columns:
            out[col] = np.nan
    return out[STANDARD_COLUMNS]


def write_table(df: pd.DataFrame, path_base: Path) -> None:
    path_base.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path_base.with_suffix(".csv"), index=False, encoding="utf-8")
    try:
        df.to_parquet(path_base.with_suffix(".parquet"), index=False)
    except Exception as exc:
        print(f"[WARN] parquet write failed for {path_base.with_suffix('.parquet')}: {exc}")


def read_table(path_base: str | Path) -> pd.DataFrame:
    path = Path(path_base)
    if path.suffix:
        if path.suffix == ".parquet":
            return pd.read_parquet(path)
        return pd.read_csv(path)
    parquet = path.with_suffix(".parquet")
    csv = path.with_suffix(".csv")
    if parquet.exists():
        return pd.read_parquet(parquet)
    if csv.exists():
        return pd.read_csv(csv)
    raise FileNotFoundError(f"{parquet} or {csv}")

