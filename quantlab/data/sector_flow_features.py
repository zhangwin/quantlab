#!/usr/bin/env python3
"""Build rolling features from normalized sector fund-flow data."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from quantlab.data.sector_flow_utils import DEFAULT_EXT_DIR, read_calendar, read_table, write_table  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build sector flow features.")
    parser.add_argument("--data-dir", default="quantlab/A_data")
    parser.add_argument("--input", default=str(DEFAULT_EXT_DIR / "normalized" / "sector_flow_daily"))
    parser.add_argument("--output", default=str(DEFAULT_EXT_DIR / "features" / "sector_flow_features"))
    parser.add_argument("--vendor", default=None)
    parser.add_argument("--sector-type", default=None, choices=["industry", "concept", "region"])
    return parser.parse_args()


def rolling_z(s: pd.Series, window: int) -> pd.Series:
    mean = s.rolling(window, min_periods=max(3, window // 3)).mean()
    std = s.rolling(window, min_periods=max(3, window // 3)).std()
    return (s - mean) / (std + 1e-12)


def build_features(df: pd.DataFrame, calendar: list[str]) -> pd.DataFrame:
    df = df.copy()
    df["date"] = df["date"].astype(str)
    df = df[df["date"].isin(set(calendar))]
    df["sector_key"] = (
        df["vendor"].astype(str)
        + ":"
        + df["sector_type"].astype(str)
        + ":"
        + df["sector_code"].fillna("").astype(str)
        + ":"
        + df["sector_name"].astype(str)
    )
    df = df.sort_values(["sector_key", "date"]).reset_index(drop=True)

    frames = []
    for _, g in df.groupby("sector_key", sort=False):
        g = g.copy()
        g["net_amount_5d"] = g["net_amount"].rolling(5, min_periods=3).sum()
        g["net_amount_10d"] = g["net_amount"].rolling(10, min_periods=5).sum()
        g["net_amount_z_20d"] = rolling_z(g["net_amount"], 20)
        g["net_pct_z_20d"] = rolling_z(g["net_pct"], 20)
        g["flow_accel_5d"] = g["net_amount"] - g["net_amount"].rolling(5, min_periods=3).mean()
        g["flow_persistence_5d"] = (g["net_amount"] > 0).rolling(5, min_periods=3).sum()
        g["price_confirm"] = ((g["pct_change"] > 0) & (g["net_amount"] > 0)).astype(int)
        g["flow_reversal"] = ((g["net_amount_5d"].shift(1) < 0) & (g["net_amount"] > 0)).astype(int)
        frames.append(g)

    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if out.empty:
        return out

    rank_groups = ["date", "vendor", "sector_type"]
    out["flow_rank_1d"] = out.groupby(rank_groups)["net_amount"].rank(ascending=False, method="min")
    out["flow_rank_5d"] = out.groupby(rank_groups)["net_amount_5d"].rank(ascending=False, method="min")
    out["flow_rank_10d"] = out.groupby(rank_groups)["net_amount_10d"].rank(ascending=False, method="min")
    out["n_sectors"] = out.groupby(rank_groups)["sector_key"].transform("count")
    out["flow_pctile_1d"] = 1.0 - (out["flow_rank_1d"] - 1) / (out["n_sectors"] - 1).replace(0, np.nan)
    out["flow_pctile_5d"] = 1.0 - (out["flow_rank_5d"] - 1) / (out["n_sectors"] - 1).replace(0, np.nan)
    out["hot_sector"] = ((out["flow_pctile_1d"] >= 0.80) & (out["net_pct"] > 0)).astype(int)
    out["cold_sector"] = ((out["flow_pctile_1d"] <= 0.20) & (out["net_pct"] < 0)).astype(int)
    return out.sort_values(["date", "vendor", "sector_type", "flow_rank_1d"]).reset_index(drop=True)


def main() -> int:
    args = parse_args()
    df = read_table(args.input)
    if args.vendor:
        df = df[df["vendor"] == args.vendor]
    if args.sector_type:
        df = df[df["sector_type"] == args.sector_type]
    calendar = read_calendar(args.data_dir)
    out = build_features(df, calendar)
    if out.empty:
        raise SystemExit("[ERROR] no features generated")

    output = Path(args.output)
    write_table(out, output)
    print("[done] sector flow features")
    print(f"  rows      : {len(out)}")
    print(f"  dates     : {out['date'].min()} ~ {out['date'].max()}")
    print(f"  sectors   : {out['sector_key'].nunique()}")
    print(f"  output    : {output.with_suffix('.csv')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
