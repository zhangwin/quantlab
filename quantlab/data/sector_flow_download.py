#!/usr/bin/env python3
"""Download and normalize sector fund-flow data.

The default provider is AKShare because it needs no token. Tushare DC support is
included for environments with TUSHARE_TOKEN configured.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

if __package__ in {None, ""}:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from quantlab.data.sector_flow_utils import (  # noqa: E402
    DEFAULT_EXT_DIR,
    ensure_standard_columns,
    find_col,
    normalize_date,
    read_calendar,
    to_float,
    write_table,
)


AK_SECTOR_TYPE = {
    "industry": "行业资金流",
    "concept": "概念资金流",
    "region": "地域资金流",
}

TS_CONTENT_TYPE = {
    "industry": "行业",
    "concept": "概念",
    "region": "地域",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download sector fund-flow data.")
    parser.add_argument("--provider", default="akshare", choices=["akshare", "tushare_dc"])
    parser.add_argument("--data-dir", default="quantlab/A_data", help="Qlib data dir for calendar alignment.")
    parser.add_argument("--output-root", default=str(DEFAULT_EXT_DIR))
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--sector-type", default="industry", choices=["industry", "concept", "region"])
    parser.add_argument("--mode", default="hist", choices=["rank", "hist"], help="AKShare mode.")
    parser.add_argument("--indicators", default="今日,5日,10日", help="AKShare rank indicators.")
    parser.add_argument("--max-sectors", type=int, default=0, help="Limit AKShare historical sector fetches.")
    parser.add_argument("--sleep", type=float, default=0.2)
    parser.add_argument("--token", default=None, help="Tushare token; falls back to TUSHARE_TOKEN.")
    parser.add_argument("--force", action="store_true", help="Overwrite raw files.")
    return parser.parse_args()


def _filter_dates(df: pd.DataFrame, start: str | None, end: str | None) -> pd.DataFrame:
    if df.empty or "date" not in df.columns:
        return df
    out = df.copy()
    if start:
        out = out[out["date"] >= start]
    if end:
        out = out[out["date"] <= end]
    return out


def normalize_ak_hist(df: pd.DataFrame, sector_name: str, sector_type: str) -> pd.DataFrame:
    cols = df.columns
    date_col = find_col(cols, "日期")
    net_amount_col = find_col(cols, "主力净流入-净额", "主力净流入净额")
    net_pct_col = find_col(cols, "主力净流入-净占比", "主力净流入净占比")
    elg_amount_col = find_col(cols, "超大单净流入-净额")
    elg_pct_col = find_col(cols, "超大单净流入-净占比")
    lg_amount_col = find_col(cols, "大单净流入-净额")
    lg_pct_col = find_col(cols, "大单净流入-净占比")
    md_amount_col = find_col(cols, "中单净流入-净额")
    md_pct_col = find_col(cols, "中单净流入-净占比")
    sm_amount_col = find_col(cols, "小单净流入-净额")
    sm_pct_col = find_col(cols, "小单净流入-净占比")

    rows = []
    for _, row in df.iterrows():
        date = normalize_date(row.get(date_col)) if date_col else None
        if not date:
            continue
        rows.append(
            {
                "date": date,
                "vendor": "akshare_em",
                "sector_type": sector_type,
                "sector_code": "",
                "sector_name": sector_name,
                "pct_change": float("nan"),
                "close": float("nan"),
                "net_amount": to_float(row.get(net_amount_col)) if net_amount_col else float("nan"),
                "net_pct": to_float(row.get(net_pct_col)) if net_pct_col else float("nan"),
                "buy_elg_amount": to_float(row.get(elg_amount_col)) if elg_amount_col else float("nan"),
                "buy_elg_pct": to_float(row.get(elg_pct_col)) if elg_pct_col else float("nan"),
                "buy_lg_amount": to_float(row.get(lg_amount_col)) if lg_amount_col else float("nan"),
                "buy_lg_pct": to_float(row.get(lg_pct_col)) if lg_pct_col else float("nan"),
                "buy_md_amount": to_float(row.get(md_amount_col)) if md_amount_col else float("nan"),
                "buy_md_pct": to_float(row.get(md_pct_col)) if md_pct_col else float("nan"),
                "buy_sm_amount": to_float(row.get(sm_amount_col)) if sm_amount_col else float("nan"),
                "buy_sm_pct": to_float(row.get(sm_pct_col)) if sm_pct_col else float("nan"),
                "source_indicator": "hist",
            }
        )
    return ensure_standard_columns(pd.DataFrame(rows))


def normalize_ak_rank(df: pd.DataFrame, sector_type: str, indicator: str, asof_date: str) -> pd.DataFrame:
    cols = df.columns
    name_col = find_col(cols, "名称", "板块名称")
    code_col = find_col(cols, "代码", "板块代码")
    pct_col = find_col(cols, "涨跌幅")
    close_col = find_col(cols, "最新价", "指数", "收盘")
    net_amount_col = find_col(cols, "主力净流入-净额", "主力净流入净额", "净流入")
    net_pct_col = find_col(cols, "主力净流入-净占比", "主力净流入净占比", "净占比")

    rows = []
    for _, row in df.iterrows():
        sector_name = str(row.get(name_col, "")).strip() if name_col else ""
        if not sector_name:
            continue
        rows.append(
            {
                "date": asof_date,
                "vendor": "akshare_em",
                "sector_type": sector_type,
                "sector_code": str(row.get(code_col, "")).strip() if code_col else "",
                "sector_name": sector_name,
                "pct_change": to_float(row.get(pct_col)) if pct_col else float("nan"),
                "close": to_float(row.get(close_col)) if close_col else float("nan"),
                "net_amount": to_float(row.get(net_amount_col)) if net_amount_col else float("nan"),
                "net_pct": to_float(row.get(net_pct_col)) if net_pct_col else float("nan"),
                "source_indicator": indicator,
            }
        )
    return ensure_standard_columns(pd.DataFrame(rows))


def fetch_ak_rank(ak, sector_type: str, indicator: str) -> pd.DataFrame:
    # AKShare has changed argument names across versions, so try common forms.
    attempts = [
        {"indicator": indicator, "sector_type": AK_SECTOR_TYPE[sector_type]},
        {"indicator": indicator, "sector": AK_SECTOR_TYPE[sector_type]},
        {"symbol": indicator, "sector_type": AK_SECTOR_TYPE[sector_type]},
        {"indicator": indicator},
    ]
    last_exc = None
    for kwargs in attempts:
        try:
            return ak.stock_sector_fund_flow_rank(**kwargs)
        except TypeError as exc:
            last_exc = exc
            continue
    raise RuntimeError(f"AKShare stock_sector_fund_flow_rank failed: {last_exc}")


def download_akshare(args: argparse.Namespace, calendar: list[str]) -> pd.DataFrame:
    try:
        import akshare as ak
    except ImportError as exc:
        raise SystemExit("akshare is required. Install with: pip install akshare") from exc

    output_root = Path(args.output_root)
    raw_dir = output_root / "raw" / "akshare_em"
    raw_dir.mkdir(parents=True, exist_ok=True)

    start = args.start
    end = args.end or calendar[-1]
    asof_date = end

    frames: list[pd.DataFrame] = []
    indicators = [x.strip() for x in args.indicators.split(",") if x.strip()]

    print(f"[akshare] provider mode={args.mode} sector_type={args.sector_type} start={start} end={end}")

    rank_frames = []
    for indicator in indicators:
        raw_path = raw_dir / f"rank_{args.sector_type}_{indicator}.csv"
        if raw_path.exists() and not args.force:
            raw = pd.read_csv(raw_path)
        else:
            print(f"[akshare] rank {args.sector_type} {indicator}")
            raw = fetch_ak_rank(ak, args.sector_type, indicator)
            raw.to_csv(raw_path, index=False, encoding="utf-8")
        rank_frames.append(raw)
        if args.mode == "rank":
            frames.append(normalize_ak_rank(raw, args.sector_type, indicator, asof_date))

    if args.mode == "hist":
        if args.sector_type == "region":
            raise SystemExit("AKShare historical region sector flow is not supported by this script.")
        if not rank_frames:
            raise SystemExit("[ERROR] no rank data to discover sectors")
        name_col = find_col(rank_frames[0].columns, "名称", "板块名称")
        if not name_col:
            raise SystemExit(f"[ERROR] cannot find sector name column in rank data: {list(rank_frames[0].columns)}")
        sector_names = sorted(set(str(x).strip() for x in rank_frames[0][name_col].dropna() if str(x).strip()))
        if args.max_sectors > 0:
            sector_names = sector_names[: args.max_sectors]
        print(f"[akshare] historical sectors: {len(sector_names)}")

        for i, name in enumerate(sector_names, start=1):
            safe_name = name.replace("/", "_").replace("\\", "_")
            raw_path = raw_dir / f"hist_{args.sector_type}_{safe_name}.csv"
            try:
                if raw_path.exists() and not args.force:
                    raw = pd.read_csv(raw_path)
                else:
                    print(f"[akshare] hist {i}/{len(sector_names)} {name}")
                    if args.sector_type == "industry":
                        raw = ak.stock_sector_fund_flow_hist(symbol=name)
                    else:
                        raw = ak.stock_concept_fund_flow_hist(symbol=name)
                    raw.to_csv(raw_path, index=False, encoding="utf-8")
                    time.sleep(args.sleep)
                norm = normalize_ak_hist(raw, name, args.sector_type)
                frames.append(_filter_dates(norm, start, end))
            except Exception as exc:
                print(f"[WARN] hist failed {name}: {exc}")

    if not frames:
        return pd.DataFrame()
    return ensure_standard_columns(pd.concat(frames, ignore_index=True))


def download_tushare_dc(args: argparse.Namespace, calendar: list[str]) -> pd.DataFrame:
    try:
        import tushare as ts
    except ImportError as exc:
        raise SystemExit("tushare is required. Install with: pip install tushare") from exc

    import os

    token = args.token or os.environ.get("TUSHARE_TOKEN")
    if not token:
        raise SystemExit("TUSHARE_TOKEN is required for provider=tushare_dc")
    ts.set_token(token)
    pro = ts.pro_api()

    start = (args.start or calendar[0]).replace("-", "")
    end = (args.end or calendar[-1]).replace("-", "")
    content_type = TS_CONTENT_TYPE[args.sector_type]
    print(f"[tushare_dc] {content_type} {start} ~ {end}")
    raw = pro.moneyflow_ind_dc(start_date=start, end_date=end, content_type=content_type)

    output_root = Path(args.output_root)
    raw_dir = output_root / "raw" / "tushare_dc"
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw.to_csv(raw_dir / f"moneyflow_ind_dc_{args.sector_type}_{start}_{end}.csv", index=False, encoding="utf-8")

    rows = []
    for _, row in raw.iterrows():
        rows.append(
            {
                "date": normalize_date(row.get("trade_date")),
                "vendor": "tushare_dc",
                "sector_type": args.sector_type,
                "sector_code": row.get("ts_code", ""),
                "sector_name": row.get("name", ""),
                "pct_change": to_float(row.get("pct_change")),
                "close": to_float(row.get("close")),
                "net_amount": to_float(row.get("net_amount")),
                "net_pct": to_float(row.get("net_pct")),
                "source_indicator": "hist",
            }
        )
    return ensure_standard_columns(pd.DataFrame(rows).dropna(subset=["date", "sector_name"]))


def main() -> int:
    args = parse_args()
    calendar = read_calendar(args.data_dir)
    if args.end is None:
        args.end = calendar[-1]

    if args.provider == "akshare":
        df = download_akshare(args, calendar)
    else:
        df = download_tushare_dc(args, calendar)

    if df.empty:
        raise SystemExit("[ERROR] no sector flow data downloaded")

    calendar_set = set(calendar)
    before = len(df)
    df = df[df["date"].isin(calendar_set)].copy()
    df = df.drop_duplicates(["date", "vendor", "sector_type", "sector_code", "sector_name", "source_indicator"])
    df = df.sort_values(["date", "vendor", "sector_type", "sector_name"]).reset_index(drop=True)

    output_root = Path(args.output_root)
    norm_dir = output_root / "normalized"
    write_table(df, norm_dir / "sector_flow_daily")

    print()
    print("[done] normalized sector flow")
    print(f"  rows before calendar filter: {before}")
    print(f"  rows after calendar filter : {len(df)}")
    print(f"  date range                 : {df['date'].min()} ~ {df['date'].max()}")
    print(f"  sectors                    : {df['sector_name'].nunique()}")
    print(f"  output                     : {norm_dir / 'sector_flow_daily.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

