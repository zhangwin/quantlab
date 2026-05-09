"""I/O helpers for building agentic review contexts from QuantLab outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from quantlab.agentic.schemas import CandidateStock, PortfolioContext, ReviewContext


def load_industry_map(path: str | Path) -> dict[str, str]:
    p = Path(path)
    if not p.exists():
        return {}
    df = pd.read_csv(p, dtype=str)
    if "symbol" not in df.columns or "industry" not in df.columns:
        return {}
    return dict(zip(df["symbol"].str.upper(), df["industry"]))


def load_sector_flow_for_date(
    path: str | Path,
    date: str,
    industry_names: Iterable[str] | None = None,
) -> dict[str, dict[str, float]]:
    p = Path(path)
    if not p.exists():
        return {}
    df = pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)
    df["date"] = df["date"].astype(str)
    df = df[df["date"] == date]
    if industry_names is not None:
        industry_set = set(industry_names)
        df = df[df["sector_name"].isin(industry_set)]

    out: dict[str, dict[str, float]] = {}
    numeric_cols = [
        "flow_pctile_1d",
        "flow_pctile_5d",
        "flow_pctile_10d",
        "flow_rank_1d",
        "flow_rank_5d",
        "flow_rank_10d",
        "net_amount",
        "net_pct",
        "net_amount_z_20d",
        "net_pct_z_20d",
        "flow_accel_5d",
        "flow_persistence_5d",
    ]
    for _, row in df.iterrows():
        name = str(row.get("sector_name", "")).strip()
        if not name:
            continue
        out[name] = {
            col: float(row[col])
            for col in numeric_cols
            if col in row.index and pd.notna(row[col])
        }
    return out


def context_from_pipeline_meta(
    date: str,
    meta_path: str | Path,
    industry_map: dict[str, str] | None = None,
    sector_flow: dict[str, dict[str, float]] | None = None,
    target_k: int = 5,
    candidate_n: int = 20,
) -> ReviewContext:
    df = pd.read_csv(meta_path, index_col=0)
    df.index = df.index.astype(str).str.upper()
    if "rank_by_selector" in df.columns:
        df = df.sort_values("rank_by_selector")
    elif "selector_score" in df.columns:
        df = df.sort_values("selector_score", ascending=False)
    else:
        df = df.head(candidate_n)

    df = df.head(candidate_n)
    industry_map = industry_map or {}
    candidates = []
    for i, (symbol, row) in enumerate(df.iterrows(), start=1):
        features = {
            col: float(row[col])
            for col in [
                "dir_agreement",
                "slope_agreement",
                "path_consistency",
                "pred_endpoint",
                "pred_mean",
                "pred_slope",
                "uncertainty",
                "selector_score",
            ]
            if col in df.columns and pd.notna(row[col])
        }
        candidates.append(
            CandidateStock(
                symbol=symbol,
                rank=int(row["rank_by_selector"]) if "rank_by_selector" in df.columns else i,
                score=float(row["selector_score"]) if "selector_score" in df.columns and pd.notna(row["selector_score"]) else float(-i),
                selected=bool(row["selected"]) if "selected" in df.columns else i <= target_k,
                industry=industry_map.get(symbol),
                last_close=float(row["last_close"]) if "last_close" in df.columns and pd.notna(row["last_close"]) else None,
                features=features,
            )
        )

    return ReviewContext(
        date=date,
        candidates=candidates,
        portfolio=PortfolioContext(target_k=target_k),
        sector_flow=sector_flow or {},
        metadata={"meta_path": str(meta_path), "candidate_n": candidate_n},
    )

