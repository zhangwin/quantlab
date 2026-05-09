#!/usr/bin/env python3
"""Run agentic review on pipeline_daily candidates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

if __package__ in {None, ""}:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from quantlab.agentic.engine import AgenticReviewEngine  # noqa: E402
from quantlab.agentic.io import (  # noqa: E402
    context_from_pipeline_meta,
    load_industry_map,
    load_sector_flow_for_date,
)
from quantlab.agentic.reviewers import ConcentrationReviewer, LiquidityReviewer, SectorFlowReviewer  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run QuantLab agentic review.")
    parser.add_argument("--date", required=True)
    parser.add_argument("--meta-path", required=True)
    parser.add_argument("--industry-map", default="quantlab/industry_map.csv")
    parser.add_argument("--sector-flow", default="quantlab/A_data_ext/sector_flow/features/sector_flow_features.csv")
    parser.add_argument("--output-dir", default="quantlab/experiments/reports/agentic_review")
    parser.add_argument("--target-k", type=int, default=5)
    parser.add_argument("--candidate-n", type=int, default=20)
    parser.add_argument("--no-replacement", action="store_true")
    parser.add_argument("--disable-liquidity", action="store_true")
    parser.add_argument("--disable-sector-flow", action="store_true")
    parser.add_argument("--disable-concentration", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    industry_map = load_industry_map(args.industry_map)
    sector_flow = load_sector_flow_for_date(args.sector_flow, args.date)
    context = context_from_pipeline_meta(
        date=args.date,
        meta_path=args.meta_path,
        industry_map=industry_map,
        sector_flow=sector_flow,
        target_k=args.target_k,
        candidate_n=args.candidate_n,
    )

    reviewers = []
    if not args.disable_liquidity:
        reviewers.append(LiquidityReviewer())
    if not args.disable_sector_flow:
        reviewers.append(SectorFlowReviewer())
    if not args.disable_concentration:
        reviewers.append(ConcentrationReviewer())

    engine = AgenticReviewEngine(
        reviewers=reviewers,
        target_k=args.target_k,
        allow_replacement=not args.no_replacement,
    )
    decision = engine.review(context)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{args.date}_review.json"
    csv_path = out_dir / f"{args.date}_findings.csv"
    json_path.write_text(
        json.dumps(decision.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    pd.DataFrame([f.to_dict() for f in decision.findings]).to_csv(csv_path, index=False)

    print("[done] agentic review")
    print(f"  date            : {args.date}")
    print(f"  action          : {decision.action.value}")
    print(f"  position_scale  : {decision.position_scale}")
    print(f"  final_candidates: {','.join(decision.final_candidates)}")
    print(f"  findings        : {len(decision.findings)}")
    print(f"  json            : {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

