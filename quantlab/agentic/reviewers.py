"""Deterministic reviewers for the agentic layer.

These reviewers are deliberately rule-based. They make the review layer
backtestable today while leaving room for LLM reviewers that implement the same
BaseReviewer interface later.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import Counter

from quantlab.agentic.schemas import (
    AgentAction,
    ReviewContext,
    ReviewFinding,
    ReviewSeverity,
)


class BaseReviewer(ABC):
    name = "base"

    @abstractmethod
    def review(self, context: ReviewContext) -> list[ReviewFinding]:
        """Return structured findings for a review context."""


class LiquidityReviewer(BaseReviewer):
    name = "liquidity"

    def __init__(
        self,
        min_avg_amount_20d: float = 30_000_000.0,
        critical_avg_amount_20d: float = 10_000_000.0,
    ):
        self.min_avg_amount_20d = min_avg_amount_20d
        self.critical_avg_amount_20d = critical_avg_amount_20d

    def review(self, context: ReviewContext) -> list[ReviewFinding]:
        findings: list[ReviewFinding] = []
        veto = []
        review = []
        for item in context.selected_candidates():
            amount = item.avg_amount_20d
            if amount is None:
                continue
            if amount < self.critical_avg_amount_20d:
                veto.append(item.symbol)
            elif amount < self.min_avg_amount_20d:
                review.append(item.symbol)

        if veto:
            findings.append(
                ReviewFinding(
                    reviewer=self.name,
                    severity=ReviewSeverity.CRITICAL,
                    action=AgentAction.VETO,
                    message="Selected stocks include critically low 20-day average amount.",
                    symbols=veto,
                    details={"critical_avg_amount_20d": self.critical_avg_amount_20d},
                )
            )
        if review:
            findings.append(
                ReviewFinding(
                    reviewer=self.name,
                    severity=ReviewSeverity.WARNING,
                    action=AgentAction.REVIEW,
                    message="Selected stocks include low liquidity names.",
                    symbols=review,
                    details={"min_avg_amount_20d": self.min_avg_amount_20d},
                )
            )
        return findings


class SectorFlowReviewer(BaseReviewer):
    name = "sector_flow"

    def __init__(
        self,
        cold_pctile_threshold: float = 0.20,
        hot_pctile_threshold: float = 0.80,
        min_persistence_5d: float = 1.0,
        critical_net_pct_z: float = -1.0,
    ):
        self.cold_pctile_threshold = cold_pctile_threshold
        self.hot_pctile_threshold = hot_pctile_threshold
        self.min_persistence_5d = min_persistence_5d
        self.critical_net_pct_z = critical_net_pct_z

    def review(self, context: ReviewContext) -> list[ReviewFinding]:
        cold_symbols = []
        cold_industries = set()
        hot_symbols = []
        for item in context.selected_candidates():
            if not item.industry:
                continue
            flow = context.sector_flow.get(item.industry)
            if not flow:
                continue
            pctile = flow.get("flow_pctile_5d", flow.get("flow_pctile_1d"))
            persistence = flow.get("flow_persistence_5d")
            net_pct_z = flow.get("net_pct_z_20d")
            if (
                pctile is not None
                and pctile <= self.cold_pctile_threshold
                and (persistence is None or persistence <= self.min_persistence_5d)
                and (net_pct_z is None or net_pct_z <= self.critical_net_pct_z)
            ):
                cold_symbols.append(item.symbol)
                cold_industries.add(item.industry)
            elif pctile is not None and pctile >= self.hot_pctile_threshold:
                hot_symbols.append(item.symbol)

        findings: list[ReviewFinding] = []
        if cold_symbols:
            findings.append(
                ReviewFinding(
                    reviewer=self.name,
                    severity=ReviewSeverity.WARNING,
                    action=AgentAction.REDUCE,
                    message="Selected stocks have exposure to cold sector flow.",
                    symbols=cold_symbols,
                    industries=sorted(cold_industries),
                    position_scale=0.75,
                    details={
                        "cold_pctile_threshold": self.cold_pctile_threshold,
                        "critical_net_pct_z": self.critical_net_pct_z,
                    },
                )
            )
        if hot_symbols:
            findings.append(
                ReviewFinding(
                    reviewer=self.name,
                    severity=ReviewSeverity.INFO,
                    action=AgentAction.APPROVE,
                    message="Some selected stocks are aligned with strong sector flow.",
                    symbols=hot_symbols,
                    details={"hot_pctile_threshold": self.hot_pctile_threshold},
                )
            )
        return findings


class ConcentrationReviewer(BaseReviewer):
    name = "concentration"

    def __init__(self, max_selected_industry_share: float = 0.60, reduce_scale: float = 0.75):
        self.max_selected_industry_share = max_selected_industry_share
        self.reduce_scale = reduce_scale

    def review(self, context: ReviewContext) -> list[ReviewFinding]:
        selected = context.selected_candidates()
        industries = [item.industry or "unknown" for item in selected]
        if not industries:
            return []
        counts = Counter(industries)
        industry, count = counts.most_common(1)[0]
        share = count / len(selected)
        if share <= self.max_selected_industry_share:
            return []
        symbols = [item.symbol for item in selected if (item.industry or "unknown") == industry]
        return [
            ReviewFinding(
                reviewer=self.name,
                severity=ReviewSeverity.WARNING,
                action=AgentAction.REDUCE,
                message="Selected candidates are concentrated in one industry.",
                symbols=symbols,
                industries=[industry],
                position_scale=self.reduce_scale,
                details={
                    "industry": industry,
                    "share": share,
                    "max_selected_industry_share": self.max_selected_industry_share,
                },
            )
        ]

