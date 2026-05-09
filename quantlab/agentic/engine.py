"""Agentic review engine."""

from __future__ import annotations

from quantlab.agentic.reviewers import BaseReviewer
from quantlab.agentic.schemas import (
    AgentAction,
    ReviewContext,
    ReviewDecision,
    ReviewFinding,
    ReviewSeverity,
)


class AgenticReviewEngine:
    """Run reviewers and aggregate their findings into a structured decision."""

    def __init__(
        self,
        reviewers: list[BaseReviewer],
        target_k: int | None = None,
        allow_replacement: bool = True,
    ):
        self.reviewers = reviewers
        self.target_k = target_k
        self.allow_replacement = allow_replacement

    def review(self, context: ReviewContext) -> ReviewDecision:
        findings: list[ReviewFinding] = []
        for reviewer in self.reviewers:
            findings.extend(reviewer.review(context))

        veto_symbols = sorted({s for f in findings if f.action == AgentAction.VETO for s in f.symbols})
        reduce_symbols = sorted({s for f in findings if f.action == AgentAction.REDUCE for s in f.symbols})
        review_symbols = sorted({s for f in findings if f.action == AgentAction.REVIEW for s in f.symbols})

        position_scale = 1.0
        for finding in findings:
            if finding.position_scale is not None:
                position_scale = min(position_scale, finding.position_scale)

        if any(f.severity == ReviewSeverity.CRITICAL for f in findings):
            action = AgentAction.VETO
        elif any(f.action == AgentAction.REDUCE for f in findings):
            action = AgentAction.REDUCE
        elif any(f.action == AgentAction.REVIEW for f in findings):
            action = AgentAction.REVIEW
        else:
            action = AgentAction.APPROVE

        final_candidates = self._build_final_candidates(context, veto_symbols)
        return ReviewDecision(
            date=context.date,
            action=action,
            position_scale=position_scale,
            veto_symbols=veto_symbols,
            reduce_symbols=reduce_symbols,
            review_symbols=review_symbols,
            findings=findings,
            final_candidates=final_candidates,
            metadata={
                "reviewer_count": len(self.reviewers),
                "finding_count": len(findings),
                "allow_replacement": self.allow_replacement,
            },
        )

    def _build_final_candidates(self, context: ReviewContext, veto_symbols: list[str]) -> list[str]:
        target_k = self.target_k or context.portfolio.target_k
        veto = set(veto_symbols)
        selected = [item for item in context.selected_candidates() if item.symbol not in veto]
        if not self.allow_replacement:
            return [item.symbol for item in selected]

        selected_symbols = {item.symbol for item in selected}
        replacements = []
        for item in sorted(context.candidates, key=lambda c: c.rank):
            if len(selected) + len(replacements) >= target_k:
                break
            if item.symbol in veto or item.symbol in selected_symbols:
                continue
            replacements.append(item)
        return [item.symbol for item in selected + replacements][:target_k]

