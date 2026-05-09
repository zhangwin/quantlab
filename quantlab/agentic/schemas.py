"""Structured schemas for agentic review.

These dataclasses are the boundary between quantitative selectors and any
agentic review implementation. Keep them serializable and deterministic.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class ReviewSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class AgentAction(str, Enum):
    APPROVE = "approve"
    REDUCE = "reduce"
    VETO = "veto"
    REVIEW = "review"


@dataclass
class CandidateStock:
    symbol: str
    rank: int
    score: float
    selected: bool = False
    industry: str | None = None
    last_close: float | None = None
    avg_amount_20d: float | None = None
    features: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PortfolioContext:
    total_value: float | None = None
    cash: float | None = None
    current_positions: dict[str, float] = field(default_factory=dict)
    target_k: int = 5
    max_industry_weight: float = 0.30
    max_single_weight: float = 0.20

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ReviewContext:
    date: str
    candidates: list[CandidateStock]
    portfolio: PortfolioContext = field(default_factory=PortfolioContext)
    sector_flow: dict[str, dict[str, float]] = field(default_factory=dict)
    market_state: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def selected_candidates(self) -> list[CandidateStock]:
        selected = [c for c in self.candidates if c.selected]
        if selected:
            return selected
        return sorted(self.candidates, key=lambda c: c.rank)[: self.portfolio.target_k]

    def to_dict(self) -> dict[str, Any]:
        return {
            "date": self.date,
            "candidates": [c.to_dict() for c in self.candidates],
            "portfolio": self.portfolio.to_dict(),
            "sector_flow": self.sector_flow,
            "market_state": self.market_state,
            "metadata": self.metadata,
        }


@dataclass
class ReviewFinding:
    reviewer: str
    severity: ReviewSeverity
    action: AgentAction
    message: str
    symbols: list[str] = field(default_factory=list)
    industries: list[str] = field(default_factory=list)
    position_scale: float | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["severity"] = self.severity.value
        data["action"] = self.action.value
        return data


@dataclass
class ReviewDecision:
    date: str
    action: AgentAction
    position_scale: float
    veto_symbols: list[str] = field(default_factory=list)
    reduce_symbols: list[str] = field(default_factory=list)
    review_symbols: list[str] = field(default_factory=list)
    findings: list[ReviewFinding] = field(default_factory=list)
    final_candidates: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "date": self.date,
            "action": self.action.value,
            "position_scale": self.position_scale,
            "veto_symbols": self.veto_symbols,
            "reduce_symbols": self.reduce_symbols,
            "review_symbols": self.review_symbols,
            "findings": [f.to_dict() for f in self.findings],
            "final_candidates": self.final_candidates,
            "metadata": self.metadata,
        }

