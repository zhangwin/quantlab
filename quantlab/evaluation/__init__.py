"""Evaluation layer: performance, trade analysis, signal attribution, regime diagnosis.

M8: evaluation — 绩效计算 + 交易分析 + 信号归因 + 分环境诊断
"""

from quantlab.evaluation.evaluation import (
    CostBreakdown,
    EvaluationPipeline,
    EvaluationReport,
    HoldingAnalysis,
    PerformanceCalculator,
    PerformanceSummary,
    RegimeAnalyzer,
    RiskImpactReport,
    SignalAttributor,
    TradeAnalyzer,
    TradeSummary,
)

__all__ = [
    "PerformanceSummary",
    "PerformanceCalculator",
    "TradeSummary",
    "CostBreakdown",
    "HoldingAnalysis",
    "TradeAnalyzer",
    "SignalAttributor",
    "RegimeAnalyzer",
    "RiskImpactReport",
    "EvaluationReport",
    "EvaluationPipeline",
]
