# Release Notes

## Unreleased

### Added

- Added `quantlab.agentic`, a reusable agentic review layer for QuantLab selection pipelines.
  - Defines serializable review schemas: `ReviewContext`, `CandidateStock`, `ReviewFinding`, and `ReviewDecision`.
  - Adds deterministic reviewers for liquidity, sector-flow exposure, and industry concentration.
  - Adds `AgenticReviewEngine` to aggregate reviewer findings into structured `approve`, `reduce`, `veto`, or `review` decisions.
  - Adds `review_cli.py` for running daily reviews from pipeline meta files without embedding agent logic in experiment scripts.
- Added sector fund-flow data utilities under `quantlab.data`.
  - `sector_flow_download.py` supports AKShare and Tushare DC providers.
  - `sector_flow_features.py` builds rolling sector-flow ranks, z-scores, persistence, and hot/cold sector labels.
  - `sector_flow_utils.py` centralizes calendar alignment, standard columns, and table I/O.
- Added `docs/agentic_feature_article.html`, a detailed HTML article introducing the agentic review layer, sector-flow alignment, daily decision flow, and experiment path.
- Added `docs/project_architecture.html`, a static architecture overview for the full QuantLab pipeline.
- Linked the architecture overview from `README.md` and `README_CN.md` using a rendered HTML preview, with the GitHub source file kept as a secondary link.

### Changed

- Ignored external TradingAgents reference artifacts via `.gitignore`; QuantLab now keeps only the derived `quantlab.agentic` module rather than vendoring the upstream project.
- Updated README module descriptions to include the independent agentic review layer and sector fund-flow data layer.

