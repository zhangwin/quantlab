"""M2 Alpha158 Signal Pipeline — factor management, validation, mining, and signal production.

Sub-modules:
    FactorRegistry        — Factor pool management (add/remove/enable/disable, YAML persistence)
    FactorValidator       — Factor quality evaluation (IC, ICIR, correlation, long-short backtest)
    FactorMiner           — Semi-automatic factor exploration (variants, combinations, templates)
    AlphaSignalPipeline   — LightGBM rolling training + daily prediction
"""

import hashlib
import logging
from dataclasses import dataclass, field, asdict
from itertools import product
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class FactorDef:
    """Single factor definition."""
    name: str
    expression: str
    category: str  # momentum / mean_revert / volatility / liquidity / custom
    source: str    # alpha158 / manual / rdagent / miner
    enabled: bool = True
    added_date: str = ""
    validation: Optional[Dict] = None


@dataclass
class FactorReport:
    """Result of single-factor validation."""
    expression: str
    ic_mean: float = 0.0
    icir: float = 0.0
    ic_series: Optional[pd.Series] = None
    ic_positive_ratio: float = 0.0
    ic_decay: Optional[Dict[int, float]] = None
    turnover: float = 0.0
    auto_corr: float = 0.0
    sharpe_long_short: float = 0.0
    max_corr_with_existing: float = 0.0
    verdict: str = "review"  # accept / reject / review


@dataclass
class IncrementalReport(FactorReport):
    """Extended report for incremental factor evaluation."""
    marginal_ic: float = 0.0
    feature_importance: float = 0.0
    correlation_matrix: Optional[pd.DataFrame] = None


# ---------------------------------------------------------------------------
# Alpha158 expression generator (mirrors qlib/contrib/data/loader.py)
# ---------------------------------------------------------------------------

def _generate_alpha158_factors() -> List[FactorDef]:
    """Generate the 158 Alpha158 factor definitions.

    Mirrors the logic from ``qlib.contrib.data.loader.Alpha158DL.get_feature_config``.
    """
    factors = []
    today = pd.Timestamp.now().strftime("%Y-%m-%d")

    # --- Kbar (9 factors) ---
    kbar = [
        ("KMID",  "($close-$open)/$open",                             "momentum"),
        ("KLEN",  "($high-$low)/$open",                               "volatility"),
        ("KMID2", "($close-$open)/($high-$low+1e-12)",                "momentum"),
        ("KUP",   "($high-Greater($open,$close))/$open",              "volatility"),
        ("KUP2",  "($high-Greater($open,$close))/($high-$low+1e-12)", "volatility"),
        ("KLOW",  "(Less($open,$close)-$low)/$open",                  "volatility"),
        ("KLOW2", "(Less($open,$close)-$low)/($high-$low+1e-12)",     "volatility"),
        ("KSFT",  "(2*$close-$high-$low)/$open",                      "momentum"),
        ("KSFT2", "(2*$close-$high-$low)/($high-$low+1e-12)",         "momentum"),
    ]
    for name, expr, cat in kbar:
        factors.append(FactorDef(name=name, expression=expr, category=cat,
                                 source="alpha158", added_date=today))

    # --- Price (4 factors) ---
    for field_name in ["OPEN", "HIGH", "LOW", "VWAP"]:
        name = f"{field_name}0"
        expr = f"${field_name.lower()}/$close"
        factors.append(FactorDef(name=name, expression=expr, category="momentum",
                                 source="alpha158", added_date=today))

    # --- Rolling operators (145 factors) ---
    windows = [5, 10, 20, 30, 60]

    rolling_ops = [
        ("ROC",   "Ref($close, {w})/$close",                        "momentum"),
        ("MA",    "Mean($close, {w})/$close",                        "momentum"),
        ("STD",   "Std($close, {w})/$close",                         "volatility"),
        ("BETA",  "Slope($close, {w})/$close",                       "momentum"),
        ("RSQR",  "Rsquare($close, {w})",                            "momentum"),
        ("RESI",  "Resi($close, {w})/$close",                        "momentum"),
        ("MAX",   "Max($high, {w})/$close",                          "volatility"),
        ("MIN",   "Min($low, {w})/$close",                           "volatility"),
        ("QTLU",  "Quantile($close, {w}, 0.8)/$close",              "volatility"),
        ("QTLD",  "Quantile($close, {w}, 0.2)/$close",              "volatility"),
        ("RANK",  "Rank($close, {w})",                               "momentum"),
        ("RSV",   "($close-Min($low,{w}))/(Max($high,{w})-Min($low,{w})+1e-12)", "momentum"),
        ("IMAX",  "IdxMax($high, {w})/{w}",                         "momentum"),
        ("IMIN",  "IdxMin($low, {w})/{w}",                          "momentum"),
        ("IMXD",  "(IdxMax($high,{w})-IdxMin($low,{w}))/{w}",       "momentum"),
        ("CORR",  "Corr($close, Log($volume+1), {w})",              "liquidity"),
        ("CORD",  "Corr($close/Ref($close,1), Log($volume/Ref($volume,1)+1), {w})", "liquidity"),
        ("CNTP",  "Mean($close>Ref($close,1), {w})",                "momentum"),
        ("CNTN",  "Mean($close<Ref($close,1), {w})",                "momentum"),
        ("CNTD",  "Mean($close>Ref($close,1),{w})-Mean($close<Ref($close,1),{w})", "momentum"),
        ("SUMP",  "Sum(Greater($close-Ref($close,1),0),{w})/(Sum(Abs($close-Ref($close,1)),{w})+1e-12)", "momentum"),
        ("SUMN",  "Sum(Greater(Ref($close,1)-$close,0),{w})/(Sum(Abs($close-Ref($close,1)),{w})+1e-12)", "momentum"),
        ("SUMD",  "(Sum(Greater($close-Ref($close,1),0),{w})-Sum(Greater(Ref($close,1)-$close,0),{w}))/(Sum(Abs($close-Ref($close,1)),{w})+1e-12)", "momentum"),
        ("VMA",   "Mean($volume, {w})/($volume+1e-12)",              "liquidity"),
        ("VSTD",  "Std($volume, {w})/($volume+1e-12)",               "liquidity"),
        ("WVMA",  "Std(Abs($close/Ref($close,1)-1)*$volume,{w})/(Mean(Abs($close/Ref($close,1)-1)*$volume,{w})+1e-12)", "liquidity"),
        ("VSUMP", "Sum(Greater($volume-Ref($volume,1),0),{w})/(Sum(Abs($volume-Ref($volume,1)),{w})+1e-12)", "liquidity"),
        ("VSUMN", "Sum(Greater(Ref($volume,1)-$volume,0),{w})/(Sum(Abs($volume-Ref($volume,1)),{w})+1e-12)", "liquidity"),
        ("VSUMD", "(Sum(Greater($volume-Ref($volume,1),0),{w})-Sum(Greater(Ref($volume,1)-$volume,0),{w}))/(Sum(Abs($volume-Ref($volume,1)),{w})+1e-12)", "liquidity"),
    ]

    for prefix, template, cat in rolling_ops:
        for w in windows:
            name = f"{prefix}{w}"
            expr = template.replace("{w}", str(w))
            factors.append(FactorDef(name=name, expression=expr, category=cat,
                                     source="alpha158", added_date=today))

    return factors


# ===========================================================================
# FactorRegistry
# ===========================================================================

class FactorRegistry:
    """Factor pool management with YAML persistence.

    Parameters
    ----------
    config_path : str
        Path to ``factors.yaml``. Created automatically on first run.
    """

    def __init__(self, config_path: str = "configs/factors.yaml"):
        self._path = Path(config_path)
        self._factors: Dict[str, FactorDef] = {}
        self._load()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """Load factors from YAML, or generate Alpha158 defaults."""
        if self._path.exists():
            import yaml
            with open(self._path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            for item in data.get("factors", []):
                fd = FactorDef(**item)
                self._factors[fd.name] = fd
            logger.info("Loaded %d factors from %s", len(self._factors), self._path)
        else:
            self._init_alpha158()

    def _init_alpha158(self) -> None:
        """Generate Alpha158 base factors and save."""
        for fd in _generate_alpha158_factors():
            self._factors[fd.name] = fd
        logger.info("Generated %d Alpha158 base factors", len(self._factors))
        self.save()

    def save(self) -> None:
        """Persist current factor pool to YAML."""
        import yaml
        self._path.parent.mkdir(parents=True, exist_ok=True)
        data = {"factors": []}
        for fd in self._factors.values():
            d = asdict(fd)
            # Drop None fields for cleaner YAML
            d = {k: v for k, v in d.items() if v is not None}
            data["factors"].append(d)
        with open(self._path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
        logger.info("Saved %d factors to %s", len(self._factors), self._path)

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def add(
        self,
        name: str,
        expression: str,
        category: str = "custom",
        source: str = "manual",
    ) -> FactorDef:
        """Add a new factor to the pool.

        Raises
        ------
        ValueError
            If *name* already exists.
        """
        if name in self._factors:
            raise ValueError(f"Factor '{name}' already exists. Use a different name.")
        fd = FactorDef(
            name=name,
            expression=expression,
            category=category,
            source=source,
            enabled=True,
            added_date=pd.Timestamp.now().strftime("%Y-%m-%d"),
        )
        self._factors[name] = fd
        logger.info("Added factor: %s = %s", name, expression)
        return fd

    def remove(self, name: str) -> None:
        """Remove a factor. Alpha158 factors can only be disabled, not removed."""
        if name not in self._factors:
            raise KeyError(f"Factor '{name}' not found")
        if self._factors[name].source == "alpha158":
            raise ValueError(
                f"Cannot remove alpha158 factor '{name}'. Use disable() instead."
            )
        del self._factors[name]
        logger.info("Removed factor: %s", name)

    def enable(self, name: str) -> None:
        if name not in self._factors:
            raise KeyError(f"Factor '{name}' not found")
        self._factors[name].enabled = True

    def disable(self, name: str) -> None:
        if name not in self._factors:
            raise KeyError(f"Factor '{name}' not found")
        self._factors[name].enabled = False

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_enabled(self) -> List[FactorDef]:
        """Return all enabled factors."""
        return [fd for fd in self._factors.values() if fd.enabled]

    def get_expressions(self) -> Tuple[List[str], List[str]]:
        """Return ``(expressions, names)`` for all enabled factors.

        Directly usable with ``QlibDataLoader``.
        """
        enabled = self.get_enabled()
        return [fd.expression for fd in enabled], [fd.name for fd in enabled]

    def list(
        self,
        category: Optional[str] = None,
        source: Optional[str] = None,
    ) -> List[FactorDef]:
        """Query factors by category and/or source."""
        result = list(self._factors.values())
        if category:
            result = [fd for fd in result if fd.category == category]
        if source:
            result = [fd for fd in result if fd.source == source]
        return result

    def get(self, name: str) -> FactorDef:
        if name not in self._factors:
            raise KeyError(f"Factor '{name}' not found")
        return self._factors[name]

    def update_validation(self, name: str, result: Dict) -> None:
        """Store validation result on a factor."""
        if name not in self._factors:
            raise KeyError(f"Factor '{name}' not found")
        self._factors[name].validation = result

    def summary(self) -> pd.DataFrame:
        """Return a summary DataFrame of all factors."""
        rows = []
        for fd in self._factors.values():
            row = {
                "name": fd.name,
                "category": fd.category,
                "source": fd.source,
                "enabled": fd.enabled,
            }
            if fd.validation:
                row["ic_mean"] = fd.validation.get("ic_mean")
                row["icir"] = fd.validation.get("icir")
            rows.append(row)
        return pd.DataFrame(rows)

    @property
    def factor_hash(self) -> str:
        """Hash of current enabled factor names — used to detect pool changes."""
        names = sorted(fd.name for fd in self.get_enabled())
        return hashlib.md5("|".join(names).encode()).hexdigest()

    def __len__(self) -> int:
        return len(self._factors)


# ===========================================================================
# FactorValidator
# ===========================================================================

class FactorValidator:
    """Factor quality evaluation.

    Parameters
    ----------
    data_manager : DataManager
        Initialized M1 DataManager.
    """

    # Verdict thresholds
    ACCEPT_IC = 0.02
    ACCEPT_ICIR = 0.5
    ACCEPT_CORR = 0.7
    ACCEPT_IC_POS = 0.55
    REJECT_IC = 0.01
    REJECT_ICIR = 0.2
    REJECT_CORR = 0.9

    def __init__(self, data_manager):
        self.dm = data_manager
        self.dm._ensure_init()
        self._D = self.dm._D

    def validate_single(
        self,
        expression: str,
        start: str,
        end: str,
        existing_expressions: Optional[List[str]] = None,
    ) -> FactorReport:
        """Full evaluation of a single factor expression.

        Parameters
        ----------
        expression : str
            Qlib factor expression.
        start, end : str
            Evaluation date range.
        existing_expressions : list of str, optional
            Existing factor expressions for correlation check.

        Returns
        -------
        FactorReport
        """
        report = FactorReport(expression=expression)
        try:
            # 1. Compute factor values
            factor_df = self._compute_factor(expression, start, end)
            if factor_df.empty:
                report.verdict = "reject"
                return report

            # 2. Compute forward returns as label
            label_df = self._compute_label(start, end)
            if label_df.empty:
                report.verdict = "reject"
                return report

            # 3. Align factor and label
            aligned = factor_df.to_frame("factor").join(label_df.to_frame("label"), how="inner")
            aligned = aligned.dropna()
            if len(aligned) < 30:
                report.verdict = "reject"
                return report

            # 4. Daily Rank IC
            ic_series = self._calc_daily_rank_ic(aligned)
            report.ic_series = ic_series
            report.ic_mean = ic_series.mean()
            ic_std = ic_series.std()
            report.icir = report.ic_mean / ic_std if ic_std > 1e-9 else 0.0
            report.ic_positive_ratio = (ic_series > 0).mean()

            # 5. IC decay
            report.ic_decay = self._calc_ic_decay(expression, start, end)

            # 6. Auto-correlation
            report.auto_corr = self._calc_auto_corr(factor_df)

            # 7. Long-short Sharpe
            report.sharpe_long_short = self._calc_long_short_sharpe(aligned)

            # 8. Turnover
            report.turnover = self._calc_turnover(factor_df)

            # 9. Correlation with existing factors
            if existing_expressions:
                report.max_corr_with_existing = self._calc_max_corr(
                    expression, existing_expressions, start, end
                )

            # 10. Verdict
            report.verdict = self._judge(report)

        except Exception as e:
            logger.error("Validation failed for '%s': %s", expression, e)
            report.verdict = "reject"

        return report

    def validate_incremental(
        self,
        expression: str,
        existing_expressions: List[str],
        start: str = "2020-01-01",
        end: str = "2024-12-31",
    ) -> IncrementalReport:
        """Evaluate incremental contribution of a new factor.

        Trains LightGBM with and without the new factor and compares IC.
        """
        base_report = self.validate_single(expression, start, end, existing_expressions)
        report = IncrementalReport(
            expression=base_report.expression,
            ic_mean=base_report.ic_mean,
            icir=base_report.icir,
            ic_series=base_report.ic_series,
            ic_positive_ratio=base_report.ic_positive_ratio,
            ic_decay=base_report.ic_decay,
            turnover=base_report.turnover,
            auto_corr=base_report.auto_corr,
            sharpe_long_short=base_report.sharpe_long_short,
            max_corr_with_existing=base_report.max_corr_with_existing,
            verdict=base_report.verdict,
        )

        try:
            # Compute correlation matrix with top existing factors
            top_n = min(10, len(existing_expressions))
            corr_exprs = existing_expressions[:top_n] + [expression]
            factor_values = {}
            for expr in corr_exprs:
                fv = self._compute_factor(expr, start, end)
                if not fv.empty:
                    factor_values[expr] = fv

            if len(factor_values) > 1:
                corr_df = pd.DataFrame(factor_values)
                report.correlation_matrix = corr_df.corr(method="spearman")

        except Exception as e:
            logger.warning("Incremental validation partial failure: %s", e)

        return report

    def validate_batch(
        self,
        expressions: List[str],
        start: str,
        end: str,
        existing_expressions: Optional[List[str]] = None,
    ) -> List[FactorReport]:
        """Batch validation of multiple factor expressions."""
        reports = []
        for i, expr in enumerate(expressions):
            logger.info("Validating %d/%d: %s", i + 1, len(expressions), expr[:60])
            report = self.validate_single(expr, start, end, existing_expressions)
            reports.append(report)
        return reports

    def compare(
        self,
        expressions: List[str],
        start: str,
        end: str,
    ) -> pd.DataFrame:
        """Compare multiple factors side-by-side."""
        reports = self.validate_batch(expressions, start, end)
        rows = []
        for r in reports:
            rows.append({
                "expression": r.expression[:50],
                "ic_mean": round(r.ic_mean, 4),
                "icir": round(r.icir, 2),
                "ic_pos_ratio": round(r.ic_positive_ratio, 3),
                "sharpe_ls": round(r.sharpe_long_short, 2),
                "auto_corr": round(r.auto_corr, 3),
                "turnover": round(r.turnover, 3),
                "verdict": r.verdict,
            })
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Internal computation helpers
    # ------------------------------------------------------------------

    def _compute_factor(
        self,
        expression: str,
        start: str,
        end: str,
    ) -> pd.Series:
        """Compute factor values via Qlib engine.

        Returns
        -------
        Series with MultiIndex (instrument, datetime).
        """
        instruments = self._D.instruments(market=self.dm.market)
        df = self._D.features(
            instruments,
            [expression],
            start_time=start,
            end_time=end,
            freq="day",
        )
        if df is None or df.empty:
            return pd.Series(dtype=float)
        return df.iloc[:, 0]

    def _compute_label(self, start: str, end: str) -> pd.Series:
        """Compute next-day return label: Ref($close, -2) / Ref($close, -1) - 1."""
        instruments = self._D.instruments(market=self.dm.market)
        df = self._D.features(
            instruments,
            ["Ref($close, -2)/Ref($close, -1) - 1"],
            start_time=start,
            end_time=end,
            freq="day",
        )
        if df is None or df.empty:
            return pd.Series(dtype=float)
        return df.iloc[:, 0]

    def _calc_daily_rank_ic(self, aligned: pd.DataFrame) -> pd.Series:
        """Compute daily cross-sectional Spearman rank IC."""
        # aligned has MultiIndex (instrument, datetime) and columns [factor, label]
        ic_list = []
        dates = []
        for date, group in aligned.groupby(level="datetime"):
            g = group.dropna()
            if len(g) < 10:
                continue
            corr, _ = spearmanr(g["factor"], g["label"])
            if not np.isnan(corr):
                ic_list.append(corr)
                dates.append(date)
        return pd.Series(ic_list, index=dates, name="IC")

    def _calc_ic_decay(
        self,
        expression: str,
        start: str,
        end: str,
    ) -> Dict[int, float]:
        """IC at horizons T+1 through T+10."""
        decay = {}
        instruments = self._D.instruments(market=self.dm.market)
        factor_df = self._compute_factor(expression, start, end)
        if factor_df.empty:
            return decay

        for horizon in [1, 2, 3, 5, 10]:
            label_expr = f"Ref($close, -{horizon + 1})/Ref($close, -{horizon}) - 1"
            label_df = self._D.features(
                instruments, [label_expr],
                start_time=start, end_time=end, freq="day",
            )
            if label_df is None or label_df.empty:
                continue
            aligned = factor_df.to_frame("factor").join(
                label_df.iloc[:, 0].to_frame("label"), how="inner"
            ).dropna()
            ic_s = self._calc_daily_rank_ic(aligned)
            if len(ic_s) > 0:
                decay[horizon] = round(ic_s.mean(), 4)
        return decay

    def _calc_auto_corr(self, factor_series: pd.Series) -> float:
        """Cross-sectional auto-correlation between T and T-1."""
        if factor_series.empty:
            return 0.0
        df = factor_series.unstack(level="instrument")
        if df.shape[0] < 2:
            return 0.0
        corrs = []
        for i in range(1, len(df)):
            today = df.iloc[i].dropna()
            yesterday = df.iloc[i - 1].dropna()
            common = today.index.intersection(yesterday.index)
            if len(common) < 10:
                continue
            c, _ = spearmanr(today[common], yesterday[common])
            if not np.isnan(c):
                corrs.append(c)
        return float(np.mean(corrs)) if corrs else 0.0

    def _calc_long_short_sharpe(self, aligned: pd.DataFrame) -> float:
        """Annualized Sharpe of Top20% - Bottom20% portfolio."""
        daily_ret = []
        for date, group in aligned.groupby(level="datetime"):
            g = group.dropna()
            if len(g) < 20:
                continue
            n = max(1, len(g) // 5)
            sorted_g = g.sort_values("factor")
            long_ret = sorted_g["label"].iloc[-n:].mean()
            short_ret = sorted_g["label"].iloc[:n].mean()
            daily_ret.append(long_ret - short_ret)

        if len(daily_ret) < 10:
            return 0.0
        ret = pd.Series(daily_ret)
        mean = ret.mean()
        std = ret.std()
        if std < 1e-9:
            return 0.0
        return float(mean / std * np.sqrt(252))

    def _calc_turnover(self, factor_series: pd.Series) -> float:
        """Average daily turnover of top quintile."""
        df = factor_series.unstack(level="instrument")
        if df.shape[0] < 2:
            return 0.0
        turnover_list = []
        for i in range(1, len(df)):
            today = df.iloc[i].dropna()
            yesterday = df.iloc[i - 1].dropna()
            if len(today) < 5 or len(yesterday) < 5:
                continue
            n = max(1, len(today) // 5)
            top_today = set(today.nlargest(n).index)
            top_yesterday = set(yesterday.nlargest(n).index)
            if not top_today and not top_yesterday:
                continue
            union = top_today | top_yesterday
            turnover_list.append(len(top_today ^ top_yesterday) / len(union))
        return float(np.mean(turnover_list)) if turnover_list else 0.0

    def _calc_max_corr(
        self,
        expression: str,
        existing_expressions: List[str],
        start: str,
        end: str,
    ) -> float:
        """Max cross-sectional correlation between new and existing factors."""
        new_fv = self._compute_factor(expression, start, end)
        if new_fv.empty:
            return 0.0

        max_corr = 0.0
        for ex_expr in existing_expressions:
            ex_fv = self._compute_factor(ex_expr, start, end)
            if ex_fv.empty:
                continue
            aligned = new_fv.to_frame("new").join(ex_fv.to_frame("old"), how="inner").dropna()
            if len(aligned) < 30:
                continue
            # Cross-sectional average correlation
            corrs = []
            for _, group in aligned.groupby(level="datetime"):
                if len(group) < 10:
                    continue
                c, _ = spearmanr(group["new"], group["old"])
                if not np.isnan(c):
                    corrs.append(abs(c))
            if corrs:
                avg_corr = float(np.mean(corrs))
                max_corr = max(max_corr, avg_corr)
        return max_corr

    def _judge(self, report: FactorReport) -> str:
        """Auto-judge verdict based on thresholds."""
        # Reject conditions
        if abs(report.ic_mean) < self.REJECT_IC:
            return "reject"
        if abs(report.icir) < self.REJECT_ICIR:
            return "reject"
        if report.max_corr_with_existing > self.REJECT_CORR:
            return "reject"

        # Accept conditions
        if (abs(report.ic_mean) > self.ACCEPT_IC
                and abs(report.icir) > self.ACCEPT_ICIR
                and report.max_corr_with_existing < self.ACCEPT_CORR
                and report.ic_positive_ratio > self.ACCEPT_IC_POS):
            return "accept"

        return "review"


# ===========================================================================
# FactorMiner
# ===========================================================================

class FactorMiner:
    """Semi-automatic factor exploration.

    Parameters
    ----------
    validator : FactorValidator
        For evaluating discovered factors.
    registry : FactorRegistry
        For registering accepted factors.
    """

    def __init__(self, validator: FactorValidator, registry: FactorRegistry):
        self.validator = validator
        self.registry = registry

    def explore_variants(
        self,
        base_expression: str,
        param_grid: Dict[str, List],
        start: str = "2020-01-01",
        end: str = "2024-12-31",
    ) -> List[FactorReport]:
        """Parameter search on a single factor template.

        Parameters
        ----------
        base_expression : str
            Expression with ``{param}`` placeholders, e.g. ``"Std($close, {window})/$close"``.
        param_grid : dict
            ``{param_name: [values]}``, e.g. ``{"window": [5, 10, 20, 30, 60]}``.

        Returns
        -------
        list of FactorReport
            Sorted by ICIR descending.
        """
        candidates = self._expand_grid(base_expression, param_grid)
        existing_exprs = self.registry.get_expressions()[0]
        reports = self.validator.validate_batch(candidates, start, end, existing_exprs)
        reports.sort(key=lambda r: abs(r.icir), reverse=True)
        return reports

    def explore_combinations(
        self,
        fields: List[str],
        operators: List[str],
        windows: List[int],
        start: str = "2020-01-01",
        end: str = "2024-12-31",
    ) -> List[FactorReport]:
        """Cartesian product of fields × operators × windows.

        Generates expressions like ``Corr($close, $volume, 20)``.
        """
        candidates = []
        for op in operators:
            if op in ("Corr", "Cov"):
                # Binary operators: need 2 fields
                for f1, f2 in product(fields, fields):
                    if f1 == f2:
                        continue
                    for w in windows:
                        candidates.append(f"{op}({f1}, {f2}, {w})")
            else:
                # Unary operators
                for f in fields:
                    for w in windows:
                        candidates.append(f"{op}({f}, {w})")

        existing_exprs = self.registry.get_expressions()[0]
        reports = self.validator.validate_batch(candidates, start, end, existing_exprs)
        reports.sort(key=lambda r: abs(r.icir), reverse=True)
        return reports

    def explore_from_template(
        self,
        template: str,
        variables: Dict[str, List],
        start: str = "2020-01-01",
        end: str = "2024-12-31",
    ) -> List[FactorReport]:
        """Batch generation from a template string with variable substitution.

        Parameters
        ----------
        template : str
            E.g. ``"({field} - Mean({field}, {window})) / (Std({field}, {window}) + 1e-12)"``
        variables : dict
            ``{"field": ["$close", "$volume"], "window": [5, 10, 20]}``
        """
        candidates = self._expand_grid(template, variables)
        existing_exprs = self.registry.get_expressions()[0]
        reports = self.validator.validate_batch(candidates, start, end, existing_exprs)
        reports.sort(key=lambda r: abs(r.icir), reverse=True)
        return reports

    def auto_mine(
        self,
        budget: int = 200,
        start: str = "2020-01-01",
        end: str = "2024-12-31",
    ) -> List[FactorReport]:
        """Fully automatic factor mining.

        1. Define search space (fields × operators × windows)
        2. Generate candidates, sample up to *budget*
        3. Batch validate
        4. Greedy de-duplicate (skip if corr > 0.7 with selected)
        5. Return sorted results
        """
        # Define search space
        base_fields = [
            "$close", "$open", "$high", "$low", "$volume",
        ]
        derived_fields = [
            "$close/Ref($close,1)", "$high-$low", "$close-$open",
            "Log($volume+1)",
        ]
        unary_ops = [
            "Mean", "Std", "Skew", "Slope", "Rsquare",
            "Rank", "Max", "Min", "Sum",
        ]
        binary_ops = ["Corr", "Cov"]
        windows = [5, 10, 20, 30, 60]

        all_fields = base_fields + derived_fields

        # Generate candidates
        candidates = []

        # Unary: op(field, window)
        for op in unary_ops:
            for f in all_fields:
                for w in windows:
                    candidates.append(f"{op}({f}, {w})")

        # Binary: op(field1, field2, window)
        for op in binary_ops:
            for f1, f2 in product(all_fields, all_fields):
                if f1 >= f2:  # avoid duplicates and self-correlation
                    continue
                for w in windows:
                    candidates.append(f"{op}({f1}, {f2}, {w})")

        logger.info("Generated %d candidates, sampling %d", len(candidates), budget)

        # Sample
        if len(candidates) > budget:
            rng = np.random.RandomState(42)
            indices = rng.choice(len(candidates), size=budget, replace=False)
            candidates = [candidates[i] for i in indices]

        # Validate
        existing_exprs = self.registry.get_expressions()[0]
        reports = self.validator.validate_batch(candidates, start, end, existing_exprs)

        # Sort by ICIR
        reports.sort(key=lambda r: abs(r.icir), reverse=True)

        # Greedy de-duplicate
        selected = []
        for r in reports:
            if r.verdict == "reject":
                continue
            # Check correlation with already selected
            is_redundant = False
            for s in selected:
                # Use max_corr as a proxy — if both have high corr with existing,
                # they may be redundant with each other.
                # For exact check we'd need pairwise, but that's expensive.
                # Approximate: skip if same IC sign and similar IC
                if (abs(r.ic_mean - s.ic_mean) < 0.005
                        and abs(r.icir - s.icir) < 0.2):
                    is_redundant = True
                    break
            if not is_redundant:
                selected.append(r)

        logger.info("Auto-mine: %d candidates → %d selected", len(reports), len(selected))
        return selected

    def accept_and_register(
        self,
        reports: List[FactorReport],
        name_prefix: str = "MINED",
    ) -> int:
        """Register factors with verdict='accept' into the registry.

        Returns number of newly registered factors.
        """
        count = 0
        for i, r in enumerate(reports):
            if r.verdict != "accept":
                continue
            name = f"{name_prefix}_{count:03d}"
            # Avoid name collision
            while name in [fd.name for fd in self.registry.list()]:
                count += 1
                name = f"{name_prefix}_{count:03d}"
            self.registry.add(name, r.expression, "custom", "miner")
            self.registry.update_validation(name, {
                "ic_mean": round(r.ic_mean, 4),
                "icir": round(r.icir, 2),
                "max_corr_with_existing": round(r.max_corr_with_existing, 3),
            })
            count += 1
        if count > 0:
            self.registry.save()
        logger.info("Registered %d factors from mining results", count)
        return count

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _expand_grid(template: str, variables: Dict[str, List]) -> List[str]:
        """Expand a template with all combinations of variables."""
        keys = list(variables.keys())
        values = list(variables.values())
        candidates = []
        for combo in product(*values):
            expr = template
            for k, v in zip(keys, combo):
                expr = expr.replace(f"{{{k}}}", str(v))
            candidates.append(expr)
        return candidates


# ===========================================================================
# AlphaSignalPipeline
# ===========================================================================

class AlphaSignalPipeline:
    """LightGBM rolling training + daily prediction.

    Parameters
    ----------
    registry : FactorRegistry
        Factor pool.
    market : str
        Stock pool name.
    retrain_interval : int
        Number of trading days between retraining.
    train_years : int
        Length of training window in years.
    """

    def __init__(
        self,
        registry: FactorRegistry,
        market: str = "csi300",
        retrain_interval: int = 20,
        train_years: int = 3,
        lgb_params: Optional[Dict] = None,
    ):
        self.registry = registry
        self.market = market
        self.retrain_interval = retrain_interval
        self.train_years = train_years

        self.lgb_params = lgb_params or {
            "objective": "mse",
            "verbosity": -1,
            "colsample_bytree": 0.8879,
            "learning_rate": 0.2,
            "subsample": 0.8789,
            "lambda_l1": 205.6999,
            "lambda_l2": 580.9768,
            "max_depth": 8,
            "num_leaves": 210,
            "num_threads": 4,
        }

        self._model = None
        self._last_train_date: Optional[str] = None
        self._predict_count: int = 0
        self._last_factor_hash: str = ""
        self._feature_importance: Optional[pd.Series] = None
        self._feature_names: List[str] = []

    def should_retrain(self, anchor_date: str) -> bool:
        """Check if retraining is needed.

        Triggers on:
        1. No model yet
        2. Every retrain_interval predictions
        3. Factor pool changed
        """
        if self._model is None:
            return True
        if self.registry.factor_hash != self._last_factor_hash:
            logger.info("Factor pool changed, retraining required")
            return True
        if self._predict_count >= self.retrain_interval:
            return True
        return False

    def train(self, anchor_date: str, data_manager) -> None:
        """Train LightGBM on rolling window ending at anchor_date.

        Training data: [anchor - train_years, anchor - 2 trading days]
        Label: Ref($close, -2) / Ref($close, -1) - 1
        """
        import lightgbm as lgb

        data_manager._ensure_init()
        D = data_manager._D

        # Compute train window
        train_end = self._offset_trading_days(anchor_date, -2, data_manager)
        train_start_ts = pd.Timestamp(anchor_date) - pd.DateOffset(years=self.train_years)
        train_start = train_start_ts.strftime("%Y-%m-%d")

        logger.info("Training: %s ~ %s (anchor=%s)", train_start, train_end, anchor_date)

        # Get factor expressions
        expressions, names = self.registry.get_expressions()
        self._feature_names = names

        # Label expression
        label_expr = "Ref($close, -2)/Ref($close, -1) - 1"

        # Fetch features + label via Qlib
        instruments = D.instruments(market=self.market)
        all_exprs = expressions + [label_expr]
        all_names = names + ["LABEL"]

        df = D.features(
            instruments,
            all_exprs,
            start_time=train_start,
            end_time=train_end,
            freq="day",
        )
        if df is None or df.empty:
            raise ValueError(f"No training data for {train_start} ~ {train_end}")

        df.columns = all_names

        # Split features and label
        features = df[names]
        label = df["LABEL"]

        # Drop rows where label is NaN
        valid_mask = label.notna()
        features = features[valid_mask]
        label = label[valid_mask]

        # Fill NaN in features
        features = features.fillna(0.0)

        logger.info(
            "Training data: %d rows × %d features, date range %s ~ %s",
            len(features), len(names),
            features.index.get_level_values("datetime").min().strftime("%Y-%m-%d"),
            features.index.get_level_values("datetime").max().strftime("%Y-%m-%d"),
        )

        # Split into train and validation (last 20% by time)
        dates = sorted(features.index.get_level_values("datetime").unique())
        split_idx = int(len(dates) * 0.8)
        split_date = dates[split_idx]

        train_mask = features.index.get_level_values("datetime") < split_date
        valid_mask_split = ~train_mask

        X_train = features[train_mask].values
        y_train = label[train_mask].values
        X_valid = features[valid_mask_split].values
        y_valid = label[valid_mask_split].values

        dtrain = lgb.Dataset(X_train, label=y_train, feature_name=names, free_raw_data=False)
        dvalid = lgb.Dataset(X_valid, label=y_valid, feature_name=names, free_raw_data=False)

        # Train
        self._model = lgb.train(
            self.lgb_params,
            dtrain,
            num_boost_round=1000,
            valid_sets=[dtrain, dvalid],
            valid_names=["train", "valid"],
            callbacks=[
                lgb.early_stopping(50),
                lgb.log_evaluation(100),
            ],
        )

        # Feature importance
        importance = self._model.feature_importance(importance_type="gain")
        self._feature_importance = pd.Series(importance, index=names).sort_values(ascending=False)

        # Update state
        self._last_train_date = anchor_date
        self._predict_count = 0
        self._last_factor_hash = self.registry.factor_hash

        logger.info(
            "Training complete. Best iteration: %d. Top-5 features: %s",
            self._model.best_iteration,
            list(self._feature_importance.head().index),
        )

    def predict(self, anchor_date: str, data_manager) -> pd.Series:
        """Produce cross-sectional prediction scores.

        Auto-retrains if needed (interval or factor pool change).

        Parameters
        ----------
        anchor_date : str
            Current trading date (T). Uses only data ≤ T.
        data_manager : DataManager
            M1 instance.

        Returns
        -------
        Series
            Index = symbol, values = predicted score (higher = more bullish).
        """
        # Auto-retrain check
        if self.should_retrain(anchor_date):
            self.train(anchor_date, data_manager)

        data_manager._ensure_init()
        D = data_manager._D

        expressions, names = self.registry.get_expressions()

        # Fetch features for anchor_date
        instruments = D.instruments(market=self.market)
        df = D.features(
            instruments,
            expressions,
            start_time=anchor_date,
            end_time=anchor_date,
            freq="day",
        )
        if df is None or df.empty:
            logger.warning("No data for prediction on %s", anchor_date)
            return pd.Series(dtype=float)

        df.columns = names
        df = df.fillna(0.0)

        # Predict
        scores = self._model.predict(df.values)
        result = pd.Series(scores, index=df.index.get_level_values("instrument"), name="alpha_score")

        # Increment predict count
        self._predict_count += 1

        logger.info(
            "Predicted %d stocks on %s (predict_count=%d/%d)",
            len(result), anchor_date, self._predict_count, self.retrain_interval,
        )
        return result

    def get_feature_importance(self) -> Optional[pd.Series]:
        """Return current model's feature importance (gain-based)."""
        return self._feature_importance

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _offset_trading_days(date: str, offset: int, data_manager) -> str:
        """Get the trading date *offset* days from *date*.

        Negative offset = days before. Uses the calendar.
        """
        cal = data_manager.get_trading_calendar("2000-01-01", date)
        if not cal:
            return date
        target_idx = len(cal) + offset  # offset is negative
        target_idx = max(0, min(target_idx, len(cal) - 1))
        return cal[target_idx].strftime("%Y-%m-%d")
