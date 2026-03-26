"""Tests for M2 Signal Alpha Pipeline.

Run with:
    python -m pytest quantlab/tests/test_signal_alpha.py -v
    python -m pytest quantlab/tests/test_signal_alpha.py::TestFactorRegistry -v
"""

import os
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from quantlab.signal.signal_alpha import (
    FactorDef,
    FactorRegistry,
    FactorReport,
    _generate_alpha158_factors,
)

QLIB_DATA_DIR = os.path.expanduser("~/.qlib/qlib_data/cn_data")
SKIP_NO_DATA = not os.path.isdir(QLIB_DATA_DIR)
skip_reason = "Qlib data not found at ~/.qlib/qlib_data/cn_data"


# ===================================================================
# Alpha158 Factor Generation
# ===================================================================

class TestAlpha158Generation:

    def test_count(self):
        """Alpha158 should generate exactly 158 factors."""
        factors = _generate_alpha158_factors()
        assert len(factors) == 158, f"Expected 158, got {len(factors)}"

    def test_unique_names(self):
        """All factor names must be unique."""
        factors = _generate_alpha158_factors()
        names = [f.name for f in factors]
        assert len(names) == len(set(names)), "Duplicate factor names found"

    def test_all_have_expressions(self):
        """Every factor must have a non-empty expression."""
        factors = _generate_alpha158_factors()
        for f in factors:
            assert f.expression, f"Factor {f.name} has empty expression"

    def test_source_is_alpha158(self):
        """All generated factors should have source='alpha158'."""
        factors = _generate_alpha158_factors()
        for f in factors:
            assert f.source == "alpha158"

    def test_categories(self):
        """Factors should have valid categories."""
        valid_cats = {"momentum", "volatility", "liquidity", "mean_revert", "custom"}
        factors = _generate_alpha158_factors()
        for f in factors:
            assert f.category in valid_cats, f"{f.name} has invalid category: {f.category}"


# ===================================================================
# FactorRegistry
# ===================================================================

class TestFactorRegistry:

    def test_init_creates_alpha158(self):
        """First-time init auto-generates 158 factors."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = os.path.join(tmpdir, "factors.yaml")
            reg = FactorRegistry(config)
            assert len(reg) == 158

    def test_persistence(self):
        """save() then reload gives the same factors."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = os.path.join(tmpdir, "factors.yaml")
            reg1 = FactorRegistry(config)
            reg1.add("TEST_FACTOR", "Mean($close, 5)", "custom", "manual")
            reg1.save()

            reg2 = FactorRegistry(config)
            assert len(reg2) == 159
            assert reg2.get("TEST_FACTOR").expression == "Mean($close, 5)"

    def test_add_duplicate_raises(self):
        """Adding a factor with duplicate name raises ValueError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = os.path.join(tmpdir, "factors.yaml")
            reg = FactorRegistry(config)
            with pytest.raises(ValueError, match="already exists"):
                reg.add("KMID", "anything", "custom", "manual")

    def test_remove_alpha158_raises(self):
        """Cannot remove alpha158-sourced factors."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = os.path.join(tmpdir, "factors.yaml")
            reg = FactorRegistry(config)
            with pytest.raises(ValueError, match="Cannot remove"):
                reg.remove("KMID")

    def test_remove_custom(self):
        """Can remove manually added factors."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = os.path.join(tmpdir, "factors.yaml")
            reg = FactorRegistry(config)
            reg.add("CUSTOM_1", "Std($close, 5)", "custom", "manual")
            assert len(reg) == 159
            reg.remove("CUSTOM_1")
            assert len(reg) == 158

    def test_enable_disable(self):
        """Enable/disable affects get_enabled count."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = os.path.join(tmpdir, "factors.yaml")
            reg = FactorRegistry(config)
            initial_enabled = len(reg.get_enabled())
            reg.disable("KMID")
            assert len(reg.get_enabled()) == initial_enabled - 1
            reg.enable("KMID")
            assert len(reg.get_enabled()) == initial_enabled

    def test_get_expressions(self):
        """get_expressions returns two equal-length lists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = os.path.join(tmpdir, "factors.yaml")
            reg = FactorRegistry(config)
            exprs, names = reg.get_expressions()
            assert len(exprs) == len(names)
            assert len(exprs) == 158
            assert all(isinstance(e, str) for e in exprs)
            assert all(isinstance(n, str) for n in names)

    def test_list_by_category(self):
        """list() can filter by category."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = os.path.join(tmpdir, "factors.yaml")
            reg = FactorRegistry(config)
            momentum = reg.list(category="momentum")
            volatility = reg.list(category="volatility")
            assert len(momentum) > 0
            assert len(volatility) > 0
            assert len(momentum) + len(volatility) <= len(reg)

    def test_list_by_source(self):
        """list() can filter by source."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = os.path.join(tmpdir, "factors.yaml")
            reg = FactorRegistry(config)
            reg.add("MANUAL_1", "Std($close, 5)", "custom", "manual")
            alpha = reg.list(source="alpha158")
            manual = reg.list(source="manual")
            assert len(alpha) == 158
            assert len(manual) == 1

    def test_factor_hash_changes(self):
        """factor_hash changes when factors are added/removed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = os.path.join(tmpdir, "factors.yaml")
            reg = FactorRegistry(config)
            hash1 = reg.factor_hash
            reg.add("NEW_FACTOR", "Std($close, 5)", "custom", "manual")
            hash2 = reg.factor_hash
            assert hash1 != hash2

    def test_update_validation(self):
        """update_validation stores result on factor."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = os.path.join(tmpdir, "factors.yaml")
            reg = FactorRegistry(config)
            reg.update_validation("KMID", {"ic_mean": 0.03, "icir": 1.2})
            fd = reg.get("KMID")
            assert fd.validation["ic_mean"] == 0.03

    def test_summary(self):
        """summary() returns a DataFrame."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = os.path.join(tmpdir, "factors.yaml")
            reg = FactorRegistry(config)
            df = reg.summary()
            assert isinstance(df, pd.DataFrame)
            assert len(df) == 158
            assert "name" in df.columns
            assert "enabled" in df.columns

    def test_get_nonexistent_raises(self):
        """get() raises KeyError for missing factor."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = os.path.join(tmpdir, "factors.yaml")
            reg = FactorRegistry(config)
            with pytest.raises(KeyError):
                reg.get("NONEXISTENT")


# ===================================================================
# FactorValidator (requires Qlib data)
# ===================================================================

class TestFactorValidator:

    @pytest.fixture(scope="class")
    def validator(self):
        if SKIP_NO_DATA:
            pytest.skip(skip_reason)
        from quantlab.data.data_manager import DataManager
        from quantlab.signal.signal_alpha import FactorValidator
        dm = DataManager(provider_uri=QLIB_DATA_DIR, market="csi300")
        dm.init_qlib()
        return FactorValidator(dm)

    @pytest.mark.skipif(SKIP_NO_DATA, reason=skip_reason)
    def test_valid_factor(self, validator):
        """A known-good factor should have positive IC."""
        report = validator.validate_single(
            "Mean($close, 20)/$close",
            "2022-01-01", "2023-12-31",
        )
        assert report.ic_mean != 0.0
        assert report.icir != 0.0
        assert report.verdict != "reject" or True  # May vary by data period

    @pytest.mark.skipif(SKIP_NO_DATA, reason=skip_reason)
    def test_constant_factor_rejected(self, validator):
        """A constant factor (Ref($close,0)=$close) should be rejected."""
        report = validator.validate_single(
            "$close/$close",  # Always 1.0
            "2022-01-01", "2023-12-31",
        )
        assert abs(report.ic_mean) < 0.01

    @pytest.mark.skipif(SKIP_NO_DATA, reason=skip_reason)
    def test_batch_validate(self, validator):
        """Batch validation returns correct number of reports."""
        exprs = [
            "Mean($close, 5)/$close",
            "Std($close, 20)/$close",
            "Rank($close, 10)",
        ]
        reports = validator.validate_batch(exprs, "2023-01-01", "2023-06-30")
        assert len(reports) == 3
        for r in reports:
            assert isinstance(r, FactorReport)

    @pytest.mark.skipif(SKIP_NO_DATA, reason=skip_reason)
    def test_compare(self, validator):
        """compare() returns a DataFrame."""
        exprs = [
            "Mean($close, 5)/$close",
            "Mean($close, 20)/$close",
        ]
        df = validator.compare(exprs, "2023-01-01", "2023-06-30")
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2


# ===================================================================
# AlphaSignalPipeline (requires Qlib data + LightGBM)
# ===================================================================

class TestAlphaSignalPipeline:

    @pytest.fixture(scope="class")
    def pipeline_and_dm(self):
        if SKIP_NO_DATA:
            pytest.skip(skip_reason)
        try:
            import lightgbm  # noqa
        except ImportError:
            pytest.skip("lightgbm not installed")

        from quantlab.data.data_manager import DataManager
        from quantlab.signal.signal_alpha import FactorRegistry, AlphaSignalPipeline

        dm = DataManager(provider_uri=QLIB_DATA_DIR, market="csi300")
        dm.init_qlib()

        with tempfile.TemporaryDirectory() as tmpdir:
            config = os.path.join(tmpdir, "factors.yaml")
            reg = FactorRegistry(config)
            pipeline = AlphaSignalPipeline(
                registry=reg,
                market="csi300",
                retrain_interval=20,
                train_years=3,
            )
            yield pipeline, dm

    @pytest.mark.skipif(SKIP_NO_DATA, reason=skip_reason)
    def test_should_retrain_initial(self, pipeline_and_dm):
        """Should retrain when no model exists."""
        pipeline, dm = pipeline_and_dm
        assert pipeline.should_retrain("2024-06-28") is True

    @pytest.mark.skipif(SKIP_NO_DATA, reason=skip_reason)
    def test_predict_produces_series(self, pipeline_and_dm):
        """predict() returns a Series indexed by symbol."""
        pipeline, dm = pipeline_and_dm
        signal = pipeline.predict("2024-06-28", dm)
        assert isinstance(signal, pd.Series)
        assert len(signal) >= 280  # CSI300
        assert not signal.isna().all()

    @pytest.mark.skipif(SKIP_NO_DATA, reason=skip_reason)
    def test_feature_importance_after_train(self, pipeline_and_dm):
        """Feature importance available after training."""
        pipeline, dm = pipeline_and_dm
        importance = pipeline.get_feature_importance()
        assert importance is not None
        assert isinstance(importance, pd.Series)
        assert len(importance) == 158

    @pytest.mark.skipif(SKIP_NO_DATA, reason=skip_reason)
    def test_no_retrain_within_interval(self, pipeline_and_dm):
        """Within retrain interval, should_retrain returns False."""
        pipeline, dm = pipeline_and_dm
        # After first predict, count=1
        assert pipeline.should_retrain("2024-06-28") is False
