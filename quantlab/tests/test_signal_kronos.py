"""Tests for M3 Kronos Signal Pipeline.

Run with:
    python -m pytest quantlab/tests/test_signal_kronos.py -v
    python -m pytest quantlab/tests/test_signal_kronos.py::TestFinetuneRecipe -v
"""

import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from quantlab.signal.signal_kronos import (
    FinetuneRecipe,
    KronosOutput,
    RecipeReport,
)

_QUANTLAB_DIR = Path(__file__).resolve().parent.parent
DEFAULT_RECIPES = str(_QUANTLAB_DIR / "configs" / "kronos_recipes.yaml")

# Kronos model availability check
try:
    import torch
    from quantlab._compat import import_kronos_module
    _kronos_mod = import_kronos_module("model.kronos")
    KronosTokenizer = _kronos_mod.KronosTokenizer
    Kronos = _kronos_mod.Kronos
    HAS_KRONOS = True
except ImportError:
    HAS_KRONOS = False

QLIB_DATA_DIR = os.path.expanduser("~/.qlib/qlib_data/cn_data")
SKIP_NO_DATA = not os.path.isdir(QLIB_DATA_DIR)
SKIP_NO_KRONOS = not HAS_KRONOS


# ===================================================================
# FinetuneRecipe 测试
# ===================================================================

class TestFinetuneRecipe:

    def test_default_values(self):
        """创建空 recipe，所有字段有合理默认值。"""
        r = FinetuneRecipe()
        assert r.name == "default"
        assert r.finetune_predictor is True
        assert r.finetune_tokenizer is False
        assert r.predictor_strategy == "last_n"
        assert r.predictor_unfreeze_layers == 2
        assert r.epochs == 3
        assert r.learning_rate == 2e-5
        assert r.batch_size == 64
        assert r.temperature == 0.6
        assert r.sample_count == 10
        assert r.reset_from_pretrained is True

    def test_yaml_load_all(self):
        """加载预设方案文件，7 个方案全部正确解析。"""
        recipes = FinetuneRecipe.load_all(DEFAULT_RECIPES)
        assert len(recipes) == 7
        names = [r.name for r in recipes]
        assert "conservative" in names
        assert "aggressive" in names
        assert "head_only" in names
        assert "cumulative" in names
        assert "recency_focus" in names
        assert "zero_shot" in names
        assert "s1_heavy" in names

    def test_yaml_load_single(self):
        """加载指定名称的方案。"""
        r = FinetuneRecipe.load(DEFAULT_RECIPES, "conservative")
        assert r.name == "conservative"
        assert r.finetune_predictor is True
        assert r.finetune_tokenizer is False
        assert r.predictor_strategy == "last_n"
        assert r.predictor_unfreeze_layers == 2

    def test_yaml_load_missing_raises(self):
        """加载不存在的方案抛出 KeyError。"""
        with pytest.raises(KeyError, match="not found"):
            FinetuneRecipe.load(DEFAULT_RECIPES, "nonexistent")

    def test_validation_bad_strategy(self):
        """无效的冻结策略抛出 ValueError。"""
        with pytest.raises(ValueError, match="predictor_strategy"):
            FinetuneRecipe(predictor_strategy="invalid")

    def test_validation_bad_epochs(self):
        """epochs < 0 抛出 ValueError。"""
        with pytest.raises(ValueError, match="epochs"):
            FinetuneRecipe(epochs=-1)

    def test_validation_bad_lr(self):
        """learning_rate <= 0 抛出 ValueError。"""
        with pytest.raises(ValueError, match="learning_rate"):
            FinetuneRecipe(learning_rate=0)

    def test_validation_bad_sample_strategy(self):
        """无效的采样策略抛出 ValueError。"""
        with pytest.raises(ValueError, match="sample_strategy"):
            FinetuneRecipe(sample_strategy="invalid")

    def test_save_and_reload(self):
        """save → load 往返，字段完全一致。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test_recipes.yaml")
            r1 = FinetuneRecipe(
                name="test_recipe",
                description="test",
                finetune_tokenizer=True,
                predictor_strategy="full",
                epochs=7,
                learning_rate=1e-4,
                sample_strategy="recency_weighted",
                recency_decay=0.9,
                temperature=0.8,
            )
            r1.save(path)

            r2 = FinetuneRecipe.load(path, "test_recipe")
            assert r2.name == r1.name
            assert r2.finetune_tokenizer == r1.finetune_tokenizer
            assert r2.predictor_strategy == r1.predictor_strategy
            assert r2.epochs == r1.epochs
            assert r2.learning_rate == r1.learning_rate
            assert r2.sample_strategy == r1.sample_strategy
            assert r2.recency_decay == r1.recency_decay
            assert r2.temperature == r1.temperature

    def test_zero_shot_no_finetune(self):
        """zero_shot 方案不微调任何模块。"""
        r = FinetuneRecipe.load(DEFAULT_RECIPES, "zero_shot")
        assert r.finetune_tokenizer is False
        assert r.finetune_predictor is False

    def test_aggressive_full_finetune(self):
        """aggressive 方案全部解冻。"""
        r = FinetuneRecipe.load(DEFAULT_RECIPES, "aggressive")
        assert r.finetune_tokenizer is True
        assert r.finetune_predictor is True
        assert r.tokenizer_strategy == "full"
        assert r.predictor_strategy == "full"

    def test_cumulative_no_reset(self):
        """cumulative 方案不每天重置。"""
        r = FinetuneRecipe.load(DEFAULT_RECIPES, "cumulative")
        assert r.reset_from_pretrained is False

    def test_s1_heavy_weights(self):
        """s1_heavy 方案损失权重配置正确。"""
        r = FinetuneRecipe.load(DEFAULT_RECIPES, "s1_heavy")
        assert r.s1_loss_weight == 2.0
        assert r.s2_loss_weight == 0.5


# ===================================================================
# KronosOutput 测试
# ===================================================================

class TestKronosOutput:

    def test_empty_output(self):
        """空输出的默认值。"""
        out = KronosOutput()
        assert len(out.return_1d) == 0
        assert len(out.return_5d) == 0
        assert len(out.uncertainty) == 0
        assert len(out.pred_klines) == 0

    def test_with_data(self):
        """带数据的输出。"""
        out = KronosOutput(
            return_1d=pd.Series({"A": 0.01, "B": -0.02}),
            return_5d=pd.Series({"A": 0.03, "B": -0.01}),
            uncertainty=pd.Series({"A": 0.005, "B": 0.008}),
        )
        assert len(out.return_1d) == 2
        assert out.return_1d["A"] == 0.01
        assert out.uncertainty["B"] == 0.008


# ===================================================================
# RecipeReport 测试
# ===================================================================

class TestRecipeReport:

    def test_default_report(self):
        """默认报告所有字段为 0。"""
        r = RecipeReport()
        assert r.ic_mean == 0.0
        assert r.icir == 0.0
        assert r.total_time_sec == 0.0

    def test_report_with_values(self):
        """带值的报告。"""
        r = RecipeReport(
            recipe_name="test",
            ic_mean=0.035,
            icir=1.2,
            long_short_sharpe=1.5,
        )
        assert r.recipe_name == "test"
        assert r.ic_mean == 0.035
        assert r.icir == 1.2


# ===================================================================
# KronosFinetuner 测试 (需要 Kronos 模型)
# ===================================================================

class TestKronosFinetuner:

    @pytest.fixture(scope="class")
    def finetuner(self):
        if SKIP_NO_KRONOS:
            pytest.skip("Kronos model not available")
        from quantlab.signal.signal_kronos import KronosFinetuner

        tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
        model = Kronos.from_pretrained("NeoQuasar/Kronos-base")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        return KronosFinetuner(tokenizer, model, device=device)

    @pytest.fixture
    def dummy_data(self):
        """生成模拟股票数据。"""
        symbols = {}
        for i in range(5):
            dates = pd.bdate_range("2024-01-01", periods=30)
            np.random.seed(i)
            close = 100 + np.cumsum(np.random.randn(30) * 2)
            df = pd.DataFrame({
                "open": close + np.random.randn(30) * 0.5,
                "high": close + abs(np.random.randn(30)) * 1,
                "low": close - abs(np.random.randn(30)) * 1,
                "close": close,
                "volume": np.random.randint(1000, 10000, 30).astype(float),
                "amount": np.random.randint(100000, 1000000, 30).astype(float),
            }, index=dates)
            symbols[f"SH60000{i}"] = df
        return symbols

    @pytest.mark.skipif(SKIP_NO_KRONOS, reason="Kronos model not available")
    def test_zero_shot_no_change(self, finetuner, dummy_data):
        """零样本方案不修改模型参数。"""
        recipe = FinetuneRecipe(
            name="zero_shot",
            finetune_tokenizer=False,
            finetune_predictor=False,
        )
        tok, mdl = finetuner.finetune(dummy_data, recipe)
        # 参数应与预训练完全一致
        for (n1, p1), (n2, p2) in zip(
            tok.named_parameters(),
            finetuner.base_tokenizer.named_parameters()
        ):
            assert torch.allclose(p1.cpu(), p2.cpu()), f"Tokenizer param {n1} changed"

    @pytest.mark.skipif(SKIP_NO_KRONOS, reason="Kronos model not available")
    def test_head_only_freeze(self, finetuner, dummy_data):
        """head_only 策略只有 head 参数变化。"""
        recipe = FinetuneRecipe(
            name="head_only_test",
            finetune_predictor=True,
            predictor_strategy="head_only",
            epochs=1,
            batch_size=2,
        )
        tok, mdl = finetuner.finetune(dummy_data, recipe)
        # transformer 层参数不应变化
        base_state = finetuner.base_model_state
        for name, param in mdl.named_parameters():
            if "transformer" in name and "head" not in name and "dep_layer" not in name:
                base_param = base_state[name].to(param.device)
                assert torch.allclose(param, base_param), f"Param {name} should be frozen"


# ===================================================================
# KronosInference 测试 (需要 Kronos 模型)
# ===================================================================

class TestKronosInference:

    @pytest.mark.skipif(SKIP_NO_KRONOS, reason="Kronos model not available")
    def test_output_dimensions(self):
        """输出维度与输入股票数一致。"""
        from quantlab.signal.signal_kronos import KronosInference, FinetuneRecipe

        tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
        model = Kronos.from_pretrained("NeoQuasar/Kronos-base")
        device = "cuda" if torch.cuda.is_available() else "cpu"

        inference = KronosInference(device=device)
        recipe = FinetuneRecipe(predict_horizon=5, sample_count=2, temperature=0.6)

        # 准备数据
        symbol_data = {}
        for i in range(3):
            dates = pd.bdate_range("2024-01-01", periods=60)
            close = 100 + np.cumsum(np.random.randn(60) * 2)
            df = pd.DataFrame({
                "open": close + np.random.randn(60) * 0.5,
                "high": close + abs(np.random.randn(60)) * 1,
                "low": close - abs(np.random.randn(60)) * 1,
                "close": close,
                "volume": np.random.randint(1000, 10000, 60).astype(float),
                "amount": np.random.randint(100000, 1000000, 60).astype(float),
            }, index=dates)
            symbol_data[f"SH60000{i}"] = df

        output = inference.predict_all(
            tokenizer, model, symbol_data, recipe, "2024-03-22"
        )
        assert len(output.return_1d) == len(output.return_5d)
        assert len(output.return_1d) == len(output.uncertainty)
        assert all(output.uncertainty >= 0)

    @pytest.mark.skipif(SKIP_NO_KRONOS, reason="Kronos model not available")
    def test_uncertainty_nonneg(self):
        """不确定性应全部 >= 0。"""
        # This is implicitly tested in test_output_dimensions
        pass


# ===================================================================
# KronosSignalPipeline 端到端测试 (需要 Kronos + Qlib 数据)
# ===================================================================

class TestKronosSignalPipeline:

    @pytest.mark.skipif(
        SKIP_NO_DATA or SKIP_NO_KRONOS,
        reason="Requires both Qlib data and Kronos model"
    )
    def test_daily_run_produces_output(self):
        """daily_run 返回非空 KronosOutput。"""
        from quantlab.data.data_manager import DataManager
        from quantlab.signal.signal_kronos import FinetuneRecipe, KronosSignalPipeline

        dm = DataManager(provider_uri=QLIB_DATA_DIR, market="csi300")
        dm.init_qlib()

        recipe = FinetuneRecipe.load(DEFAULT_RECIPES, "zero_shot")
        pipeline = KronosSignalPipeline(
            recipe=recipe,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )

        # 获取数据
        df = dm.get_ohlcv_before("2024-06-28", 60)
        symbol_data = {}
        if hasattr(df.index, "get_level_values"):
            for sym in df.index.get_level_values(1).unique()[:10]:
                symbol_data[sym] = df.xs(sym, level=1).copy()

        output = pipeline.daily_run(symbol_data, "2024-06-28", dm)
        assert isinstance(output, KronosOutput)
        assert len(output.return_1d) > 0

    @pytest.mark.skipif(
        SKIP_NO_DATA or SKIP_NO_KRONOS,
        reason="Requires both Qlib data and Kronos model"
    )
    def test_switch_recipe(self):
        """switch_recipe 后使用新方案。"""
        from quantlab.signal.signal_kronos import FinetuneRecipe, KronosSignalPipeline

        recipe1 = FinetuneRecipe.load(DEFAULT_RECIPES, "conservative")
        pipeline = KronosSignalPipeline(recipe=recipe1)

        recipe2 = FinetuneRecipe.load(DEFAULT_RECIPES, "aggressive")
        pipeline.switch_recipe(recipe2)
        assert pipeline.recipe.name == "aggressive"
