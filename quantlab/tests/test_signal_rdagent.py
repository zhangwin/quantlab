"""M4 RD-Agent Signal Pipeline 离线测试。

不依赖 RD-Agent、Qlib 或 GPU，只测试数据结构、注册表、静态检查等核心逻辑。

Usage:
    pytest test_signal_rdagent.py -v
    pytest test_signal_rdagent.py -k "TestCodeFactorExecutor" -v
"""

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from quantlab.signals.signal_rdagent import (
    EvolutionConfig,
    CodeFactorEntry,
    CodeFactorRegistry,
    CodeFactorExecutor,
    FactorResult,
    RDAgentOutput,
    DecayReport,
    FactorDecayStatus,
    EvolutionReport,
    RawFactor,
    RDAgentSignalPipeline,
    EvolutionRunner,
    FORBIDDEN_IMPORTS,
    FORBIDDEN_BUILTINS,
)


# ---------------------------------------------------------------------------
# TestEvolutionConfig
# ---------------------------------------------------------------------------


class TestEvolutionConfig:
    """EvolutionConfig 数据结构测试。"""

    def test_default_values(self):
        cfg = EvolutionConfig()
        assert cfg.name == "default"
        assert cfg.max_rounds == 20
        assert cfg.total_budget == 50
        assert cfg.min_ic == 0.02
        assert cfg.min_icir == 0.5
        assert len(cfg.target_directions) == 2

    def test_validation_max_rounds(self):
        with pytest.raises(ValueError, match="max_rounds"):
            EvolutionConfig(max_rounds=-1)

    def test_validation_total_budget(self):
        with pytest.raises(ValueError, match="total_budget"):
            EvolutionConfig(total_budget=-1)

    def test_validation_timeout(self):
        with pytest.raises(ValueError, match="timeout_hours"):
            EvolutionConfig(timeout_hours=0)

    def test_validation_corr(self):
        with pytest.raises(ValueError, match="max_corr_with_alpha"):
            EvolutionConfig(max_corr_with_alpha=0)

    def test_save_and_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "config.yaml")
            cfg = EvolutionConfig(
                name="test_cfg",
                max_rounds=10,
                total_budget=30,
                target_directions=["mean_revert"],
            )
            cfg.save(path)

            loaded = EvolutionConfig.load(path, "test_cfg")
            assert loaded.name == "test_cfg"
            assert loaded.max_rounds == 10
            assert loaded.total_budget == 30
            assert loaded.target_directions == ["mean_revert"]

    def test_load_not_found(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "config.yaml")
            cfg = EvolutionConfig(name="a")
            cfg.save(path)
            with pytest.raises(KeyError, match="not found"):
                EvolutionConfig.load(path, "nonexistent")

    def test_load_all(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "config.yaml")
            EvolutionConfig(name="a", max_rounds=5).save(path)
            EvolutionConfig(name="b", max_rounds=10).save(path)

            all_cfgs = EvolutionConfig.load_all(path)
            assert len(all_cfgs) == 2
            names = {c.name for c in all_cfgs}
            assert names == {"a", "b"}

    def test_save_overwrites_existing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "config.yaml")
            EvolutionConfig(name="x", max_rounds=5).save(path)
            EvolutionConfig(name="x", max_rounds=99).save(path)

            loaded = EvolutionConfig.load(path, "x")
            assert loaded.max_rounds == 99


# ---------------------------------------------------------------------------
# TestCodeFactorEntry
# ---------------------------------------------------------------------------


class TestCodeFactorEntry:
    """CodeFactorEntry 数据结构测试。"""

    def test_defaults(self):
        entry = CodeFactorEntry()
        assert entry.status == "active"
        assert entry.enabled is True
        assert entry.weight == 0.0
        assert entry.ic_history == []

    def test_custom_values(self):
        entry = CodeFactorEntry(
            name="test_factor",
            direction="mean_revert",
            ic_at_creation=0.05,
            ic_history=[("2024-01-01", 0.03), ("2024-01-02", 0.04)],
        )
        assert entry.name == "test_factor"
        assert len(entry.ic_history) == 2


# ---------------------------------------------------------------------------
# TestCodeFactorRegistry
# ---------------------------------------------------------------------------


VALID_FACTOR_CODE = '''import numpy as np
import pandas as pd

def compute_factor(ohlcv):
    """Test factor: return close/open ratio."""
    result = {}
    for sym, df in ohlcv.items():
        result[sym] = df["close"].iloc[-1] / df["open"].iloc[-1] - 1
    return pd.Series(result)
'''


class TestCodeFactorRegistry:
    """CodeFactorRegistry 测试。"""

    def _make_registry(self, tmpdir):
        code_dir = os.path.join(tmpdir, "factors")
        reg_path = os.path.join(tmpdir, "registry.yaml")
        return CodeFactorRegistry(code_dir, reg_path)

    def test_register_and_get(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            reg = self._make_registry(tmpdir)
            entry = reg.register("test_f", VALID_FACTOR_CODE, "mean_revert")
            assert entry.name == "test_f"
            assert entry.direction == "mean_revert"
            assert entry.status == "active"
            assert len(reg) == 1

    def test_register_duplicate(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            reg = self._make_registry(tmpdir)
            reg.register("dup", VALID_FACTOR_CODE, "mean_revert")
            with pytest.raises(ValueError, match="already exists"):
                reg.register("dup", VALID_FACTOR_CODE, "mean_revert")

    def test_get_active(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            reg = self._make_registry(tmpdir)
            reg.register("a", VALID_FACTOR_CODE, "mean_revert")
            reg.register("b", VALID_FACTOR_CODE.replace("test_f", "b"), "vol")
            reg.retire("b", "test")
            assert len(reg.get_active()) == 1
            assert reg.get_active()[0].name == "a"

    def test_retire(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            reg = self._make_registry(tmpdir)
            reg.register("r", VALID_FACTOR_CODE, "mean_revert")
            reg.retire("r", "decay")
            entry = reg.get("r")
            assert entry.status == "retired"
            assert entry.enabled is False

    def test_set_probation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            reg = self._make_registry(tmpdir)
            reg.register("p", VALID_FACTOR_CODE, "mean_revert")
            reg.set_probation("p")
            assert reg.get("p").status == "probation"

    def test_update_ic(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            reg = self._make_registry(tmpdir)
            reg.register("ic_test", VALID_FACTOR_CODE, "mean_revert")
            reg.update_ic("ic_test", "2024-01-01", 0.05)
            reg.update_ic("ic_test", "2024-01-02", 0.03)
            entry = reg.get("ic_test")
            assert len(entry.ic_history) == 2

    def test_update_weights(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            reg = self._make_registry(tmpdir)
            reg.register("w1", VALID_FACTOR_CODE, "a")
            reg.register("w2", VALID_FACTOR_CODE.replace("test_f", "w2"), "b")
            reg.update_weights({"w1": 2.0, "w2": 1.0})
            assert abs(reg.get("w1").weight - 2.0/3.0) < 1e-6
            assert abs(reg.get("w2").weight - 1.0/3.0) < 1e-6

    def test_update_weights_all_negative(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            reg = self._make_registry(tmpdir)
            reg.register("n1", VALID_FACTOR_CODE, "a")
            reg.register("n2", VALID_FACTOR_CODE.replace("test_f", "n2"), "b")
            reg.update_weights({"n1": -0.5, "n2": -1.0})
            # 退化为等权
            assert abs(reg.get("n1").weight - 0.5) < 1e-6
            assert abs(reg.get("n2").weight - 0.5) < 1e-6

    def test_load_code(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            reg = self._make_registry(tmpdir)
            reg.register("lc", VALID_FACTOR_CODE, "mean_revert")
            code = reg.load_code("lc")
            assert "compute_factor" in code

    def test_persistence(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            code_dir = os.path.join(tmpdir, "factors")
            reg_path = os.path.join(tmpdir, "registry.yaml")

            reg1 = CodeFactorRegistry(code_dir, reg_path)
            reg1.register("persist", VALID_FACTOR_CODE, "mean_revert")
            reg1.update_ic("persist", "2024-01-01", 0.04)
            reg1.save()

            reg2 = CodeFactorRegistry(code_dir, reg_path)
            assert len(reg2) == 1
            assert reg2.get("persist").name == "persist"
            assert len(reg2.get("persist").ic_history) == 1

    def test_summary(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            reg = self._make_registry(tmpdir)
            reg.register("s1", VALID_FACTOR_CODE, "mean_revert")
            df = reg.summary()
            assert len(df) == 1
            assert "name" in df.columns
            assert "status" in df.columns

    def test_get_nonexistent(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            reg = self._make_registry(tmpdir)
            with pytest.raises(KeyError):
                reg.get("nope")


# ---------------------------------------------------------------------------
# TestCodeFactorExecutor
# ---------------------------------------------------------------------------


class TestCodeFactorExecutor:
    """CodeFactorExecutor 静态检查测试。"""

    def test_valid_code(self):
        executor = CodeFactorExecutor()
        ok, err = executor.static_check(VALID_FACTOR_CODE)
        assert ok is True
        assert err == ""

    def test_syntax_error(self):
        executor = CodeFactorExecutor()
        ok, err = executor.static_check("def compute_factor(:\n  pass")
        assert ok is False
        assert "语法错误" in err

    def test_forbidden_import_os(self):
        executor = CodeFactorExecutor()
        code = "import os\ndef compute_factor(x): pass"
        ok, err = executor.static_check(code)
        assert ok is False
        assert "禁止 import" in err

    def test_forbidden_import_requests(self):
        executor = CodeFactorExecutor()
        code = "import requests\ndef compute_factor(x): pass"
        ok, err = executor.static_check(code)
        assert ok is False

    def test_forbidden_import_from(self):
        executor = CodeFactorExecutor()
        code = "from subprocess import run\ndef compute_factor(x): pass"
        ok, err = executor.static_check(code)
        assert ok is False

    def test_forbidden_builtin_exec(self):
        executor = CodeFactorExecutor()
        code = "def compute_factor(x): exec('1+1')"
        ok, err = executor.static_check(code)
        assert ok is False
        assert "禁止调用" in err

    def test_forbidden_builtin_eval(self):
        executor = CodeFactorExecutor()
        code = "def compute_factor(x): return eval('1')"
        ok, err = executor.static_check(code)
        assert ok is False

    def test_missing_compute_factor(self):
        executor = CodeFactorExecutor()
        code = "def my_factor(x): pass"
        ok, err = executor.static_check(code)
        assert ok is False
        assert "compute_factor" in err

    def test_too_many_lines(self):
        executor = CodeFactorExecutor()
        code = "def compute_factor(x): pass\n" + "\n".join(["x=1"] * 150)
        ok, err = executor.static_check(code, max_lines=100)
        assert ok is False
        assert "行限制" in err

    def test_allowed_imports(self):
        executor = CodeFactorExecutor()
        code = """import numpy as np
import pandas as pd
from scipy import stats

def compute_factor(ohlcv):
    return pd.Series()
"""
        ok, err = executor.static_check(code)
        assert ok is True

    def test_inprocess_execution(self):
        executor = CodeFactorExecutor(sandbox_mode="inprocess")
        ohlcv = {
            "SH600000": pd.DataFrame({
                "open": [10, 11, 12],
                "high": [11, 12, 13],
                "low": [9, 10, 11],
                "close": [10.5, 11.5, 12.5],
                "volume": [100, 200, 300],
                "amount": [1000, 2000, 3000],
            })
        }
        result = executor.execute_factor(VALID_FACTOR_CODE, ohlcv)
        assert result.success is True
        assert "SH600000" in result.values.index

    def test_inprocess_execution_error(self):
        executor = CodeFactorExecutor(sandbox_mode="inprocess")
        code = """import pandas as pd
def compute_factor(ohlcv):
    raise RuntimeError("intentional error")
"""
        result = executor.execute_factor(code, {})
        assert result.success is False

    def test_inprocess_bad_return_type(self):
        executor = CodeFactorExecutor(sandbox_mode="inprocess")
        code = """import pandas as pd
def compute_factor(ohlcv):
    return {"a": 1}
"""
        result = executor.execute_factor(code, {})
        assert result.success is False
        assert "Series" in result.error

    def test_inprocess_all_nan(self):
        executor = CodeFactorExecutor(sandbox_mode="inprocess")
        code = """import pandas as pd
import numpy as np
def compute_factor(ohlcv):
    return pd.Series([np.nan, np.nan])
"""
        result = executor.execute_factor(code, {})
        assert result.success is False
        assert "NaN" in result.error

    def test_batch_execution(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            code_dir = os.path.join(tmpdir, "factors")
            reg_path = os.path.join(tmpdir, "registry.yaml")
            registry = CodeFactorRegistry(code_dir, reg_path)
            registry.register("bf1", VALID_FACTOR_CODE, "a")

            executor = CodeFactorExecutor(sandbox_mode="inprocess")
            entries = registry.get_active()
            ohlcv = {
                "SH600000": pd.DataFrame({
                    "open": [10, 11, 12],
                    "high": [11, 12, 13],
                    "low": [9, 10, 11],
                    "close": [10.5, 11.5, 12.5],
                    "volume": [100, 200, 300],
                    "amount": [1000, 2000, 3000],
                })
            }
            results = executor.execute_batch(entries, ohlcv, registry)
            assert "bf1" in results
            assert results["bf1"].success is True


# ---------------------------------------------------------------------------
# TestRDAgentOutput / TestDecayReport
# ---------------------------------------------------------------------------


class TestDataStructures:
    """输出数据结构测试。"""

    def test_rdagent_output_empty(self):
        out = RDAgentOutput()
        assert out.signal.empty
        assert out.factor_count == 0

    def test_rdagent_output_with_data(self):
        signal = pd.Series({"A": 0.5, "B": 0.3})
        out = RDAgentOutput(
            signal=signal,
            factor_count=2,
            factor_signals={"f1": signal},
            factor_weights={"f1": 1.0},
        )
        assert len(out.signal) == 2
        assert out.factor_count == 2

    def test_decay_report(self):
        report = DecayReport(checked_date="2024-01-01")
        status = FactorDecayStatus(
            name="test",
            rolling_ic_30d=0.03,
            ic_trend="stable",
        )
        report.factor_reports.append(status)
        assert len(report.factor_reports) == 1

    def test_evolution_report(self):
        report = EvolutionReport(
            config_name="test",
            total_rounds=10,
            total_candidates=20,
            registered=5,
        )
        assert report.total_candidates == 20

    def test_factor_result(self):
        r = FactorResult(success=True, values=pd.Series([1, 2, 3]))
        assert r.success is True
        assert len(r.values) == 3

    def test_raw_factor(self):
        rf = RawFactor(code="x=1", name="test", round_idx=3)
        assert rf.round_idx == 3


# ---------------------------------------------------------------------------
# TestRDAgentSignalPipeline
# ---------------------------------------------------------------------------


class TestRDAgentSignalPipeline:
    """RDAgentSignalPipeline 离线逻辑测试。"""

    def _setup_pipeline(self, tmpdir):
        code_dir = os.path.join(tmpdir, "factors")
        reg_path = os.path.join(tmpdir, "registry.yaml")
        registry = CodeFactorRegistry(code_dir, reg_path)
        executor = CodeFactorExecutor(sandbox_mode="inprocess")
        pipeline = RDAgentSignalPipeline(
            code_registry=registry,
            executor=executor,
            window=30,
        )
        return registry, pipeline

    def test_compute_no_factors(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            registry, pipeline = self._setup_pipeline(tmpdir)
            # Mock data_manager
            class MockDM:
                def get_ohlcv_before(self, date, window):
                    return pd.DataFrame()
            out = pipeline.compute("2024-01-01", MockDM())
            assert out.signal.empty

    def test_get_factor_status_empty(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            registry, pipeline = self._setup_pipeline(tmpdir)
            df = pipeline.get_factor_status()
            assert len(df) == 0

    def test_get_factor_status_with_data(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            registry, pipeline = self._setup_pipeline(tmpdir)
            registry.register("s1", VALID_FACTOR_CODE, "mean_revert")
            df = pipeline.get_factor_status()
            assert len(df) == 1

    def test_check_decay_no_factors(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            registry, pipeline = self._setup_pipeline(tmpdir)
            report = pipeline.check_decay("2024-01-01", None)
            assert len(report.factor_reports) == 0

    def test_check_decay_suggests_evolution(self):
        """Active 因子数不足时建议进化。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            registry, pipeline = self._setup_pipeline(tmpdir)
            # 注册 2 个因子（不足 3 个阈值）
            registry.register("d1", VALID_FACTOR_CODE, "a")
            registry.register("d2", VALID_FACTOR_CODE.replace("test_f", "d2"), "b")
            report = pipeline.check_decay("2024-01-01", None)
            assert report.suggest_evolution is True


# ---------------------------------------------------------------------------
# TestEvolutionRunner
# ---------------------------------------------------------------------------


class TestEvolutionRunner:
    """EvolutionRunner 模板因子生成测试。"""

    def test_template_generation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            code_dir = os.path.join(tmpdir, "factors")
            reg_path = os.path.join(tmpdir, "registry.yaml")
            registry = CodeFactorRegistry(code_dir, reg_path)
            config = EvolutionConfig(
                name="test",
                target_directions=["mean_revert", "volatility_anomaly"],
                total_budget=10,
            )
            runner = EvolutionRunner(config=config, code_registry=registry)
            raw = runner._generate_template_factors()
            assert len(raw) > 0
            for f in raw:
                assert f.code
                assert f.name

    def test_validate_and_register(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            code_dir = os.path.join(tmpdir, "factors")
            reg_path = os.path.join(tmpdir, "registry.yaml")
            registry = CodeFactorRegistry(code_dir, reg_path)
            config = EvolutionConfig(
                name="test",
                target_directions=["mean_revert"],
            )
            runner = EvolutionRunner(config=config, code_registry=registry)

            raw = [RawFactor(
                code=VALID_FACTOR_CODE,
                name="manual_test",
                description="manual",
            )]
            registered = runner.validate_and_register(raw)
            assert len(registered) == 1
            assert registered[0].name == "manual_test"

    def test_validate_bad_code_rejected(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            code_dir = os.path.join(tmpdir, "factors")
            reg_path = os.path.join(tmpdir, "registry.yaml")
            registry = CodeFactorRegistry(code_dir, reg_path)
            config = EvolutionConfig(name="test")
            runner = EvolutionRunner(config=config, code_registry=registry)

            raw = [RawFactor(
                code="import os\ndef compute_factor(x): pass",
                name="bad_factor",
            )]
            registered = runner.validate_and_register(raw)
            assert len(registered) == 0

    def test_extract_factors_empty(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            code_dir = os.path.join(tmpdir, "factors")
            reg_path = os.path.join(tmpdir, "registry.yaml")
            registry = CodeFactorRegistry(code_dir, reg_path)
            config = EvolutionConfig(name="test")
            runner = EvolutionRunner(config=config, code_registry=registry)

            raw = runner.extract_factors(tmpdir)
            assert raw == []

    def test_run_evolution_falls_back_to_templates(self):
        """RD-Agent 不可用时退化为模板生成。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            code_dir = os.path.join(tmpdir, "factors")
            reg_path = os.path.join(tmpdir, "registry.yaml")
            registry = CodeFactorRegistry(code_dir, reg_path)
            config = EvolutionConfig(
                name="test",
                target_directions=["mean_revert", "volatility_anomaly"],
                max_rounds=2,
                total_budget=5,
            )
            runner = EvolutionRunner(config=config, code_registry=registry)
            report = runner.run_evolution()
            assert report.total_candidates > 0
            assert report.registered > 0


# ---------------------------------------------------------------------------
# 加载预置配置文件测试
# ---------------------------------------------------------------------------


class TestPresetConfigs:
    """测试预置的 rdagent_evolution.yaml 配置。"""

    _QUANTLAB_DIR = Path(__file__).resolve().parent.parent
    YAML_PATH = str(_QUANTLAB_DIR / "configs" / "rdagent_evolution.yaml")

    @pytest.mark.skipif(
        not os.path.exists(str(Path(__file__).resolve().parent.parent / "configs" / "rdagent_evolution.yaml")),
        reason="配置文件不存在",
    )
    def test_load_preset_configs(self):
        configs = EvolutionConfig.load_all(self.YAML_PATH)
        assert len(configs) >= 4

    @pytest.mark.skipif(
        not os.path.exists(str(Path(__file__).resolve().parent.parent / "configs" / "rdagent_evolution.yaml")),
        reason="配置文件不存在",
    )
    def test_preset_names(self):
        configs = EvolutionConfig.load_all(self.YAML_PATH)
        names = {c.name for c in configs}
        assert "mean_revert_focus" in names
        assert "broad_explore" in names

    @pytest.mark.skipif(
        not os.path.exists(str(Path(__file__).resolve().parent.parent / "configs" / "rdagent_evolution.yaml")),
        reason="配置文件不存在",
    )
    def test_preset_broad_explore(self):
        cfg = EvolutionConfig.load(self.YAML_PATH, "broad_explore")
        assert cfg.total_budget == 100
        assert len(cfg.target_directions) >= 5
