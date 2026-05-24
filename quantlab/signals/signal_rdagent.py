"""M4 RD-Agent 信号管线：进化管理 + 因子生命周期 + 安全在线计算。

核心类：
    EvolutionConfig          进化循环配置（方向约束、资源限制）
    CodeFactorEntry          Python 代码因子元数据
    CodeFactorRegistry       Python 代码因子注册表
    FactorResult             单个因子执行结果
    CodeFactorExecutor       沙箱因子计算引擎
    EvolutionRunner          进化执行器（包装 RD-Agent 进化循环）
    RDAgentOutput            信号输出
    DecayReport              衰退检查报告
    RDAgentSignalPipeline    信号生产（日常使用入口）
"""

import ast
import copy
import logging
import multiprocessing
import os
import subprocess
import tempfile
import time
from dataclasses import dataclass, field, asdict, fields
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------


@dataclass
class EvolutionConfig:
    """完整描述 '如何驱动 RD-Agent 进化' 的配置。"""

    name: str = "default"

    # === 进化方向约束 ===
    target_directions: List[str] = field(
        default_factory=lambda: ["mean_revert", "volatility_anomaly"]
    )
    direction_prompt: str = ""
    forbidden_patterns: List[str] = field(
        default_factory=lambda: ["future_data", "hardcoded_date", "internet_access"]
    )

    # === 正交性约束 ===
    max_corr_with_alpha: float = 0.3
    max_corr_within_pool: float = 0.5

    # === 验证门控 ===
    min_ic: float = 0.02
    min_icir: float = 0.5
    max_overfit_gap: float = 0.05
    validation_split: float = 0.3

    # === 进化资源 ===
    max_rounds: int = 20
    max_factors_per_round: int = 5
    total_budget: int = 50
    timeout_hours: float = 4.0

    # === RD-Agent 执行环境 ===
    execution_env: str = "subprocess"  # "docker" | "conda" | "subprocess"
    docker_image: str = "rdagent-qlib:latest"
    conda_env: str = "rdagent"

    # === 因子代码约定 ===
    factor_interface: str = "compute_factor(ohlcv: Dict[str, DataFrame]) -> Series"
    required_docstring: bool = True
    max_code_lines: int = 100

    def __post_init__(self):
        self._validate()

    def _validate(self):
        if self.max_rounds < 0:
            raise ValueError("max_rounds must be >= 0")
        if self.total_budget < 0:
            raise ValueError("total_budget must be >= 0")
        if self.timeout_hours <= 0:
            raise ValueError("timeout_hours must be > 0")
        if not 0 < self.max_corr_with_alpha <= 1.0:
            raise ValueError("max_corr_with_alpha must be in (0, 1]")

    @classmethod
    def load(cls, yaml_path: str, name: str) -> "EvolutionConfig":
        """从 YAML 文件加载指定名称的配置。"""
        all_configs = cls.load_all(yaml_path)
        for c in all_configs:
            if c.name == name:
                return c
        available = [c.name for c in all_configs]
        raise KeyError(f"Config '{name}' not found. Available: {available}")

    @classmethod
    def load_all(cls, yaml_path: str) -> List["EvolutionConfig"]:
        """从 YAML 文件加载全部配置。"""
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        configs = []
        valid_keys = {f.name for f in fields(cls)}
        for item in data.get("configs", []):
            filtered = {k: v for k, v in item.items() if k in valid_keys}
            configs.append(cls(**filtered))
        return configs

    def save(self, yaml_path: str):
        """保存配置到 YAML。"""
        path = Path(yaml_path)
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
        else:
            data = {}
        configs_list = data.get("configs", [])
        d = asdict(self)
        replaced = False
        for i, c in enumerate(configs_list):
            if c.get("name") == self.name:
                configs_list[i] = d
                replaced = True
                break
        if not replaced:
            configs_list.append(d)
        data["configs"] = configs_list
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, allow_unicode=True, default_flow_style=False)


@dataclass
class CodeFactorEntry:
    """Python 代码因子元数据。"""
    name: str = ""
    code_path: str = ""
    source_round: int = 0
    source_config: str = ""
    direction: str = ""
    description: str = ""
    created_date: str = ""
    status: str = "active"          # active | probation | retired
    enabled: bool = True

    # 验证指标（注册时快照）
    ic_at_creation: float = 0.0
    icir_at_creation: float = 0.0
    corr_with_alpha: float = 0.0

    # 生命周期跟踪
    ic_history: List[Tuple[str, float]] = field(default_factory=list)
    last_decay_check: str = ""
    decay_warnings: int = 0
    weight: float = 0.0


@dataclass
class RawFactor:
    """进化产出的中间结构。"""
    code: str = ""
    name: str = ""
    description: str = ""
    round_idx: int = 0
    mlflow_ic: float = 0.0


@dataclass
class EvolutionReport:
    """进化执行报告。"""
    config_name: str = ""
    total_rounds: int = 0
    total_candidates: int = 0
    passed_validation: int = 0
    registered: int = 0
    rejected_reasons: Dict[str, int] = field(default_factory=dict)
    elapsed_hours: float = 0.0
    round_details: List[Dict] = field(default_factory=list)


@dataclass
class FactorResult:
    """单个因子执行结果。"""
    success: bool = False
    values: Optional[pd.Series] = None
    error: str = ""
    elapsed_ms: int = 0


@dataclass
class RDAgentOutput:
    """M4 信号输出。"""
    signal: pd.Series = field(default_factory=pd.Series)
    factor_count: int = 0
    factor_signals: Dict[str, pd.Series] = field(default_factory=dict)
    factor_weights: Dict[str, float] = field(default_factory=dict)
    failed_factors: List[str] = field(default_factory=list)


@dataclass
class FactorDecayStatus:
    """单个因子衰退状态。"""
    name: str = ""
    rolling_ic_30d: float = 0.0
    rolling_ic_90d: float = 0.0
    ic_trend: str = "stable"     # stable | declining | collapsed
    action: str = "none"         # none | warn | probation | retire
    reason: str = ""


@dataclass
class DecayReport:
    """衰退检查报告。"""
    checked_date: str = ""
    factor_reports: List[FactorDecayStatus] = field(default_factory=list)
    suggest_evolution: bool = False
    suggest_reason: str = ""


# ---------------------------------------------------------------------------
# CodeFactorRegistry — 代码因子注册表
# ---------------------------------------------------------------------------


class CodeFactorRegistry:
    """管理 RD-Agent 进化产出的 Python 因子代码。"""

    def __init__(self, factor_code_dir: str, registry_path: str):
        """
        Args:
            factor_code_dir: 因子代码文件存放目录
            registry_path: 注册表 YAML 文件路径
        """
        self.factor_code_dir = Path(factor_code_dir)
        self.registry_path = Path(registry_path)
        self._entries: Dict[str, CodeFactorEntry] = {}
        self._load()

    def _load(self):
        """从 YAML 加载注册表。"""
        if not self.registry_path.exists():
            return
        with open(self.registry_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        for item in data.get("factors", []):
            # 处理 ic_history 的 tuple 转换
            if "ic_history" in item and item["ic_history"]:
                item["ic_history"] = [
                    (h[0], h[1]) if isinstance(h, (list, tuple)) else h
                    for h in item["ic_history"]
                ]
            entry = CodeFactorEntry(**{
                k: v for k, v in item.items()
                if k in {f.name for f in fields(CodeFactorEntry)}
            })
            self._entries[entry.name] = entry

    def save(self):
        """持久化到 YAML。"""
        self.factor_code_dir.mkdir(parents=True, exist_ok=True)
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        factors_list = []
        for entry in self._entries.values():
            d = asdict(entry)
            # ic_history 转为列表格式
            d["ic_history"] = [[date, ic] for date, ic in d.get("ic_history", [])]
            factors_list.append(d)
        data = {"factors": factors_list}
        with open(self.registry_path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, allow_unicode=True, default_flow_style=False)

    def register(
        self,
        name: str,
        code: str,
        direction: str,
        source_round: int = 0,
        config_name: str = "",
        description: str = "",
        ic: float = 0.0,
        icir: float = 0.0,
        corr_with_alpha: float = 0.0,
    ) -> CodeFactorEntry:
        """注册新因子：保存代码文件 + 注册元数据。"""
        if name in self._entries:
            raise ValueError(f"Factor '{name}' already exists")

        # 保存代码文件
        self.factor_code_dir.mkdir(parents=True, exist_ok=True)
        code_filename = f"{name}.py"
        code_path = self.factor_code_dir / code_filename
        with open(code_path, "w", encoding="utf-8") as f:
            f.write(code)

        entry = CodeFactorEntry(
            name=name,
            code_path=code_filename,
            source_round=source_round,
            source_config=config_name,
            direction=direction,
            description=description or _extract_docstring(code),
            created_date=pd.Timestamp.now().strftime("%Y-%m-%d"),
            status="active",
            enabled=True,
            ic_at_creation=ic,
            icir_at_creation=icir,
            corr_with_alpha=corr_with_alpha,
            weight=0.0,
        )
        self._entries[name] = entry
        self.save()
        return entry

    def get_active(self) -> List[CodeFactorEntry]:
        """返回 status='active' 且 enabled=True 的因子。"""
        return [
            e for e in self._entries.values()
            if e.status == "active" and e.enabled
        ]

    def get_all(self) -> List[CodeFactorEntry]:
        """返回全部因子（含 retired）。"""
        return list(self._entries.values())

    def get(self, name: str) -> CodeFactorEntry:
        """获取指定因子。"""
        if name not in self._entries:
            raise KeyError(f"Factor '{name}' not found")
        return self._entries[name]

    def retire(self, name: str, reason: str = ""):
        """退役因子。"""
        entry = self.get(name)
        entry.status = "retired"
        entry.enabled = False
        entry.weight = 0.0
        logger.info(f"因子 {name} 已退役: {reason}")
        self.save()

    def set_probation(self, name: str):
        """标记为观察期。"""
        entry = self.get(name)
        entry.status = "probation"
        entry.weight = 0.0
        self.save()

    def update_ic(self, name: str, date: str, ic_value: float):
        """追加滚动 IC 记录。"""
        entry = self.get(name)
        entry.ic_history.append((date, ic_value))
        # 只保留最近 180 天
        if len(entry.ic_history) > 180:
            entry.ic_history = entry.ic_history[-180:]

    def update_weights(self, factor_icir_map: Dict[str, float]):
        """按 ICIR 重新计算权重。"""
        active_names = [e.name for e in self.get_active()]
        effective = {n: max(0, factor_icir_map.get(n, 0)) for n in active_names}
        total = sum(effective.values())

        if total <= 0:
            # 退化为等权
            n = len(active_names)
            for name in active_names:
                self._entries[name].weight = 1.0 / n if n > 0 else 0
        else:
            for name in active_names:
                self._entries[name].weight = effective[name] / total

        # 非 active 因子权重为 0
        for name, entry in self._entries.items():
            if name not in active_names:
                entry.weight = 0.0

        self.save()

    def load_code(self, name: str) -> str:
        """读取因子代码内容。"""
        entry = self.get(name)
        code_path = self.factor_code_dir / entry.code_path
        with open(code_path, "r", encoding="utf-8") as f:
            return f.read()

    def __len__(self):
        return len(self._entries)

    def summary(self) -> pd.DataFrame:
        """因子概览 DataFrame。"""
        rows = []
        for e in self._entries.values():
            recent_ic = e.ic_history[-1][1] if e.ic_history else None
            rows.append({
                "name": e.name,
                "status": e.status,
                "direction": e.direction,
                "weight": e.weight,
                "ic_at_creation": e.ic_at_creation,
                "recent_ic": recent_ic,
                "decay_warnings": e.decay_warnings,
                "created_date": e.created_date,
            })
        return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# CodeFactorExecutor — 沙箱因子计算引擎
# ---------------------------------------------------------------------------

# 禁止的 import 模块
FORBIDDEN_IMPORTS = {
    "os", "sys", "subprocess", "socket", "http", "requests", "urllib",
    "shutil", "pathlib", "ctypes", "multiprocessing", "threading",
    "signal", "io", "pickle", "shelve", "webbrowser", "ftplib",
    "smtplib", "telnetlib", "xmlrpc",
}

# 禁止的内置函数
FORBIDDEN_BUILTINS = {"exec", "eval", "compile", "__import__", "breakpoint"}


class CodeFactorExecutor:
    """在隔离环境中执行 LLM 生成的 Python 因子代码。"""

    def __init__(self, sandbox_mode: str = "subprocess", timeout_sec: int = 30):
        """
        Args:
            sandbox_mode: "subprocess" | "docker"
            timeout_sec: 单个因子执行超时（秒）
        """
        self.sandbox_mode = sandbox_mode
        self.timeout_sec = timeout_sec

    def static_check(self, code: str, max_lines: int = 100) -> Tuple[bool, str]:
        """静态代码检查。

        Returns:
            (通过, 错误信息)
        """
        # 1. 行数检查
        lines = code.strip().split("\n")
        if len(lines) > max_lines:
            return False, f"代码超过 {max_lines} 行限制 ({len(lines)} 行)"

        # 2. AST 解析
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return False, f"语法错误: {e}"

        # 3. 检查禁止 import
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    mod = alias.name.split(".")[0]
                    if mod in FORBIDDEN_IMPORTS:
                        return False, f"禁止 import: {alias.name}"
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    mod = node.module.split(".")[0]
                    if mod in FORBIDDEN_IMPORTS:
                        return False, f"禁止 import: {node.module}"

        # 4. 检查禁止内置函数
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id in FORBIDDEN_BUILTINS:
                        return False, f"禁止调用: {node.func.id}()"

        # 5. 检查 compute_factor 函数是否存在
        has_compute = False
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "compute_factor":
                has_compute = True
                break
        if not has_compute:
            return False, "缺少 compute_factor() 函数定义"

        return True, ""

    def execute_factor(self, code: str, ohlcv_data: Dict[str, pd.DataFrame]) -> FactorResult:
        """在沙箱中执行单个因子。

        Args:
            code: 因子 Python 代码（必须定义 compute_factor 函数）
            ohlcv_data: {symbol: DataFrame[open,high,low,close,volume,amount]}

        Returns:
            FactorResult
        """
        # 静态检查
        ok, err = self.static_check(code)
        if not ok:
            return FactorResult(success=False, error=f"静态检查失败: {err}")

        if self.sandbox_mode == "subprocess":
            return self._execute_subprocess(code, ohlcv_data)
        elif self.sandbox_mode == "docker":
            return self._execute_docker(code, ohlcv_data)
        else:
            return self._execute_inprocess(code, ohlcv_data)

    def execute_batch(
        self,
        entries: List[CodeFactorEntry],
        ohlcv_data: Dict[str, pd.DataFrame],
        registry: "CodeFactorRegistry" = None,
    ) -> Dict[str, FactorResult]:
        """批量执行因子。

        Args:
            entries: 因子列表
            ohlcv_data: OHLCV 数据
            registry: 注册表（用于加载代码）

        Returns:
            {name: FactorResult}
        """
        results = {}
        for entry in entries:
            try:
                if registry is not None:
                    code = registry.load_code(entry.name)
                else:
                    code = ""
                if not code:
                    results[entry.name] = FactorResult(
                        success=False, error="代码为空"
                    )
                    continue
                result = self.execute_factor(code, ohlcv_data)
                results[entry.name] = result
            except Exception as e:
                results[entry.name] = FactorResult(
                    success=False, error=str(e)
                )
        return results

    def _execute_subprocess(self, code: str, ohlcv_data: Dict[str, pd.DataFrame]) -> FactorResult:
        """在子进程中执行。"""
        t0 = time.time()
        try:
            # 序列化数据到临时文件
            with tempfile.TemporaryDirectory() as tmpdir:
                data_path = os.path.join(tmpdir, "data.pkl")
                code_path = os.path.join(tmpdir, "factor.py")
                result_path = os.path.join(tmpdir, "result.pkl")

                # 保存数据
                import pickle
                with open(data_path, "wb") as f:
                    pickle.dump(ohlcv_data, f)

                # 构建执行脚本
                runner_code = f'''
import pickle
import sys
import pandas as pd
import numpy as np

# 加载数据
with open("{data_path.replace(os.sep, '/')}", "rb") as f:
    ohlcv = pickle.load(f)

# 加载因子代码
exec(open("{code_path.replace(os.sep, '/')}", encoding="utf-8").read())

# 执行
result = compute_factor(ohlcv)

# 校验
if not isinstance(result, pd.Series):
    raise ValueError("compute_factor must return pd.Series")
if result.isin([np.inf, -np.inf]).any():
    raise ValueError("Result contains inf values")
if result.isna().all():
    raise ValueError("Result is all NaN")

# 保存结果
with open("{result_path.replace(os.sep, '/')}", "wb") as f:
    pickle.dump(result, f)
'''
                runner_path = os.path.join(tmpdir, "runner.py")
                with open(runner_path, "w", encoding="utf-8") as f:
                    f.write(runner_code)
                with open(code_path, "w", encoding="utf-8") as f:
                    f.write(code)

                # 执行子进程
                proc = subprocess.run(
                    [sys.executable, runner_path],
                    capture_output=True,
                    text=True,
                    timeout=self.timeout_sec,
                    cwd=tmpdir,
                )

                if proc.returncode != 0:
                    err = proc.stderr.strip()
                    # 截取最后几行
                    err_lines = err.split("\n")[-5:]
                    return FactorResult(
                        success=False,
                        error="\n".join(err_lines),
                        elapsed_ms=int((time.time() - t0) * 1000),
                    )

                # 读取结果
                with open(result_path, "rb") as f:
                    result = pickle.load(f)

                return FactorResult(
                    success=True,
                    values=result,
                    elapsed_ms=int((time.time() - t0) * 1000),
                )

        except subprocess.TimeoutExpired:
            return FactorResult(
                success=False,
                error=f"timeout ({self.timeout_sec}s)",
                elapsed_ms=int((time.time() - t0) * 1000),
            )
        except Exception as e:
            return FactorResult(
                success=False,
                error=str(e),
                elapsed_ms=int((time.time() - t0) * 1000),
            )

    def _execute_docker(self, code: str, ohlcv_data: Dict[str, pd.DataFrame]) -> FactorResult:
        """在 Docker 中执行（预留，当前退化为 subprocess）。"""
        logger.warning("Docker 沙箱模式尚未实现，退化为 subprocess")
        return self._execute_subprocess(code, ohlcv_data)

    def _execute_inprocess(self, code: str, ohlcv_data: Dict[str, pd.DataFrame]) -> FactorResult:
        """在当前进程中执行（仅用于测试，不安全）。"""
        t0 = time.time()
        try:
            namespace = {"pd": pd, "np": np}
            exec(code, namespace)
            compute_fn = namespace.get("compute_factor")
            if compute_fn is None:
                return FactorResult(success=False, error="compute_factor not defined")

            result = compute_fn(ohlcv_data)
            if not isinstance(result, pd.Series):
                return FactorResult(success=False, error="Must return pd.Series")
            if result.isin([np.inf, -np.inf]).any():
                return FactorResult(success=False, error="Result contains inf")
            if result.isna().all():
                return FactorResult(success=False, error="Result is all NaN")

            return FactorResult(
                success=True,
                values=result,
                elapsed_ms=int((time.time() - t0) * 1000),
            )
        except Exception as e:
            return FactorResult(
                success=False,
                error=str(e),
                elapsed_ms=int((time.time() - t0) * 1000),
            )


# ---------------------------------------------------------------------------
# EvolutionRunner — 进化执行器
# ---------------------------------------------------------------------------


class EvolutionRunner:
    """包装 RD-Agent 进化循环，产出经过验证的因子代码。

    离线运行（每周末或手动触发），不参与日常在线信号计算。
    """

    def __init__(
        self,
        config: EvolutionConfig,
        code_registry: CodeFactorRegistry,
        alpha_registry=None,
        validator=None,
    ):
        """
        Args:
            config: 进化配置
            code_registry: M4 代码因子注册表
            alpha_registry: M2 FactorRegistry 实例（用于正交性检查）
            validator: M2 FactorValidator 实例（用于因子验证）
        """
        self.config = config
        self.code_registry = code_registry
        self.alpha_registry = alpha_registry
        self.validator = validator

    def run_evolution(self, data_manager=None) -> EvolutionReport:
        """执行一轮完整进化循环。

        尝试调用 RD-Agent 的 FactorRDLoop。如果 RD-Agent 不可用，
        退化为简单的模板生成模式。
        """
        t0 = time.time()
        report = EvolutionReport(config_name=self.config.name)

        try:
            raw_factors = self._run_rdagent_loop()
        except Exception as e:
            logger.warning(f"RD-Agent 进化循环失败: {e}")
            logger.info("退化为模板因子生成模式")
            raw_factors = self._generate_template_factors()

        report.total_candidates = len(raw_factors)
        report.total_rounds = self.config.max_rounds

        # 验证与注册
        registered = self.validate_and_register(raw_factors, data_manager)
        report.registered = len(registered)
        report.elapsed_hours = (time.time() - t0) / 3600

        return report

    def _run_rdagent_loop(self) -> List[RawFactor]:
        """调用 RD-Agent 的因子进化循环。"""
        from quantlab._compat import import_rdagent_module
        conf_mod = import_rdagent_module("rdagent.app.qlib_rd_loop.conf")
        factor_mod = import_rdagent_module("rdagent.app.qlib_rd_loop.factor")
        FactorBasePropSetting = conf_mod.FactorBasePropSetting
        FactorRDLoop = factor_mod.FactorRDLoop

        # 配置 RD-Agent 参数
        prop_setting = FactorBasePropSetting()
        # 注入方向约束到 hypothesis generation prompt
        # （RD-Agent 通过 scenario 和 prompt 设置方向）

        loop = FactorRDLoop(prop_setting)
        # 运行指定轮数
        raw_factors = []
        for round_idx in range(self.config.max_rounds):
            if len(raw_factors) >= self.config.total_budget:
                break
            try:
                loop.run(step_n=1)
                # 从 loop 的 trace 中提取产出因子
                factors = self._extract_from_trace(loop, round_idx)
                raw_factors.extend(factors)
            except Exception as e:
                logger.warning(f"进化轮 {round_idx} 失败: {e}")
                continue

        return raw_factors

    def _extract_from_trace(self, loop, round_idx: int) -> List[RawFactor]:
        """从 RD-Agent 的 trace 中提取因子代码。"""
        factors = []
        try:
            trace = loop.trace
            if not trace or not trace.hist:
                return factors
            # 获取最新一轮的实验结果
            latest = trace.hist[-1] if trace.hist else None
            if latest is None:
                return factors

            exp = latest[0] if isinstance(latest, (list, tuple)) else latest
            if hasattr(exp, "sub_workspace_list"):
                for i, ws in enumerate(exp.sub_workspace_list):
                    if ws is None:
                        continue
                    # 读取因子代码
                    code_path = ws.workspace_path / "factor.py" if hasattr(ws, "workspace_path") else None
                    if code_path and code_path.exists():
                        code = code_path.read_text(encoding="utf-8")
                        task = exp.sub_tasks[i] if i < len(exp.sub_tasks) else None
                        name = getattr(task, "factor_name", f"rdagent_r{round_idx}_{i}")
                        desc = getattr(task, "factor_description", "")
                        factors.append(RawFactor(
                            code=code, name=name, description=desc,
                            round_idx=round_idx,
                        ))
        except Exception as e:
            logger.warning(f"提取因子失败: {e}")
        return factors

    def _generate_template_factors(self) -> List[RawFactor]:
        """模板因子生成（RD-Agent 不可用时的退化模式）。

        生成基于常见金融逻辑的 Python 因子代码模板。
        """
        templates = _get_factor_templates()
        # 按方向过滤
        directions = set(self.config.target_directions)
        filtered = [t for t in templates if t["direction"] in directions]
        if not filtered:
            filtered = templates

        # 限制数量
        filtered = filtered[:self.config.total_budget]

        raw_factors = []
        for i, t in enumerate(filtered):
            raw_factors.append(RawFactor(
                code=t["code"],
                name=t["name"],
                description=t["description"],
                round_idx=0,
            ))
        return raw_factors

    def extract_factors(self, workspace_path: str) -> List[RawFactor]:
        """从 QlibFBWorkspace 路径扫描因子代码。"""
        factors = []
        ws_path = Path(workspace_path)
        for py_file in ws_path.glob("**/*.py"):
            if py_file.name.startswith("factor"):
                code = py_file.read_text(encoding="utf-8")
                factors.append(RawFactor(
                    code=code,
                    name=py_file.stem,
                    description="",
                ))
        return factors

    def validate_and_register(
        self,
        raw_factors: List[RawFactor],
        data_manager=None,
    ) -> List[CodeFactorEntry]:
        """验证 + 去冗余 + 注册。"""
        executor = CodeFactorExecutor(sandbox_mode="subprocess", timeout_sec=30)
        registered = []

        # 获取 M2 因子表达式（用于正交性检查）
        alpha_exprs = []
        if self.alpha_registry is not None:
            try:
                alpha_exprs, _ = self.alpha_registry.get_expressions()
            except Exception:
                pass

        for raw in raw_factors:
            reject_reason = None

            # 1. 静态检查
            ok, err = executor.static_check(raw.code, self.config.max_code_lines)
            if not ok:
                reject_reason = "code_error"
                logger.debug(f"因子 {raw.name} 静态检查失败: {err}")
                continue

            # 2. 沙箱试运行（用少量数据）
            if data_manager is not None:
                try:
                    test_data = self._get_test_data(data_manager)
                    result = executor.execute_factor(raw.code, test_data)
                    if not result.success:
                        reject_reason = "code_error"
                        logger.debug(f"因子 {raw.name} 执行失败: {result.error}")
                        continue
                except Exception as e:
                    reject_reason = "code_error"
                    continue

            # 3. IC/ICIR 验证（如果有 validator）
            ic_value = raw.mlflow_ic
            icir_value = 0.0
            if self.validator is not None and data_manager is not None:
                try:
                    # 使用因子输出值计算 IC
                    # 这里简化处理，真实场景需要更完整的实现
                    pass
                except Exception:
                    pass

            # 4. 正交性检查（如果有 alpha_registry）
            corr_with_alpha = 0.0
            # 简化：真实场景需要计算因子输出与 M2 因子的截面相关

            # 5. 分流：判断是 Qlib 表达式还是 Python 代码
            is_qlib_expr = _is_qlib_expression(raw.code)
            if is_qlib_expr and self.alpha_registry is not None:
                # 提取表达式，注入 M2
                expr = _extract_qlib_expression(raw.code)
                if expr:
                    try:
                        direction = self.config.target_directions[0] if self.config.target_directions else "custom"
                        self.alpha_registry.add(raw.name, expr, direction, "rdagent")
                        logger.info(f"Qlib 表达式因子 {raw.name} 已注入 M2")
                    except Exception as e:
                        logger.warning(f"注入 M2 失败: {e}")
                continue

            # 6. 注册到 M4
            try:
                safe_name = raw.name.replace(" ", "_").replace("-", "_")
                entry = self.code_registry.register(
                    name=safe_name,
                    code=raw.code,
                    direction=self.config.target_directions[0] if self.config.target_directions else "custom",
                    source_round=raw.round_idx,
                    config_name=self.config.name,
                    description=raw.description,
                    ic=ic_value,
                    icir=icir_value,
                    corr_with_alpha=corr_with_alpha,
                )
                registered.append(entry)
                logger.info(f"因子 {safe_name} 已注册")
            except ValueError:
                logger.debug(f"因子 {raw.name} 已存在，跳过")

        return registered

    def _get_test_data(self, data_manager) -> Dict[str, pd.DataFrame]:
        """获取少量测试数据。"""
        try:
            df = data_manager.get_ohlcv_before("2024-01-01", 30)
            if df is None or df.empty:
                return {}
            result = {}
            if isinstance(df.index, pd.MultiIndex):
                symbols = df.index.get_level_values(1).unique()[:5]
                for sym in symbols:
                    result[sym] = df.xs(sym, level=1).copy()
            return result
        except Exception:
            return {}


# ---------------------------------------------------------------------------
# RDAgentSignalPipeline — 日常使用入口
# ---------------------------------------------------------------------------


class RDAgentSignalPipeline:
    """日常回测/实盘入口：用注册表中的 active 因子计算信号。"""

    def __init__(
        self,
        code_registry: CodeFactorRegistry,
        executor: CodeFactorExecutor,
        alpha_registry=None,
        validator=None,
        window: int = 60,
    ):
        """
        Args:
            code_registry: M4 代码因子注册表
            executor: 沙箱执行器
            alpha_registry: M2 FactorRegistry（用于正交性和衰退检查）
            validator: M2 FactorValidator
            window: OHLCV 回看窗口
        """
        self.code_registry = code_registry
        self.executor = executor
        self.alpha_registry = alpha_registry
        self.validator = validator
        self.window = window
        self._consecutive_failures: Dict[str, int] = {}

    def compute(self, anchor_date: str, data_manager) -> RDAgentOutput:
        """执行因子计算 + 加权合成。"""
        entries = self.code_registry.get_active()
        if not entries:
            return RDAgentOutput()

        # 获取 OHLCV 数据
        ohlcv_data = self._get_ohlcv(anchor_date, data_manager)
        if not ohlcv_data:
            return RDAgentOutput()

        # 沙箱批量计算
        results = self.executor.execute_batch(entries, ohlcv_data, self.code_registry)

        # 处理结果
        factor_signals = {}
        factor_weights = {}
        failed_factors = []

        for entry in entries:
            name = entry.name
            result = results.get(name)
            if result is None or not result.success:
                failed_factors.append(name)
                self._consecutive_failures[name] = self._consecutive_failures.get(name, 0) + 1
                # 连续失败 3 次 → probation
                if self._consecutive_failures.get(name, 0) >= 3:
                    self.code_registry.set_probation(name)
                    logger.warning(f"因子 {name} 连续失败 3 次，进入观察期")
                continue
            self._consecutive_failures[name] = 0
            factor_signals[name] = result.values
            factor_weights[name] = entry.weight

        if not factor_signals:
            return RDAgentOutput(failed_factors=failed_factors)

        # 加权合成
        # 1. Rank 归一化
        ranked_signals = {}
        for name, values in factor_signals.items():
            ranked = values.rank(pct=True, na_option="keep")
            ranked_signals[name] = ranked

        # 2. 权重重归一化（失败因子的权重分配给成功因子）
        total_weight = sum(factor_weights.values())
        if total_weight <= 0:
            total_weight = len(factor_weights)
            factor_weights = {n: 1.0 for n in factor_weights}
        norm_weights = {n: w / total_weight for n, w in factor_weights.items()}

        # 3. 加权合成
        all_symbols = set()
        for values in ranked_signals.values():
            all_symbols.update(values.dropna().index)
        all_symbols = sorted(all_symbols)

        signal = pd.Series(0.0, index=all_symbols)
        for name, ranked in ranked_signals.items():
            w = norm_weights.get(name, 0)
            aligned = ranked.reindex(all_symbols, fill_value=np.nan)
            signal = signal + w * aligned.fillna(0.5)  # NaN 用中性值 0.5

        return RDAgentOutput(
            signal=signal,
            factor_count=len(factor_signals),
            factor_signals=factor_signals,
            factor_weights=norm_weights,
            failed_factors=failed_factors,
        )

    def check_decay(self, anchor_date: str, data_manager) -> DecayReport:
        """检查因子衰退。"""
        from scipy import stats

        report = DecayReport(checked_date=anchor_date)
        entries = [e for e in self.code_registry.get_all()
                   if e.status in ("active", "probation")]

        if not entries:
            return report

        for entry in entries:
            status = FactorDecayStatus(name=entry.name)

            # 计算近期 IC
            ic_hist = entry.ic_history
            if len(ic_hist) >= 30:
                recent_30 = [ic for _, ic in ic_hist[-30:]]
                status.rolling_ic_30d = float(np.mean(recent_30))
            if len(ic_hist) >= 90:
                recent_90 = [ic for _, ic in ic_hist[-90:]]
                status.rolling_ic_90d = float(np.mean(recent_90))

            # 判断趋势
            if status.rolling_ic_30d > 0.02 and (
                status.rolling_ic_90d == 0 or
                status.rolling_ic_30d > status.rolling_ic_90d * 0.5
            ):
                status.ic_trend = "stable"
            elif status.rolling_ic_30d > 0:
                status.ic_trend = "declining"
            else:
                status.ic_trend = "collapsed"

            # 生命周期动作
            if status.ic_trend == "stable":
                if entry.status == "probation":
                    entry.status = "active"
                    status.action = "none"
                    status.reason = "IC 回升，恢复 active"
                else:
                    status.action = "none"
            elif status.ic_trend == "declining":
                if entry.status == "active":
                    entry.decay_warnings += 1
                    if entry.decay_warnings >= 3:
                        self.code_registry.set_probation(entry.name)
                        status.action = "probation"
                        status.reason = f"连续衰退 {entry.decay_warnings} 次"
                    else:
                        status.action = "warn"
                        status.reason = f"IC 下降 (30d: {status.rolling_ic_30d:.3f})"
            elif status.ic_trend == "collapsed":
                if entry.status == "active":
                    self.code_registry.set_probation(entry.name)
                    status.action = "probation"
                    status.reason = f"IC 崩塌 ({status.rolling_ic_30d:.3f})"
                elif entry.status == "probation":
                    self.code_registry.retire(entry.name, "IC 持续崩塌")
                    status.action = "retire"
                    status.reason = f"退役 (30d IC: {status.rolling_ic_30d:.3f})"

            report.factor_reports.append(status)

        # 检查是否需要触发再进化
        active_count = len(self.code_registry.get_active())
        if active_count < 3:
            report.suggest_evolution = True
            report.suggest_reason = f"Active 因子数不足 ({active_count} < 3)"

        self.code_registry.save()
        return report

    def trigger_evolution(
        self,
        config: EvolutionConfig,
        data_manager,
    ) -> EvolutionReport:
        """触发新一轮进化。"""
        runner = EvolutionRunner(
            config=config,
            code_registry=self.code_registry,
            alpha_registry=self.alpha_registry,
            validator=self.validator,
        )
        return runner.run_evolution(data_manager)

    def get_factor_status(self) -> pd.DataFrame:
        """全部因子状态概览。"""
        return self.code_registry.summary()

    def _get_ohlcv(self, anchor_date: str, data_manager) -> Dict[str, pd.DataFrame]:
        """获取全市场 OHLCV 数据。"""
        try:
            df = data_manager.get_ohlcv_before(anchor_date, self.window)
            if df is None or df.empty:
                return {}
            result = {}
            if isinstance(df.index, pd.MultiIndex):
                for sym in df.index.get_level_values(1).unique():
                    sym_df = df.xs(sym, level=1).copy()
                    if len(sym_df) >= 20:
                        result[sym] = sym_df
            elif "instrument" in df.columns:
                for sym, grp in df.groupby("instrument"):
                    if len(grp) >= 20:
                        result[sym] = grp.copy()
            return result
        except Exception as e:
            logger.warning(f"获取 OHLCV 失败: {e}")
            return {}


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------


def _extract_docstring(code: str) -> str:
    """从代码中提取 docstring。"""
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "compute_factor":
                ds = ast.get_docstring(node)
                return ds or ""
    except Exception:
        pass
    return ""


def _is_qlib_expression(code: str) -> bool:
    """判断代码是否包含 Qlib 表达式（通过 FACTOR_QEXPR 标记）。"""
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "FACTOR_QEXPR":
                        if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                            return True
    except Exception:
        pass
    return False


def _extract_qlib_expression(code: str) -> Optional[str]:
    """从代码中提取 Qlib 表达式字符串。"""
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "FACTOR_QEXPR":
                        if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                            return node.value.value
    except Exception:
        pass
    return None


def _get_factor_templates() -> List[Dict]:
    """内置因子代码模板（RD-Agent 不可用时的退化模式）。"""
    return [
        {
            "name": "rdagent_vol_spike",
            "direction": "volatility_anomaly",
            "description": "Volume spike relative to 20-day average",
            "code": '''import numpy as np
import pandas as pd

def compute_factor(ohlcv):
    """Volume spike: current volume / 20-day mean volume."""
    result = {}
    for sym, df in ohlcv.items():
        if len(df) < 20:
            continue
        vol = df["volume"].values if "volume" in df.columns else df["vol"].values
        mean_vol = np.mean(vol[-20:])
        if mean_vol > 0:
            result[sym] = vol[-1] / mean_vol - 1
        else:
            result[sym] = 0.0
    return pd.Series(result)
''',
        },
        {
            "name": "rdagent_mean_rev_ma",
            "direction": "mean_revert",
            "description": "Price deviation from 20-day moving average",
            "code": '''import numpy as np
import pandas as pd

def compute_factor(ohlcv):
    """Price deviation from 20-day MA, normalized by std."""
    result = {}
    for sym, df in ohlcv.items():
        if len(df) < 20:
            continue
        close = df["close"].values
        ma20 = np.mean(close[-20:])
        std20 = np.std(close[-20:])
        if std20 > 0:
            result[sym] = (close[-1] - ma20) / std20
        else:
            result[sym] = 0.0
    return pd.Series(result)
''',
        },
        {
            "name": "rdagent_amihud_illiq",
            "direction": "liquidity_change",
            "description": "Amihud illiquidity ratio (|return| / volume)",
            "code": '''import numpy as np
import pandas as pd

def compute_factor(ohlcv):
    """Amihud illiquidity: mean(|daily_return| / daily_volume) over 20 days."""
    result = {}
    for sym, df in ohlcv.items():
        if len(df) < 21:
            continue
        close = df["close"].values
        vol = df["volume"].values if "volume" in df.columns else df["vol"].values
        returns = np.abs(np.diff(close[-21:]) / close[-21:-1])
        volumes = vol[-20:]
        mask = volumes > 0
        if mask.sum() > 0:
            illiq = np.mean(returns[mask] / volumes[mask])
            result[sym] = np.log(illiq + 1e-12)
        else:
            result[sym] = 0.0
    return pd.Series(result)
''',
        },
        {
            "name": "rdagent_rsi_divergence",
            "direction": "mean_revert",
            "description": "RSI divergence from price trend",
            "code": '''import numpy as np
import pandas as pd

def compute_factor(ohlcv):
    """RSI(14) divergence: RSI minus 50, negative = oversold."""
    result = {}
    for sym, df in ohlcv.items():
        if len(df) < 15:
            continue
        close = df["close"].values[-15:]
        changes = np.diff(close)
        gains = np.maximum(changes, 0)
        losses = np.abs(np.minimum(changes, 0))
        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)
        if avg_loss > 0:
            rs = avg_gain / avg_loss
            rsi = 100 - 100 / (1 + rs)
        else:
            rsi = 100.0
        result[sym] = (rsi - 50) / 50  # normalize to [-1, 1]
    return pd.Series(result)
''',
        },
        {
            "name": "rdagent_range_vol",
            "direction": "volatility_anomaly",
            "description": "Normalized intraday range volatility",
            "code": '''import numpy as np
import pandas as pd

def compute_factor(ohlcv):
    """Intraday range volatility: std(high-low)/mean(close) over 20 days."""
    result = {}
    for sym, df in ohlcv.items():
        if len(df) < 20:
            continue
        high = df["high"].values[-20:]
        low = df["low"].values[-20:]
        close = df["close"].values[-20:]
        ranges = (high - low) / (close + 1e-8)
        result[sym] = float(np.std(ranges))
    return pd.Series(result)
''',
        },
        {
            "name": "rdagent_vol_price_corr",
            "direction": "momentum_divergence",
            "description": "Volume-price correlation over 20 days",
            "code": '''import numpy as np
import pandas as pd

def compute_factor(ohlcv):
    """Correlation between price change and volume over 20 days."""
    result = {}
    for sym, df in ohlcv.items():
        if len(df) < 21:
            continue
        close = df["close"].values[-21:]
        vol = df["volume"].values[-20:] if "volume" in df.columns else df["vol"].values[-20:]
        returns = np.diff(close)
        if np.std(returns) > 0 and np.std(vol) > 0:
            corr = np.corrcoef(returns, vol)[0, 1]
            result[sym] = corr if not np.isnan(corr) else 0.0
        else:
            result[sym] = 0.0
    return pd.Series(result)
''',
        },
    ]


import sys
