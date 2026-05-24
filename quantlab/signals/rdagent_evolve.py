"""RD-Agent 进化执行：触发因子进化循环，验证并注册新因子。

Usage:
    # 使用 mean_revert_focus 配置运行进化
    python rdagent_evolve.py --config mean_revert_focus

    # 使用指定配置文件
    python rdagent_evolve.py --config broad_explore --configs-file ./configs/rdagent_evolution.yaml

    # 限制轮数和预算
    python rdagent_evolve.py --config volatility_anomaly --max-rounds 5 --budget 10

    # 只使用模板因子（不调用 RD-Agent）
    python rdagent_evolve.py --config mean_revert_focus --template-only

    # 指定注册表和代码目录
    python rdagent_evolve.py --config mean_revert_focus --registry ./data/rdagent/registry.yaml --code-dir ./data/rdagent/factors

    # 指定 Qlib 数据目录（用于沙箱试运行验证）
    python rdagent_evolve.py --config mean_revert_focus --data-dir ~/.qlib/qlib_data/cn_data --market csi300
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd

_QUANTLAB_DIR = Path(__file__).resolve().parent.parent

QLIB_DATA_DIR = str(Path.home() / ".qlib" / "qlib_data" / "cn_data")
DEFAULT_CONFIGS = str(_QUANTLAB_DIR / "configs" / "rdagent_evolution.yaml")
DEFAULT_REGISTRY = str(_QUANTLAB_DIR.parent / "data" / "rdagent" / "registry.yaml")
DEFAULT_CODE_DIR = str(_QUANTLAB_DIR.parent / "data" / "rdagent" / "factors")
DEFAULT_OUTPUT = str(_QUANTLAB_DIR.parent / "outputs" / "rdagent")


def main():
    parser = argparse.ArgumentParser(description="RD-Agent 因子进化执行")

    # 配置
    parser.add_argument("--config", required=True, help="进化配置名称")
    parser.add_argument("--configs-file", default=DEFAULT_CONFIGS, help="进化配置文件路径")

    # 注册表
    parser.add_argument("--registry", default=DEFAULT_REGISTRY, help="因子注册表路径")
    parser.add_argument("--code-dir", default=DEFAULT_CODE_DIR, help="因子代码目录")

    # 进化参数覆盖
    parser.add_argument("--max-rounds", type=int, default=None, help="覆盖最大轮数")
    parser.add_argument("--budget", type=int, default=None, help="覆盖总预算")
    parser.add_argument("--template-only", action="store_true", help="只使用模板因子，不调用 RD-Agent")

    # 数据（用于验证）
    parser.add_argument("--data-dir", default=QLIB_DATA_DIR, help="Qlib 数据目录")
    parser.add_argument("--market", default="csi300", help="股票池")

    # 输出
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="输出目录")

    args = parser.parse_args()

    from quantlab.signals.signal_rdagent import (
        EvolutionConfig,
        CodeFactorRegistry,
        EvolutionRunner,
    )

    # 加载配置
    config = EvolutionConfig.load(args.configs_file, args.config)
    print(f"进化配置: {config.name}")
    print(f"  方向: {config.target_directions}")
    print(f"  最大轮数: {config.max_rounds}")
    print(f"  总预算: {config.total_budget}")
    print(f"  正交约束: corr_alpha<{config.max_corr_with_alpha}, corr_pool<{config.max_corr_within_pool}")
    print()

    # 应用参数覆盖
    if args.max_rounds is not None:
        config.max_rounds = args.max_rounds
        print(f"  [覆盖] max_rounds = {args.max_rounds}")
    if args.budget is not None:
        config.total_budget = args.budget
        print(f"  [覆盖] total_budget = {args.budget}")

    # 初始化注册表
    registry = CodeFactorRegistry(args.code_dir, args.registry)
    print(f"注册表: {args.registry}")
    print(f"  已有因子: {len(registry)}")
    active = registry.get_active()
    print(f"  活跃因子: {len(active)}")
    print()

    # 初始化数据管理器（可选）
    data_manager = None
    try:
        from quantlab.data.data_manager import DataManager
        data_manager = DataManager(provider_uri=args.data_dir, market=args.market)
        data_manager.init_qlib()
        print(f"数据源: {args.data_dir} ({args.market})")
    except Exception as e:
        print(f"数据源不可用: {e}")
        print("跳过沙箱验证，仅做静态检查")
    print()

    # 执行进化
    runner = EvolutionRunner(config=config, code_registry=registry)

    if args.template_only:
        print("模式: 模板因子生成（不调用 RD-Agent）")
        raw_factors = runner._generate_template_factors()
        registered = runner.validate_and_register(raw_factors, data_manager)
        print(f"\n模板因子: {len(raw_factors)}")
        print(f"注册成功: {len(registered)}")
        for entry in registered:
            print(f"  + {entry.name} ({entry.direction})")
    else:
        print("开始进化循环...")
        t0 = time.time()
        report = runner.run_evolution(data_manager)
        elapsed = time.time() - t0

        print(f"\n进化完成 ({elapsed:.1f}s)")
        print(f"  总轮数: {report.total_rounds}")
        print(f"  候选因子: {report.total_candidates}")
        print(f"  注册成功: {report.registered}")
        print(f"  耗时: {report.elapsed_hours:.2f} h")

    # 保存报告
    os.makedirs(args.output, exist_ok=True)
    summary = registry.summary()
    if not summary.empty:
        print(f"\n当前因子池:")
        print(summary.to_string(index=False))
        summary.to_csv(os.path.join(args.output, "factor_summary.csv"), index=False)

    # 保存执行记录
    record = {
        "config": config.name,
        "registry_size": len(registry),
        "active_count": len(registry.get_active()),
        "template_only": args.template_only,
        "timestamp": pd.Timestamp.now().isoformat(),
    }
    record_path = os.path.join(args.output, "evolve_record.json")
    with open(record_path, "w") as f:
        json.dump(record, f, indent=2, ensure_ascii=False)
    print(f"\n执行记录: {record_path}")


if __name__ == "__main__":
    main()
