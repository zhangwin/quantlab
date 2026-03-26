"""RD-Agent 数据准备：初始化因子目录、导入外部因子、检查状态。

Usage:
    # 初始化因子目录和注册表
    python rdagent_data.py init

    # 指定自定义目录
    python rdagent_data.py init --code-dir ./data/rdagent/factors --registry ./data/rdagent/registry.yaml

    # 从目录批量导入因子代码
    python rdagent_data.py import-factors --source-dir ./external_factors --direction mean_revert

    # 从 RD-Agent workspace 导入因子
    python rdagent_data.py import-workspace --workspace-path ./rdagent_output/workspace_001

    # 查看当前状态
    python rdagent_data.py status

    # 验证所有因子代码
    python rdagent_data.py validate

    # 清理退役因子代码文件
    python rdagent_data.py cleanup --remove-retired
"""

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

_QUANTLAB_DIR = Path(__file__).resolve().parent.parent

DEFAULT_REGISTRY = str(_QUANTLAB_DIR.parent / "data" / "rdagent" / "registry.yaml")
DEFAULT_CODE_DIR = str(_QUANTLAB_DIR.parent / "data" / "rdagent" / "factors")
DEFAULT_CONFIGS = str(_QUANTLAB_DIR / "configs" / "rdagent_evolution.yaml")


def cmd_init(args):
    """初始化因子目录和注册表。"""
    code_dir = Path(args.code_dir)
    registry_path = Path(args.registry)

    # 创建目录
    code_dir.mkdir(parents=True, exist_ok=True)
    registry_path.parent.mkdir(parents=True, exist_ok=True)

    # 创建空注册表
    if not registry_path.exists():
        import yaml
        with open(registry_path, "w", encoding="utf-8") as f:
            yaml.dump({"factors": []}, f, allow_unicode=True)
        print(f"注册表已创建: {registry_path}")
    else:
        print(f"注册表已存在: {registry_path}")

    print(f"因子代码目录: {code_dir}")

    # 检查进化配置文件
    configs_path = Path(args.configs_file)
    if configs_path.exists():
        from quantlab.signal.signal_rdagent import EvolutionConfig
        configs = EvolutionConfig.load_all(str(configs_path))
        print(f"进化配置文件: {configs_path} ({len(configs)} 个配置)")
    else:
        print(f"进化配置文件不存在: {configs_path}")
        print("  可从 quantlab/configs/rdagent_evolution.yaml 复制")

    print("\n初始化完成。")
    print("下一步:")
    print("  1. 运行 rdagent_evolve.py --config <name> 生成因子")
    print("  2. 或 rdagent_registry_cli.py register 手动注册因子")
    print("  3. 然后 rdagent_compute.py --anchor-date <date> 计算信号")


def cmd_import_factors(args):
    """从目录批量导入因子代码。"""
    from quantlab.signal.signal_rdagent import CodeFactorRegistry, CodeFactorExecutor

    source_dir = Path(args.source_dir)
    if not source_dir.is_dir():
        print(f"源目录不存在: {source_dir}")
        return

    registry = CodeFactorRegistry(args.code_dir, args.registry)
    executor = CodeFactorExecutor()

    py_files = sorted(source_dir.glob("*.py"))
    print(f"扫描到 {len(py_files)} 个 Python 文件")

    imported = 0
    skipped = 0
    failed = 0

    for py_file in py_files:
        code = py_file.read_text(encoding="utf-8")
        name = py_file.stem

        # 静态检查
        ok, err = executor.static_check(code)
        if not ok:
            print(f"  ✗ {name}: {err}")
            failed += 1
            continue

        # 注册
        try:
            registry.register(
                name=name,
                code=code,
                direction=args.direction,
                description=args.description or f"Imported from {py_file.name}",
            )
            print(f"  ✓ {name}")
            imported += 1
        except ValueError:
            print(f"  - {name}: 已存在")
            skipped += 1

    print(f"\n导入完成: {imported} 成功, {skipped} 跳过, {failed} 失败")


def cmd_import_workspace(args):
    """从 RD-Agent workspace 导入因子。"""
    from quantlab.signal.signal_rdagent import (
        CodeFactorRegistry,
        EvolutionRunner,
        EvolutionConfig,
    )

    ws_path = Path(args.workspace_path)
    if not ws_path.is_dir():
        print(f"Workspace 不存在: {ws_path}")
        return

    registry = CodeFactorRegistry(args.code_dir, args.registry)

    # 使用默认配置
    config = EvolutionConfig(name="import")
    if args.direction:
        config.target_directions = [args.direction]

    runner = EvolutionRunner(config=config, code_registry=registry)
    raw_factors = runner.extract_factors(str(ws_path))

    print(f"从 workspace 扫描到 {len(raw_factors)} 个因子")
    registered = runner.validate_and_register(raw_factors)

    print(f"注册成功: {len(registered)}")
    for entry in registered:
        print(f"  + {entry.name}")


def cmd_status(args):
    """查看因子系统状态。"""
    from quantlab.signal.signal_rdagent import CodeFactorRegistry

    code_dir = Path(args.code_dir)
    registry_path = Path(args.registry)

    # 检查目录
    if not code_dir.is_dir():
        print(f"因子目录不存在: {code_dir}")
        print("请先运行: python rdagent_data.py init")
        return

    # 检查注册表
    if not registry_path.exists():
        print(f"注册表不存在: {registry_path}")
        print("请先运行: python rdagent_data.py init")
        return

    registry = CodeFactorRegistry(str(code_dir), str(registry_path))
    all_factors = registry.get_all()
    active = registry.get_active()

    print(f"因子目录: {code_dir}")
    print(f"注册表: {registry_path}")
    print(f"\n因子总数: {len(all_factors)}")

    # 按状态统计
    status_counts = {}
    for e in all_factors:
        status_counts[e.status] = status_counts.get(e.status, 0) + 1
    for status, count in sorted(status_counts.items()):
        print(f"  {status}: {count}")

    # 按方向统计
    if all_factors:
        dir_counts = {}
        for e in all_factors:
            dir_counts[e.direction] = dir_counts.get(e.direction, 0) + 1
        print(f"\n方向分布:")
        for d, count in sorted(dir_counts.items()):
            print(f"  {d}: {count}")

    # 权重分布
    if active:
        weights = [e.weight for e in active]
        print(f"\n活跃因子权重:")
        print(f"  最大: {max(weights):.4f}")
        print(f"  最小: {min(weights):.4f}")
        print(f"  总和: {sum(weights):.4f}")

    # 代码文件状态
    py_files = list(code_dir.glob("*.py"))
    registered_files = {e.code_path for e in all_factors}
    orphan_files = [f for f in py_files if f.name not in registered_files]
    if orphan_files:
        print(f"\n孤立代码文件 (未注册): {len(orphan_files)}")
        for f in orphan_files[:5]:
            print(f"  {f.name}")

    # 检查进化配置
    configs_path = Path(args.configs_file)
    if configs_path.exists():
        from quantlab.signal.signal_rdagent import EvolutionConfig
        configs = EvolutionConfig.load_all(str(configs_path))
        print(f"\n进化配置: {len(configs)} 个")
        for c in configs:
            print(f"  {c.name}: {c.target_directions}")


def cmd_validate(args):
    """验证所有因子代码。"""
    from quantlab.signal.signal_rdagent import CodeFactorRegistry, CodeFactorExecutor

    registry = CodeFactorRegistry(args.code_dir, args.registry)
    executor = CodeFactorExecutor()
    entries = registry.get_all()

    if not entries:
        print("没有因子")
        return

    print(f"验证 {len(entries)} 个因子...")
    passed = 0
    failed = 0

    for entry in entries:
        try:
            code = registry.load_code(entry.name)
            ok, err = executor.static_check(code)
            if ok:
                print(f"  ✓ {entry.name}")
                passed += 1
            else:
                print(f"  ✗ {entry.name}: {err}")
                failed += 1
        except FileNotFoundError:
            print(f"  ✗ {entry.name}: 代码文件不存在")
            failed += 1

    print(f"\n结果: {passed} 通过, {failed} 失败")


def cmd_cleanup(args):
    """清理退役因子。"""
    from quantlab.signal.signal_rdagent import CodeFactorRegistry

    registry = CodeFactorRegistry(args.code_dir, args.registry)
    retired = [e for e in registry.get_all() if e.status == "retired"]

    if not retired:
        print("没有退役因子")
        return

    print(f"退役因子: {len(retired)}")
    for e in retired:
        print(f"  {e.name} (创建: {e.created_date})")

    if args.remove_retired:
        code_dir = Path(args.code_dir)
        removed = 0
        for e in retired:
            code_path = code_dir / e.code_path
            if code_path.exists():
                code_path.unlink()
                print(f"  已删除: {code_path}")
                removed += 1
        print(f"\n已删除 {removed} 个代码文件")
        print("注意: 注册表记录仍然保留（便于审计追溯）")
    else:
        print("\n使用 --remove-retired 删除代码文件")


def main():
    parser = argparse.ArgumentParser(description="RD-Agent 数据准备")
    parser.add_argument("--registry", default=DEFAULT_REGISTRY, help="注册表路径")
    parser.add_argument("--code-dir", default=DEFAULT_CODE_DIR, help="因子代码目录")
    parser.add_argument("--configs-file", default=DEFAULT_CONFIGS, help="进化配置文件")

    sub = parser.add_subparsers(dest="command")

    # init
    sub.add_parser("init", help="初始化因子目录和注册表")

    # import-factors
    p_imp = sub.add_parser("import-factors", help="从目录导入因子")
    p_imp.add_argument("--source-dir", required=True, help="源目录")
    p_imp.add_argument("--direction", default="custom", help="因子方向")
    p_imp.add_argument("--description", default="", help="描述")

    # import-workspace
    p_ws = sub.add_parser("import-workspace", help="从 RD-Agent workspace 导入")
    p_ws.add_argument("--workspace-path", required=True, help="workspace 路径")
    p_ws.add_argument("--direction", default=None, help="因子方向")

    # status
    sub.add_parser("status", help="查看状态")

    # validate
    sub.add_parser("validate", help="验证因子代码")

    # cleanup
    p_clean = sub.add_parser("cleanup", help="清理退役因子")
    p_clean.add_argument("--remove-retired", action="store_true", help="删除退役因子代码文件")

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return

    cmds = {
        "init": cmd_init,
        "import-factors": cmd_import_factors,
        "import-workspace": cmd_import_workspace,
        "status": cmd_status,
        "validate": cmd_validate,
        "cleanup": cmd_cleanup,
    }
    cmds[args.command](args)


if __name__ == "__main__":
    main()
