"""RD-Agent 因子注册表管理：查看、注册、退役、导出因子。

Usage:
    # 查看全部因子概览
    python rdagent_registry_cli.py list

    # 只看 active 因子
    python rdagent_registry_cli.py list --status active

    # 查看某个因子详情（含代码）
    python rdagent_registry_cli.py show vol_spike

    # 手动注册因子（从文件）
    python rdagent_registry_cli.py register --name my_factor --code-file ./factors/my_factor.py --direction mean_revert

    # 手动注册因子（从代码字符串）
    python rdagent_registry_cli.py register --name my_factor --code "import pandas as pd..." --direction mean_revert

    # 退役因子
    python rdagent_registry_cli.py retire vol_spike --reason "IC 持续衰退"

    # 恢复因子（probation -> active）
    python rdagent_registry_cli.py activate vol_spike

    # 更新因子权重（基于指定 ICIR）
    python rdagent_registry_cli.py update-weights --icir "vol_spike:1.2,mean_rev:0.8,amihud:0.5"

    # 导出注册表为 JSON
    python rdagent_registry_cli.py export --format json --output ./registry_export.json

    # 查看进化配置
    python rdagent_registry_cli.py configs

    # 查看某个进化配置详情
    python rdagent_registry_cli.py configs --name mean_revert_focus
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

_QUANTLAB_DIR = Path(__file__).resolve().parent.parent

DEFAULT_CONFIGS = str(_QUANTLAB_DIR / "configs" / "rdagent_evolution.yaml")
DEFAULT_REGISTRY = str(_QUANTLAB_DIR.parent / "data" / "rdagent" / "registry.yaml")
DEFAULT_CODE_DIR = str(_QUANTLAB_DIR.parent / "data" / "rdagent" / "factors")


def cmd_list(args):
    """列出注册表中的因子。"""
    from quantlab.signal.signal_rdagent import CodeFactorRegistry

    registry = CodeFactorRegistry(args.code_dir, args.registry)
    entries = registry.get_all()

    if args.status:
        entries = [e for e in entries if e.status == args.status]

    if not entries:
        print("没有因子")
        return

    summary = registry.summary()
    if args.status:
        summary = summary[summary["status"] == args.status]

    pd.set_option("display.max_colwidth", 30)
    pd.set_option("display.width", 120)
    print(f"因子数: {len(entries)}")
    print()
    print(summary.to_string(index=False))


def cmd_show(args):
    """查看因子详情。"""
    from quantlab.signal.signal_rdagent import CodeFactorRegistry
    from dataclasses import asdict

    registry = CodeFactorRegistry(args.code_dir, args.registry)
    try:
        entry = registry.get(args.name)
    except KeyError:
        print(f"因子不存在: {args.name}")
        return

    print(f"因子: {entry.name}")
    print(f"  状态: {entry.status}")
    print(f"  方向: {entry.direction}")
    print(f"  来源配置: {entry.source_config}")
    print(f"  来源轮次: {entry.source_round}")
    print(f"  创建日期: {entry.created_date}")
    print(f"  描述: {entry.description}")
    print(f"  权重: {entry.weight:.4f}")
    print(f"  创建时 IC: {entry.ic_at_creation:.4f}")
    print(f"  创建时 ICIR: {entry.icir_at_creation:.4f}")
    print(f"  与 Alpha 相关: {entry.corr_with_alpha:.4f}")
    print(f"  衰退警告: {entry.decay_warnings}")

    if entry.ic_history:
        print(f"  IC 历史 (最近 10):")
        for date, ic in entry.ic_history[-10:]:
            print(f"    {date}: {ic:.4f}")

    # 显示代码
    if args.show_code:
        try:
            code = registry.load_code(args.name)
            print(f"\n代码 ({entry.code_path}):")
            print("-" * 60)
            print(code)
            print("-" * 60)
        except FileNotFoundError:
            print(f"\n代码文件不存在: {entry.code_path}")


def cmd_register(args):
    """手动注册因子。"""
    from quantlab.signal.signal_rdagent import CodeFactorRegistry, CodeFactorExecutor

    registry = CodeFactorRegistry(args.code_dir, args.registry)

    # 获取代码
    if args.code_file:
        with open(args.code_file, "r", encoding="utf-8") as f:
            code = f.read()
    elif args.code:
        code = args.code
    else:
        print("请指定 --code-file 或 --code")
        return

    # 静态检查
    executor = CodeFactorExecutor()
    ok, err = executor.static_check(code)
    if not ok:
        print(f"代码检查失败: {err}")
        if not args.force:
            return
        print("  --force 模式，继续注册")

    # 注册
    try:
        entry = registry.register(
            name=args.name,
            code=code,
            direction=args.direction,
            description=args.description or "",
        )
        print(f"注册成功: {entry.name}")
        print(f"  代码路径: {entry.code_path}")
        print(f"  方向: {entry.direction}")
    except ValueError as e:
        print(f"注册失败: {e}")


def cmd_retire(args):
    """退役因子。"""
    from quantlab.signal.signal_rdagent import CodeFactorRegistry

    registry = CodeFactorRegistry(args.code_dir, args.registry)
    try:
        registry.retire(args.name, args.reason or "手动退役")
        print(f"已退役: {args.name}")
    except KeyError:
        print(f"因子不存在: {args.name}")


def cmd_activate(args):
    """恢复因子为 active。"""
    from quantlab.signal.signal_rdagent import CodeFactorRegistry

    registry = CodeFactorRegistry(args.code_dir, args.registry)
    try:
        entry = registry.get(args.name)
        entry.status = "active"
        entry.enabled = True
        entry.decay_warnings = 0
        registry.save()
        print(f"已恢复: {args.name} -> active")
    except KeyError:
        print(f"因子不存在: {args.name}")


def cmd_update_weights(args):
    """更新因子权重。"""
    from quantlab.signal.signal_rdagent import CodeFactorRegistry

    registry = CodeFactorRegistry(args.code_dir, args.registry)

    # 解析 ICIR 字符串: "name1:val1,name2:val2"
    icir_map = {}
    for pair in args.icir.split(","):
        pair = pair.strip()
        if ":" not in pair:
            continue
        name, val = pair.split(":", 1)
        icir_map[name.strip()] = float(val.strip())

    registry.update_weights(icir_map)
    print(f"权重已更新 ({len(icir_map)} 个因子)")

    # 显示结果
    for entry in registry.get_active():
        print(f"  {entry.name}: {entry.weight:.4f}")


def cmd_export(args):
    """导出注册表。"""
    from quantlab.signal.signal_rdagent import CodeFactorRegistry
    from dataclasses import asdict

    registry = CodeFactorRegistry(args.code_dir, args.registry)
    entries = registry.get_all()

    if args.format == "json":
        data = []
        for entry in entries:
            d = asdict(entry)
            d["ic_history"] = [[date, ic] for date, ic in d.get("ic_history", [])]
            if args.include_code:
                try:
                    d["code"] = registry.load_code(entry.name)
                except Exception:
                    d["code"] = ""
            data.append(d)
        output = args.output or "registry_export.json"
        with open(output, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"已导出: {output} ({len(data)} 个因子)")

    elif args.format == "csv":
        df = registry.summary()
        output = args.output or "registry_export.csv"
        df.to_csv(output, index=False)
        print(f"已导出: {output} ({len(df)} 个因子)")


def cmd_configs(args):
    """查看进化配置。"""
    from quantlab.signal.signal_rdagent import EvolutionConfig

    configs = EvolutionConfig.load_all(args.configs_file)

    if args.name:
        # 显示指定配置详情
        matched = [c for c in configs if c.name == args.name]
        if not matched:
            print(f"配置不存在: {args.name}")
            print(f"可用配置: {[c.name for c in configs]}")
            return
        c = matched[0]
        from dataclasses import asdict
        for k, v in asdict(c).items():
            print(f"  {k}: {v}")
    else:
        # 列出全部
        print(f"进化配置 ({len(configs)}):")
        for c in configs:
            print(f"\n  [{c.name}]")
            print(f"    方向: {c.target_directions}")
            print(f"    轮数: {c.max_rounds}, 预算: {c.total_budget}")
            print(f"    IC>{c.min_ic}, ICIR>{c.min_icir}")
            print(f"    corr_alpha<{c.max_corr_with_alpha}, corr_pool<{c.max_corr_within_pool}")


def main():
    parser = argparse.ArgumentParser(description="RD-Agent 因子注册表管理")
    parser.add_argument("--registry", default=DEFAULT_REGISTRY, help="注册表路径")
    parser.add_argument("--code-dir", default=DEFAULT_CODE_DIR, help="因子代码目录")
    parser.add_argument("--configs-file", default=DEFAULT_CONFIGS, help="进化配置文件")

    sub = parser.add_subparsers(dest="command")

    # list
    p_list = sub.add_parser("list", help="列出因子")
    p_list.add_argument("--status", choices=["active", "probation", "retired"], help="按状态过滤")

    # show
    p_show = sub.add_parser("show", help="查看因子详情")
    p_show.add_argument("name", help="因子名称")
    p_show.add_argument("--show-code", action="store_true", help="显示代码")

    # register
    p_reg = sub.add_parser("register", help="手动注册因子")
    p_reg.add_argument("--name", required=True, help="因子名称")
    p_reg.add_argument("--code-file", help="因子代码文件路径")
    p_reg.add_argument("--code", help="因子代码字符串")
    p_reg.add_argument("--direction", required=True, help="因子方向")
    p_reg.add_argument("--description", default="", help="描述")
    p_reg.add_argument("--force", action="store_true", help="跳过静态检查")

    # retire
    p_ret = sub.add_parser("retire", help="退役因子")
    p_ret.add_argument("name", help="因子名称")
    p_ret.add_argument("--reason", default="", help="退役原因")

    # activate
    p_act = sub.add_parser("activate", help="恢复因子为 active")
    p_act.add_argument("name", help="因子名称")

    # update-weights
    p_uw = sub.add_parser("update-weights", help="更新因子权重")
    p_uw.add_argument("--icir", required=True, help="ICIR 值 (格式: name1:val1,name2:val2)")

    # export
    p_exp = sub.add_parser("export", help="导出注册表")
    p_exp.add_argument("--format", default="json", choices=["json", "csv"], help="导出格式")
    p_exp.add_argument("--output", help="输出文件路径")
    p_exp.add_argument("--include-code", action="store_true", help="包含因子代码")

    # configs
    p_cfg = sub.add_parser("configs", help="查看进化配置")
    p_cfg.add_argument("--name", help="配置名称（不指定则列出全部）")

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return

    cmds = {
        "list": cmd_list,
        "show": cmd_show,
        "register": cmd_register,
        "retire": cmd_retire,
        "activate": cmd_activate,
        "update-weights": cmd_update_weights,
        "export": cmd_export,
        "configs": cmd_configs,
    }
    cmds[args.command](args)


if __name__ == "__main__":
    main()
