"""管理因子池：查看、添加、删除、启用、禁用因子。

Usage:
    python factor_registry_cli.py list                              # 查看全部因子
    python factor_registry_cli.py list --source manual              # 按来源筛选
    python factor_registry_cli.py list --category momentum          # 按分类筛选
    python factor_registry_cli.py summary                           # 因子概览表
    python factor_registry_cli.py add --name MY_FACTOR --expr "Corr($close, $volume, 20)" --category liquidity
    python factor_registry_cli.py remove --name MY_FACTOR
    python factor_registry_cli.py enable --name STD20
    python factor_registry_cli.py disable --name STD20
    python factor_registry_cli.py init                              # 重新生成 Alpha158 基础因子
"""

import argparse
import sys
from pathlib import Path

_QUANTLAB_DIR = Path(__file__).resolve().parent.parent

DEFAULT_CONFIG = str(_QUANTLAB_DIR / "configs" / "factors.yaml")


def cmd_list(args):
    from quantlab.signal.signal_alpha import FactorRegistry
    reg = FactorRegistry(args.config)
    factors = reg.list(category=args.category, source=args.source)
    print(f"共 {len(factors)} 个因子 (总计 {len(reg)} 个)")
    print()
    print(f"{'Name':<20} {'Category':<14} {'Source':<10} {'Enabled':<8} Expression")
    print("-" * 100)
    for fd in factors:
        expr_short = fd.expression[:50] + ("..." if len(fd.expression) > 50 else "")
        print(f"{fd.name:<20} {fd.category:<14} {fd.source:<10} {'Y' if fd.enabled else 'N':<8} {expr_short}")


def cmd_summary(args):
    from quantlab.signal.signal_alpha import FactorRegistry
    reg = FactorRegistry(args.config)
    df = reg.summary()
    print(df.to_string(index=False))
    print()
    enabled = df["enabled"].sum()
    print(f"总计: {len(df)} 个因子, {enabled} 个已启用")
    by_source = df.groupby("source").size()
    print(f"来源分布: {dict(by_source)}")
    by_cat = df.groupby("category").size()
    print(f"分类分布: {dict(by_cat)}")


def cmd_add(args):
    from quantlab.signal.signal_alpha import FactorRegistry
    reg = FactorRegistry(args.config)
    fd = reg.add(args.name, args.expr, args.category, args.source)
    reg.save()
    print(f"已添加因子: {fd.name} = {fd.expression}")
    print(f"分类: {fd.category}, 来源: {fd.source}")


def cmd_remove(args):
    from quantlab.signal.signal_alpha import FactorRegistry
    reg = FactorRegistry(args.config)
    reg.remove(args.name)
    reg.save()
    print(f"已删除因子: {args.name}")


def cmd_enable(args):
    from quantlab.signal.signal_alpha import FactorRegistry
    reg = FactorRegistry(args.config)
    reg.enable(args.name)
    reg.save()
    print(f"已启用因子: {args.name}")


def cmd_disable(args):
    from quantlab.signal.signal_alpha import FactorRegistry
    reg = FactorRegistry(args.config)
    reg.disable(args.name)
    reg.save()
    print(f"已禁用因子: {args.name}")


def cmd_init(args):
    from quantlab.signal.signal_alpha import FactorRegistry
    config_path = Path(args.config)
    if config_path.exists():
        confirm = input(f"{config_path} 已存在，是否覆盖？[y/N] ").strip().lower()
        if confirm != "y":
            print("已取消")
            return
        config_path.unlink()
    reg = FactorRegistry(args.config)
    print(f"已生成 {len(reg)} 个 Alpha158 基础因子 → {args.config}")


def main():
    parser = argparse.ArgumentParser(description="因子注册表管理")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="因子配置文件路径")
    sub = parser.add_subparsers(dest="command")

    p_list = sub.add_parser("list", help="查看因子列表")
    p_list.add_argument("--category", help="按分类筛选")
    p_list.add_argument("--source", help="按来源筛选")
    p_list.set_defaults(func=cmd_list)

    p_sum = sub.add_parser("summary", help="因子概览表")
    p_sum.set_defaults(func=cmd_summary)

    p_add = sub.add_parser("add", help="添加新因子")
    p_add.add_argument("--name", required=True, help="因子名称")
    p_add.add_argument("--expr", required=True, help="Qlib 表达式")
    p_add.add_argument("--category", default="custom", help="分类（默认 custom）")
    p_add.add_argument("--source", default="manual", help="来源（默认 manual）")
    p_add.set_defaults(func=cmd_add)

    p_rm = sub.add_parser("remove", help="删除因子")
    p_rm.add_argument("--name", required=True, help="因子名称")
    p_rm.set_defaults(func=cmd_remove)

    p_en = sub.add_parser("enable", help="启用因子")
    p_en.add_argument("--name", required=True, help="因子名称")
    p_en.set_defaults(func=cmd_enable)

    p_dis = sub.add_parser("disable", help="禁用因子")
    p_dis.add_argument("--name", required=True, help="因子名称")
    p_dis.set_defaults(func=cmd_disable)

    p_init = sub.add_parser("init", help="重新生成 Alpha158 基础因子")
    p_init.set_defaults(func=cmd_init)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return
    args.func(args)


if __name__ == "__main__":
    main()
