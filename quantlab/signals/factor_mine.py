"""因子挖掘：参数搜索、组合探索、全自动挖掘。

Usage:
    # 参数搜索：对一个因子模板搜索最优窗口
    python factor_mine.py variants --template "Std($close, {window})/$close" --param window=5,10,20,30,60

    # 组合探索：字段 × 算子 × 窗口 笛卡尔积
    python factor_mine.py combinations --fields "$close,$volume" --operators "Corr,Cov" --windows 10,20,30

    # 模板探索
    python factor_mine.py template --template "({field}-Mean({field},{w}))/(Std({field},{w})+1e-12)" --variables "field=$close,$volume;w=5,10,20"

    # 全自动挖掘
    python factor_mine.py auto --budget 200

    # 全自动挖掘并自动注册通过的因子
    python factor_mine.py auto --budget 200 --register
"""

import argparse
import sys
from pathlib import Path

_QUANTLAB_DIR = Path(__file__).resolve().parent.parent

QLIB_DATA_DIR = str(Path.home() / ".qlib" / "qlib_data" / "cn_data")
DEFAULT_CONFIG = str(_QUANTLAB_DIR / "configs" / "factors.yaml")


def _print_reports(reports, top_n=20):
    """Print top-N factor reports."""
    print(f"\n{'Rank':<5} {'ICIR':<8} {'IC':<8} {'Sharpe':<8} {'Verdict':<8} Expression")
    print("-" * 90)
    for i, r in enumerate(reports[:top_n]):
        expr_short = r.expression[:45] + ("..." if len(r.expression) > 45 else "")
        print(f"{i+1:<5} {r.icir:<8.2f} {r.ic_mean:<8.4f} {r.sharpe_long_short:<8.2f} {r.verdict:<8} {expr_short}")


def cmd_variants(args):
    from quantlab.data.data_manager import DataManager
    from quantlab.signals.signal_alpha import FactorValidator, FactorRegistry, FactorMiner

    dm = DataManager(provider_uri=args.data_dir, market=args.market)
    dm.init_qlib()
    reg = FactorRegistry(args.config)
    validator = FactorValidator(dm)
    miner = FactorMiner(validator, reg)

    # Parse param: "window=5,10,20,30,60"
    param_grid = {}
    for p in args.param:
        key, vals = p.split("=")
        param_grid[key] = [int(v) if v.isdigit() else v for v in vals.split(",")]

    print(f"参数搜索: {args.template}")
    print(f"参数: {param_grid}")
    reports = miner.explore_variants(args.template, param_grid, args.start, args.end)
    _print_reports(reports)


def cmd_combinations(args):
    from quantlab.data.data_manager import DataManager
    from quantlab.signals.signal_alpha import FactorValidator, FactorRegistry, FactorMiner

    dm = DataManager(provider_uri=args.data_dir, market=args.market)
    dm.init_qlib()
    reg = FactorRegistry(args.config)
    validator = FactorValidator(dm)
    miner = FactorMiner(validator, reg)

    fields = [f.strip() for f in args.fields.split(",")]
    operators = [o.strip() for o in args.operators.split(",")]
    windows = [int(w) for w in args.windows.split(",")]

    print(f"组合探索: {len(fields)} 字段 × {len(operators)} 算子 × {len(windows)} 窗口")
    reports = miner.explore_combinations(fields, operators, windows, args.start, args.end)
    _print_reports(reports)


def cmd_template(args):
    from quantlab.data.data_manager import DataManager
    from quantlab.signals.signal_alpha import FactorValidator, FactorRegistry, FactorMiner

    dm = DataManager(provider_uri=args.data_dir, market=args.market)
    dm.init_qlib()
    reg = FactorRegistry(args.config)
    validator = FactorValidator(dm)
    miner = FactorMiner(validator, reg)

    # Parse variables: "field=$close,$volume;w=5,10,20"
    variables = {}
    for pair in args.variables.split(";"):
        key, vals = pair.split("=")
        variables[key.strip()] = [v.strip() for v in vals.split(",")]

    print(f"模板探索: {args.template}")
    print(f"变量: {variables}")
    reports = miner.explore_from_template(args.template, variables, args.start, args.end)
    _print_reports(reports)


def cmd_auto(args):
    from quantlab.data.data_manager import DataManager
    from quantlab.signals.signal_alpha import FactorValidator, FactorRegistry, FactorMiner

    dm = DataManager(provider_uri=args.data_dir, market=args.market)
    dm.init_qlib()
    reg = FactorRegistry(args.config)
    validator = FactorValidator(dm)
    miner = FactorMiner(validator, reg)

    print(f"全自动挖掘: budget={args.budget}")
    reports = miner.auto_mine(budget=args.budget, start=args.start, end=args.end)
    _print_reports(reports, top_n=30)

    accepted = [r for r in reports if r.verdict == "accept"]
    print(f"\n共 {len(accepted)} 个因子通过验证（accept）")

    if args.register and accepted:
        count = miner.accept_and_register(reports)
        print(f"已注册 {count} 个新因子到因子池")


def main():
    parser = argparse.ArgumentParser(description="因子挖掘工具")
    parser.add_argument("--data-dir", default=QLIB_DATA_DIR, help="Qlib 数据目录")
    parser.add_argument("--market", default="csi300", help="股票池")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="因子配置文件")
    parser.add_argument("--start", default="2020-01-01", help="验证起始日期")
    parser.add_argument("--end", default="2024-12-31", help="验证截止日期")

    sub = parser.add_subparsers(dest="command")

    p_var = sub.add_parser("variants", help="参数搜索")
    p_var.add_argument("--template", required=True, help="因子模板，如 Std($close, {window})/$close")
    p_var.add_argument("--param", nargs="+", required=True, help="参数定义，如 window=5,10,20,30,60")
    p_var.set_defaults(func=cmd_variants)

    p_comb = sub.add_parser("combinations", help="组合探索")
    p_comb.add_argument("--fields", required=True, help="字段列表，逗号分隔")
    p_comb.add_argument("--operators", required=True, help="算子列表，逗号分隔")
    p_comb.add_argument("--windows", required=True, help="窗口列表，逗号分隔")
    p_comb.set_defaults(func=cmd_combinations)

    p_tpl = sub.add_parser("template", help="模板探索")
    p_tpl.add_argument("--template", required=True, help="模板字符串")
    p_tpl.add_argument("--variables", required=True, help="变量定义，如 field=$close,$volume;w=5,10,20")
    p_tpl.set_defaults(func=cmd_template)

    p_auto = sub.add_parser("auto", help="全自动挖掘")
    p_auto.add_argument("--budget", type=int, default=200, help="候选因子数量上限")
    p_auto.add_argument("--register", action="store_true", help="自动注册通过验证的因子")
    p_auto.set_defaults(func=cmd_auto)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return
    args.func(args)


if __name__ == "__main__":
    main()
