"""验证因子质量：计算 IC、ICIR、多空夏普等指标。

Usage:
    # 验证单个因子
    python factor_validate.py --expr "Mean($close, 20)/$close" --start 2020-01-01 --end 2024-12-31

    # 验证因子并检查与现有因子的相关性
    python factor_validate.py --expr "Corr($close, $volume, 20)" --start 2020-01-01 --end 2024-12-31 --check-corr

    # 批量对比多个因子
    python factor_validate.py --compare "Mean($close,5)/$close" "Mean($close,20)/$close" "Mean($close,60)/$close" --start 2020-01-01 --end 2024-12-31
"""

import argparse
import sys
from pathlib import Path

_QUANTLAB_DIR = Path(__file__).resolve().parent.parent

QLIB_DATA_DIR = str(Path.home() / ".qlib" / "qlib_data" / "cn_data")
DEFAULT_CONFIG = str(_QUANTLAB_DIR / "configs" / "factors.yaml")


def main():
    parser = argparse.ArgumentParser(description="因子质量验证")
    parser.add_argument("--expr", help="单个因子表达式")
    parser.add_argument("--compare", nargs="+", help="多个因子表达式（横向对比）")
    parser.add_argument("--start", default="2020-01-01", help="验证起始日期")
    parser.add_argument("--end", default="2024-12-31", help="验证截止日期")
    parser.add_argument("--check-corr", action="store_true", help="检查与现有因子池的相关性")
    parser.add_argument("--data-dir", default=QLIB_DATA_DIR, help="Qlib 数据目录")
    parser.add_argument("--market", default="csi300", help="股票池")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="因子配置文件")
    args = parser.parse_args()

    if not args.expr and not args.compare:
        parser.print_help()
        return

    from quantlab.data.data_manager import DataManager
    from quantlab.signals.signal_alpha import FactorValidator, FactorRegistry

    dm = DataManager(provider_uri=args.data_dir, market=args.market)
    dm.init_qlib()
    validator = FactorValidator(dm)

    if args.compare:
        print(f"对比 {len(args.compare)} 个因子 ({args.start} ~ {args.end})")
        print()
        df = validator.compare(args.compare, args.start, args.end)
        print(df.to_string(index=False))
        return

    # Single factor validation
    expr = args.expr
    print(f"验证因子: {expr}")
    print(f"区间: {args.start} ~ {args.end}")
    print(f"股票池: {args.market}")
    print()

    existing_exprs = None
    if args.check_corr:
        reg = FactorRegistry(args.config)
        existing_exprs = reg.get_expressions()[0]
        print(f"对比现有因子池: {len(existing_exprs)} 个因子")

    report = validator.validate_single(expr, args.start, args.end, existing_exprs)

    print("=" * 50)
    print(f"  判定结果: {report.verdict.upper()}")
    print("=" * 50)
    print(f"  IC 均值:        {report.ic_mean:.4f}")
    print(f"  ICIR:           {report.icir:.2f}")
    print(f"  IC 正比率:      {report.ic_positive_ratio:.1%}")
    print(f"  多空夏普:       {report.sharpe_long_short:.2f}")
    print(f"  自相关:         {report.auto_corr:.3f}")
    print(f"  换手率:         {report.turnover:.3f}")
    if report.max_corr_with_existing > 0:
        print(f"  最大相关:       {report.max_corr_with_existing:.3f}")
    if report.ic_decay:
        print(f"  IC 衰减:        {report.ic_decay}")
    print()

    # Verdict explanation
    if report.verdict == "accept":
        print("  → 因子质量良好，建议纳入因子池")
    elif report.verdict == "reject":
        print("  → 因子质量不佳，建议放弃")
    else:
        print("  → 因子质量一般，建议人工判断")


if __name__ == "__main__":
    main()
