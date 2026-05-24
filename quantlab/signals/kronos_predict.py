"""Kronos 信号预测：微调 + 推理产出日频信号。

Usage:
    # 单日预测
    python kronos_predict.py --anchor-date 2024-06-28

    # 批量回测
    python kronos_predict.py --start 2024-01-01 --end 2024-06-30

    # 使用指定方案
    python kronos_predict.py --anchor-date 2024-06-28 --recipe aggressive

    # 零样本推理（不微调）
    python kronos_predict.py --anchor-date 2024-06-28 --recipe zero_shot
"""

import argparse
import sys
from pathlib import Path

_QUANTLAB_DIR = Path(__file__).resolve().parent.parent

QLIB_DATA_DIR = str(Path.home() / ".qlib" / "qlib_data" / "cn_data")
DEFAULT_RECIPES = str(_QUANTLAB_DIR / "configs" / "kronos_recipes.yaml")


def main():
    parser = argparse.ArgumentParser(description="Kronos 信号预测")
    parser.add_argument("--anchor-date", help="单日预测日期")
    parser.add_argument("--start", help="批量回测起始日期")
    parser.add_argument("--end", help="批量回测截止日期")
    parser.add_argument("--recipe", default="conservative", help="微调方案名称（默认 conservative）")
    parser.add_argument("--recipes-file", default=DEFAULT_RECIPES, help="方案配置文件路径")
    parser.add_argument("--top-k", type=int, default=20, help="显示 Top-K 股票")
    parser.add_argument("--output", help="信号输出 CSV 路径")
    parser.add_argument("--data-dir", default=QLIB_DATA_DIR, help="Qlib 数据目录")
    parser.add_argument("--market", default="csi300", help="股票池")
    parser.add_argument("--device", default="cuda", help="计算设备")
    parser.add_argument("--max-context", type=int, default=512, help="最大上下文长度")
    args = parser.parse_args()

    if not args.anchor_date and not (args.start and args.end):
        parser.print_help()
        return

    from quantlab.data.data_manager import DataManager
    from quantlab.signals.signal_kronos import FinetuneRecipe, KronosSignalPipeline

    # 加载方案
    recipe = FinetuneRecipe.load(args.recipes_file, args.recipe)
    print(f"使用方案: {recipe.name} ({recipe.description})")

    # 初始化
    dm = DataManager(provider_uri=args.data_dir, market=args.market)
    dm.init_qlib()

    pipeline = KronosSignalPipeline(
        recipe=recipe,
        device=args.device,
        max_context=args.max_context,
    )

    if args.anchor_date:
        # 获取全市场数据
        symbol_data = _get_symbol_data(dm, args.anchor_date, recipe.data_lookback)
        print(f"数据: {len(symbol_data)} 只股票")

        output = pipeline.daily_run(symbol_data, args.anchor_date, dm)

        print(f"\n信号输出: {args.anchor_date} ({len(output.return_1d)} 只股票)")

        # Top-K
        if not output.return_1d.empty:
            top = output.return_1d.nlargest(args.top_k)
            print(f"\nTop-{args.top_k} 看多 (T+1 预测收益):")
            for sym, score in top.items():
                unc = output.uncertainty.get(sym, 0)
                print(f"  {sym}: {score:+.4f}  (不确定性: {unc:.4f})")

            bottom = output.return_1d.nsmallest(5)
            print(f"\nBottom-5 看空:")
            for sym, score in bottom.items():
                print(f"  {sym}: {score:+.4f}")

            # 统计
            print(f"\n信号统计:")
            print(f"  均值: {output.return_1d.mean():.4f}")
            print(f"  标准差: {output.return_1d.std():.4f}")
            print(f"  平均不确定性: {output.uncertainty.mean():.4f}")

        if args.output:
            import pandas as pd
            result = pd.DataFrame({
                "return_1d": output.return_1d,
                "return_5d": output.return_5d,
                "uncertainty": output.uncertainty,
            })
            result.to_csv(args.output)
            print(f"\n信号已保存 -> {args.output}")

    else:
        # 批量回测
        import pandas as pd
        cal = dm.get_trading_calendar(args.start, args.end)
        print(f"批量预测: {args.start} ~ {args.end} ({len(cal)} 个交易日)")

        all_signals = {}
        for i, date in enumerate(cal):
            date_str = date.strftime("%Y-%m-%d")
            symbol_data = _get_symbol_data(dm, date_str, recipe.data_lookback)
            output = pipeline.daily_run(symbol_data, date_str, dm)
            all_signals[date_str] = output.return_1d
            if (i + 1) % 5 == 0:
                print(f"  进度: {i+1}/{len(cal)}")

        merged = pd.DataFrame(all_signals)
        print(f"\n预测完成: {merged.shape[0]} 只股票 x {merged.shape[1]} 天")

        if args.output:
            merged.to_csv(args.output)
            print(f"信号已保存 -> {args.output}")


def _get_symbol_data(dm, date_str, lookback):
    """从 DataManager 获取分股票的 OHLCV 数据。"""
    df = dm.get_ohlcv_before(date_str, lookback)
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


import pandas as pd

if __name__ == "__main__":
    main()
