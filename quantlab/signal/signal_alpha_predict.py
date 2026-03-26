"""Alpha158 信号预测：滚动训练 LightGBM 并产出日频信号。

Usage:
    # 单日预测（自动判断是否需要重训）
    python signal_alpha_predict.py --anchor-date 2024-06-28

    # 批量回测：对一段区间逐日产出信号
    python signal_alpha_predict.py --start 2024-01-01 --end 2024-06-30

    # 查看特征重要性
    python signal_alpha_predict.py --anchor-date 2024-06-28 --show-importance
"""

import argparse
import sys
from pathlib import Path

_QUANTLAB_DIR = Path(__file__).resolve().parent.parent

QLIB_DATA_DIR = str(Path.home() / ".qlib" / "qlib_data" / "cn_data")
DEFAULT_CONFIG = str(_QUANTLAB_DIR / "configs" / "factors.yaml")


def main():
    parser = argparse.ArgumentParser(description="Alpha158 信号预测")
    parser.add_argument("--anchor-date", help="单日预测日期")
    parser.add_argument("--start", help="批量回测起始日期")
    parser.add_argument("--end", help="批量回测截止日期")
    parser.add_argument("--show-importance", action="store_true", help="显示特征重要性")
    parser.add_argument("--top-k", type=int, default=20, help="显示 Top-K 股票（默认 20）")
    parser.add_argument("--output", help="信号输出 CSV 路径")
    parser.add_argument("--data-dir", default=QLIB_DATA_DIR, help="Qlib 数据目录")
    parser.add_argument("--market", default="csi300", help="股票池")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="因子配置文件")
    parser.add_argument("--retrain-interval", type=int, default=20, help="重训间隔（交易日）")
    parser.add_argument("--train-years", type=int, default=3, help="训练窗口（年）")
    args = parser.parse_args()

    if not args.anchor_date and not (args.start and args.end):
        parser.print_help()
        return

    from quantlab.data.data_manager import DataManager
    from quantlab.signal.signal_alpha import FactorRegistry, AlphaSignalPipeline

    dm = DataManager(provider_uri=args.data_dir, market=args.market)
    dm.init_qlib()
    reg = FactorRegistry(args.config)

    pipeline = AlphaSignalPipeline(
        registry=reg,
        market=args.market,
        retrain_interval=args.retrain_interval,
        train_years=args.train_years,
    )

    if args.anchor_date:
        # Single day prediction
        signal = pipeline.predict(args.anchor_date, dm)
        print(f"\n信号输出: {args.anchor_date} ({len(signal)} 只股票)")
        print()

        # Show top and bottom
        top = signal.nlargest(args.top_k)
        print(f"Top-{args.top_k} 看多:")
        for sym, score in top.items():
            print(f"  {sym}: {score:.4f}")

        bottom = signal.nsmallest(5)
        print(f"\nBottom-5 看空:")
        for sym, score in bottom.items():
            print(f"  {sym}: {score:.4f}")

        if args.show_importance:
            importance = pipeline.get_feature_importance()
            if importance is not None:
                print(f"\nTop-20 特征重要性:")
                for name, imp in importance.head(20).items():
                    print(f"  {name:<20} {imp:.1f}")

        if args.output:
            signal.to_csv(args.output)
            print(f"\n信号已保存 → {args.output}")

    else:
        # Batch backtest mode
        import pandas as pd
        cal = dm.get_trading_calendar(args.start, args.end)
        print(f"批量预测: {args.start} ~ {args.end} ({len(cal)} 个交易日)")

        all_signals = {}
        for i, date in enumerate(cal):
            date_str = date.strftime("%Y-%m-%d")
            signal = pipeline.predict(date_str, dm)
            all_signals[date_str] = signal
            if (i + 1) % 10 == 0:
                print(f"  进度: {i+1}/{len(cal)}")

        # Merge to DataFrame
        merged = pd.DataFrame(all_signals)
        print(f"\n预测完成: {merged.shape[0]} 只股票 × {merged.shape[1]} 天")

        if args.output:
            merged.to_csv(args.output)
            print(f"信号已保存 → {args.output}")

        # Summary statistics
        print(f"\n信号统计:")
        print(f"  均值: {merged.values[~pd.isna(merged.values)].mean():.4f}")
        print(f"  标准差: {merged.values[~pd.isna(merged.values)].std():.4f}")


if __name__ == "__main__":
    main()
