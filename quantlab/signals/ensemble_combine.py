"""信号融合：合并三条管线信号产出最终选股信号。

Usage:
    # 单日融合（需先生成各管线信号 CSV）
    python ensemble_combine.py --anchor-date 2024-06-28 \
        --alpha-signal ./outputs/alpha/signal.csv \
        --kronos-signal ./outputs/kronos/signal.csv \
        --rdagent-signal ./outputs/rdagent/signal.csv

    # 批量融合（信号 CSV 为 date×symbol 矩阵）
    python ensemble_combine.py --start 2024-01-01 --end 2024-06-30 \
        --alpha-signal ./outputs/alpha/signals.csv \
        --kronos-signal ./outputs/kronos/signals.csv

    # 调整融合参数
    python ensemble_combine.py --anchor-date 2024-06-28 \
        --alpha-signal ./outputs/alpha/signal.csv \
        --ic-lookback 90 --uncertainty-penalty 0.15 \
        --min-weight 0.15 --max-weight 0.5

    # 使用实时管线计算（调用 M2/M3/M4 pipeline）
    python ensemble_combine.py --anchor-date 2024-06-28 --live \
        --data-dir ~/.qlib/qlib_data/cn_data --market csi300

    # 输出融合信号
    python ensemble_combine.py --anchor-date 2024-06-28 \
        --alpha-signal ./outputs/alpha/signal.csv \
        --output ./outputs/ensemble/signal.csv --top-k 30
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

QLIB_DATA_DIR = str(Path.home() / ".qlib" / "qlib_data" / "cn_data")


def load_signal_csv(path: str, anchor_date: str = None) -> pd.Series:
    """从 CSV 加载信号。

    支持两种格式：
    - 单列 CSV（index=symbol, column=signal）
    - 矩阵 CSV（index=date, columns=symbol），取 anchor_date 行
    """
    if not path or not os.path.exists(path):
        return pd.Series(dtype=float)
    df = pd.read_csv(path, index_col=0)
    if anchor_date and anchor_date in df.index:
        return df.loc[anchor_date].dropna()
    elif len(df.columns) == 1:
        return df.iloc[:, 0].dropna()
    elif anchor_date:
        # 尝试最近一个日期
        dates = sorted(df.index)
        valid = [d for d in dates if d <= anchor_date]
        if valid:
            return df.loc[valid[-1]].dropna()
    # 返回最后一行
    if len(df) > 0:
        return df.iloc[-1].dropna()
    return pd.Series(dtype=float)


def combine_single_day(ensemble, signals, anchor_date, top_k):
    """单日融合。"""
    from quantlab.signals.signal_ensemble import EnsembleOutput

    output = ensemble.combine(signals, anchor_date)

    print(f"\n日期: {anchor_date}")
    print(f"  模式: {output.mode}")
    print(f"  不确定性惩罚: {'是' if output.uncertainty_adjusted else '否'}")
    print(f"  参与管线: {list(output.weights.keys())}")
    print(f"  权重:")
    for name, w in sorted(output.weights.items(), key=lambda x: -x[1]):
        print(f"    {name}: {w:.4f}")

    if not output.signal.empty:
        print(f"  覆盖股票: {len(output.signal)}")
        sorted_sig = output.signal.sort_values(ascending=False)
        print(f"\n  Top-{top_k} 股票:")
        for sym, val in sorted_sig.head(top_k).items():
            print(f"    {sym}: {val:.4f}")
        print(f"\n  Bottom-5 股票:")
        for sym, val in sorted_sig.tail(5).items():
            print(f"    {sym}: {val:.4f}")
        print(f"\n  统计: mean={output.signal.mean():.4f}, "
              f"std={output.signal.std():.4f}")
    else:
        print("  无信号输出")

    return output


def main():
    parser = argparse.ArgumentParser(description="信号融合")

    # 日期
    parser.add_argument("--anchor-date", help="单日融合日期")
    parser.add_argument("--start", help="批量融合起始日期")
    parser.add_argument("--end", help="批量融合截止日期")

    # 信号输入（CSV 模式）
    parser.add_argument("--alpha-signal", help="M2 Alpha 信号 CSV")
    parser.add_argument("--kronos-signal", help="M3 Kronos 信号 CSV")
    parser.add_argument("--kronos-uncertainty", help="M3 Kronos 不确定性 CSV")
    parser.add_argument("--rdagent-signal", help="M4 RD-Agent 信号 CSV")

    # 实时模式
    parser.add_argument("--live", action="store_true",
                        help="实时模式：调用 M2/M3/M4 pipeline 计算")
    parser.add_argument("--data-dir", default=QLIB_DATA_DIR, help="Qlib 数据目录")
    parser.add_argument("--market", default="csi300", help="股票池")

    # 融合参数
    parser.add_argument("--ic-lookback", type=int, default=60, help="IC 窗口长度")
    parser.add_argument("--uncertainty-penalty", type=float, default=0.1,
                        help="不确定性惩罚系数")
    parser.add_argument("--min-weight", type=float, default=0.1, help="单管线最低权重")
    parser.add_argument("--max-weight", type=float, default=0.6, help="单管线最高权重")

    # 输出
    parser.add_argument("--top-k", type=int, default=20, help="显示 Top-K 股票")
    parser.add_argument("--output", help="信号输出 CSV 路径")

    args = parser.parse_args()

    if not args.anchor_date and not args.start:
        parser.error("请指定 --anchor-date（单日）或 --start/--end（批量）")

    from quantlab.signals.signal_ensemble import SignalEnsemblePipeline

    ensemble = SignalEnsemblePipeline(
        ic_lookback=args.ic_lookback,
        uncertainty_penalty=args.uncertainty_penalty,
        min_weight=args.min_weight,
        max_weight=args.max_weight,
    )

    all_signals = {}

    if args.anchor_date:
        # 单日模式
        signals = {}
        if args.alpha_signal:
            signals["alpha"] = load_signal_csv(args.alpha_signal, args.anchor_date)
        if args.kronos_signal:
            signals["kronos"] = load_signal_csv(args.kronos_signal, args.anchor_date)
        if args.kronos_uncertainty:
            signals["kronos_uncertainty"] = load_signal_csv(
                args.kronos_uncertainty, args.anchor_date
            )
        if args.rdagent_signal:
            signals["rdagent"] = load_signal_csv(args.rdagent_signal, args.anchor_date)

        if not any(s is not None and len(s) > 0 for s in signals.values()):
            print("没有可用的管线信号")
            return

        output = combine_single_day(ensemble, signals, args.anchor_date, args.top_k)
        if not output.signal.empty:
            all_signals[args.anchor_date] = output.signal

    elif args.start:
        end = args.end or args.start
        # 加载信号矩阵
        alpha_df = pd.read_csv(args.alpha_signal, index_col=0) if args.alpha_signal and os.path.exists(args.alpha_signal) else pd.DataFrame()
        kronos_df = pd.read_csv(args.kronos_signal, index_col=0) if args.kronos_signal and os.path.exists(args.kronos_signal) else pd.DataFrame()
        rdagent_df = pd.read_csv(args.rdagent_signal, index_col=0) if args.rdagent_signal and os.path.exists(args.rdagent_signal) else pd.DataFrame()
        unc_df = pd.read_csv(args.kronos_uncertainty, index_col=0) if args.kronos_uncertainty and os.path.exists(args.kronos_uncertainty) else pd.DataFrame()

        # 收集日期
        all_dates = set()
        for df in [alpha_df, kronos_df, rdagent_df]:
            if not df.empty:
                all_dates.update(df.index.astype(str))
        dates = sorted(d for d in all_dates if args.start <= d <= end)

        print(f"批量融合: {len(dates)} 个交易日 ({args.start} ~ {end})")

        for date in dates:
            signals = {}
            if not alpha_df.empty and date in alpha_df.index:
                signals["alpha"] = alpha_df.loc[date].dropna()
            if not kronos_df.empty and date in kronos_df.index:
                signals["kronos"] = kronos_df.loc[date].dropna()
            if not unc_df.empty and date in unc_df.index:
                signals["kronos_uncertainty"] = unc_df.loc[date].dropna()
            if not rdagent_df.empty and date in rdagent_df.index:
                signals["rdagent"] = rdagent_df.loc[date].dropna()

            if not signals:
                continue

            output = combine_single_day(ensemble, signals, date, args.top_k)
            if not output.signal.empty:
                all_signals[date] = output.signal

    # 保存输出
    if args.output and all_signals:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        df = pd.DataFrame(all_signals).T
        df.index.name = "date"
        df.to_csv(args.output)
        print(f"\n信号已保存: {args.output} ({df.shape[0]} 日 × {df.shape[1]} 只股票)")

    if all_signals:
        n_days = len(all_signals)
        n_stocks = len(set().union(*[s.index for s in all_signals.values()]))
        print(f"\n统计: {n_days} 个交易日, 覆盖 {n_stocks} 只股票")


if __name__ == "__main__":
    main()
