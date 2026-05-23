"""信号融合监控：贡献度分析、告警检查、IC 历史查看。

Usage:
    # 查看贡献度报告（从融合结果 CSV 回算）
    python ensemble_monitor.py report \
        --signal-csv ./outputs/ensemble/signals.csv \
        --return-csv ./outputs/returns.csv \
        --lookback 60

    # 检查告警
    python ensemble_monitor.py check \
        --signal-csv ./outputs/ensemble/signals.csv \
        --return-csv ./outputs/returns.csv

    # 查看 IC 历史
    python ensemble_monitor.py ic-history \
        --alpha-signal ./outputs/alpha/signals.csv \
        --kronos-signal ./outputs/kronos/signals.csv \
        --rdagent-signal ./outputs/rdagent/signals.csv \
        --return-csv ./outputs/returns.csv

    # 模拟回测监控（从各管线信号 CSV 重建融合过程）
    python ensemble_monitor.py backtest \
        --alpha-signal ./outputs/alpha/signals.csv \
        --kronos-signal ./outputs/kronos/signals.csv \
        --rdagent-signal ./outputs/rdagent/signals.csv \
        --return-csv ./outputs/returns.csv \
        --start 2024-01-01 --end 2024-06-30
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd



def load_matrix_csv(path: str) -> pd.DataFrame:
    """加载 date×symbol 矩阵 CSV。"""
    if not path or not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path, index_col=0)


def cmd_report(args):
    """贡献度报告。"""
    from quantlab.signal.signal_ensemble import (
        SignalEnsemblePipeline, ICWeightEngine, EnsembleMonitor,
    )
    from scipy import stats as sp_stats

    signal_df = load_matrix_csv(args.signal_csv)
    return_df = load_matrix_csv(args.return_csv)

    if signal_df.empty:
        print("信号 CSV 为空")
        return
    if return_df.empty:
        print("收益 CSV 为空")
        return

    # 计算融合信号 IC
    common_dates = sorted(set(signal_df.index) & set(return_df.index))
    lookback = min(args.lookback, len(common_dates))
    recent_dates = common_dates[-lookback:]

    print(f"贡献度报告 ({recent_dates[0]} ~ {recent_dates[-1]}, {len(recent_dates)} 天)")
    print("=" * 60)

    ics = []
    for date in recent_dates:
        sig = signal_df.loc[date].dropna()
        ret = return_df.loc[date].dropna()
        common = sig.index.intersection(ret.index)
        if len(common) >= 10:
            ic = sp_stats.spearmanr(sig.loc[common], ret.loc[common])[0]
            if not np.isnan(ic):
                ics.append(ic)

    if ics:
        print(f"\n融合信号 IC:")
        print(f"  均值: {np.mean(ics):.4f}")
        print(f"  标准差: {np.std(ics):.4f}")
        print(f"  ICIR: {np.mean(ics) / (np.std(ics) + 1e-8):.4f}")
        print(f"  正比例: {sum(1 for ic in ics if ic > 0) / len(ics):.1%}")
    else:
        print("  IC 计算不足（数据太少）")


def cmd_check(args):
    """告警检查。"""
    from quantlab.signal.signal_ensemble import EnsembleMonitor
    from scipy import stats as sp_stats

    signal_df = load_matrix_csv(args.signal_csv)
    return_df = load_matrix_csv(args.return_csv)

    if signal_df.empty or return_df.empty:
        print("数据不足，无法检查")
        return

    monitor = EnsembleMonitor()
    common_dates = sorted(set(signal_df.index) & set(return_df.index))

    for date in common_dates:
        sig = signal_df.loc[date].dropna()
        ret = return_df.loc[date].dropna()
        # 用等权模拟
        monitor.record(date, {"combined": 1.0}, sig, ret)

    alerts = monitor.check_anomaly()
    if alerts:
        print(f"发现 {len(alerts)} 个告警:")
        for a in alerts:
            print(f"  {a}")
    else:
        print("无告警")


def cmd_ic_history(args):
    """IC 历史。"""
    from scipy import stats as sp_stats

    return_df = load_matrix_csv(args.return_csv)
    if return_df.empty:
        print("收益 CSV 为空")
        return

    pipeline_csvs = {}
    if args.alpha_signal:
        pipeline_csvs["alpha"] = load_matrix_csv(args.alpha_signal)
    if args.kronos_signal:
        pipeline_csvs["kronos"] = load_matrix_csv(args.kronos_signal)
    if args.rdagent_signal:
        pipeline_csvs["rdagent"] = load_matrix_csv(args.rdagent_signal)

    if not pipeline_csvs:
        print("没有管线信号 CSV")
        return

    # 逐日计算 IC
    all_dates = sorted(return_df.index)
    rows = []
    for date in all_dates:
        ret = return_df.loc[date].dropna()
        if len(ret) < 10:
            continue
        row = {"date": date}
        for name, df in pipeline_csvs.items():
            if date in df.index:
                sig = df.loc[date].dropna()
                common = sig.index.intersection(ret.index)
                if len(common) >= 10:
                    ic = sp_stats.spearmanr(sig.loc[common], ret.loc[common])[0]
                    row[name] = ic if not np.isnan(ic) else np.nan
                else:
                    row[name] = np.nan
            else:
                row[name] = np.nan
        rows.append(row)

    if not rows:
        print("IC 历史为空")
        return

    ic_df = pd.DataFrame(rows).set_index("date")

    print(f"IC 历史 ({ic_df.index[0]} ~ {ic_df.index[-1]}, {len(ic_df)} 天)")
    print("=" * 60)

    for name in ic_df.columns:
        valid = ic_df[name].dropna()
        if len(valid) > 0:
            print(f"\n  {name}:")
            print(f"    均值 IC: {valid.mean():.4f}")
            print(f"    IC 标准差: {valid.std():.4f}")
            print(f"    ICIR: {valid.mean() / (valid.std() + 1e-8):.4f}")
            print(f"    正比例: {(valid > 0).mean():.1%}")
            # 最近 20 天
            recent = valid.tail(20)
            print(f"    近20日 IC: {recent.mean():.4f}")
        else:
            print(f"\n  {name}: 无数据")

    if args.output:
        ic_df.to_csv(args.output)
        print(f"\nIC 历史已保存: {args.output}")


def cmd_backtest(args):
    """回测监控：重建融合过程并分析。"""
    from quantlab.signal.signal_ensemble import SignalEnsemblePipeline

    return_df = load_matrix_csv(args.return_csv)
    alpha_df = load_matrix_csv(args.alpha_signal) if args.alpha_signal else pd.DataFrame()
    kronos_df = load_matrix_csv(args.kronos_signal) if args.kronos_signal else pd.DataFrame()
    rdagent_df = load_matrix_csv(args.rdagent_signal) if args.rdagent_signal else pd.DataFrame()

    if return_df.empty:
        print("收益 CSV 为空")
        return

    ensemble = SignalEnsemblePipeline(
        ic_lookback=args.ic_lookback,
        uncertainty_penalty=args.uncertainty_penalty,
        min_weight=args.min_weight,
        max_weight=args.max_weight,
    )

    all_dates = sorted(return_df.index.astype(str))
    start = args.start or all_dates[0]
    end = args.end or all_dates[-1]
    dates = [d for d in all_dates if start <= d <= end]

    print(f"回测监控: {len(dates)} 个交易日 ({start} ~ {end})")
    print("=" * 60)

    weight_history = []
    ic_history = []

    for i, date in enumerate(dates):
        # 构建信号
        signals = {}
        if not alpha_df.empty and date in alpha_df.index:
            signals["alpha"] = alpha_df.loc[date].dropna()
        if not kronos_df.empty and date in kronos_df.index:
            signals["kronos"] = kronos_df.loc[date].dropna()
        if not rdagent_df.empty and date in rdagent_df.index:
            signals["rdagent"] = rdagent_df.loc[date].dropna()

        if not signals:
            continue

        # 融合
        output = ensemble.combine(signals, date)

        # 更新历史（用前一日信号 vs 当日收益）
        if i > 0 and date in return_df.index:
            prev_date = dates[i - 1]
            prev_signals = {}
            if not alpha_df.empty and prev_date in alpha_df.index:
                prev_signals["alpha"] = alpha_df.loc[prev_date].dropna()
            if not kronos_df.empty and prev_date in kronos_df.index:
                prev_signals["kronos"] = kronos_df.loc[prev_date].dropna()
            if not rdagent_df.empty and prev_date in rdagent_df.index:
                prev_signals["rdagent"] = rdagent_df.loc[prev_date].dropna()
            if prev_signals:
                ret = return_df.loc[date].dropna() if date in return_df.index else pd.Series()
                ensemble.update_history(prev_date, prev_signals, ret)

        weight_history.append({"date": date, **output.weights, "mode": output.mode})

        # 每 20 天打印一次
        if (i + 1) % 20 == 0 or i == len(dates) - 1:
            print(f"\n  [{date}] mode={output.mode}")
            for name, w in sorted(output.weights.items(), key=lambda x: -x[1]):
                print(f"    {name}: {w:.4f}")

    # 告警检查
    alerts = ensemble.get_monitor().check_anomaly()
    if alerts:
        print(f"\n告警 ({len(alerts)}):")
        for a in alerts:
            print(f"  {a}")

    # 贡献度报告
    report = ensemble.get_monitor().get_contribution_report(lookback=60)
    print(f"\n贡献度报告 ({report.period}):")
    for name, stat in report.pipeline_stats.items():
        print(f"  {name}: avg_w={stat.avg_weight:.3f}, "
              f"trend={stat.weight_trend}")

    # 保存权重历史
    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        pd.DataFrame(weight_history).to_csv(args.output, index=False)
        print(f"\n权重历史已保存: {args.output}")


def main():
    parser = argparse.ArgumentParser(description="信号融合监控")

    sub = parser.add_subparsers(dest="command")

    # report
    p_rep = sub.add_parser("report", help="贡献度报告")
    p_rep.add_argument("--signal-csv", required=True, help="融合信号 CSV")
    p_rep.add_argument("--return-csv", required=True, help="实际收益 CSV")
    p_rep.add_argument("--lookback", type=int, default=60, help="分析窗口")

    # check
    p_chk = sub.add_parser("check", help="告警检查")
    p_chk.add_argument("--signal-csv", required=True, help="融合信号 CSV")
    p_chk.add_argument("--return-csv", required=True, help="实际收益 CSV")

    # ic-history
    p_ic = sub.add_parser("ic-history", help="IC 历史")
    p_ic.add_argument("--alpha-signal", help="M2 Alpha 信号 CSV")
    p_ic.add_argument("--kronos-signal", help="M3 Kronos 信号 CSV")
    p_ic.add_argument("--rdagent-signal", help="M4 RD-Agent 信号 CSV")
    p_ic.add_argument("--return-csv", required=True, help="实际收益 CSV")
    p_ic.add_argument("--output", help="IC 历史输出 CSV")

    # backtest
    p_bt = sub.add_parser("backtest", help="回测监控")
    p_bt.add_argument("--alpha-signal", help="M2 Alpha 信号 CSV")
    p_bt.add_argument("--kronos-signal", help="M3 Kronos 信号 CSV")
    p_bt.add_argument("--rdagent-signal", help="M4 RD-Agent 信号 CSV")
    p_bt.add_argument("--return-csv", required=True, help="实际收益 CSV")
    p_bt.add_argument("--start", help="起始日期")
    p_bt.add_argument("--end", help="截止日期")
    p_bt.add_argument("--ic-lookback", type=int, default=60, help="IC 窗口")
    p_bt.add_argument("--uncertainty-penalty", type=float, default=0.1,
                       help="不确定性惩罚系数")
    p_bt.add_argument("--min-weight", type=float, default=0.1, help="最低权重")
    p_bt.add_argument("--max-weight", type=float, default=0.6, help="最高权重")
    p_bt.add_argument("--output", help="权重历史输出 CSV")

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return

    cmds = {
        "report": cmd_report,
        "check": cmd_check,
        "ic-history": cmd_ic_history,
        "backtest": cmd_backtest,
    }
    cmds[args.command](args)


if __name__ == "__main__":
    main()
