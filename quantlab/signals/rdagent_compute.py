"""RD-Agent 日常信号计算：用注册表中的 active 因子计算加权合成信号。

Usage:
    # 单日计算
    python rdagent_compute.py --anchor-date 2024-06-28

    # 批量回测
    python rdagent_compute.py --start 2024-01-01 --end 2024-06-30

    # 指定注册表
    python rdagent_compute.py --anchor-date 2024-06-28 --registry ./data/rdagent/registry.yaml

    # 显示 Top-K 股票
    python rdagent_compute.py --anchor-date 2024-06-28 --top-k 30

    # 输出信号到 CSV
    python rdagent_compute.py --anchor-date 2024-06-28 --output ./outputs/rdagent/signals.csv

    # 同时执行衰退检查
    python rdagent_compute.py --anchor-date 2024-06-28 --check-decay
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_QUANTLAB_DIR = Path(__file__).resolve().parent.parent

QLIB_DATA_DIR = str(Path.home() / ".qlib" / "qlib_data" / "cn_data")
DEFAULT_REGISTRY = str(_QUANTLAB_DIR.parent / "data" / "rdagent" / "registry.yaml")
DEFAULT_CODE_DIR = str(_QUANTLAB_DIR.parent / "data" / "rdagent" / "factors")
DEFAULT_OUTPUT = str(_QUANTLAB_DIR.parent / "outputs" / "rdagent")


def compute_single_day(pipeline, anchor_date, data_manager, top_k, check_decay):
    """计算单日信号。"""
    print(f"\n日期: {anchor_date}")
    print("-" * 50)

    output = pipeline.compute(anchor_date, data_manager)

    if output.signal.empty:
        print("  无信号输出（可能没有活跃因子或数据不可用）")
        return output

    print(f"  因子数: {output.factor_count}")
    print(f"  失败因子: {output.failed_factors or '无'}")
    print(f"  覆盖股票: {len(output.signal)}")

    # 权重信息
    if output.factor_weights:
        print(f"  因子权重:")
        for name, w in sorted(output.factor_weights.items(), key=lambda x: -x[1]):
            print(f"    {name}: {w:.4f}")

    # Top-K
    sorted_signal = output.signal.sort_values(ascending=False)
    print(f"\n  Top-{top_k} 股票:")
    for sym, val in sorted_signal.head(top_k).items():
        print(f"    {sym}: {val:.4f}")
    print(f"\n  Bottom-{min(5, len(sorted_signal))} 股票:")
    for sym, val in sorted_signal.tail(min(5, len(sorted_signal))).items():
        print(f"    {sym}: {val:.4f}")

    # 衰退检查
    if check_decay:
        print(f"\n  衰退检查:")
        report = pipeline.check_decay(anchor_date, data_manager)
        for fr in report.factor_reports:
            status_mark = {"stable": "✓", "declining": "⚠", "collapsed": "✗"}.get(fr.ic_trend, "?")
            print(f"    [{status_mark}] {fr.name}: {fr.ic_trend} (30d IC: {fr.rolling_ic_30d:.3f})")
            if fr.action != "none":
                print(f"        动作: {fr.action} - {fr.reason}")
        if report.suggest_evolution:
            print(f"    建议触发进化: {report.suggest_reason}")

    return output


def main():
    parser = argparse.ArgumentParser(description="RD-Agent 日常信号计算")

    # 日期
    parser.add_argument("--anchor-date", help="单日计算日期")
    parser.add_argument("--start", help="批量回测起始日期")
    parser.add_argument("--end", help="批量回测截止日期")

    # 注册表
    parser.add_argument("--registry", default=DEFAULT_REGISTRY, help="因子注册表路径")
    parser.add_argument("--code-dir", default=DEFAULT_CODE_DIR, help="因子代码目录")

    # 数据
    parser.add_argument("--data-dir", default=QLIB_DATA_DIR, help="Qlib 数据目录")
    parser.add_argument("--market", default="csi300", help="股票池")
    parser.add_argument("--window", type=int, default=60, help="OHLCV 回看窗口天数")

    # 计算参数
    parser.add_argument("--sandbox", default="subprocess",
                        choices=["subprocess", "inprocess"],
                        help="沙箱模式")
    parser.add_argument("--timeout", type=int, default=30, help="因子执行超时（秒）")

    # 输出
    parser.add_argument("--top-k", type=int, default=20, help="显示 Top-K 股票")
    parser.add_argument("--output", default=None, help="信号输出 CSV 路径")
    parser.add_argument("--check-decay", action="store_true", help="同时执行衰退检查")

    args = parser.parse_args()

    if not args.anchor_date and not args.start:
        parser.error("请指定 --anchor-date（单日）或 --start/--end（批量回测）")

    from quantlab.signals.signal_rdagent import (
        CodeFactorRegistry,
        CodeFactorExecutor,
        RDAgentSignalPipeline,
    )

    # 初始化
    registry = CodeFactorRegistry(args.code_dir, args.registry)
    executor = CodeFactorExecutor(sandbox_mode=args.sandbox, timeout_sec=args.timeout)
    pipeline = RDAgentSignalPipeline(
        code_registry=registry,
        executor=executor,
        window=args.window,
    )

    print(f"注册表: {args.registry}")
    active = registry.get_active()
    print(f"活跃因子: {len(active)}")
    if not active:
        print("没有活跃因子，请先运行 rdagent_evolve.py 生成因子")
        return
    for e in active:
        print(f"  - {e.name} (w={e.weight:.3f}, {e.direction})")

    # 初始化数据管理器
    from quantlab.data.data_manager import DataManager
    data_manager = DataManager(provider_uri=args.data_dir, market=args.market)
    data_manager.init_qlib()
    print(f"数据源: {args.data_dir} ({args.market})")

    # 单日 or 批量
    all_signals = {}

    if args.anchor_date:
        output = compute_single_day(
            pipeline, args.anchor_date, data_manager, args.top_k, args.check_decay
        )
        if not output.signal.empty:
            all_signals[args.anchor_date] = output.signal

    elif args.start:
        end = args.end or args.start
        # 获取交易日历
        from qlib.data import D
        cal = D.calendar()
        dates = [
            d.strftime("%Y-%m-%d")
            for d in cal
            if args.start <= d.strftime("%Y-%m-%d") <= end
        ]
        print(f"\n批量回测: {len(dates)} 个交易日 ({args.start} ~ {end})")

        for date in dates:
            output = compute_single_day(
                pipeline, date, data_manager, args.top_k, args.check_decay
            )
            if not output.signal.empty:
                all_signals[date] = output.signal

    # 保存输出
    if args.output and all_signals:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        df = pd.DataFrame(all_signals).T
        df.index.name = "date"
        df.to_csv(args.output)
        print(f"\n信号已保存: {args.output} ({df.shape[0]} 日 × {df.shape[1]} 只股票)")

    # 打印统计
    if all_signals:
        n_days = len(all_signals)
        n_stocks = len(set().union(*[s.index for s in all_signals.values()]))
        print(f"\n统计: {n_days} 个交易日, 覆盖 {n_stocks} 只股票")


if __name__ == "__main__":
    main()
