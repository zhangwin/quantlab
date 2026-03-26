"""交易执行：从融合信号生成订单并模拟成交。

Usage:
    # 单日执行（T 日信号 → T+1 成交）
    python execution_run.py --anchor-date 2024-06-28 \
        --signal ./outputs/ensemble/signal.csv \
        --close-prices ./data/close.csv \
        --open-prices ./data/open_next.csv \
        --limit-up ./data/limit_up.csv \
        --limit-down ./data/limit_down.csv

    # 多日回测
    python execution_run.py --start 2024-01-01 --end 2024-06-30 \
        --signal ./outputs/ensemble/signals.csv \
        --close-prices ./data/close.csv \
        --open-prices ./data/open.csv \
        --limit-up ./data/limit_up.csv \
        --limit-down ./data/limit_down.csv

    # 自定义参数
    python execution_run.py --anchor-date 2024-06-28 \
        --signal ./outputs/ensemble/signal.csv \
        --close-prices ./data/close.csv \
        --open-prices ./data/open_next.csv \
        --initial-cash 2000000 --max-positions 30 \
        --max-single-weight 0.08 --target-buy 5 --target-sell 5

    # 追加止损强卖
    python execution_run.py --anchor-date 2024-06-28 \
        --signal ./outputs/ensemble/signal.csv \
        --close-prices ./data/close.csv \
        --open-prices ./data/open_next.csv \
        --force-sell SH600000,SH601318 --force-sell-reason stop_loss

    # 输出成交记录
    python execution_run.py --anchor-date 2024-06-28 \
        --signal ./outputs/ensemble/signal.csv \
        --close-prices ./data/close.csv \
        --open-prices ./data/open_next.csv \
        --output ./outputs/execution/trades.csv
"""

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

from quantlab.execution.execution import ExecutionPipeline


def load_price_csv(path: str) -> pd.DataFrame:
    """加载价格 CSV，index=date, columns=symbol。"""
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.index = df.index.strftime("%Y-%m-%d")
    return df


def load_signal_csv(path: str) -> pd.DataFrame:
    """加载信号 CSV，index=date, columns=symbol 或 index=symbol。"""
    df = pd.read_csv(path, index_col=0)
    return df


def run_single_day(
    pipeline: ExecutionPipeline,
    signal: pd.Series,
    close_prices: dict,
    open_prices: dict,
    limit_up: dict,
    limit_down: dict,
    anchor_date: str,
    exec_date: str,
    industry_map: dict = None,
    force_sell: list = None,
    force_sell_reason: str = "stop_loss",
):
    """单日执行：T 日生成订单 → T+1 成交。"""
    # T 日收盘后生成订单
    orders = pipeline.generate_orders(signal, close_prices, industry_map, anchor_date)

    # 追加风控强卖
    if force_sell:
        orders = pipeline.add_force_sell_orders(
            orders, force_sell, close_prices, force_sell_reason,
        )

    # T+1 成交判定
    records = pipeline.execute_orders(orders, open_prices, limit_up, limit_down, exec_date)

    return orders, records


def print_trade_summary(records, exec_date: str, pipeline: ExecutionPipeline, close_prices: dict):
    """打印当日交易摘要。"""
    summary = pipeline.get_daily_summary(close_prices, exec_date)

    filled = [r for r in records if r.status in ("filled", "partial_filled")]
    failed = [r for r in records if r.status.startswith("failed")]

    print(f"\n{'='*60}")
    print(f"  交易日: {exec_date}")
    print(f"{'='*60}")
    print(f"  总资产: {summary.total_value:>14,.0f}")
    print(f"  现  金: {summary.cash:>14,.0f}")
    print(f"  市  值: {summary.market_value:>14,.0f}")
    print(f"  持仓数: {summary.position_count:>14d}")
    print(f"  当日PnL: {summary.daily_pnl:>+13,.0f}")
    print(f"  日收益: {summary.daily_return:>13.4%}")
    print(f"  买入: {summary.buy_count} 笔  卖出: {summary.sell_count} 笔")
    print(f"  佣金: {summary.total_commission:,.2f}  印花税: {summary.total_stamp_tax:,.2f}")

    if filled:
        print(f"\n  成交明细:")
        for r in filled:
            tag = "买" if r.direction == "buy" else "卖"
            print(f"    [{tag}] {r.symbol}  {r.shares}股 × {r.exec_price:.2f}  "
                  f"金额={r.amount:,.0f}  费用={r.total_cost:,.1f}  "
                  f"[{r.status}] ({r.reason})")

    if failed:
        print(f"\n  失败订单:")
        for r in failed:
            tag = "买" if r.direction == "buy" else "卖"
            print(f"    [{tag}] {r.symbol}  {r.shares}股  → {r.status} ({r.reason})")

    print()


def main():
    parser = argparse.ArgumentParser(description="M6 交易执行")
    parser.add_argument("--signal", required=True, help="融合信号 CSV")
    parser.add_argument("--close-prices", required=True, help="收盘价 CSV (date×symbol)")
    parser.add_argument("--open-prices", required=True, help="开盘价 CSV (date×symbol)")
    parser.add_argument("--limit-up", default=None, help="涨停价 CSV (date×symbol)")
    parser.add_argument("--limit-down", default=None, help="跌停价 CSV (date×symbol)")
    parser.add_argument("--industry-map", default=None, help="行业映射 CSV (symbol,industry)")

    # 日期
    parser.add_argument("--anchor-date", default=None, help="T 日（单日模式）")
    parser.add_argument("--start", default=None, help="起始日期（多日模式）")
    parser.add_argument("--end", default=None, help="结束日期（多日模式）")

    # 执行参数
    parser.add_argument("--initial-cash", type=float, default=1_000_000, help="初始资金 (默认 100万)")
    parser.add_argument("--max-positions", type=int, default=20, help="最大持仓数")
    parser.add_argument("--max-single-weight", type=float, default=0.1, help="单票仓位上限")
    parser.add_argument("--target-buy", type=int, default=3, help="每日目标买入数")
    parser.add_argument("--target-sell", type=int, default=3, help="每日目标卖出数")

    # 风控强卖
    parser.add_argument("--force-sell", default=None, help="强制卖出的股票代码，逗号分隔")
    parser.add_argument("--force-sell-reason", default="stop_loss",
                        choices=["stop_loss", "industry_limit", "circuit_breaker"],
                        help="强卖原因")

    # 输出
    parser.add_argument("--output", default=None, help="输出成交记录 CSV")
    parser.add_argument("--summary-output", default=None, help="输出每日摘要 CSV")
    parser.add_argument("--quiet", action="store_true", help="静默模式")

    args = parser.parse_args()

    # 加载数据
    signal_df = load_signal_csv(args.signal)
    close_df = load_price_csv(args.close_prices)
    open_df = load_price_csv(args.open_prices)

    limit_up_df = load_price_csv(args.limit_up) if args.limit_up else None
    limit_down_df = load_price_csv(args.limit_down) if args.limit_down else None

    industry_map = None
    if args.industry_map:
        ind_df = pd.read_csv(args.industry_map)
        industry_map = dict(zip(ind_df.iloc[:, 0], ind_df.iloc[:, 1]))

    force_sell = args.force_sell.split(",") if args.force_sell else None

    # 创建 pipeline
    pipeline = ExecutionPipeline(
        initial_cash=args.initial_cash,
        max_positions=args.max_positions,
        max_single_weight=args.max_single_weight,
        target_buy_count=args.target_buy,
        target_sell_count=args.target_sell,
    )

    # 确定交易日期
    if args.anchor_date:
        dates = [args.anchor_date]
    elif args.start and args.end:
        all_dates = sorted(close_df.index.unique())
        dates = [d for d in all_dates if args.start <= d <= args.end]
    else:
        print("错误: 请指定 --anchor-date 或 --start/--end")
        sys.exit(1)

    if not dates:
        print("错误: 指定日期范围内无交易日")
        sys.exit(1)

    all_records = []
    all_summaries = []

    for i, date in enumerate(dates):
        # 获取 T 日信号
        if date in signal_df.index:
            signal = signal_df.loc[date]
        elif signal_df.shape[0] == 1:
            signal = signal_df.iloc[0]
        else:
            if not args.quiet:
                print(f"  跳过 {date}: 无信号数据")
            continue

        # T 日收盘价
        if date not in close_df.index:
            if not args.quiet:
                print(f"  跳过 {date}: 无收盘价数据")
            continue
        close_prices = close_df.loc[date].dropna().to_dict()

        # T+1 日期和开盘价
        all_dates_sorted = sorted(open_df.index.unique())
        t1_candidates = [d for d in all_dates_sorted if d > date]
        if not t1_candidates:
            if not args.quiet:
                print(f"  跳过 {date}: 无 T+1 日数据")
            continue
        exec_date = t1_candidates[0]

        if exec_date not in open_df.index:
            if not args.quiet:
                print(f"  跳过 {date}: T+1 日 {exec_date} 无开盘价")
            continue
        open_prices = open_df.loc[exec_date].dropna().to_dict()

        # 涨跌停价
        limit_up = {}
        limit_down = {}
        if limit_up_df is not None and exec_date in limit_up_df.index:
            limit_up = limit_up_df.loc[exec_date].dropna().to_dict()
        if limit_down_df is not None and exec_date in limit_down_df.index:
            limit_down = limit_down_df.loc[exec_date].dropna().to_dict()

        # 仅首日或指定时使用 force_sell
        fs = force_sell if (i == 0 and force_sell) else None

        orders, records = run_single_day(
            pipeline, signal, close_prices, open_prices,
            limit_up, limit_down, date, exec_date,
            industry_map, fs, args.force_sell_reason,
        )

        all_records.extend(records)

        # 用 T+1 收盘价计算摘要（如有），否则用开盘价
        t1_close = close_df.loc[exec_date].dropna().to_dict() if exec_date in close_df.index else open_prices
        summary = pipeline.get_daily_summary(t1_close, exec_date)
        all_summaries.append(summary)

        if not args.quiet:
            print_trade_summary(records, exec_date, pipeline, t1_close)

    # 输出成交记录
    if args.output and all_records:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        rows = []
        for r in all_records:
            rows.append({
                "date": r.date, "symbol": r.symbol, "direction": r.direction,
                "status": r.status, "order_price": r.order_price,
                "exec_price": r.exec_price, "shares": r.shares,
                "amount": r.amount, "commission": r.commission,
                "stamp_tax": r.stamp_tax, "total_cost": r.total_cost,
                "reason": r.reason,
            })
        pd.DataFrame(rows).to_csv(args.output, index=False)
        print(f"成交记录已保存至 {args.output}")

    # 输出每日摘要
    if args.summary_output and all_summaries:
        os.makedirs(os.path.dirname(args.summary_output) or ".", exist_ok=True)
        rows = []
        for s in all_summaries:
            rows.append({
                "date": s.date, "total_value": s.total_value, "cash": s.cash,
                "market_value": s.market_value, "position_count": s.position_count,
                "daily_pnl": s.daily_pnl, "daily_return": s.daily_return,
                "buy_count": s.buy_count, "sell_count": s.sell_count,
                "total_commission": s.total_commission, "total_stamp_tax": s.total_stamp_tax,
            })
        pd.DataFrame(rows).to_csv(args.summary_output, index=False)
        print(f"每日摘要已保存至 {args.summary_output}")

    # 最终汇总
    if not args.quiet and all_summaries:
        first = all_summaries[0]
        last = all_summaries[-1]
        total_return = (last.total_value / args.initial_cash - 1)
        total_commission = sum(s.total_commission for s in all_summaries)
        total_tax = sum(s.total_stamp_tax for s in all_summaries)
        print(f"{'='*60}")
        print(f"  回测汇总 ({dates[0]} ~ {dates[-1]})")
        print(f"{'='*60}")
        print(f"  初始资金: {args.initial_cash:>14,.0f}")
        print(f"  最终资产: {last.total_value:>14,.0f}")
        print(f"  总收益率: {total_return:>13.4%}")
        print(f"  交易天数: {len(all_summaries):>14d}")
        print(f"  累计佣金: {total_commission:>14,.2f}")
        print(f"  累计印花税: {total_tax:>12,.2f}")
        print(f"  最终持仓: {last.position_count:>14d} 只")


if __name__ == "__main__":
    main()
