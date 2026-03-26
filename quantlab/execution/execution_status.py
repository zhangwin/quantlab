"""账户状态查询与持仓分析。

Usage:
    # 查看当前持仓
    python execution_status.py holdings --trades ./outputs/execution/trades.csv \
        --close-prices ./data/close.csv --date 2024-06-28

    # 查看交易历史
    python execution_status.py history --trades ./outputs/execution/trades.csv \
        --start 2024-06-01 --end 2024-06-28

    # 查看每日摘要
    python execution_status.py summary --summary ./outputs/execution/daily_summary.csv

    # 统计交易成本
    python execution_status.py cost --trades ./outputs/execution/trades.csv

    # 成功率分析
    python execution_status.py stats --trades ./outputs/execution/trades.csv
"""

import argparse
import sys
from pathlib import Path

import pandas as pd



def cmd_holdings(args):
    """查看持仓状态（从成交记录重建）。"""
    trades = pd.read_csv(args.trades)
    trades = trades[trades["status"].isin(["filled", "partial_filled"])]

    if args.date:
        trades = trades[trades["date"] <= args.date]

    # 重建持仓
    holdings = {}  # symbol -> {"shares": int, "cost": float, "cost_price": float}
    for _, row in trades.iterrows():
        sym = row["symbol"]
        if row["direction"] == "buy":
            if sym not in holdings:
                holdings[sym] = {"shares": 0, "total_cost": 0.0}
            h = holdings[sym]
            h["shares"] += row["shares"]
            h["total_cost"] += row["amount"] + row.get("commission", 0)
        elif row["direction"] == "sell":
            if sym in holdings:
                h = holdings[sym]
                h["shares"] -= row["shares"]
                if h["shares"] <= 0:
                    del holdings[sym]

    if not holdings:
        print("当前无持仓")
        return

    # 加载收盘价（可选）
    close_prices = {}
    if args.close_prices and args.date:
        close_df = pd.read_csv(args.close_prices, index_col=0, parse_dates=True)
        close_df.index = close_df.index.strftime("%Y-%m-%d")
        if args.date in close_df.index:
            close_prices = close_df.loc[args.date].dropna().to_dict()

    print(f"\n{'='*70}")
    print(f"  持仓状态 ({args.date or '最新'})")
    print(f"{'='*70}")
    print(f"  {'股票':<12} {'股数':>8} {'成本价':>10} {'现价':>10} {'市值':>12} {'盈亏':>12} {'盈亏%':>8}")
    print(f"  {'-'*64}")

    total_mv = 0
    total_cost = 0
    for sym, h in sorted(holdings.items()):
        cost_price = h["total_cost"] / h["shares"] if h["shares"] > 0 else 0
        current = close_prices.get(sym, cost_price)
        mv = h["shares"] * current
        pnl = mv - h["total_cost"]
        pnl_pct = pnl / h["total_cost"] if h["total_cost"] > 0 else 0
        total_mv += mv
        total_cost += h["total_cost"]
        print(f"  {sym:<12} {h['shares']:>8d} {cost_price:>10.2f} {current:>10.2f} "
              f"{mv:>12,.0f} {pnl:>+12,.0f} {pnl_pct:>+7.2%}")

    total_pnl = total_mv - total_cost
    print(f"  {'-'*64}")
    print(f"  {'合计':<12} {'':>8} {'':>10} {'':>10} "
          f"{total_mv:>12,.0f} {total_pnl:>+12,.0f}")
    print(f"  持仓 {len(holdings)} 只\n")


def cmd_history(args):
    """查看交易历史。"""
    trades = pd.read_csv(args.trades)

    if args.start:
        trades = trades[trades["date"] >= args.start]
    if args.end:
        trades = trades[trades["date"] <= args.end]
    if args.status:
        trades = trades[trades["status"] == args.status]

    if trades.empty:
        print("无匹配交易记录")
        return

    print(f"\n{'='*90}")
    print(f"  交易历史 ({args.start or '开始'} ~ {args.end or '最新'})")
    print(f"{'='*90}")
    print(f"  {'日期':<12} {'方向':>4} {'股票':<12} {'状态':<20} "
          f"{'股数':>8} {'成交价':>10} {'金额':>12} {'费用':>8}")
    print(f"  {'-'*84}")

    for _, row in trades.iterrows():
        tag = "买入" if row["direction"] == "buy" else "卖出"
        print(f"  {row['date']:<12} {tag:>4} {row['symbol']:<12} {row['status']:<20} "
              f"{row['shares']:>8} {row.get('exec_price', 0):>10.2f} "
              f"{row.get('amount', 0):>12,.0f} {row.get('total_cost', 0):>8,.1f}")

    print(f"\n  共 {len(trades)} 笔记录\n")


def cmd_summary(args):
    """查看每日摘要。"""
    df = pd.read_csv(args.summary)

    print(f"\n{'='*90}")
    print(f"  每日摘要")
    print(f"{'='*90}")
    print(f"  {'日期':<12} {'总资产':>14} {'现金':>14} {'市值':>14} "
          f"{'持仓':>4} {'日PnL':>12} {'日收益':>8}")
    print(f"  {'-'*84}")

    for _, row in df.iterrows():
        print(f"  {row['date']:<12} {row['total_value']:>14,.0f} {row['cash']:>14,.0f} "
              f"{row['market_value']:>14,.0f} {row['position_count']:>4} "
              f"{row['daily_pnl']:>+12,.0f} {row['daily_return']:>+7.4%}")

    print()


def cmd_cost(args):
    """统计交易成本。"""
    trades = pd.read_csv(args.trades)
    filled = trades[trades["status"].isin(["filled", "partial_filled"])]

    if filled.empty:
        print("无成交记录")
        return

    buys = filled[filled["direction"] == "buy"]
    sells = filled[filled["direction"] == "sell"]

    total_commission = filled["commission"].sum()
    total_stamp_tax = filled["stamp_tax"].sum()
    total_amount = filled["amount"].sum()

    print(f"\n{'='*50}")
    print(f"  交易成本统计")
    print(f"{'='*50}")
    print(f"  总成交笔数:   {len(filled):>10d}")
    print(f"    买入:       {len(buys):>10d}")
    print(f"    卖出:       {len(sells):>10d}")
    print(f"  总成交金额:   {total_amount:>14,.0f}")
    print(f"  总佣金:       {total_commission:>14,.2f}")
    print(f"  总印花税:     {total_stamp_tax:>14,.2f}")
    print(f"  总费用:       {total_commission + total_stamp_tax:>14,.2f}")
    print(f"  费率:         {(total_commission + total_stamp_tax) / total_amount:.4%}" if total_amount > 0 else "")
    print()


def cmd_stats(args):
    """成功率/失败原因分析。"""
    trades = pd.read_csv(args.trades)

    total = len(trades)
    status_counts = trades["status"].value_counts()
    reason_counts = trades["reason"].value_counts()

    print(f"\n{'='*50}")
    print(f"  交易统计")
    print(f"{'='*50}")
    print(f"  总订单数: {total}")
    print(f"\n  按状态:")
    for status, count in status_counts.items():
        pct = count / total * 100
        print(f"    {status:<25} {count:>6}  ({pct:.1f}%)")
    print(f"\n  按原因:")
    for reason, count in reason_counts.items():
        pct = count / total * 100
        print(f"    {reason:<25} {count:>6}  ({pct:.1f}%)")
    print()


def main():
    parser = argparse.ArgumentParser(description="M6 账户状态查询")
    subparsers = parser.add_subparsers(dest="command", help="子命令")

    # holdings
    p_holdings = subparsers.add_parser("holdings", help="查看持仓")
    p_holdings.add_argument("--trades", required=True, help="成交记录 CSV")
    p_holdings.add_argument("--close-prices", default=None, help="收盘价 CSV")
    p_holdings.add_argument("--date", default=None, help="查询日期")

    # history
    p_history = subparsers.add_parser("history", help="交易历史")
    p_history.add_argument("--trades", required=True, help="成交记录 CSV")
    p_history.add_argument("--start", default=None, help="起始日期")
    p_history.add_argument("--end", default=None, help="结束日期")
    p_history.add_argument("--status", default=None, help="筛选状态")

    # summary
    p_summary = subparsers.add_parser("summary", help="每日摘要")
    p_summary.add_argument("--summary", required=True, help="每日摘要 CSV")

    # cost
    p_cost = subparsers.add_parser("cost", help="成本统计")
    p_cost.add_argument("--trades", required=True, help="成交记录 CSV")

    # stats
    p_stats = subparsers.add_parser("stats", help="成功率分析")
    p_stats.add_argument("--trades", required=True, help="成交记录 CSV")

    args = parser.parse_args()

    if args.command == "holdings":
        cmd_holdings(args)
    elif args.command == "history":
        cmd_history(args)
    elif args.command == "summary":
        cmd_summary(args)
    elif args.command == "cost":
        cmd_cost(args)
    elif args.command == "stats":
        cmd_stats(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
