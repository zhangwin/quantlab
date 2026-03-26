"""画 K 线图并保存为 HTML 文件。

Usage:
    python data_plot.py --symbol SH600519 --start 2024-01-01 --end 2024-06-30
    python data_plot.py --symbol SH600519 --start 2024-01-01 --end 2024-06-30 --ma 5,20,60
    python data_plot.py --symbol SH600519 --start 2024-01-01 --end 2024-06-30 --output maotai.html
    python data_plot.py --symbol SH600519 --start 2024-01-01 --end 2024-06-30 --show
"""

import argparse
import sys
from pathlib import Path

QLIB_DATA_DIR = str(Path.home() / ".qlib" / "qlib_data" / "cn_data")


def main():
    parser = argparse.ArgumentParser(description="画 K 线图")
    parser.add_argument("--symbol", required=True, help="股票代码，如 SH600519")
    parser.add_argument("--start", required=True, help="起始日期")
    parser.add_argument("--end", required=True, help="截止日期")
    parser.add_argument("--ma", default="5,10,20", help="均线周期，逗号分隔（默认 5,10,20）")
    parser.add_argument("--output", default=None, help="输出文件路径（默认 {symbol}_{start}_{end}.html）")
    parser.add_argument("--show", action="store_true", help="同时在浏览器中打开")
    parser.add_argument("--data-dir", default=QLIB_DATA_DIR, help="Qlib 数据目录")
    parser.add_argument("--market", default="csi300", help="股票池: csi300/csi500/all")
    args = parser.parse_args()

    from quantlab.data.data_manager import DataManager
    from quantlab.data.data_viewer import DataViewer

    dm = DataManager(provider_uri=args.data_dir, market=args.market)
    dm.init_qlib()
    viewer = DataViewer(dm)

    output = args.output or f"{args.symbol}_{args.start}_{args.end}.html"
    ma_windows = [int(x) for x in args.ma.split(",")]

    print(f"绘制 K 线图: {args.symbol}")
    print(f"日期范围: {args.start} ~ {args.end}")
    print(f"均线: MA{ma_windows}")

    fig = viewer.plot_kline(args.symbol, args.start, args.end, ma_windows=ma_windows)
    fig.write_html(output)
    print(f"图表已保存 → {output}")

    if args.show:
        fig.show()


if __name__ == "__main__":
    main()
