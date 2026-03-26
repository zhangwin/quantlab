"""导出股票 OHLCV 数据为 CSV 文件。

Usage:
    python data_export.py --symbols SH600519 --start 2024-01-01 --end 2024-06-30
    python data_export.py --symbols SH600519,SH601318,SZ000001 --start 2024-01-01 --end 2024-06-30
    python data_export.py --symbols SH600519,SH601318 --start 2024-01-01 --end 2024-06-30 --merge
    python data_export.py --symbols SH600519 --start 2024-01-01 --end 2024-06-30 --output ./my_data
"""

import argparse
import sys
from pathlib import Path

QLIB_DATA_DIR = str(Path.home() / ".qlib" / "qlib_data" / "cn_data")


def main():
    parser = argparse.ArgumentParser(description="导出股票 OHLCV 数据为 CSV")
    parser.add_argument("--symbols", required=True, help="股票代码，逗号分隔，如 SH600519,SH601318")
    parser.add_argument("--start", required=True, help="起始日期，如 2024-01-01")
    parser.add_argument("--end", required=True, help="截止日期，如 2024-06-30")
    parser.add_argument("--output", default="./export", help="输出目录（默认 ./export）")
    parser.add_argument("--merge", action="store_true", help="合并多只股票为一个文件")
    parser.add_argument("--data-dir", default=QLIB_DATA_DIR, help="Qlib 数据目录")
    parser.add_argument("--market", default="csi300", help="股票池: csi300/csi500/all")
    args = parser.parse_args()

    import pandas as pd
    from quantlab.data.data_manager import DataManager
    from quantlab.data.data_viewer import DataViewer

    dm = DataManager(provider_uri=args.data_dir, market=args.market)
    dm.init_qlib()
    viewer = DataViewer(dm)

    symbols = [s.strip() for s in args.symbols.split(",")]
    print(f"导出股票: {symbols}")
    print(f"日期范围: {args.start} ~ {args.end}")
    print(f"输出目录: {args.output}")

    if args.merge:
        out_path = str(Path(args.output) / "portfolio.csv")
        viewer.export_portfolio_csv(symbols, args.start, args.end, output_path=out_path)
        print(f"合并导出完成 → {out_path}")
    else:
        paths = viewer.export_csv(symbols, args.start, args.end, output_dir=args.output)
        print(f"导出 {len(paths)} 个文件:")
        for p in paths:
            df = pd.read_csv(p, index_col=0)
            print(f"  {p.name}: {len(df)} 行")


if __name__ == "__main__":
    main()
