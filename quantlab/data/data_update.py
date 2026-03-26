"""增量更新行情数据到指定日期。

Usage:
    python data_update.py                                  # 更新到今天（Yahoo）
    python data_update.py --end-date 2025-03-10            # 更新到指定日期
    python data_update.py --source baostock                # 用 baostock 更新（推荐 A股）
    python data_update.py --source csv                     # 从本地 CSV 导入
"""

import argparse
import sys
from pathlib import Path

_QUANTLAB_DIR = Path(__file__).resolve().parent.parent

QLIB_DATA_DIR = str(_QUANTLAB_DIR / "A_data")


def main():
    parser = argparse.ArgumentParser(description="增量更新行情数据")
    parser.add_argument("--end-date", default=None, help="更新到哪天，格式 YYYY-MM-DD（默认今天）")
    parser.add_argument("--source", default="yahoo", choices=["yahoo", "baostock", "csv"], help="数据源（默认 yahoo）")
    parser.add_argument("--data-dir", default=QLIB_DATA_DIR, help="Qlib 数据目录")
    parser.add_argument("--market", default="csi300", help="股票池: csi300/csi500/all")
    args = parser.parse_args()

    import pandas as pd
    from quantlab.data.data_manager import DataManager

    dm = DataManager(
        provider_uri=args.data_dir,
        market=args.market,
        update_source=args.source,
    )

    end_display = args.end_date or pd.Timestamp.now().strftime("%Y-%m-%d")
    print(f"数据源: {args.source}")
    print(f"更新到: {end_display}")

    updated = dm.ensure_data_updated(args.end_date)
    if updated:
        print("更新完成!")
    else:
        print("数据已是最新，无需更新。")


if __name__ == "__main__":
    main()
