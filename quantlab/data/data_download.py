"""下载 Qlib A股日频数据到本地。

Usage:
    python data_download.py                          # 默认下载到 quantlab/A_data
    python data_download.py --target-dir /data/cn    # 指定目标目录
    python data_download.py --region us              # 下载美股数据
    python data_download.py --force                  # 强制重新下载
"""

import argparse
import sys
from pathlib import Path

_QUANTLAB_DIR = Path(__file__).resolve().parent.parent

QLIB_DATA_DIR = str(_QUANTLAB_DIR / "A_data")


def main():
    parser = argparse.ArgumentParser(description="下载 Qlib A股日频数据")
    parser.add_argument("--target-dir", default=QLIB_DATA_DIR, help="下载目标目录（默认 quantlab/A_data）")
    parser.add_argument("--region", default="cn", choices=["cn", "us"], help="数据区域（默认 cn）")
    parser.add_argument("--force", action="store_true", help="强制重新下载，覆盖已有数据")
    args = parser.parse_args()

    target = Path(args.target_dir)
    print(f"目标目录: {target}")
    print(f"区域: {args.region}")

    if target.exists() and (target / "calendars" / "day.txt").exists():
        if not args.force:
            print("数据目录已存在。如需重新下载请加 --force")
            return
        print("--force: 将覆盖已有数据")

    print("开始下载（约 200~500MB）...")
    try:
        from qlib.tests.data import GetData
        getter = GetData(delete_zip_file=True)
        getter.qlib_data(
            name="qlib_data",
            target_dir=str(target),
            region=args.region,
            interval="1d",
            exists_skip=not args.force,
        )
        print("下载完成!")
    except Exception as e:
        print(f"自动下载失败: {e}")
        print()
        print("手动下载方法:")
        print(f"  python {_QUANTLAB_DIR.parent / 'qlib' / 'scripts' / 'get_data.py'} qlib_data \\")
        print(f"      --target_dir {target} --region {args.region}")
        sys.exit(1)


if __name__ == "__main__":
    main()
