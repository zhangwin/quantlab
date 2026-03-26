"""查看本地数据状态：日期范围、股票统计、数据质量、磁盘占用等。

Usage:
    python data_status.py                          # 默认查看 CSI300
    python data_status.py --market all             # 查看全 A股
    python data_status.py --data-dir /data/cn      # 指定数据目录
"""

import argparse
import sys
from pathlib import Path

_QUANTLAB_DIR = Path(__file__).resolve().parent.parent

QLIB_DATA_DIR = str(_QUANTLAB_DIR / "A_data")


def _fmt_size(size_bytes: int) -> str:
    """Format bytes to human-readable string."""
    if size_bytes >= 1024 ** 3:
        return f"{size_bytes / 1024 ** 3:.2f} GB"
    elif size_bytes >= 1024 ** 2:
        return f"{size_bytes / 1024 ** 2:.1f} MB"
    elif size_bytes >= 1024:
        return f"{size_bytes / 1024:.1f} KB"
    return f"{size_bytes} B"


def main():
    parser = argparse.ArgumentParser(description="查看本地数据状态")
    parser.add_argument("--data-dir", default=QLIB_DATA_DIR, help="Qlib 数据目录（默认 quantlab/A_data）")
    parser.add_argument("--market", default="csi300", help="股票池: csi300/csi500/all")
    args = parser.parse_args()

    data_path = Path(args.data_dir)

    print("=" * 60)
    print("  本地数据状态报告")
    print("=" * 60)
    print(f"数据目录:   {data_path}")
    print()

    if not data_path.exists():
        print("[ERROR] 数据目录不存在")
        print("请先运行: python data_download.py")
        return

    # ------------------------------------------------------------------
    # 1. 日历信息
    # ------------------------------------------------------------------
    print("--- 日历信息 ---")
    cal_path = data_path / "calendars" / "day.txt"
    if not cal_path.exists():
        print("[ERROR] 日历文件不存在: calendars/day.txt")
        return

    with open(cal_path) as f:
        cal_lines = [line.strip() for line in f if line.strip()]
    total_days = len(cal_lines)

    if total_days == 0:
        print("[ERROR] 日历文件为空")
        return

    import pandas as pd

    first_date = pd.Timestamp(cal_lines[0])
    latest_date = pd.Timestamp(cal_lines[-1])
    today = pd.Timestamp.now().normalize()
    data_lag = (today - latest_date).days

    print(f"起始日期:     {first_date.strftime('%Y-%m-%d')}")
    print(f"最新日期:     {latest_date.strftime('%Y-%m-%d')}")
    print(f"总交易日数:   {total_days}")
    print(f"数据时间跨度: {(latest_date - first_date).days} 天")
    if data_lag > 1:
        print(f"数据滞后:     {data_lag} 天（距今）⚠️  建议运行 python data_update.py")
    else:
        print(f"数据滞后:     {data_lag} 天（已是最新）")
    print()

    # ------------------------------------------------------------------
    # 2. 股票池与价格统计
    # ------------------------------------------------------------------
    print("--- 股票池统计 ---")
    from quantlab.data.data_manager import DataManager
    dm = DataManager(provider_uri=args.data_dir, market=args.market)
    dm.init_qlib()

    latest_str = latest_date.strftime("%Y-%m-%d")
    try:
        close = dm.get_close_prices(latest_str)
        valid_close = close.dropna()
        print(f"股票池:       {args.market}")
        print(f"当前股票数:   {len(close)}")
        print(f"有效价格数:   {len(valid_close)}  (NaN: {len(close) - len(valid_close)})")
        if len(valid_close) > 0:
            print(f"价格范围:     {valid_close.min():.2f} ~ {valid_close.max():.2f}")
            print(f"价格中位数:   {valid_close.median():.2f}")
            print(f"价格均值:     {valid_close.mean():.2f}")
    except Exception as e:
        print(f"[ERROR] 股票数据读取失败: {e}")
    print()

    # ------------------------------------------------------------------
    # 3. 数据质量检查
    # ------------------------------------------------------------------
    print("--- 数据质量 ---")
    try:
        # 检查最近几天的数据覆盖率
        cal = dm.get_trading_calendar(
            start=(latest_date - pd.Timedelta(days=30)).strftime("%Y-%m-%d"),
            end=latest_str,
        )
        recent_days = len(cal)
        if recent_days > 0:
            # 取最近5个交易日的覆盖率
            check_days = cal[-min(5, len(cal)):]
            coverage_list = []
            for day in check_days:
                day_str = day.strftime("%Y-%m-%d")
                day_close = dm.get_close_prices(day_str)
                valid_count = day_close.dropna().shape[0]
                total_count = day_close.shape[0]
                coverage = valid_count / total_count * 100 if total_count > 0 else 0
                coverage_list.append(coverage)
                print(f"  {day_str}: {valid_count}/{total_count} 只有数据 ({coverage:.1f}%)")

            avg_coverage = sum(coverage_list) / len(coverage_list) if coverage_list else 0
            print(f"近期平均覆盖率: {avg_coverage:.1f}%")
    except Exception as e:
        print(f"[WARN] 数据质量检查失败: {e}")
    print()

    # ------------------------------------------------------------------
    # 4. instruments 文件统计
    # ------------------------------------------------------------------
    print("--- Instruments 文件 ---")
    inst_dir = data_path / "instruments"
    if inst_dir.exists():
        for inst_file in sorted(inst_dir.glob("*.txt")):
            with open(inst_file) as f:
                lines = [line for line in f if line.strip()]
            print(f"  {inst_file.name}: {len(lines)} 条记录")
    else:
        print("  instruments 目录不存在")
    print()

    # ------------------------------------------------------------------
    # 5. 行业数据
    # ------------------------------------------------------------------
    print("--- 行业映射 ---")
    industry_path = data_path.parent / "industry_map.csv"
    if industry_path.exists():
        industry_df = pd.read_csv(industry_path, dtype=str)
        n_industries = industry_df["industry"].nunique() if "industry" in industry_df.columns else 0
        print(f"状态: 已配置 ({industry_path.name})")
        print(f"覆盖股票数: {len(industry_df)}, 行业数: {n_industries}")
    else:
        print("状态: 未配置（将使用 dummy 映射）")
        print(f"如需行业风控，请准备: {industry_path}")
    print()

    # ------------------------------------------------------------------
    # 6. features (bin 文件) 统计
    # ------------------------------------------------------------------
    print("--- 数据文件统计 ---")
    features_dir = data_path / "features"
    if features_dir.exists():
        stock_dirs = [d for d in features_dir.iterdir() if d.is_dir()]
        print(f"股票数据目录: {len(stock_dirs)} 个")
        # 统计 bin 文件数和字段
        if stock_dirs:
            sample_dir = stock_dirs[0]
            bin_files = list(sample_dir.glob("*.bin"))
            field_names = sorted([f.stem for f in bin_files])
            print(f"每只股票字段: {len(bin_files)} 个")
            print(f"字段列表:     {', '.join(field_names)}")
    else:
        print("features 目录不存在")

    # 磁盘占用按子目录统计
    print()
    print("--- 磁盘占用 ---")
    total_size = 0
    sub_sizes = {}
    for item in sorted(data_path.iterdir()):
        if item.is_dir():
            dir_size = sum(f.stat().st_size for f in item.rglob("*") if f.is_file())
            sub_sizes[item.name] = dir_size
            total_size += dir_size
        elif item.is_file():
            total_size += item.stat().st_size

    for name, size in sorted(sub_sizes.items(), key=lambda x: -x[1]):
        print(f"  {name + '/':20s} {_fmt_size(size)}")
    print(f"  {'总计':20s} {_fmt_size(total_size)}")

    print()
    print("=" * 60)


if __name__ == "__main__":
    main()
