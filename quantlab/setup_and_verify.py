"""QuantLab 环境搭建与数据验证脚本

Usage:
    # Step 1: 安装依赖（先执行 pip install）
    # Step 2: 运行本脚本验证
    python quantlab/setup_and_verify.py

本脚本会依次执行:
  1. 检查依赖是否安装
  2. 下载 Qlib A股数据（如果不存在）
  3. 初始化 Qlib 并验证数据完整性
  4. 导出示例 CSV
  5. 运行基础测试
"""

import os
import sys
import time
from pathlib import Path

# 用于用户提示信息中的路径显示
PROJECT_ROOT = Path(__file__).resolve().parent.parent

QLIB_DATA_DIR = Path.home() / ".qlib" / "qlib_data" / "cn_data"


def step(n, title):
    print(f"\n{'='*60}")
    print(f"  Step {n}: {title}")
    print(f"{'='*60}")


def check_import(module_name, pip_name=None):
    """尝试导入模块，失败则给出安装提示。"""
    try:
        __import__(module_name)
        print(f"  [OK] {module_name}")
        return True
    except ImportError:
        pip_name = pip_name or module_name
        print(f"  [FAIL] {module_name} — 请运行: pip install {pip_name}")
        return False


# ==================================================================
# Step 1: 检查依赖
# ==================================================================
def step1_check_deps():
    step(1, "检查依赖")
    all_ok = True
    all_ok &= check_import("numpy")
    all_ok &= check_import("pandas")
    all_ok &= check_import("scipy")
    all_ok &= check_import("yaml", "pyyaml")
    all_ok &= check_import("qlib")
    all_ok &= check_import("fire")
    all_ok &= check_import("tqdm")
    all_ok &= check_import("plotly")
    all_ok &= check_import("yfinance")

    if not all_ok:
        print("\n  !! 存在缺失依赖，请先安装:")
        print(f"     pip install -r {PROJECT_ROOT / 'quantlab' / 'requirements.txt'}")
        print(f"     cd {PROJECT_ROOT / 'qlib'} && pip install -e .")
        return False

    print("\n  所有依赖已就绪!")
    return True


# ==================================================================
# Step 2: 下载 Qlib 数据
# ==================================================================
def step2_download_data():
    step(2, "下载 Qlib A股数据")

    if QLIB_DATA_DIR.exists() and (QLIB_DATA_DIR / "calendars" / "day.txt").exists():
        print(f"  数据目录已存在: {QLIB_DATA_DIR}")
        print("  跳过下载（如需重新下载，请删除该目录后重试）")
        return True

    print(f"  目标目录: {QLIB_DATA_DIR}")
    print("  即将从 GitHub 下载 A股日频数据（约 200-500MB）...")
    print("  如果下载缓慢，可手动下载后解压到上述目录")
    print()

    confirm = input("  是否继续下载？[y/N] ").strip().lower()
    if confirm != "y":
        print("  已跳过下载")
        return False

    try:
        from qlib.tests.data import GetData
        getter = GetData(delete_zip_file=True)
        getter.qlib_data(
            name="qlib_data",
            target_dir=str(QLIB_DATA_DIR),
            region="cn",
            interval="1d",
            exists_skip=True,
        )
        print("  数据下载完成!")
        return True
    except Exception as e:
        print(f"  下载失败: {e}")
        print()
        print("  === 手动下载方法 ===")
        print(f"  python {PROJECT_ROOT / 'qlib' / 'scripts' / 'get_data.py'} qlib_data \\")
        print(f"      --target_dir {QLIB_DATA_DIR} --region cn")
        return False


# ==================================================================
# Step 3: 初始化 Qlib + 验证数据
# ==================================================================
def step3_verify_data():
    step(3, "验证数据完整性")

    if not QLIB_DATA_DIR.exists():
        print("  数据目录不存在，请先完成 Step 2")
        return False

    from quantlab.data.data_manager import DataManager

    dm = DataManager(provider_uri=str(QLIB_DATA_DIR), market="csi300")

    # 3.1 检查日历
    print("\n  --- 3.1 日历文件 ---")
    latest = dm.get_latest_date()
    print(f"  数据最新日期: {latest.strftime('%Y-%m-%d')}")

    cal_path = Path(dm.provider_uri) / "calendars" / "day.txt"
    with open(cal_path) as f:
        total_days = len(f.read().strip().splitlines())
    print(f"  总交易日数: {total_days}")

    # 3.2 初始化 Qlib
    print("\n  --- 3.2 初始化 Qlib ---")
    dm.init_qlib()
    print("  Qlib 初始化成功!")

    # 3.3 交易日历
    print("\n  --- 3.3 交易日历 ---")
    cal = dm.get_trading_calendar("2024-01-01", "2024-12-31")
    print(f"  2024年交易日数: {len(cal)}")
    print(f"  首个交易日: {cal[0].strftime('%Y-%m-%d')}")
    print(f"  最后交易日: {cal[-1].strftime('%Y-%m-%d')}")

    # 3.4 股票池
    print("\n  --- 3.4 CSI300 股票池 ---")
    close = dm.get_close_prices(cal[-1].strftime("%Y-%m-%d"))
    print(f"  股票数量: {len(close)}")
    print(f"  价格范围: {close.min():.2f} ~ {close.max():.2f}")
    top5 = close.nlargest(5)
    print(f"  最贵5只: {dict(zip(top5.index, top5.values.round(2)))}")

    # 3.5 OHLCV 时间隔离
    print("\n  --- 3.5 时间隔离验证 ---")
    test_anchor = "2024-06-28"
    ohlcv = dm.get_ohlcv_before(test_anchor, lookback_days=30)
    print(f"  anchor={test_anchor}, lookback=30")
    print(f"  返回股票数: {len(ohlcv)}")
    if ohlcv:
        sym = list(ohlcv.keys())[0]
        max_date = ohlcv[sym].index.max()
        print(f"  示例 {sym}: {len(ohlcv[sym])} 行, 最大日期={max_date.strftime('%Y-%m-%d')}")
        assert max_date <= pd.Timestamp(test_anchor), "!! 时间隔离失败 !!"
        print(f"  时间隔离: PASS (最大日期 ≤ anchor)")

    # 3.6 涨跌停价
    print("\n  --- 3.6 涨跌停价 ---")
    lu, ld = dm.get_limit_prices(cal[-1].strftime("%Y-%m-%d"))
    print(f"  涨停价股数: {len(lu)}, 跌停价股数: {len(ld)}")
    if len(lu) > 0:
        sym = lu.index[0]
        print(f"  示例 {sym}: 涨停={lu[sym]:.2f}, 跌停={ld[sym]:.2f}")

    # 3.7 日收益率
    print("\n  --- 3.7 日收益率 ---")
    rets = dm.get_daily_returns(cal[-1].strftime("%Y-%m-%d"))
    print(f"  股票数: {len(rets)}")
    valid_rets = rets.dropna()
    print(f"  均值: {valid_rets.mean():.4f}, 标准差: {valid_rets.std():.4f}")
    print(f"  最大: {valid_rets.max():.4f}, 最小: {valid_rets.min():.4f}")

    print("\n  所有数据验证通过!")
    return True


# ==================================================================
# Step 4: 导出示例 CSV
# ==================================================================
def step4_export_csv():
    step(4, "导出示例 CSV")

    from quantlab.data.data_manager import DataManager
    from quantlab.data.data_viewer import DataViewer

    dm = DataManager(provider_uri=str(QLIB_DATA_DIR), market="csi300")
    dm.init_qlib()
    viewer = DataViewer(dm)

    export_dir = PROJECT_ROOT / "quantlab" / "export"
    paths = viewer.export_csv(
        symbols=["SH600519", "SH601318", "SZ000001"],
        start="2024-01-01",
        end="2024-06-30",
        output_dir=str(export_dir),
    )
    print(f"  导出 {len(paths)} 个文件到 {export_dir}/")
    for p in paths:
        import pandas as pd
        df = pd.read_csv(p, index_col=0)
        print(f"    {p.name}: {len(df)} 行")

    # 展示前几行
    if paths:
        import pandas as pd
        df = pd.read_csv(paths[0], index_col=0, parse_dates=True)
        print(f"\n  {paths[0].name} 前5行:")
        print(df.head().to_string(index=True))

    return True


# ==================================================================
# Step 5: 运行测试
# ==================================================================
def step5_run_tests():
    step(5, "运行测试")

    import subprocess
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "quantlab/tests/", "-v", "--tb=short"],
        cwd=str(PROJECT_ROOT),
        capture_output=False,
    )
    return result.returncode == 0


# ==================================================================
# Main
# ==================================================================
if __name__ == "__main__":
    import pandas as pd  # noqa: used in step3

    print("=" * 60)
    print("  QuantLab 环境搭建与数据验证")
    print("=" * 60)
    print(f"  项目根目录: {PROJECT_ROOT}")
    print(f"  Qlib 数据目录: {QLIB_DATA_DIR}")

    ok = step1_check_deps()
    if not ok:
        sys.exit(1)

    ok = step2_download_data()
    if not ok:
        print("\n  数据未就绪，后续步骤跳过")
        sys.exit(1)

    ok = step3_verify_data()
    if not ok:
        sys.exit(1)

    step4_export_csv()

    step5_run_tests()

    print("\n" + "=" * 60)
    print("  全部完成! QuantLab 环境已就绪。")
    print("=" * 60)
