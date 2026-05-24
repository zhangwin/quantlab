"""Kronos 微调方案对比实验。

Usage:
    # 运行全部预设方案对比
    python kronos_experiment.py --all

    # 运行指定方案
    python kronos_experiment.py --recipes conservative,aggressive,zero_shot

    # 自定义评估区间
    python kronos_experiment.py --all --start 2024-07-01 --end 2024-12-31

    # 保存结果
    python kronos_experiment.py --all --output ./experiment_results
"""

import argparse
import sys
from pathlib import Path

_QUANTLAB_DIR = Path(__file__).resolve().parent.parent

QLIB_DATA_DIR = str(Path.home() / ".qlib" / "qlib_data" / "cn_data")
DEFAULT_RECIPES = str(_QUANTLAB_DIR / "configs" / "kronos_recipes.yaml")


def main():
    parser = argparse.ArgumentParser(description="Kronos 微调方案对比实验")
    parser.add_argument("--all", action="store_true", help="运行全部预设方案")
    parser.add_argument("--recipes", help="指定方案名称，逗号分隔")
    parser.add_argument("--recipes-file", default=DEFAULT_RECIPES, help="方案配置文件路径")
    parser.add_argument("--start", default="2024-07-01", help="评估起始日期")
    parser.add_argument("--end", default="2024-12-31", help="评估截止日期")
    parser.add_argument("--output", help="结果输出目录")
    parser.add_argument("--data-dir", default=QLIB_DATA_DIR, help="Qlib 数据目录")
    parser.add_argument("--market", default="csi300", help="股票池")
    parser.add_argument("--device", default="cuda", help="计算设备")
    args = parser.parse_args()

    if not args.all and not args.recipes:
        parser.print_help()
        return

    from quantlab.data.data_manager import DataManager
    from quantlab.signals.signal_kronos import FinetuneRecipe, FinetuneExperiment

    dm = DataManager(provider_uri=args.data_dir, market=args.market)
    dm.init_qlib()

    # 加载方案
    all_recipes = FinetuneRecipe.load_all(args.recipes_file)
    if args.recipes:
        names = [n.strip() for n in args.recipes.split(",")]
        recipes = [r for r in all_recipes if r.name in names]
        missing = set(names) - {r.name for r in recipes}
        if missing:
            print(f"警告: 方案未找到: {missing}")
    else:
        recipes = all_recipes

    print(f"实验方案: {[r.name for r in recipes]}")
    print(f"评估区间: {args.start} ~ {args.end}")
    print()

    exp = FinetuneExperiment(
        data_manager=dm,
        recipes=recipes,
        eval_start=args.start,
        eval_end=args.end,
        device=args.device,
    )

    exp.run_all()

    # 打印对比表
    df = exp.compare()
    print("\n" + "=" * 80)
    print("方案对比:")
    print("=" * 80)
    print(df.to_string())
    print()

    best = exp.get_best_recipe()
    print(f"最优方案: {best.name} (ICIR 最高)")

    if args.output:
        exp.save_results(args.output)
        print(f"\n结果已保存 -> {args.output}")


if __name__ == "__main__":
    main()
