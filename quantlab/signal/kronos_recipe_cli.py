"""Kronos 微调方案管理。

Usage:
    # 查看所有预设方案
    python kronos_recipe_cli.py list

    # 查看方案详情
    python kronos_recipe_cli.py show conservative

    # 创建自定义方案
    python kronos_recipe_cli.py create my_recipe --predictor-strategy last_n --unfreeze-layers 3 --epochs 5

    # 删除自定义方案
    python kronos_recipe_cli.py delete my_recipe
"""

import argparse
import sys
from pathlib import Path

_QUANTLAB_DIR = Path(__file__).resolve().parent.parent

DEFAULT_RECIPES = str(_QUANTLAB_DIR / "configs" / "kronos_recipes.yaml")


def cmd_list(args):
    from quantlab.signal.signal_kronos import FinetuneRecipe
    recipes = FinetuneRecipe.load_all(args.recipes_file)

    print(f"共 {len(recipes)} 个方案:\n")
    print(f"{'名称':<20} {'Tok':<5} {'Pred':<5} {'策略':<12} {'Epochs':<7} {'LR':<10} {'采样':<18} {'描述'}")
    print("-" * 110)
    for r in recipes:
        tok = "Y" if r.finetune_tokenizer else "N"
        pred = "Y" if r.finetune_predictor else "N"
        strategy = r.predictor_strategy
        if strategy == "last_n":
            strategy = f"last_{r.predictor_unfreeze_layers}"
        print(
            f"{r.name:<20} {tok:<5} {pred:<5} {strategy:<12} "
            f"{r.epochs:<7} {r.learning_rate:<10.1e} {r.sample_strategy:<18} {r.description}"
        )


def cmd_show(args):
    from quantlab.signal.signal_kronos import FinetuneRecipe
    from dataclasses import asdict
    recipe = FinetuneRecipe.load(args.recipes_file, args.name)
    print(f"方案: {recipe.name}")
    print(f"描述: {recipe.description}")
    print()
    d = asdict(recipe)
    for k, v in d.items():
        if k in ("name", "description"):
            continue
        print(f"  {k:<30} {v}")


def cmd_create(args):
    from quantlab.signal.signal_kronos import FinetuneRecipe

    kwargs = {"name": args.name}
    if args.description:
        kwargs["description"] = args.description
    if args.finetune_tokenizer:
        kwargs["finetune_tokenizer"] = True
    if args.predictor_strategy:
        kwargs["predictor_strategy"] = args.predictor_strategy
    if args.unfreeze_layers is not None:
        kwargs["predictor_unfreeze_layers"] = args.unfreeze_layers
    if args.epochs is not None:
        kwargs["epochs"] = args.epochs
    if args.lr is not None:
        kwargs["learning_rate"] = args.lr
    if args.sample_strategy:
        kwargs["sample_strategy"] = args.sample_strategy
    if args.temperature is not None:
        kwargs["temperature"] = args.temperature
    if args.sample_count is not None:
        kwargs["sample_count"] = args.sample_count

    recipe = FinetuneRecipe(**kwargs)
    recipe.save(args.recipes_file)
    print(f"方案 '{args.name}' 已创建并保存到 {args.recipes_file}")


def cmd_delete(args):
    import yaml
    path = Path(args.recipes_file)
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    recipes = data.get("recipes", [])
    new_recipes = [r for r in recipes if r.get("name") != args.name]
    if len(new_recipes) == len(recipes):
        print(f"方案 '{args.name}' 不存在")
        return
    data["recipes"] = new_recipes
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, allow_unicode=True, default_flow_style=False)
    print(f"方案 '{args.name}' 已删除")


def main():
    parser = argparse.ArgumentParser(description="Kronos 微调方案管理")
    parser.add_argument("--recipes-file", default=DEFAULT_RECIPES, help="方案配置文件")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("list", help="列出所有方案")

    p_show = sub.add_parser("show", help="查看方案详情")
    p_show.add_argument("name", help="方案名称")

    p_create = sub.add_parser("create", help="创建自定义方案")
    p_create.add_argument("name", help="方案名称")
    p_create.add_argument("--description", default="", help="方案描述")
    p_create.add_argument("--finetune-tokenizer", action="store_true")
    p_create.add_argument("--predictor-strategy", choices=["none", "last_n", "head_only", "full"])
    p_create.add_argument("--unfreeze-layers", type=int)
    p_create.add_argument("--epochs", type=int)
    p_create.add_argument("--lr", type=float)
    p_create.add_argument("--sample-strategy", choices=["uniform", "recency_weighted", "volatility_stratified"])
    p_create.add_argument("--temperature", type=float)
    p_create.add_argument("--sample-count", type=int)

    p_del = sub.add_parser("delete", help="删除方案")
    p_del.add_argument("name", help="方案名称")

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return

    cmds = {"list": cmd_list, "show": cmd_show, "create": cmd_create, "delete": cmd_delete}
    cmds[args.command](args)


if __name__ == "__main__":
    main()
