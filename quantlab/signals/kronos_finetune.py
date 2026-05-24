"""Kronos 微调训练：基于 recipe 配置执行 tokenizer/predictor 微调。

与 Kronos 原版 finetune/ 目录的区别：
  - 原版使用 DDP 多卡训练，适合长期大规模微调（30 epoch）
  - 本脚本使用单卡训练，适合日频快速微调（1-5 epoch）+ recipe 管理
  - 支持从 recipe 配置自动设置冻结策略、损失权重、采样方式

Usage:
    # 使用 conservative 方案微调（默认用预处理好的 pkl 数据）
    python kronos_finetune.py --recipe conservative

    # 指定数据目录和输出目录
    python kronos_finetune.py --recipe aggressive --data-path ./data/kronos --output ./outputs/kronos

    # 使用实时 Qlib 数据微调（不用预处理 pkl）
    python kronos_finetune.py --recipe conservative --use-qlib --anchor-date 2024-06-28

    # 只微调 tokenizer
    python kronos_finetune.py --recipe aggressive --stage tokenizer

    # 只微调 predictor（使用已微调的 tokenizer）
    python kronos_finetune.py --recipe conservative --stage predictor --finetuned-tokenizer ./outputs/kronos/tokenizer

    # 两阶段顺序微调（先 tokenizer 后 predictor）
    python kronos_finetune.py --recipe aggressive --stage both

    # 保存微调后的模型
    python kronos_finetune.py --recipe conservative --output ./outputs/kronos --save-model
"""

import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

_QUANTLAB_DIR = Path(__file__).resolve().parent.parent

QLIB_DATA_DIR = str(Path.home() / ".qlib" / "qlib_data" / "cn_data")
DEFAULT_RECIPES = str(_QUANTLAB_DIR / "configs" / "kronos_recipes.yaml")
DEFAULT_DATA_PATH = str(_QUANTLAB_DIR.parent / "data" / "kronos")
DEFAULT_OUTPUT = str(_QUANTLAB_DIR.parent / "outputs" / "kronos")


def load_pkl_data(data_path: str, data_type: str = "train") -> dict:
    """加载预处理好的 pkl 数据集。"""
    path = os.path.join(data_path, f"{data_type}_data.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"数据文件不存在: {path}\n"
            f"请先运行: python kronos_data.py prepare"
        )
    with open(path, "rb") as f:
        data = pickle.load(f)
    print(f"加载 {data_type} 数据: {len(data)} 只股票")
    return data


def load_qlib_data(data_dir: str, market: str, anchor_date: str, lookback: int) -> dict:
    """从 Qlib 实时加载数据（用于日频微调）。"""
    from quantlab.data.data_manager import DataManager

    dm = DataManager(provider_uri=data_dir, market=market)
    dm.init_qlib()

    df = dm.get_ohlcv_before(anchor_date, lookback)
    if df is None or df.empty:
        return {}

    result = {}
    cols = ["open", "high", "low", "close", "volume", "amount"]
    # 重命名适配 Kronos
    rename_map = {"volume": "vol"}

    if isinstance(df.index, pd.MultiIndex):
        for sym in df.index.get_level_values(1).unique():
            sym_df = df.xs(sym, level=1).copy()
            if len(sym_df) >= 20:
                # 确保列名匹配
                for col in cols:
                    if col not in sym_df.columns:
                        if col == "amount" and "amt" not in sym_df.columns:
                            sym_df["amount"] = sym_df.get("close", 0) * sym_df.get("volume", 0)
                sym_df = sym_df.rename(columns={"volume": "vol"})
                if "amt" not in sym_df.columns and "amount" in sym_df.columns:
                    sym_df["amt"] = sym_df["amount"]
                result[sym] = sym_df
    return result


def finetune_with_recipe(args):
    """使用 recipe 配置执行微调。"""
    import torch
    import torch.nn.functional as F
    from quantlab._compat import import_kronos_module
    kronos_mod = import_kronos_module("model.kronos")
    KronosTokenizer, Kronos = kronos_mod.KronosTokenizer, kronos_mod.Kronos
    from quantlab.signals.signal_kronos import FinetuneRecipe

    # 加载方案
    recipe = FinetuneRecipe.load(args.recipes_file, args.recipe)
    print(f"微调方案: {recipe.name}")
    print(f"  描述: {recipe.description}")
    print(f"  Tokenizer: {'微调' if recipe.finetune_tokenizer else '冻结'} ({recipe.tokenizer_strategy})")
    print(f"  Predictor: {'微调' if recipe.finetune_predictor else '冻结'} ({recipe.predictor_strategy})")
    print(f"  Epochs: {recipe.epochs}, LR: {recipe.learning_rate}")
    print(f"  采样策略: {recipe.sample_strategy}")
    print()

    # 设备
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
        print("CUDA 不可用，使用 CPU")
    print(f"设备: {device}")

    # 加载数据
    if args.use_qlib:
        print(f"从 Qlib 加载数据: {args.anchor_date}")
        symbol_data = load_qlib_data(
            args.data_dir, args.market, args.anchor_date, recipe.data_lookback,
        )
    else:
        symbol_data = load_pkl_data(args.data_path, "train")

    if not symbol_data:
        print("没有可用数据，退出")
        return

    # 加载模型
    print(f"\n加载预训练模型...")
    tokenizer_path = args.tokenizer_path or "NeoQuasar/Kronos-Tokenizer-base"
    predictor_path = args.predictor_path or "NeoQuasar/Kronos-base"

    tokenizer = KronosTokenizer.from_pretrained(tokenizer_path)
    model = Kronos.from_pretrained(predictor_path)

    # 确定微调阶段
    stage = args.stage
    if stage == "both":
        stages = ["tokenizer", "predictor"]
    elif stage == "tokenizer":
        stages = ["tokenizer"]
    elif stage == "predictor":
        stages = ["predictor"]
    else:
        # 根据 recipe 自动判断
        stages = []
        if recipe.finetune_tokenizer:
            stages.append("tokenizer")
        if recipe.finetune_predictor:
            stages.append("predictor")
    if not stages:
        print("方案不需要微调任何模块（零样本方案）")
        return

    print(f"微调阶段: {stages}")

    # 如果使用已微调的 tokenizer
    if args.finetuned_tokenizer and "predictor" in stages:
        print(f"加载已微调的 Tokenizer: {args.finetuned_tokenizer}")
        tokenizer = KronosTokenizer.from_pretrained(args.finetuned_tokenizer)

    # 创建输出目录
    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)

    # 使用 signal_kronos 的 KronosFinetuner 进行微调
    from quantlab.signals.signal_kronos import KronosFinetuner
    finetuner = KronosFinetuner(tokenizer, model, device=device)

    # 转换数据格式（pkl 格式可能需要适配）
    adapted_data = _adapt_data(symbol_data, recipe)

    print(f"\n开始微调 ({len(adapted_data)} 只股票)...")
    t0 = time.time()
    ft_tokenizer, ft_model = finetuner.finetune(adapted_data, recipe)
    elapsed = time.time() - t0
    print(f"微调完成: {elapsed:.1f}s")

    # 保存模型
    if args.save_model:
        tok_save = os.path.join(output_dir, "tokenizer")
        mdl_save = os.path.join(output_dir, "predictor")

        ft_tokenizer.cpu().save_pretrained(tok_save)
        ft_model.cpu().save_pretrained(mdl_save)

        print(f"\n模型已保存:")
        print(f"  Tokenizer: {tok_save}")
        print(f"  Predictor: {mdl_save}")

    # 保存训练记录
    record = {
        "recipe": recipe.name,
        "stages": stages,
        "n_symbols": len(adapted_data),
        "elapsed_sec": elapsed,
        "device": device,
        "timestamp": pd.Timestamp.now().isoformat(),
    }
    record_path = os.path.join(output_dir, "finetune_record.json")
    with open(record_path, "w") as f:
        json.dump(record, f, indent=2, ensure_ascii=False)
    print(f"训练记录: {record_path}")

    # 可选：在验证集上评估
    if args.evaluate and not args.use_qlib:
        print("\n在验证集上评估...")
        _evaluate_on_val(ft_tokenizer, ft_model, args.data_path, recipe, device)


def _adapt_data(symbol_data: dict, recipe) -> dict:
    """适配数据格式，确保有 open/high/low/close/volume/amount 列。"""
    adapted = {}
    required_cols = ["open", "high", "low", "close", "volume", "amount"]
    alt_map = {"vol": "volume", "amt": "amount"}

    for sym, df in symbol_data.items():
        df = df.copy()
        # 列名映射
        for alt, std in alt_map.items():
            if alt in df.columns and std not in df.columns:
                df[std] = df[alt]

        # 补缺失列
        if "amount" not in df.columns and "volume" in df.columns and "close" in df.columns:
            df["amount"] = df["volume"] * df["close"]

        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            continue

        if len(df) < 20:
            continue

        adapted[sym] = df[required_cols]

    return adapted


def _evaluate_on_val(tokenizer, model, data_path, recipe, device):
    """在验证集上评估重建损失。"""
    import torch
    import torch.nn.functional as F

    val_data = load_pkl_data(data_path, "val")
    adapted = _adapt_data(val_data, recipe)

    tokenizer = tokenizer.to(device).eval()
    model = model.to(device).eval()

    total_recon_loss = 0.0
    n_samples = 0

    with torch.no_grad():
        for sym, df in list(adapted.items())[:100]:  # 取前100只
            values = df.values.astype(np.float32)
            mean = values.mean(axis=0)
            std = values.std(axis=0) + 1e-5
            values = (values - mean) / std
            values = np.clip(values, -5, 5)

            x = torch.tensor(values, dtype=torch.float32).unsqueeze(0).to(device)
            (z_pre, z_full), bsq_loss, _, _ = tokenizer(x)
            recon = F.mse_loss(z_full, x).item()
            total_recon_loss += recon
            n_samples += 1

    if n_samples > 0:
        avg_loss = total_recon_loss / n_samples
        print(f"  验证集重建损失: {avg_loss:.4f} ({n_samples} 只股票)")
    else:
        print("  验证集为空")


def main():
    parser = argparse.ArgumentParser(description="Kronos 微调训练")

    # 方案配置
    parser.add_argument("--recipe", required=True, help="微调方案名称")
    parser.add_argument("--recipes-file", default=DEFAULT_RECIPES, help="方案配置文件")

    # 数据来源
    parser.add_argument("--data-path", default=DEFAULT_DATA_PATH, help="预处理数据目录（pkl 格式）")
    parser.add_argument("--use-qlib", action="store_true", help="从 Qlib 实时加载数据")
    parser.add_argument("--data-dir", default=QLIB_DATA_DIR, help="Qlib 数据目录")
    parser.add_argument("--market", default="csi300", help="股票池")
    parser.add_argument("--anchor-date", help="实时数据的锚定日期（--use-qlib 时必需）")

    # 模型路径
    parser.add_argument("--tokenizer-path", default=None, help="预训练 Tokenizer 路径/ID")
    parser.add_argument("--predictor-path", default=None, help="预训练 Predictor 路径/ID")
    parser.add_argument("--finetuned-tokenizer", default=None, help="已微调 Tokenizer 路径（微调 predictor 时使用）")

    # 微调阶段
    parser.add_argument(
        "--stage", default="auto",
        choices=["auto", "tokenizer", "predictor", "both"],
        help="微调阶段: auto=根据recipe, tokenizer=只调tokenizer, predictor=只调predictor, both=两阶段",
    )

    # 输出
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="输出目录")
    parser.add_argument("--save-model", action="store_true", help="保存微调后的模型")
    parser.add_argument("--evaluate", action="store_true", help="在验证集上评估")
    parser.add_argument("--device", default="cuda", help="计算设备")

    args = parser.parse_args()

    if args.use_qlib and not args.anchor_date:
        parser.error("--use-qlib 需要指定 --anchor-date")

    finetune_with_recipe(args)


if __name__ == "__main__":
    main()
