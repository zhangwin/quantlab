"""Kronos 数据准备：下载预训练模型、预处理 Qlib 数据为训练集。

Usage:
    # 下载预训练模型（从 HuggingFace）
    python kronos_data.py download-models

    # 下载指定版本的模型
    python kronos_data.py download-models --tokenizer NeoQuasar/Kronos-Tokenizer-base --predictor NeoQuasar/Kronos-base

    # 从 Qlib 数据生成训练/验证/测试集（pkl 格式）
    python kronos_data.py prepare --start 2011-01-01 --end 2025-01-01

    # 自定义参数
    python kronos_data.py prepare --market csi300 --lookback 90 --horizon 10 --output ./data/kronos

    # 查看已有数据集状态
    python kronos_data.py status

    # 查看已有数据集状态（指定目录）
    python kronos_data.py status --data-path ./data/kronos
"""

import argparse
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_QUANTLAB_DIR = Path(__file__).resolve().parent.parent

QLIB_DATA_DIR = str(Path.home() / ".qlib" / "qlib_data" / "cn_data")
DEFAULT_OUTPUT = str(_QUANTLAB_DIR.parent / "data" / "kronos")


# ===================================================================
# 下载预训练模型
# ===================================================================

def cmd_download_models(args):
    """从 HuggingFace Hub 下载 Kronos 预训练模型。"""
    from quantlab._compat import import_kronos_module

    kronos_mod = import_kronos_module("model.kronos")
    KronosTokenizer = kronos_mod.KronosTokenizer

    print(f"下载 Kronos Tokenizer: {args.tokenizer}")
    tokenizer = KronosTokenizer.from_pretrained(args.tokenizer)
    tok_save = os.path.join(args.save_dir, "tokenizer")
    tokenizer.save_pretrained(tok_save)
    print(f"  已保存 -> {tok_save}")

    print(f"下载 Kronos Predictor: {args.predictor}")
    Kronos = kronos_mod.Kronos
    model = Kronos.from_pretrained(args.predictor)
    mdl_save = os.path.join(args.save_dir, "predictor")
    model.save_pretrained(mdl_save)
    print(f"  已保存 -> {mdl_save}")

    print("\n下载完成。模型已缓存到本地。")
    print(f"  Tokenizer: {tok_save}")
    print(f"  Predictor: {mdl_save}")


# ===================================================================
# 数据预处理
# ===================================================================

def cmd_prepare(args):
    """从 Qlib 加载数据，预处理并切分为训练/验证/测试集。"""
    import qlib
    from qlib.config import REG_CN
    from qlib.data import D
    from qlib.data.dataset.loader import QlibDataLoader

    print(f"初始化 Qlib: {args.data_dir}")
    qlib.init(provider_uri=args.data_dir, region=REG_CN)

    feature_list = ["open", "high", "low", "close", "vol", "amt"]
    data_fields = ["open", "close", "high", "low", "volume", "vwap"]
    data_fields_qlib = ["$" + f for f in data_fields]

    # 获取交易日历
    cal = D.calendar()
    start_idx = cal.searchsorted(pd.Timestamp(args.start))
    end_idx = cal.searchsorted(pd.Timestamp(args.end))
    adjusted_start = max(start_idx - args.lookback, 0)
    adjusted_end = min(end_idx + args.horizon, len(cal) - 1)
    real_start = cal[adjusted_start]
    real_end = cal[adjusted_end]

    print(f"加载数据: {args.market}, {real_start.date()} ~ {real_end.date()}")

    # 加载数据
    data_df = QlibDataLoader(config=data_fields_qlib).load(
        args.market, real_start, real_end
    )
    data_df = data_df.stack().unstack(level=1)

    # 按股票处理
    symbol_list = list(data_df.columns)
    print(f"处理 {len(symbol_list)} 只股票...")

    all_data = {}
    min_len = args.lookback + args.horizon + 1

    for i, symbol in enumerate(symbol_list):
        if (i + 1) % 100 == 0:
            print(f"  进度: {i+1}/{len(symbol_list)}")

        sym_df = data_df[symbol]
        sym_df = sym_df.reset_index().rename(columns={"level_1": "field"})
        sym_df = pd.pivot(sym_df, index="datetime", columns="field", values=symbol)
        sym_df = sym_df.rename(columns={f"${f}": f for f in data_fields})

        # 计算 vol 和 amt
        sym_df["vol"] = sym_df["volume"]
        sym_df["amt"] = (
            (sym_df["open"] + sym_df["high"] + sym_df["low"] + sym_df["close"]) / 4
            * sym_df["vol"]
        )
        sym_df = sym_df[feature_list]
        sym_df = sym_df.dropna()

        if len(sym_df) < min_len:
            continue

        all_data[symbol] = sym_df

    print(f"有效股票: {len(all_data)} 只")

    # 切分数据集
    train_end = args.train_end or "2022-12-31"
    val_start = args.val_start or "2022-09-01"
    val_end = args.val_end or "2024-06-30"
    test_start = args.test_start or "2024-04-01"

    train_data, val_data, test_data = {}, {}, {}
    for symbol, df in all_data.items():
        t_mask = (df.index >= args.start) & (df.index <= train_end)
        v_mask = (df.index >= val_start) & (df.index <= val_end)
        te_mask = (df.index >= test_start) & (df.index <= args.end)

        if t_mask.sum() >= min_len:
            train_data[symbol] = df[t_mask]
        if v_mask.sum() > 0:
            val_data[symbol] = df[v_mask]
        if te_mask.sum() > 0:
            test_data[symbol] = df[te_mask]

    # 保存
    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)

    for name, data in [("train", train_data), ("val", val_data), ("test", test_data)]:
        path = os.path.join(output_dir, f"{name}_data.pkl")
        with open(path, "wb") as f:
            pickle.dump(data, f)
        n_symbols = len(data)
        total_rows = sum(len(df) for df in data.values())
        print(f"  {name}: {n_symbols} 只股票, {total_rows} 行 -> {path}")

    # 保存元信息
    meta = {
        "market": args.market,
        "start": args.start,
        "end": args.end,
        "train_end": train_end,
        "val_start": val_start,
        "val_end": val_end,
        "test_start": test_start,
        "lookback": args.lookback,
        "horizon": args.horizon,
        "feature_list": feature_list,
        "n_train_symbols": len(train_data),
        "n_val_symbols": len(val_data),
        "n_test_symbols": len(test_data),
    }
    import json
    with open(os.path.join(output_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"\n元信息已保存 -> {os.path.join(output_dir, 'meta.json')}")
    print("数据准备完成。")


# ===================================================================
# 状态查看
# ===================================================================

def cmd_status(args):
    """查看已有数据集的状态。"""
    data_path = args.data_path

    if not os.path.isdir(data_path):
        print(f"数据目录不存在: {data_path}")
        return

    # 元信息
    meta_path = os.path.join(data_path, "meta.json")
    if os.path.exists(meta_path):
        import json
        with open(meta_path) as f:
            meta = json.load(f)
        print("数据集元信息:")
        for k, v in meta.items():
            print(f"  {k}: {v}")
        print()

    # 各数据集大小
    for name in ["train", "val", "test"]:
        path = os.path.join(data_path, f"{name}_data.pkl")
        if os.path.exists(path):
            size_mb = os.path.getsize(path) / 1024 / 1024
            with open(path, "rb") as f:
                data = pickle.load(f)
            n_symbols = len(data)
            total_rows = sum(len(df) for df in data.values())
            avg_len = total_rows / n_symbols if n_symbols > 0 else 0
            print(f"  {name}: {n_symbols} 只股票, {total_rows} 行, 平均 {avg_len:.0f} 行/股, {size_mb:.1f} MB")
        else:
            print(f"  {name}: 文件不存在")

    # 预训练模型
    print()
    for name in ["tokenizer", "predictor"]:
        model_dir = os.path.join(data_path, "..", "models", name)
        if os.path.isdir(model_dir):
            print(f"  本地模型 {name}: {model_dir}")
        else:
            print(f"  本地模型 {name}: 未下载")

    # HuggingFace 缓存
    hf_cache = Path.home() / ".cache" / "huggingface" / "hub"
    if hf_cache.exists():
        kronos_models = [d for d in hf_cache.iterdir() if "kronos" in d.name.lower()]
        if kronos_models:
            print(f"\n  HuggingFace 缓存中的 Kronos 模型:")
            for m in kronos_models:
                size = sum(f.stat().st_size for f in m.rglob("*") if f.is_file()) / 1024 / 1024
                print(f"    {m.name}: {size:.1f} MB")


def main():
    parser = argparse.ArgumentParser(description="Kronos 数据准备")
    sub = parser.add_subparsers(dest="command")

    # download-models
    p_dl = sub.add_parser("download-models", help="下载预训练模型")
    p_dl.add_argument(
        "--tokenizer", default="NeoQuasar/Kronos-Tokenizer-base",
        help="Tokenizer 模型路径/ID",
    )
    p_dl.add_argument(
        "--predictor", default="NeoQuasar/Kronos-base",
        help="Predictor 模型路径/ID",
    )
    p_dl.add_argument(
        "--save-dir", default=str(_QUANTLAB_DIR.parent / "data" / "models"),
        help="模型保存目录",
    )

    # prepare
    p_prep = sub.add_parser("prepare", help="预处理 Qlib 数据为训练集")
    p_prep.add_argument("--data-dir", default=QLIB_DATA_DIR, help="Qlib 数据目录")
    p_prep.add_argument("--market", default="csi300", help="股票池")
    p_prep.add_argument("--start", default="2011-01-01", help="数据起始日期")
    p_prep.add_argument("--end", default="2025-01-01", help="数据截止日期")
    p_prep.add_argument("--train-end", default=None, help="训练集截止日期（默认 2022-12-31）")
    p_prep.add_argument("--val-start", default=None, help="验证集起始日期（默认 2022-09-01）")
    p_prep.add_argument("--val-end", default=None, help="验证集截止日期（默认 2024-06-30）")
    p_prep.add_argument("--test-start", default=None, help="测试集起始日期（默认 2024-04-01）")
    p_prep.add_argument("--lookback", type=int, default=90, help="回看窗口天数")
    p_prep.add_argument("--horizon", type=int, default=10, help="预测窗口天数")
    p_prep.add_argument("--output", default=DEFAULT_OUTPUT, help="输出目录")

    # status
    p_st = sub.add_parser("status", help="查看数据集状态")
    p_st.add_argument("--data-path", default=DEFAULT_OUTPUT, help="数据目录")

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return

    cmds = {
        "download-models": cmd_download_models,
        "prepare": cmd_prepare,
        "status": cmd_status,
    }
    cmds[args.command](args)


if __name__ == "__main__":
    main()
