"""M3 Kronos 信号管线：微调实验管理 + 每日微调推理 + 信号输出。

核心类：
    FinetuneRecipe         微调方案定义
    RecipeReport           方案评估报告
    KronosOutput           推理输出
    KronosFinetuner        微调执行器
    KronosInference        推理执行器
    FinetuneExperiment     实验管理（对比多方案）
    KronosSignalPipeline   信号生产（日常使用入口）
"""

import copy
import hashlib
import logging
import math
import time
from dataclasses import dataclass, field, asdict, fields
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------


@dataclass
class FinetuneRecipe:
    """完整描述 '如何微调 Kronos' 的方案定义。"""

    name: str = "default"
    description: str = ""

    # === 模块选择 ===
    finetune_tokenizer: bool = False
    finetune_predictor: bool = True

    # === 冻结策略 ===
    tokenizer_strategy: str = "none"       # none | last_n | head_only | full
    tokenizer_unfreeze_layers: int = 0
    predictor_strategy: str = "last_n"     # none | last_n | head_only | full
    predictor_unfreeze_layers: int = 2

    # === 损失配置 — Tokenizer ===
    recon_pre_weight: float = 1.0
    recon_full_weight: float = 1.0
    bsq_weight: float = 1.0
    bsq_beta: float = 0.05
    bsq_gamma0: float = 1.0
    bsq_gamma: float = 1.1

    # === 损失配置 — Predictor ===
    s1_loss_weight: float = 1.0
    s2_loss_weight: float = 1.0

    # === 数据采样 ===
    data_lookback: int = 30
    sample_strategy: str = "uniform"  # uniform | recency_weighted | volatility_stratified
    recency_decay: float = 0.95

    # === 训练超参 ===
    epochs: int = 3
    learning_rate: float = 2e-5
    batch_size: int = 64
    weight_decay: float = 0.1
    accumulation_steps: int = 1
    warmup_ratio: float = 0.1

    # === 推理超参 ===
    predict_horizon: int = 5
    sample_count: int = 10
    temperature: float = 0.6
    top_p: float = 0.9
    top_k: int = 0

    # === 每日重置 ===
    reset_from_pretrained: bool = True

    def __post_init__(self):
        self._validate()

    def _validate(self):
        valid_strategies = {"none", "last_n", "head_only", "full"}
        if self.tokenizer_strategy not in valid_strategies:
            raise ValueError(f"tokenizer_strategy must be one of {valid_strategies}")
        if self.predictor_strategy not in valid_strategies:
            raise ValueError(f"predictor_strategy must be one of {valid_strategies}")
        valid_sample = {"uniform", "recency_weighted", "volatility_stratified"}
        if self.sample_strategy not in valid_sample:
            raise ValueError(f"sample_strategy must be one of {valid_sample}")
        if self.epochs < 0:
            raise ValueError("epochs must be >= 0")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be > 0")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be > 0")

    @classmethod
    def load(cls, yaml_path: str, name: str) -> "FinetuneRecipe":
        """从 YAML 文件加载指定名称的方案。"""
        all_recipes = cls.load_all(yaml_path)
        for r in all_recipes:
            if r.name == name:
                return r
        available = [r.name for r in all_recipes]
        raise KeyError(f"Recipe '{name}' not found. Available: {available}")

    @classmethod
    def load_all(cls, yaml_path: str) -> List["FinetuneRecipe"]:
        """从 YAML 文件加载全部方案。"""
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        recipes = []
        for item in data.get("recipes", []):
            # 只保留 FinetuneRecipe 有的字段
            valid_keys = {f.name for f in fields(cls)}
            filtered = {k: v for k, v in item.items() if k in valid_keys}
            recipes.append(cls(**filtered))
        return recipes

    def save(self, yaml_path: str):
        """保存单个方案到 YAML。"""
        path = Path(yaml_path)
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
        else:
            data = {}
        recipes_list = data.get("recipes", [])
        # 替换或追加
        d = asdict(self)
        replaced = False
        for i, r in enumerate(recipes_list):
            if r.get("name") == self.name:
                recipes_list[i] = d
                replaced = True
                break
        if not replaced:
            recipes_list.append(d)
        data["recipes"] = recipes_list
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, allow_unicode=True, default_flow_style=False)


@dataclass
class RecipeReport:
    """方案评估报告。"""
    recipe_name: str = ""
    # 预测质量
    ic_mean: float = 0.0
    icir: float = 0.0
    ic_1d: float = 0.0
    ic_5d: float = 0.0
    # 回测绩效
    long_short_sharpe: float = 0.0
    long_only_return: float = 0.0
    # 计算效率
    finetune_time_sec: float = 0.0
    predict_time_sec: float = 0.0
    total_time_sec: float = 0.0
    # 稳定性
    ic_std: float = 0.0
    worst_week_ic: float = 0.0


@dataclass
class KronosOutput:
    """Kronos 推理输出。"""
    return_1d: pd.Series = field(default_factory=pd.Series)      # symbol → float
    return_5d: pd.Series = field(default_factory=pd.Series)      # symbol → float
    uncertainty: pd.Series = field(default_factory=pd.Series)     # symbol → float
    pred_klines: Dict[str, pd.DataFrame] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# KronosFinetuner — 微调执行器
# ---------------------------------------------------------------------------


class KronosFinetuner:
    """按给定 FinetuneRecipe 对 Kronos 模型执行一次微调。"""

    def __init__(self, base_tokenizer, base_model, device: str = "cuda"):
        """
        Args:
            base_tokenizer: 预训练 KronosTokenizer 实例
            base_model: 预训练 Kronos 实例
            device: 计算设备
        """
        import torch
        self.device = device
        # 保存预训练权重的副本（不会被修改）
        self.base_tokenizer_state = copy.deepcopy(base_tokenizer.state_dict())
        self.base_model_state = copy.deepcopy(base_model.state_dict())
        self.base_tokenizer = base_tokenizer
        self.base_model = base_model

        # 累积微调模式下保存上一轮权重
        self._last_tokenizer_state = None
        self._last_model_state = None

    def finetune(
        self,
        symbol_data: Dict[str, pd.DataFrame],
        recipe: FinetuneRecipe,
    ) -> Tuple:
        """
        执行微调，返回微调后的 (tokenizer, model)。

        Args:
            symbol_data: {symbol: DataFrame[open,high,low,close,volume,amount]}
                         每个 DataFrame 包含最近 data_lookback 天的 OHLCV
            recipe: 微调方案

        Returns:
            (tokenizer, model) 微调后的模型
        """
        import torch
        import torch.nn as nn
        import torch.nn.functional as F

        # 如果不需要微调任何模块，直接返回预训练模型
        if not recipe.finetune_tokenizer and not recipe.finetune_predictor:
            tokenizer = copy.deepcopy(self.base_tokenizer)
            model = copy.deepcopy(self.base_model)
            return tokenizer, model

        # 1. 准备模型
        tokenizer = copy.deepcopy(self.base_tokenizer)
        model = copy.deepcopy(self.base_model)

        if recipe.reset_from_pretrained:
            tokenizer.load_state_dict(copy.deepcopy(self.base_tokenizer_state))
            model.load_state_dict(copy.deepcopy(self.base_model_state))
        else:
            # 累积模式：使用上一次微调后的权重
            if self._last_tokenizer_state is not None:
                tokenizer.load_state_dict(copy.deepcopy(self._last_tokenizer_state))
            if self._last_model_state is not None:
                model.load_state_dict(copy.deepcopy(self._last_model_state))

        tokenizer = tokenizer.to(self.device)
        model = model.to(self.device)

        # 2. 应用冻结策略
        self._apply_freeze(tokenizer, model, recipe)

        # 3. 构建数据集
        dataset = self._build_dataset(symbol_data, recipe)
        if len(dataset) == 0:
            logger.warning("微调数据集为空，跳过微调")
            return tokenizer, model

        # 4. 构建采样器
        sampler = self._build_sampler(dataset, recipe)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=recipe.batch_size,
            sampler=sampler,
            drop_last=True,
        )

        # 5. 配置优化器
        trainable_params = []
        if recipe.finetune_tokenizer:
            trainable_params += [p for p in tokenizer.parameters() if p.requires_grad]
        if recipe.finetune_predictor:
            trainable_params += [p for p in model.parameters() if p.requires_grad]

        if not trainable_params:
            logger.warning("没有可训练参数，跳过微调")
            return tokenizer, model

        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=recipe.learning_rate,
            weight_decay=recipe.weight_decay,
        )

        # 学习率调度：warmup + cosine decay
        total_steps = len(dataloader) * recipe.epochs // recipe.accumulation_steps
        warmup_steps = max(1, int(total_steps * recipe.warmup_ratio))
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: _lr_schedule(step, warmup_steps, total_steps),
        )

        # 6. 训练循环
        tokenizer.train()
        model.train()
        global_step = 0

        for epoch in range(recipe.epochs):
            epoch_loss = 0.0
            for batch_idx, batch in enumerate(dataloader):
                x_batch = batch["x"].to(self.device)           # (B, seq_len, 6)
                x_stamp = batch["x_stamp"].to(self.device)     # (B, seq_len, 5)

                loss = torch.tensor(0.0, device=self.device)

                # Tokenizer 损失
                if recipe.finetune_tokenizer:
                    (z_pre, z_full), bsq_loss, quantized, z_indices = tokenizer(x_batch)
                    recon_pre_loss = F.mse_loss(z_pre, x_batch)
                    recon_full_loss = F.mse_loss(z_full, x_batch)
                    tok_loss = (
                        recon_pre_loss * recipe.recon_pre_weight
                        + recon_full_loss * recipe.recon_full_weight
                        + bsq_loss * recipe.bsq_weight
                    )
                    loss = loss + tok_loss
                    # 编码 token 用于 predictor
                    with torch.no_grad():
                        tokens = tokenizer.encode(x_batch, half=True)
                else:
                    with torch.no_grad():
                        tokens = tokenizer.encode(x_batch, half=True)

                # Predictor 损失
                if recipe.finetune_predictor:
                    s1_ids, s2_ids = tokens[0], tokens[1]
                    # 用 teacher forcing 训练
                    s1_logits, s2_logits = model(
                        s1_ids[:, :-1], s2_ids[:, :-1],
                        stamp=x_stamp[:, :-1],
                        use_teacher_forcing=True,
                        s1_targets=s1_ids[:, 1:],
                    )
                    s1_targets = s1_ids[:, 1:]
                    s2_targets = s2_ids[:, 1:]
                    s1_ce = F.cross_entropy(
                        s1_logits.reshape(-1, s1_logits.size(-1)),
                        s1_targets.reshape(-1),
                    )
                    s2_ce = F.cross_entropy(
                        s2_logits.reshape(-1, s2_logits.size(-1)),
                        s2_targets.reshape(-1),
                    )
                    pred_loss = s1_ce * recipe.s1_loss_weight + s2_ce * recipe.s2_loss_weight
                    loss = loss + pred_loss

                loss = loss / recipe.accumulation_steps
                loss.backward()

                if (batch_idx + 1) % recipe.accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1

                epoch_loss += loss.item() * recipe.accumulation_steps

            avg_loss = epoch_loss / max(len(dataloader), 1)
            logger.info(f"Epoch {epoch+1}/{recipe.epochs}, loss={avg_loss:.4f}")

        # 保存累积模式权重
        self._last_tokenizer_state = copy.deepcopy(tokenizer.state_dict())
        self._last_model_state = copy.deepcopy(model.state_dict())

        tokenizer.eval()
        model.eval()
        return tokenizer, model

    def _apply_freeze(self, tokenizer, model, recipe: FinetuneRecipe):
        """根据 recipe 的冻结策略设置 requires_grad。"""
        # 先全部冻结
        for p in tokenizer.parameters():
            p.requires_grad = False
        for p in model.parameters():
            p.requires_grad = False

        # 解冻 tokenizer
        if recipe.finetune_tokenizer:
            self._unfreeze_module(
                tokenizer, recipe.tokenizer_strategy,
                recipe.tokenizer_unfreeze_layers, module_type="tokenizer",
            )

        # 解冻 predictor
        if recipe.finetune_predictor:
            self._unfreeze_module(
                model, recipe.predictor_strategy,
                recipe.predictor_unfreeze_layers, module_type="predictor",
            )

    def _unfreeze_module(self, module, strategy: str, unfreeze_layers: int, module_type: str):
        """根据策略解冻模块参数。"""
        if strategy == "none":
            return  # 全部保持冻结

        if strategy == "full":
            for p in module.parameters():
                p.requires_grad = True
            return

        if strategy == "head_only":
            if module_type == "tokenizer":
                # 解冻 tokenizer 的 decoder head
                for name, p in module.named_parameters():
                    if "head" in name or "post_quant" in name:
                        p.requires_grad = True
            else:
                # 解冻 predictor 的 DualHead + DependencyAwareLayer
                for name, p in module.named_parameters():
                    if "head" in name or "dep_layer" in name:
                        p.requires_grad = True
            return

        if strategy == "last_n":
            n = unfreeze_layers
            if module_type == "tokenizer":
                # 解冻最后 N 个 decoder block + head
                decoder_blocks = list(module.decoder)
                for block in decoder_blocks[-n:]:
                    for p in block.parameters():
                        p.requires_grad = True
                for name, p in module.named_parameters():
                    if "head" in name or "post_quant" in name:
                        p.requires_grad = True
            else:
                # 解冻最后 N 个 transformer block + DualHead + DependencyAwareLayer
                transformer_blocks = list(module.transformer)
                for block in transformer_blocks[-n:]:
                    for p in block.parameters():
                        p.requires_grad = True
                for name, p in module.named_parameters():
                    if "head" in name or "dep_layer" in name or "norm" in name:
                        p.requires_grad = True

    def _build_dataset(self, symbol_data: Dict[str, pd.DataFrame], recipe: FinetuneRecipe):
        """构建微调数据集：每只股票取滑动窗口片段。"""
        import torch
        from torch.utils.data import TensorDataset

        cols = ["open", "high", "low", "close", "volume", "amount"]
        all_x = []
        all_stamps = []
        min_seq_len = 20  # 最小有效序列长度

        for sym, df in symbol_data.items():
            if len(df) < min_seq_len:
                continue
            df = df.tail(recipe.data_lookback).copy()
            if len(df) < min_seq_len:
                continue

            # 提取 OHLCV
            values = df[cols].values.astype(np.float32)
            # per-series z-score 归一化
            mean = values.mean(axis=0)
            std = values.std(axis=0) + 1e-5
            values = (values - mean) / std
            values = np.clip(values, -5, 5)

            # 时间戳特征
            if hasattr(df.index, 'minute'):
                stamps = np.stack([
                    df.index.minute if hasattr(df.index, 'minute') else np.zeros(len(df)),
                    df.index.hour if hasattr(df.index, 'hour') else np.zeros(len(df)),
                    df.index.weekday,
                    df.index.day,
                    df.index.month,
                ], axis=1).astype(np.float32)
            else:
                # 尝试从 index 或 date 列提取
                try:
                    idx = pd.DatetimeIndex(df.index)
                except Exception:
                    idx = pd.DatetimeIndex(pd.date_range("2024-01-01", periods=len(df), freq="B"))
                stamps = np.stack([
                    np.zeros(len(df)),  # minute
                    np.zeros(len(df)),  # hour
                    idx.weekday,
                    idx.day,
                    idx.month,
                ], axis=1).astype(np.float32)

            all_x.append(values)
            all_stamps.append(stamps)

        if not all_x:
            return TensorDataset(
                torch.zeros(0, min_seq_len, 6),
                torch.zeros(0, min_seq_len, 5),
            )

        # 填充到相同长度
        max_len = max(x.shape[0] for x in all_x)
        padded_x = []
        padded_stamps = []
        for x, s in zip(all_x, all_stamps):
            pad_len = max_len - x.shape[0]
            if pad_len > 0:
                x = np.pad(x, ((pad_len, 0), (0, 0)), mode="edge")
                s = np.pad(s, ((pad_len, 0), (0, 0)), mode="edge")
            padded_x.append(x)
            padded_stamps.append(s)

        x_tensor = torch.tensor(np.array(padded_x), dtype=torch.float32)
        stamp_tensor = torch.tensor(np.array(padded_stamps), dtype=torch.float32)

        return _DictDataset(x_tensor, stamp_tensor)

    def _build_sampler(self, dataset, recipe: FinetuneRecipe):
        """根据采样策略构建采样器。"""
        import torch

        n = len(dataset)
        if n == 0:
            return None

        if recipe.sample_strategy == "uniform":
            return torch.utils.data.RandomSampler(dataset)

        if recipe.sample_strategy == "recency_weighted":
            # 假设数据按添加顺序排列，越后面越近期
            weights = np.array([recipe.recency_decay ** (n - 1 - i) for i in range(n)])
            weights = weights / weights.sum()
            return torch.utils.data.WeightedRandomSampler(
                weights=weights.tolist(),
                num_samples=n,
                replacement=True,
            )

        if recipe.sample_strategy == "volatility_stratified":
            # 按波动率分层：计算每个样本的标准差作为波动率代理
            vols = []
            for i in range(n):
                item = dataset[i]
                x = item["x"]  # (seq_len, 6)
                # close 列（index 3）的标准差
                vol = x[:, 3].std().item()
                vols.append(vol)
            vols = np.array(vols)

            # 分为3组
            terciles = np.quantile(vols, [1/3, 2/3])
            group = np.digitize(vols, terciles)  # 0, 1, 2
            weights = np.zeros(n)
            for g in range(3):
                mask = group == g
                count = mask.sum()
                if count > 0:
                    weights[mask] = 1.0 / count
            weights = weights / weights.sum()
            return torch.utils.data.WeightedRandomSampler(
                weights=weights.tolist(),
                num_samples=n,
                replacement=True,
            )

        return torch.utils.data.RandomSampler(dataset)


class _DictDataset:
    """简单的 Dict 风格数据集。"""

    def __init__(self, x: "torch.Tensor", stamps: "torch.Tensor"):
        self.x = x
        self.stamps = stamps

    def __len__(self):
        return self.x.size(0)

    def __getitem__(self, idx):
        return {"x": self.x[idx], "x_stamp": self.stamps[idx]}


# ---------------------------------------------------------------------------
# KronosInference — 推理执行器
# ---------------------------------------------------------------------------


class KronosInference:
    """批量推理执行器。"""

    def __init__(self, device: str = "cuda", max_context: int = 512):
        self.device = device
        self.max_context = max_context

    def predict_all(
        self,
        tokenizer,
        model,
        symbol_data: Dict[str, pd.DataFrame],
        recipe: FinetuneRecipe,
        anchor_date: str,
    ) -> KronosOutput:
        """
        批量推理全市场。

        Args:
            tokenizer: 微调后的 KronosTokenizer
            model: 微调后的 Kronos
            symbol_data: {symbol: DataFrame[date, open, high, low, close, volume, amount]}
            recipe: 推理超参来源
            anchor_date: 锚定日期

        Returns:
            KronosOutput
        """
        from quantlab._compat import import_kronos_module
        kronos_mod = import_kronos_module("model.kronos")
        KronosPredictor = kronos_mod.KronosPredictor

        predictor = KronosPredictor(
            model, tokenizer,
            device=self.device,
            max_context=self.max_context,
        )

        anchor_dt = pd.Timestamp(anchor_date)
        pred_horizon = recipe.predict_horizon

        # 准备批量输入
        symbols = []
        df_list = []
        x_timestamp_list = []
        y_timestamp_list = []

        cols = ["open", "high", "low", "close", "volume", "amount"]

        for sym, df in symbol_data.items():
            # 取 anchor_date 之前的数据
            if isinstance(df.index, pd.DatetimeIndex):
                df_before = df[df.index <= anchor_dt]
            elif "date" in df.columns:
                df_before = df[pd.to_datetime(df["date"]) <= anchor_dt]
            else:
                df_before = df

            if len(df_before) < 20:
                continue

            # 取最近 max_context 天
            lookback = min(len(df_before), self.max_context)
            df_slice = df_before.tail(lookback).copy()

            # 确保有所需列
            missing = [c for c in cols if c not in df_slice.columns]
            if missing:
                continue

            # x_timestamp
            if isinstance(df_slice.index, pd.DatetimeIndex):
                x_ts = pd.Series(df_slice.index)
            elif "date" in df_slice.columns:
                x_ts = pd.Series(pd.to_datetime(df_slice["date"]).values)
            else:
                x_ts = pd.Series(pd.bdate_range(
                    end=anchor_dt, periods=lookback,
                ))

            # y_timestamp
            y_ts = pd.Series(pd.bdate_range(
                start=anchor_dt + pd.Timedelta(days=1),
                periods=pred_horizon,
            ))

            symbols.append(sym)
            df_list.append(df_slice[cols].reset_index(drop=True))
            x_timestamp_list.append(x_ts.reset_index(drop=True))
            y_timestamp_list.append(y_ts)

        if not symbols:
            return KronosOutput()

        # 分批推理（按序列长度分组，predict_batch 要求同长度）
        return_1d_all = {}
        return_5d_all = {}
        uncertainty_all = {}
        pred_klines_all = {}

        # 按序列长度分组
        length_groups: Dict[int, List[int]] = {}
        for i, df in enumerate(df_list):
            l = len(df)
            length_groups.setdefault(l, []).append(i)

        for seq_len, indices in length_groups.items():
            batch_dfs = [df_list[i] for i in indices]
            batch_x_ts = [x_timestamp_list[i] for i in indices]
            batch_y_ts = [y_timestamp_list[i] for i in indices]
            batch_syms = [symbols[i] for i in indices]

            # 多次采样推理
            all_sample_preds = []
            for s in range(recipe.sample_count):
                try:
                    pred_dfs = predictor.predict_batch(
                        batch_dfs, batch_x_ts, batch_y_ts,
                        pred_len=pred_horizon,
                        T=recipe.temperature,
                        top_p=recipe.top_p,
                        top_k=recipe.top_k,
                        sample_count=1,
                        verbose=False,
                    )
                    all_sample_preds.append(pred_dfs)
                except Exception as e:
                    logger.warning(f"推理采样 {s+1} 失败: {e}")

            if not all_sample_preds:
                continue

            # 聚合
            for j, sym in enumerate(batch_syms):
                last_close = batch_dfs[j]["close"].iloc[-1]
                if last_close == 0 or np.isnan(last_close):
                    continue

                sample_returns_1d = []
                sample_returns_5d = []

                for sample_preds in all_sample_preds:
                    pred_df = sample_preds[j]
                    # T+1 收益
                    r1d = pred_df["close"].iloc[0] / last_close - 1
                    sample_returns_1d.append(r1d)
                    # T+1~T+5 平均收益
                    r5d = pred_df["close"].mean() / last_close - 1
                    sample_returns_5d.append(r5d)

                return_1d_all[sym] = np.mean(sample_returns_1d)
                return_5d_all[sym] = np.mean(sample_returns_5d)
                uncertainty_all[sym] = np.std(sample_returns_5d)

                # 保留最后一次采样的预测 K 线
                pred_klines_all[sym] = all_sample_preds[-1][j]

        return KronosOutput(
            return_1d=pd.Series(return_1d_all),
            return_5d=pd.Series(return_5d_all),
            uncertainty=pd.Series(uncertainty_all),
            pred_klines=pred_klines_all,
        )


# ---------------------------------------------------------------------------
# FinetuneExperiment — 实验管理
# ---------------------------------------------------------------------------


class FinetuneExperiment:
    """对比多个 FinetuneRecipe 的效果，找出最优方案。"""

    def __init__(
        self,
        data_manager,
        recipes: List[FinetuneRecipe],
        eval_start: str = "2024-07-01",
        eval_end: str = "2024-12-31",
        tokenizer_path: str = "NeoQuasar/Kronos-Tokenizer-base",
        model_path: str = "NeoQuasar/Kronos-base",
        device: str = "cuda",
        max_context: int = 512,
    ):
        self.dm = data_manager
        self.recipes = recipes
        self.eval_start = eval_start
        self.eval_end = eval_end
        self.tokenizer_path = tokenizer_path
        self.model_path = model_path
        self.device = device
        self.max_context = max_context
        self.reports: List[RecipeReport] = []

    def _load_base_models(self):
        """加载预训练模型。"""
        from quantlab._compat import import_kronos_module
        kronos_mod = import_kronos_module("model.kronos")
        KronosTokenizer, Kronos = kronos_mod.KronosTokenizer, kronos_mod.Kronos

        tokenizer = KronosTokenizer.from_pretrained(self.tokenizer_path)
        model = Kronos.from_pretrained(self.model_path)
        return tokenizer, model

    def run_experiment(self, recipe: FinetuneRecipe) -> RecipeReport:
        """在 eval_period 上运行一个方案，返回评估报告。"""
        from scipy import stats

        base_tokenizer, base_model = self._load_base_models()
        finetuner = KronosFinetuner(base_tokenizer, base_model, device=self.device)
        inference = KronosInference(device=self.device, max_context=self.max_context)

        cal = self.dm.get_trading_calendar(self.eval_start, self.eval_end)
        if len(cal) == 0:
            return RecipeReport(recipe_name=recipe.name)

        daily_ics = []
        daily_signals = {}
        total_ft_time = 0.0
        total_pred_time = 0.0

        for date in cal:
            date_str = date.strftime("%Y-%m-%d") if hasattr(date, "strftime") else str(date)

            # 获取全市场数据
            symbol_data = self._get_symbol_data(date_str, recipe.data_lookback)
            if not symbol_data:
                continue

            # 微调
            t0 = time.time()
            tok, mdl = finetuner.finetune(symbol_data, recipe)
            ft_time = time.time() - t0
            total_ft_time += ft_time

            # 推理
            t0 = time.time()
            output = inference.predict_all(tok, mdl, symbol_data, recipe, date_str)
            pred_time = time.time() - t0
            total_pred_time += pred_time

            if output.return_1d.empty:
                continue

            daily_signals[date_str] = output.return_1d

            # 计算与次日实际收益的 Rank IC
            actual_ret = self._get_next_day_return(date_str)
            if actual_ret is not None:
                common = output.return_1d.index.intersection(actual_ret.index)
                if len(common) >= 30:
                    ic, _ = stats.spearmanr(
                        output.return_1d[common].values,
                        actual_ret[common].values,
                    )
                    if not np.isnan(ic):
                        daily_ics.append(ic)

        # 汇总
        report = RecipeReport(recipe_name=recipe.name)
        if daily_ics:
            ic_arr = np.array(daily_ics)
            report.ic_mean = float(np.mean(ic_arr))
            report.ic_std = float(np.std(ic_arr))
            report.icir = float(np.mean(ic_arr) / (np.std(ic_arr) + 1e-8))
            report.ic_1d = report.ic_mean  # 日频 IC
            # 最差一周 IC
            if len(ic_arr) >= 5:
                weekly_ics = [np.mean(ic_arr[i:i+5]) for i in range(0, len(ic_arr)-4)]
                report.worst_week_ic = float(min(weekly_ics))
            # 简化多空夏普
            if len(daily_signals) >= 5:
                report.long_short_sharpe = self._calc_long_short_sharpe(daily_signals)

        n_days = max(len(cal), 1)
        report.finetune_time_sec = total_ft_time / n_days
        report.predict_time_sec = total_pred_time / n_days
        report.total_time_sec = (total_ft_time + total_pred_time) / n_days

        return report

    def run_all(self) -> List[RecipeReport]:
        """逐个运行全部方案。"""
        self.reports = []
        for recipe in self.recipes:
            logger.info(f"运行实验: {recipe.name}")
            report = self.run_experiment(recipe)
            self.reports.append(report)
            logger.info(f"  IC={report.ic_mean:.4f}, ICIR={report.icir:.2f}")
        return self.reports

    def compare(self) -> pd.DataFrame:
        """横向对比全部方案的核心指标。"""
        if not self.reports:
            return pd.DataFrame()
        rows = []
        for r in self.reports:
            rows.append({
                "recipe": r.recipe_name,
                "ic_mean": r.ic_mean,
                "icir": r.icir,
                "ic_1d": r.ic_1d,
                "long_short_sharpe": r.long_short_sharpe,
                "total_time_sec": r.total_time_sec,
                "ic_std": r.ic_std,
                "worst_week_ic": r.worst_week_ic,
            })
        return pd.DataFrame(rows).set_index("recipe")

    def get_best_recipe(self) -> FinetuneRecipe:
        """返回 ICIR 最优的方案。"""
        if not self.reports:
            raise RuntimeError("请先运行 run_all()")
        best_idx = max(range(len(self.reports)), key=lambda i: self.reports[i].icir)
        return self.recipes[best_idx]

    def save_results(self, output_dir: str):
        """保存对比表到文件。"""
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        df = self.compare()
        df.to_csv(path / "recipe_comparison.csv")
        # 保存每个报告
        for r in self.reports:
            with open(path / f"report_{r.recipe_name}.yaml", "w") as f:
                yaml.dump(asdict(r) if hasattr(r, '__dataclass_fields__') else vars(r), f)

    def _get_symbol_data(self, date_str: str, lookback: int) -> Dict[str, pd.DataFrame]:
        """获取全市场 OHLCV 数据。"""
        try:
            df = self.dm.get_ohlcv_before(date_str, lookback)
            if df is None or df.empty:
                return {}
            # 按 symbol 分组
            result = {}
            if isinstance(df.index, pd.MultiIndex):
                # (datetime, instrument) 多层索引
                for sym in df.index.get_level_values(1).unique():
                    sym_df = df.xs(sym, level=1).copy()
                    if len(sym_df) >= 20:
                        result[sym] = sym_df
            else:
                # 单一 DataFrame
                if "instrument" in df.columns:
                    for sym, grp in df.groupby("instrument"):
                        if len(grp) >= 20:
                            result[sym] = grp.copy()
            return result
        except Exception as e:
            logger.warning(f"获取数据失败 {date_str}: {e}")
            return {}

    def _get_next_day_return(self, date_str: str) -> Optional[pd.Series]:
        """获取次日收益率。"""
        try:
            close_today = self.dm.get_close_prices(date_str)
            # 获取下一个交易日
            cal = self.dm.get_trading_calendar(date_str, None)
            if cal is None or len(cal) < 2:
                return None
            next_date = cal[1].strftime("%Y-%m-%d") if hasattr(cal[1], "strftime") else str(cal[1])
            close_next = self.dm.get_close_prices(next_date)
            if close_today is None or close_next is None:
                return None
            common = close_today.index.intersection(close_next.index)
            ret = (close_next[common] / close_today[common]) - 1
            return ret
        except Exception:
            return None

    def _calc_long_short_sharpe(self, daily_signals: Dict[str, pd.Series]) -> float:
        """简化多空夏普计算。"""
        try:
            daily_returns = []
            for date_str, signal in sorted(daily_signals.items()):
                actual = self._get_next_day_return(date_str)
                if actual is None:
                    continue
                common = signal.index.intersection(actual.index)
                if len(common) < 20:
                    continue
                # Top 10% long, Bottom 10% short
                n = len(common)
                top_n = max(1, n // 10)
                sorted_syms = signal[common].sort_values(ascending=False)
                long_ret = actual[sorted_syms.index[:top_n]].mean()
                short_ret = actual[sorted_syms.index[-top_n:]].mean()
                daily_returns.append(long_ret - short_ret)
            if not daily_returns:
                return 0.0
            arr = np.array(daily_returns)
            return float(np.mean(arr) / (np.std(arr) + 1e-8) * np.sqrt(252))
        except Exception:
            return 0.0


# ---------------------------------------------------------------------------
# KronosSignalPipeline — 日常使用入口
# ---------------------------------------------------------------------------


class KronosSignalPipeline:
    """日常回测/实盘的入口：使用选定 recipe 执行微调 + 推理。"""

    def __init__(
        self,
        recipe: FinetuneRecipe,
        tokenizer_path: str = "NeoQuasar/Kronos-Tokenizer-base",
        model_path: str = "NeoQuasar/Kronos-base",
        device: str = "cuda",
        max_context: int = 512,
    ):
        self.recipe = recipe
        self.tokenizer_path = tokenizer_path
        self.model_path = model_path
        self.device = device
        self.max_context = max_context
        self._finetuner = None
        self._inference = None
        self._base_tokenizer = None
        self._base_model = None

    def _ensure_loaded(self):
        """延迟加载模型。"""
        if self._base_tokenizer is not None:
            return

        from quantlab._compat import import_kronos_module
        kronos_mod = import_kronos_module("model.kronos")
        KronosTokenizer, Kronos = kronos_mod.KronosTokenizer, kronos_mod.Kronos

        logger.info(f"加载 Kronos 预训练模型: {self.tokenizer_path}, {self.model_path}")
        self._base_tokenizer = KronosTokenizer.from_pretrained(self.tokenizer_path)
        self._base_model = Kronos.from_pretrained(self.model_path)
        self._finetuner = KronosFinetuner(
            self._base_tokenizer, self._base_model, device=self.device,
        )
        self._inference = KronosInference(
            device=self.device, max_context=self.max_context,
        )

    def daily_run(
        self,
        symbol_data: Dict[str, pd.DataFrame],
        anchor_date: str,
        data_manager=None,
    ) -> KronosOutput:
        """
        执行微调 + 推理，返回信号。

        Args:
            symbol_data: {symbol: DataFrame} 全市场 OHLCV
            anchor_date: 锚定日期
            data_manager: M1 DataManager 实例（备用）

        Returns:
            KronosOutput
        """
        self._ensure_loaded()

        # 微调
        logger.info(f"[{anchor_date}] 微调开始 (recipe={self.recipe.name})")
        t0 = time.time()
        tok, mdl = self._finetuner.finetune(symbol_data, self.recipe)
        ft_time = time.time() - t0
        logger.info(f"[{anchor_date}] 微调完成 ({ft_time:.1f}s)")

        # 推理
        logger.info(f"[{anchor_date}] 推理开始")
        t0 = time.time()
        output = self._inference.predict_all(
            tok, mdl, symbol_data, self.recipe, anchor_date,
        )
        pred_time = time.time() - t0
        logger.info(
            f"[{anchor_date}] 推理完成 ({pred_time:.1f}s), "
            f"{len(output.return_1d)} 只股票"
        )

        return output

    def switch_recipe(self, recipe: FinetuneRecipe):
        """切换微调方案。"""
        self.recipe = recipe
        # 重置累积微调状态
        if self._finetuner is not None:
            self._finetuner._last_tokenizer_state = None
            self._finetuner._last_model_state = None
        logger.info(f"已切换到方案: {recipe.name}")


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------


def _lr_schedule(step: int, warmup_steps: int, total_steps: int) -> float:
    """Warmup + cosine decay 学习率调度。"""
    if step < warmup_steps:
        return step / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return 0.5 * (1 + math.cos(math.pi * progress))
