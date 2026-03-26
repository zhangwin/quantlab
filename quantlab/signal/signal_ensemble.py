"""M5 信号融合：Horizon 对齐 + IC 动态加权 + 信号质量监控。

核心类：
    SignalNormalizer          信号预处理与归一化
    DayRecord                 单日历史记录
    ICWeightEngine            IC 动态加权引擎
    PipelineStat              管线贡献度统计
    ContributionReport        贡献度分析报告
    EnsembleMonitor           融合质量监控
    EnsembleOutput            融合输出
    SignalEnsemblePipeline    融合入口
"""

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# SignalNormalizer — 信号预处理
# ---------------------------------------------------------------------------


class SignalNormalizer:
    """将各管线原始信号统一到可比较的尺度。"""

    @staticmethod
    def rank_normalize(signal: pd.Series) -> pd.Series:
        """截面 rank 归一化到 (0, 1]。NaN 保持为 NaN。"""
        if signal.empty:
            return signal.copy()
        return signal.rank(pct=True, na_option="keep")

    @staticmethod
    def winsorize(
        signal: pd.Series,
        lower: float = 0.01,
        upper: float = 0.99,
    ) -> pd.Series:
        """缩尾处理：将超出分位数边界的值截断。"""
        if signal.empty:
            return signal.copy()
        s = signal.copy()
        valid = s.dropna()
        if len(valid) < 2:
            return s
        lo = valid.quantile(lower)
        hi = valid.quantile(upper)
        s = s.clip(lower=lo, upper=hi)
        return s

    @staticmethod
    def align_index(signals: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
        """对齐股票 index，缺失填 NaN。"""
        if not signals:
            return {}
        all_idx = set()
        for s in signals.values():
            if s is not None and not s.empty:
                all_idx.update(s.index)
        all_idx = sorted(all_idx)
        if not all_idx:
            return signals
        result = {}
        for name, s in signals.items():
            if s is not None and not s.empty:
                result[name] = s.reindex(all_idx)
            else:
                result[name] = pd.Series(np.nan, index=all_idx)
        return result


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------


@dataclass
class DayRecord:
    """单日历史记录（供 ICWeightEngine 使用）。"""
    date: str = ""
    signals: Dict[str, pd.Series] = field(default_factory=dict)
    actual_return: Optional[pd.Series] = None


@dataclass
class PipelineStat:
    """单个管线的贡献度统计。"""
    name: str = ""
    avg_weight: float = 0.0
    avg_ic: float = 0.0
    marginal_ic: float = 0.0
    weight_trend: str = "stable"  # "increasing" | "stable" | "decreasing"


@dataclass
class ContributionReport:
    """各管线贡献度分析报告。"""
    period: str = ""
    pipeline_stats: Dict[str, PipelineStat] = field(default_factory=dict)


@dataclass
class EnsembleOutput:
    """融合输出。"""
    signal: pd.Series = field(default_factory=pd.Series)
    weights: Dict[str, float] = field(default_factory=dict)
    pipeline_ranks: Dict[str, pd.Series] = field(default_factory=dict)
    uncertainty_adjusted: bool = False
    mode: str = "cold_start"  # "ic_weighted" | "cold_start" | "blending"


# ---------------------------------------------------------------------------
# ICWeightEngine — IC 动态加权引擎
# ---------------------------------------------------------------------------


class ICWeightEngine:
    """基于各管线历史预测准确度动态分配权重。

    核心逻辑：
    1. 滚动 Rank IC 评估各管线预测能力
    2. 冷启动 → 混合过渡 → 纯 IC 加权
    3. 相关性感知降权
    4. 权重钳制
    """

    def __init__(
        self,
        ic_lookback: int = 60,
        min_weight: float = 0.1,
        max_weight: float = 0.6,
        correlation_penalty: float = 0.3,
    ):
        self.ic_lookback = ic_lookback
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.correlation_penalty = correlation_penalty
        self.history: deque = deque(maxlen=ic_lookback)

    def update(
        self,
        date: str,
        signals: Dict[str, pd.Series],
        actual_return: pd.Series,
    ):
        """追加一天的历史记录。

        Args:
            date: 信号日期（T-1）
            signals: 各管线当日信号（T-1 生成）
            actual_return: 对应实际收益（T 日收益）
        """
        record = DayRecord(
            date=date,
            signals={k: v.copy() for k, v in signals.items() if v is not None},
            actual_return=actual_return.copy() if actual_return is not None else None,
        )
        self.history.append(record)

    def compute_ic(self, signal_name: str) -> float:
        """计算某管线近 ic_lookback 天的平均 Rank IC。"""
        ics = []
        for record in self.history:
            if signal_name not in record.signals:
                continue
            if record.actual_return is None or record.actual_return.empty:
                continue
            sig = record.signals[signal_name]
            ret = record.actual_return
            common = sig.dropna().index.intersection(ret.dropna().index)
            if len(common) < 10:
                continue
            ic = sp_stats.spearmanr(sig.loc[common], ret.loc[common])[0]
            if not np.isnan(ic):
                ics.append(ic)
        return float(np.mean(ics)) if ics else 0.0

    def compute_weights(self) -> Tuple[Dict[str, float], str]:
        """计算各管线当前权重。

        Returns:
            (weights_dict, mode_string)
        """
        # 收集有效管线名称
        pipeline_names = set()
        for record in self.history:
            pipeline_names.update(record.signals.keys())
        pipeline_names = sorted(pipeline_names)

        if not pipeline_names:
            return {}, "cold_start"

        n = len(pipeline_names)
        n_history = len(self.history)

        # --- 冷启动 ---
        if n_history < 20:
            equal_w = {name: 1.0 / n for name in pipeline_names}
            return equal_w, "cold_start"

        # 计算各管线 IC
        ic_map = {name: self.compute_ic(name) for name in pipeline_names}

        # --- IC 加权 ---
        effective_ic = {name: max(0, ic) for name, ic in ic_map.items()}
        total_ic = sum(effective_ic.values())

        if total_ic <= 0:
            raw_weights = {name: 1.0 / n for name in pipeline_names}
        else:
            raw_weights = {name: effective_ic[name] / total_ic for name in pipeline_names}

        # --- 混合过渡 ---
        if n_history < self.ic_lookback:
            alpha = (n_history - 20) / max(self.ic_lookback - 20, 1)
            alpha = min(max(alpha, 0.0), 1.0)
            equal_w = 1.0 / n
            raw_weights = {
                name: alpha * raw_weights[name] + (1 - alpha) * equal_w
                for name in pipeline_names
            }
            mode = "blending"
        else:
            mode = "ic_weighted"

        # --- 相关性感知调整 ---
        raw_weights = self._apply_correlation_penalty(raw_weights, ic_map)

        # --- 权重钳制 ---
        raw_weights = self._clamp_weights(raw_weights)

        return raw_weights, mode

    def _apply_correlation_penalty(
        self,
        weights: Dict[str, float],
        ic_map: Dict[str, float],
    ) -> Dict[str, float]:
        """信号高度相关时，降低 IC 较低管线的权重。"""
        names = list(weights.keys())
        if len(names) < 2:
            return weights

        # 计算管线间平均截面 Spearman 相关
        corr_pairs = {}
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                corrs = []
                for record in self.history:
                    s_i = record.signals.get(names[i])
                    s_j = record.signals.get(names[j])
                    if s_i is None or s_j is None:
                        continue
                    common = s_i.dropna().index.intersection(s_j.dropna().index)
                    if len(common) < 10:
                        continue
                    c = sp_stats.spearmanr(s_i.loc[common], s_j.loc[common])[0]
                    if not np.isnan(c):
                        corrs.append(abs(c))
                avg_corr = float(np.mean(corrs)) if corrs else 0.0
                corr_pairs[(names[i], names[j])] = avg_corr

        adjusted = dict(weights)
        for (a, b), corr in corr_pairs.items():
            if corr > 0.5:
                # 降低 IC 较低管线的权重
                if ic_map.get(a, 0) >= ic_map.get(b, 0):
                    loser = b
                else:
                    loser = a
                penalty = 1 - self.correlation_penalty * corr
                adjusted[loser] = adjusted[loser] * max(penalty, 0.1)

        # 重新归一化
        total = sum(adjusted.values())
        if total > 0:
            adjusted = {k: v / total for k, v in adjusted.items()}
        return adjusted

    def _clamp_weights(self, weights: Dict[str, float]) -> Dict[str, float]:
        """权重钳制并重新归一化。"""
        clamped = {
            name: min(max(w, self.min_weight), self.max_weight)
            for name, w in weights.items()
        }
        total = sum(clamped.values())
        if total > 0:
            clamped = {k: v / total for k, v in clamped.items()}
        return clamped

    def get_ic_history(self) -> pd.DataFrame:
        """各管线逐日 IC 时序。"""
        rows = []
        for record in self.history:
            if record.actual_return is None or record.actual_return.empty:
                continue
            row = {"date": record.date}
            for name, sig in record.signals.items():
                common = sig.dropna().index.intersection(
                    record.actual_return.dropna().index
                )
                if len(common) >= 10:
                    ic = sp_stats.spearmanr(
                        sig.loc[common], record.actual_return.loc[common]
                    )[0]
                    row[name] = ic if not np.isnan(ic) else np.nan
                else:
                    row[name] = np.nan
            rows.append(row)
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows).set_index("date")
        return df


# ---------------------------------------------------------------------------
# EnsembleMonitor — 融合质量监控
# ---------------------------------------------------------------------------


class EnsembleMonitor:
    """跟踪融合信号和各管线贡献度变化，检测异常。"""

    def __init__(self, max_history: int = 200):
        self._records: deque = deque(maxlen=max_history)

    def record(
        self,
        date: str,
        weights: Dict[str, float],
        combined_signal: pd.Series,
        actual_return: Optional[pd.Series] = None,
    ):
        """每日记录融合结果。"""
        self._records.append({
            "date": date,
            "weights": dict(weights),
            "combined_signal": combined_signal.copy() if combined_signal is not None else pd.Series(),
            "actual_return": actual_return.copy() if actual_return is not None else None,
        })

    def update_actual_return(self, date: str, actual_return: pd.Series):
        """回填某日的实际收益（次日补填）。"""
        for rec in self._records:
            if rec["date"] == date:
                rec["actual_return"] = actual_return.copy()
                return

    def get_contribution_report(self, lookback: int = 60) -> ContributionReport:
        """各管线贡献度分析。"""
        recent = list(self._records)[-lookback:]
        if not recent:
            return ContributionReport(period="empty")

        period = f"{recent[0]['date']} ~ {recent[-1]['date']}"
        report = ContributionReport(period=period)

        # 收集管线名称
        all_names = set()
        for rec in recent:
            all_names.update(rec["weights"].keys())

        for name in sorted(all_names):
            stat = PipelineStat(name=name)

            # 平均权重
            ws = [rec["weights"].get(name, 0) for rec in recent]
            stat.avg_weight = float(np.mean(ws))

            # 权重趋势
            if len(ws) >= 10:
                first_half = np.mean(ws[:len(ws) // 2])
                second_half = np.mean(ws[len(ws) // 2:])
                if second_half > first_half * 1.1:
                    stat.weight_trend = "increasing"
                elif second_half < first_half * 0.9:
                    stat.weight_trend = "decreasing"
                else:
                    stat.weight_trend = "stable"

            # 平均 IC（融合信号 vs 实际收益）
            ics = []
            for rec in recent:
                ret = rec.get("actual_return")
                sig = rec.get("combined_signal")
                if ret is None or sig is None or ret.empty or sig.empty:
                    continue
                common = sig.dropna().index.intersection(ret.dropna().index)
                if len(common) >= 10:
                    ic = sp_stats.spearmanr(sig.loc[common], ret.loc[common])[0]
                    if not np.isnan(ic):
                        ics.append(ic)
            stat.avg_ic = float(np.mean(ics)) if ics else 0.0

            # 边际贡献 IC：简化为 avg_weight × avg_ic
            stat.marginal_ic = stat.avg_weight * stat.avg_ic

            report.pipeline_stats[name] = stat

        return report

    def check_anomaly(self) -> List[str]:
        """返回告警消息列表。"""
        alerts = []
        recent = list(self._records)

        if not recent:
            return alerts

        # 1. 融合 IC 连续 10 天 < 0
        last_10 = recent[-10:] if len(recent) >= 10 else []
        if last_10:
            neg_count = 0
            for rec in last_10:
                ret = rec.get("actual_return")
                sig = rec.get("combined_signal")
                if ret is None or sig is None or ret.empty or sig.empty:
                    continue
                common = sig.dropna().index.intersection(ret.dropna().index)
                if len(common) >= 10:
                    ic = sp_stats.spearmanr(sig.loc[common], ret.loc[common])[0]
                    if not np.isnan(ic) and ic < 0:
                        neg_count += 1
            if neg_count >= 10:
                alerts.append(
                    "WARN: Combined signal IC negative for 10 consecutive days"
                )

        # 2. 某管线权重连续 20 天 = min_weight
        if len(recent) >= 20:
            all_names = set()
            for rec in recent:
                all_names.update(rec["weights"].keys())
            last_20 = recent[-20:]
            for name in all_names:
                min_w_count = sum(
                    1 for rec in last_20
                    if rec["weights"].get(name, 0) <= 0.1 + 1e-6
                )
                if min_w_count >= 20:
                    alerts.append(
                        f"WARN: Pipeline {name} at minimum weight for 20 days, "
                        "consider disabling"
                    )

        # 3. 所有管线 IC 同时下降（比较前半段和后半段）
        if len(recent) >= 20:
            half = len(recent) // 2
            first_half = recent[:half]
            second_half = recent[half:]
            all_names = set()
            for rec in recent:
                all_names.update(rec["weights"].keys())
            all_declining = True
            for name in all_names:
                # 简化：比较权重趋势
                w_first = np.mean([rec["weights"].get(name, 0) for rec in first_half])
                w_second = np.mean([rec["weights"].get(name, 0) for rec in second_half])
                if w_second >= w_first:
                    all_declining = False
                    break
            if all_declining and all_names:
                alerts.append(
                    "WARN: All pipelines IC declining, possible regime change"
                )

        # 4. 融合信号与单一管线相关 > 0.95
        # 检查最近一天
        if recent:
            last_rec = recent[-1]
            combined = last_rec.get("combined_signal")
            if combined is not None and not combined.empty:
                for name, w in last_rec["weights"].items():
                    if w > 0.55:  # 只检查高权重管线
                        # 无法直接获取管线信号，通过权重集中度间接判断
                        if w > 0.95:
                            alerts.append(
                                f"WARN: Combined signal dominated by {name}, "
                                "diversification lost"
                            )

        return alerts


# ---------------------------------------------------------------------------
# SignalEnsemblePipeline — 融合入口
# ---------------------------------------------------------------------------


class SignalEnsemblePipeline:
    """日常回测/实盘入口：接收三条管线信号，输出融合选股信号。

    Usage::

        ensemble = SignalEnsemblePipeline(
            ic_lookback=60, uncertainty_penalty=0.1,
            min_weight=0.1, max_weight=0.6,
        )
        output = ensemble.combine(signals, "2024-06-28")
        # output.signal → M6 选股
    """

    def __init__(
        self,
        ic_lookback: int = 60,
        uncertainty_penalty: float = 0.1,
        min_weight: float = 0.1,
        max_weight: float = 0.6,
        correlation_penalty: float = 0.3,
    ):
        self.uncertainty_penalty = uncertainty_penalty
        self.normalizer = SignalNormalizer()
        self.ic_engine = ICWeightEngine(
            ic_lookback=ic_lookback,
            min_weight=min_weight,
            max_weight=max_weight,
            correlation_penalty=correlation_penalty,
        )
        self.monitor = EnsembleMonitor()
        # 保存各管线 rank 信号用于 monitor 相关性检查
        self._last_pipeline_signals: Dict[str, pd.Series] = {}

    def combine(
        self,
        signals: Dict[str, Optional[pd.Series]],
        anchor_date: str,
    ) -> EnsembleOutput:
        """融合三条管线信号。

        Args:
            signals: {
                "alpha": Series,            管线A 信号
                "kronos": Series,           管线B return_1d
                "kronos_uncertainty": Series, 管线B uncertainty（可选）
                "rdagent": Series or None,  管线C（可为空）
            }
            anchor_date: 锚定日期

        Returns:
            EnsembleOutput
        """
        # 1. 过滤空管线
        kronos_unc = signals.get("kronos_uncertainty")
        active_signals = {}
        for name in ("alpha", "kronos", "rdagent"):
            s = signals.get(name)
            if s is not None and not s.empty:
                active_signals[name] = s

        if not active_signals:
            return EnsembleOutput(mode="cold_start")

        # 2. 预处理：缩尾 + rank 归一化
        ranked = {}
        for name, s in active_signals.items():
            if name in ("alpha", "rdagent"):
                s = self.normalizer.winsorize(s)
            ranked[name] = self.normalizer.rank_normalize(s)

        # 3. 对齐股票 index
        ranked = self.normalizer.align_index(ranked)

        # 填充缺失值为 0.5（中性值）
        for name in ranked:
            ranked[name] = ranked[name].fillna(0.5)

        # 4. 计算权重
        weights, mode = self.ic_engine.compute_weights()

        # 如果 engine 还没有历史（首次调用），用当前活跃管线等权
        if not weights:
            n = len(ranked)
            weights = {name: 1.0 / n for name in ranked}
            mode = "cold_start"
        else:
            # 过滤掉不在当前活跃管线中的权重
            weights = {k: v for k, v in weights.items() if k in ranked}
            # 重新归一化
            total = sum(weights.values())
            if total > 0:
                weights = {k: v / total for k, v in weights.items()}
            else:
                n = len(ranked)
                weights = {name: 1.0 / n for name in ranked}

        # 5. 加权融合
        all_idx = ranked[list(ranked.keys())[0]].index
        raw_score = pd.Series(0.0, index=all_idx)
        for name, rank_s in ranked.items():
            w = weights.get(name, 0)
            raw_score = raw_score + w * rank_s

        # 6. 不确定性惩罚
        uncertainty_adjusted = False
        if kronos_unc is not None and not kronos_unc.empty and self.uncertainty_penalty > 0:
            unc_rank = self.normalizer.rank_normalize(kronos_unc)
            unc_rank = unc_rank.reindex(all_idx, fill_value=0.5)
            penalty = unc_rank * self.uncertainty_penalty
            raw_score = raw_score - penalty
            uncertainty_adjusted = True

        # 7. 最终归一化
        final_signal = self.normalizer.rank_normalize(raw_score)

        # 8. 记录到 monitor
        self.monitor.record(anchor_date, weights, final_signal, None)
        self._last_pipeline_signals = ranked

        # 9. 返回
        return EnsembleOutput(
            signal=final_signal,
            weights=weights,
            pipeline_ranks=ranked,
            uncertainty_adjusted=uncertainty_adjusted,
            mode=mode,
        )

    def update_history(
        self,
        date: str,
        signals: Dict[str, Optional[pd.Series]],
        actual_return: pd.Series,
    ):
        """追加历史记录（评估 T-1 日信号 vs T 日收益）。

        应在 combine 之后、次日收益已知时调用。
        """
        clean_signals = {
            k: v for k, v in signals.items()
            if k in ("alpha", "kronos", "rdagent") and v is not None and not v.empty
        }
        self.ic_engine.update(date, clean_signals, actual_return)
        # 回填 monitor 中的 actual_return
        self.monitor.update_actual_return(date, actual_return)

    def get_weights(self) -> Dict[str, float]:
        """当前各管线权重。"""
        weights, _ = self.ic_engine.compute_weights()
        return weights

    def get_monitor(self) -> EnsembleMonitor:
        """获取监控器实例。"""
        return self.monitor

    def get_ic_history(self) -> pd.DataFrame:
        """各管线逐日 IC 时序。"""
        return self.ic_engine.get_ic_history()
