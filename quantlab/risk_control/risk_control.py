"""M7 风险控制：三级级联风控体系。

三级结构：
    Level 1 — StopLossChecker    个股止损（回撤 ≥ 8% 强卖）
    Level 2 — ExposureChecker    结构控制（行业 ≤ 30%、个股 ≤ 20%）
    Level 3 — CircuitBreaker     组合熔断（NAV 回撤 ≥ 10% → 暂停 5 天 + 恢复期 3 天）

辅助：
    RiskEventLog     风控事件记录
    RiskController   统一入口（daily_check）
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import pandas as pd

from quantlab.execution.execution import Position

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------


@dataclass
class StopLossAction:
    """止损动作。"""
    symbol: str
    action: str              # "force_sell" | "suggest_sell"
    reason: str              # 如 "drawdown_8.5%" | "hold_expired_loss"
    drawdown: float          # 当前回撤幅度
    highest_price: float
    current_price: float
    hold_days: int


@dataclass
class ExposureAction:
    """敞口控制动作。"""
    symbol: str
    action: str = "reduce"   # "reduce"
    reason: str = ""         # 如 "industry_over_30%: 银行业 35.2%"
    current_pct: float = 0.0
    limit_pct: float = 0.0
    excess_value: float = 0.0


@dataclass
class RiskEvent:
    """风控事件记录。"""
    date: str
    event_type: str          # stop_loss | industry_reduce | concentration_reduce
                             # | circuit_breaker | recovery_start | recovery_end
    symbol: Optional[str] = None
    details: Dict = field(default_factory=dict)


@dataclass
class RiskCheckResult:
    """每日风控检查结果。"""
    force_sell_symbols: List[str] = field(default_factory=list)
    suggest_sell_symbols: List[str] = field(default_factory=list)
    force_sell_reasons: Dict[str, str] = field(default_factory=dict)
    circuit_breaker_triggered: bool = False
    position_limit: float = 1.0    # 0.0（暂停）| 0.5（恢复期）| 1.0（正常）
    events: List[RiskEvent] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Level 1 — StopLossChecker（个股止损）
# ---------------------------------------------------------------------------


class StopLossChecker:
    """个股止损检查器。

    规则：
    - 持仓回撤 ≥ stop_loss_pct（默认 8%）→ force_sell
    - 持仓超 max_hold_days 且亏损 → suggest_sell
    - 持仓超 max_hold_days 但盈利 → 不限时
    """

    def __init__(self, stop_loss_pct: float = 0.08, max_hold_days: int = 60):
        self.stop_loss_pct = stop_loss_pct
        self.max_hold_days = max_hold_days

    def check(
        self,
        positions: Dict[str, Position],
        current_prices: Dict[str, float],
        current_date: str,
    ) -> List[StopLossAction]:
        actions = []
        for sym, pos in positions.items():
            price = current_prices.get(sym)
            if price is None or price <= 0:
                continue

            highest = pos.highest_price
            if highest <= 0:
                highest = pos.cost_price
            if highest <= 0:
                continue

            # 回撤 = (最高价 - 当前价) / 最高价
            drawdown = (highest - price) / highest

            hold_days = (
                pd.Timestamp(current_date) - pd.Timestamp(pos.entry_date)
            ).days

            # 规则 1：回撤达到止损线
            if drawdown >= self.stop_loss_pct:
                actions.append(StopLossAction(
                    symbol=sym,
                    action="force_sell",
                    reason=f"drawdown_{drawdown * 100:.1f}%",
                    drawdown=drawdown,
                    highest_price=highest,
                    current_price=price,
                    hold_days=hold_days,
                ))
                continue

            # 规则 2：超期持仓且亏损
            if hold_days > self.max_hold_days and price < pos.cost_price:
                actions.append(StopLossAction(
                    symbol=sym,
                    action="suggest_sell",
                    reason="hold_expired_loss",
                    drawdown=drawdown,
                    highest_price=highest,
                    current_price=price,
                    hold_days=hold_days,
                ))

        return actions


# ---------------------------------------------------------------------------
# Level 2 — ExposureChecker（结构控制）
# ---------------------------------------------------------------------------


class ExposureChecker:
    """行业与个股集中度检查。

    规则：
    - 单一行业占比 ≤ max_industry_pct（默认 30%）
    - 单一个股占比 ≤ max_single_pct（默认 20%）
    """

    def __init__(
        self,
        max_industry_pct: float = 0.30,
        max_single_pct: float = 0.20,
    ):
        self.max_industry_pct = max_industry_pct
        self.max_single_pct = max_single_pct

    def check_industry(
        self,
        positions: Dict[str, Position],
        current_prices: Dict[str, float],
        industry_map: Dict[str, str],
        total_value: float,
    ) -> List[ExposureAction]:
        """检查行业集中度，超限时返回需减持的股票。"""
        if total_value <= 0 or not positions:
            return []

        # 按行业汇总市值
        industry_values: Dict[str, List[Tuple[str, float]]] = {}
        for sym, pos in positions.items():
            price = current_prices.get(sym, pos.cost_price)
            mv = pos.shares * price
            ind = industry_map.get(sym, "unknown")
            industry_values.setdefault(ind, []).append((sym, mv))

        actions = []
        for ind, holdings in industry_values.items():
            ind_value = sum(mv for _, mv in holdings)
            ind_pct = ind_value / total_value

            if ind_pct <= self.max_industry_pct:
                continue

            excess = ind_value - total_value * self.max_industry_pct

            # 从行业内市值最小的股票开始减持
            holdings_sorted = sorted(holdings, key=lambda x: x[1])
            remaining_excess = excess
            for sym, mv in holdings_sorted:
                if remaining_excess <= 0:
                    break
                reduce_value = min(mv, remaining_excess)
                actions.append(ExposureAction(
                    symbol=sym,
                    action="reduce",
                    reason=f"industry_over_{self.max_industry_pct * 100:.0f}%: {ind} {ind_pct * 100:.1f}%",
                    current_pct=ind_pct,
                    limit_pct=self.max_industry_pct,
                    excess_value=reduce_value,
                ))
                remaining_excess -= reduce_value

        return actions

    def check_concentration(
        self,
        positions: Dict[str, Position],
        current_prices: Dict[str, float],
        total_value: float,
    ) -> List[ExposureAction]:
        """检查个股集中度。"""
        if total_value <= 0 or not positions:
            return []

        actions = []
        for sym, pos in positions.items():
            price = current_prices.get(sym, pos.cost_price)
            mv = pos.shares * price
            pct = mv / total_value

            if pct > self.max_single_pct:
                excess = mv - total_value * self.max_single_pct
                actions.append(ExposureAction(
                    symbol=sym,
                    action="reduce",
                    reason=f"single_over_{self.max_single_pct * 100:.0f}%: {pct * 100:.1f}%",
                    current_pct=pct,
                    limit_pct=self.max_single_pct,
                    excess_value=excess,
                ))

        return actions


# ---------------------------------------------------------------------------
# Level 3 — CircuitBreaker（组合熔断）
# ---------------------------------------------------------------------------


class CircuitBreaker:
    """组合级熔断器。

    状态机：
        Normal → check() 触发 → Pause（5 交易日）→ Recovery（3 交易日）→ Normal
    """

    def __init__(
        self,
        drawdown_pct: float = 0.10,
        pause_days: int = 5,
        recovery_position_limit: float = 0.5,
        recovery_days: int = 3,
    ):
        self.drawdown_pct = drawdown_pct
        self.pause_days = pause_days
        self.recovery_position_limit = recovery_position_limit
        self.recovery_days = recovery_days

        self.high_watermark: float = 0.0
        self.pause_until: Optional[str] = None      # 暂停截止日（含）
        self.recovery_until: Optional[str] = None    # 恢复期截止日（含）
        self.trigger_count: int = 0
        self.trigger_history: List[Tuple[str, float]] = []

    def update_high_watermark(self, value: float):
        """更新净值高水位（暂停期间不更新）。"""
        if self.pause_until is not None:
            return
        if value > self.high_watermark:
            self.high_watermark = value

    def check(self, current_value: float) -> bool:
        """检查是否触发熔断。"""
        if self.high_watermark <= 0:
            return False
        drawdown = (self.high_watermark - current_value) / self.high_watermark
        return drawdown >= self.drawdown_pct

    def trigger(self, current_date: str, calendar: List[str]):
        """触发熔断，计算暂停和恢复期截止日。"""
        self.trigger_count += 1
        self.trigger_history.append((current_date, self.high_watermark))

        try:
            idx = calendar.index(current_date)
        except ValueError:
            # 当前日期不在日历中，向后查找最近的
            idx = 0
            for i, d in enumerate(calendar):
                if d >= current_date:
                    idx = i
                    break

        # 暂停期：往后 pause_days 个交易日
        pause_end_idx = min(idx + self.pause_days, len(calendar) - 1)
        self.pause_until = calendar[pause_end_idx]

        # 恢复期：暂停结束后 recovery_days 个交易日
        recovery_end_idx = min(pause_end_idx + self.recovery_days, len(calendar) - 1)
        self.recovery_until = calendar[recovery_end_idx]

        logger.warning(
            "熔断触发 [%s] 高水位=%.2f, 暂停至 %s, 恢复期至 %s",
            current_date, self.high_watermark, self.pause_until, self.recovery_until,
        )

    def is_paused(self, current_date: str) -> bool:
        """当前日期是否处于暂停期。"""
        if self.pause_until is None:
            return False
        return current_date <= self.pause_until

    def is_recovery_mode(self, current_date: str) -> bool:
        """当前日期是否处于恢复期（暂停结束后、恢复截止前）。"""
        if self.pause_until is None or self.recovery_until is None:
            return False
        return self.pause_until < current_date <= self.recovery_until

    def get_position_limit(self, current_date: str) -> float:
        """获取当前仓位限制系数。

        Returns:
            0.0 — 暂停期，禁止交易
            0.5 — 恢复期，半仓
            1.0 — 正常
        """
        if self.is_paused(current_date):
            return 0.0
        if self.is_recovery_mode(current_date):
            return self.recovery_position_limit
        # 恢复期结束，清理状态
        if self.recovery_until is not None and current_date > self.recovery_until:
            self.pause_until = None
            self.recovery_until = None
        return 1.0


# ---------------------------------------------------------------------------
# RiskEventLog — 风控事件日志
# ---------------------------------------------------------------------------


class RiskEventLog:
    """记录和查询风控事件。"""

    def __init__(self):
        self._events: List[RiskEvent] = []

    def log(self, date: str, event_type: str, symbol: Optional[str] = None,
            details: Optional[Dict] = None):
        event = RiskEvent(
            date=date,
            event_type=event_type,
            symbol=symbol,
            details=details or {},
        )
        self._events.append(event)
        logger.info("风控事件 [%s] %s %s %s", date, event_type, symbol or "", details or "")
        return event

    def get_events(
        self,
        event_type: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> List[RiskEvent]:
        result = self._events
        if event_type is not None:
            result = [e for e in result if e.event_type == event_type]
        if start_date is not None:
            result = [e for e in result if e.date >= start_date]
        if end_date is not None:
            result = [e for e in result if e.date <= end_date]
        return result

    def summary(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for e in self._events:
            counts[e.event_type] = counts.get(e.event_type, 0) + 1
        return counts

    @property
    def events(self) -> List[RiskEvent]:
        return list(self._events)


# ---------------------------------------------------------------------------
# RiskController — 统一入口
# ---------------------------------------------------------------------------


class RiskController:
    """三级风控统一入口。

    Usage::

        risk = RiskController()
        result = risk.daily_check(
            positions, current_prices, industry_map,
            total_value, current_date, calendar,
        )
        if result.circuit_breaker_triggered:
            # M6 全仓清算
        if result.force_sell_symbols:
            # M6 强制卖出
        # M6 根据 result.position_limit 限制买入规模
    """

    def __init__(
        self,
        stop_loss_pct: float = 0.08,
        max_hold_days: int = 60,
        max_industry_pct: float = 0.30,
        max_single_pct: float = 0.20,
        circuit_breaker_pct: float = 0.10,
        pause_days: int = 5,
        recovery_position_limit: float = 0.5,
        recovery_days: int = 3,
    ):
        self.stop_loss_checker = StopLossChecker(stop_loss_pct, max_hold_days)
        self.exposure_checker = ExposureChecker(max_industry_pct, max_single_pct)
        self.circuit_breaker = CircuitBreaker(
            circuit_breaker_pct, pause_days, recovery_position_limit, recovery_days,
        )
        self.event_log = RiskEventLog()

    def daily_check(
        self,
        positions: Dict[str, Position],
        current_prices: Dict[str, float],
        industry_map: Dict[str, str],
        total_value: float,
        current_date: str,
        calendar: List[str],
    ) -> RiskCheckResult:
        """执行每日风控检查。

        Args:
            positions: 当前持仓（symbol → Position）
            current_prices: T 日收盘价
            industry_map: 股票 → 行业映射
            total_value: 组合净值（现金 + 市值）
            current_date: T 日日期
            calendar: 交易日历列表

        Returns:
            RiskCheckResult: 包含强卖列表、熔断状态、仓位限制等
        """
        events: List[RiskEvent] = []
        force_sell: List[str] = []
        suggest_sell: List[str] = []
        force_sell_reasons: Dict[str, str] = {}

        # ---- Step 1: 更新高水位 ----
        self.circuit_breaker.update_high_watermark(total_value)

        # ---- Step 2: 检查暂停/恢复状态 ----
        if self.circuit_breaker.is_paused(current_date):
            logger.info("熔断暂停中 [%s]，跳过交易", current_date)
            return RiskCheckResult(
                position_limit=0.0,
                events=events,
            )

        # 恢复期开始事件
        if self.circuit_breaker.is_recovery_mode(current_date):
            # 只在恢复期第一天记录
            existing = self.event_log.get_events("recovery_start")
            if not existing or existing[-1].date != current_date:
                # 检查是否已记录过本轮恢复
                if (not existing
                        or existing[-1].date < (self.circuit_breaker.pause_until or "")):
                    evt = self.event_log.log(current_date, "recovery_start")
                    events.append(evt)

        # ---- Step 3: Level 1 — 个股止损 ----
        sl_actions = self.stop_loss_checker.check(
            positions, current_prices, current_date,
        )
        for a in sl_actions:
            if a.action == "force_sell":
                force_sell.append(a.symbol)
                force_sell_reasons[a.symbol] = a.reason
                evt = self.event_log.log(
                    current_date, "stop_loss", a.symbol,
                    {"drawdown": a.drawdown, "highest": a.highest_price,
                     "current": a.current_price, "hold_days": a.hold_days},
                )
                events.append(evt)
            elif a.action == "suggest_sell":
                suggest_sell.append(a.symbol)

        # ---- Step 4: Level 2 — 结构控制 ----
        ind_actions = self.exposure_checker.check_industry(
            positions, current_prices, industry_map, total_value,
        )
        for a in ind_actions:
            if a.symbol not in force_sell:
                force_sell.append(a.symbol)
                force_sell_reasons[a.symbol] = a.reason
            evt = self.event_log.log(
                current_date, "industry_reduce", a.symbol,
                {"current_pct": a.current_pct, "limit_pct": a.limit_pct,
                 "excess_value": a.excess_value},
            )
            events.append(evt)

        conc_actions = self.exposure_checker.check_concentration(
            positions, current_prices, total_value,
        )
        for a in conc_actions:
            if a.symbol not in force_sell:
                force_sell.append(a.symbol)
                force_sell_reasons[a.symbol] = a.reason
            evt = self.event_log.log(
                current_date, "concentration_reduce", a.symbol,
                {"current_pct": a.current_pct, "limit_pct": a.limit_pct,
                 "excess_value": a.excess_value},
            )
            events.append(evt)

        # ---- Step 5: Level 3 — 组合熔断 ----
        circuit_breaker_triggered = False
        # 恢复期内不重复触发熔断
        in_recovery = self.circuit_breaker.is_recovery_mode(current_date)
        if not in_recovery and self.circuit_breaker.check(total_value):
            self.circuit_breaker.trigger(current_date, calendar)
            circuit_breaker_triggered = True
            evt = self.event_log.log(
                current_date, "circuit_breaker", None,
                {"high_watermark": self.circuit_breaker.high_watermark,
                 "current_value": total_value,
                 "drawdown_pct": (self.circuit_breaker.high_watermark - total_value)
                                 / self.circuit_breaker.high_watermark,
                 "pause_until": self.circuit_breaker.pause_until,
                 "recovery_until": self.circuit_breaker.recovery_until},
            )
            events.append(evt)

        # ---- Step 6: 获取仓位限制 ----
        position_limit = self.circuit_breaker.get_position_limit(current_date)

        return RiskCheckResult(
            force_sell_symbols=list(set(force_sell)),
            suggest_sell_symbols=suggest_sell,
            force_sell_reasons=force_sell_reasons,
            circuit_breaker_triggered=circuit_breaker_triggered,
            position_limit=position_limit,
            events=events,
        )

    def is_paused(self, current_date: str) -> bool:
        """查询当前是否处于熔断暂停期。"""
        return self.circuit_breaker.is_paused(current_date)

    def get_position_limit(self, current_date: str) -> float:
        """获取当前允许的仓位系数。"""
        return self.circuit_breaker.get_position_limit(current_date)

    def get_event_log(self) -> RiskEventLog:
        """获取风控事件日志。"""
        return self.event_log
