"""M6 交易执行：订单生成 + 成交判定 + 账户管理。

核心类：
    Position               持仓数据
    TradeOrder             交易订单
    TradeRecord            成交记录
    DailySummary           当日账户摘要
    PortfolioAccount       账户与持仓管理
    OrderGenerator         订单生成器
    OrderExecutor          成交判定引擎
    ExecutionPipeline      执行入口

A 股交易约束：
    · 佣金: 买卖双向万 2.5，最低 5 元
    · 印花税: 仅卖出万 5
    · 滑点: 千 1
    · T+1: 当日买入次日才能卖出
    · 涨停: 一字涨停买不到
    · 跌停: 一字跌停卖不掉
    · 最小交易单位: 100 股
"""

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

COMMISSION_RATE = 0.00025      # 佣金费率（万 2.5）
STAMP_TAX_RATE = 0.0005        # 印花税率（万 5，仅卖出）
MIN_COMMISSION = 5.0           # 最低佣金（元）
SLIPPAGE_RATE = 0.001          # 滑点（千 1）
LOT_SIZE = 100                 # 最小交易单位（1 手）


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------


@dataclass
class Position:
    """持仓数据。"""
    symbol: str = ""
    shares: int = 0
    cost_price: float = 0.0        # 买入均价（含佣金摊入）
    entry_date: str = ""           # 建仓日期
    highest_price: float = 0.0     # 持仓期间最高价（止损用）
    last_buy_date: str = ""        # 最近一次买入日期（T+1 限制）


@dataclass
class TradeOrder:
    """交易订单。"""
    symbol: str = ""
    direction: str = "buy"         # "buy" | "sell"
    target_price: float = 0.0      # 买入上限 / 卖出下限
    shares: int = 0                # 目标股数
    signal_score: float = 0.0      # 融合信号得分
    reason: str = "signal"         # "signal" | "stop_loss" | "industry_limit" | "circuit_breaker"
    priority: int = 5              # 低值优先


@dataclass
class TradeRecord:
    """成交记录。"""
    date: str = ""
    symbol: str = ""
    direction: str = "buy"
    status: str = "filled"         # filled | failed_* | partial_filled
    order_price: float = 0.0       # 目标价
    exec_price: float = 0.0        # 实际成交价（含滑点）
    shares: int = 0                # 成交股数
    amount: float = 0.0            # 成交金额
    commission: float = 0.0        # 佣金
    stamp_tax: float = 0.0         # 印花税（买入为 0）
    total_cost: float = 0.0        # 佣金 + 印花税
    reason: str = "signal"


@dataclass
class DailySummary:
    """当日账户摘要。"""
    date: str = ""
    total_value: float = 0.0       # 总资产
    cash: float = 0.0
    market_value: float = 0.0      # 持仓市值
    position_count: int = 0        # 持仓股票数
    daily_pnl: float = 0.0        # 当日盈亏
    daily_return: float = 0.0     # 当日收益率
    buy_count: int = 0
    sell_count: int = 0
    total_commission: float = 0.0
    total_stamp_tax: float = 0.0


# ---------------------------------------------------------------------------
# PortfolioAccount — 账户与持仓管理
# ---------------------------------------------------------------------------


class PortfolioAccount:
    """管理现金、持仓和交易记录。"""

    def __init__(self, initial_cash: float = 1_000_000.0):
        self._cash: float = initial_cash
        self._initial_cash: float = initial_cash
        self._holdings: Dict[str, Position] = {}
        self._trade_history: List[TradeRecord] = []
        self._prev_total_value: float = initial_cash

    def get_cash(self) -> float:
        return self._cash

    def get_holdings(self) -> Dict[str, Position]:
        return dict(self._holdings)

    def get_portfolio_value(self, current_prices: Dict[str, float]) -> float:
        """现金 + 持仓市值。"""
        market_value = sum(
            pos.shares * current_prices.get(sym, pos.cost_price)
            for sym, pos in self._holdings.items()
        )
        return self._cash + market_value

    def get_market_value(self, current_prices: Dict[str, float]) -> float:
        return sum(
            pos.shares * current_prices.get(sym, pos.cost_price)
            for sym, pos in self._holdings.items()
        )

    def get_holding_weight(self, symbol: str, current_prices: Dict[str, float]) -> float:
        """该股票占总资产比例。"""
        total = self.get_portfolio_value(current_prices)
        if total <= 0 or symbol not in self._holdings:
            return 0.0
        pos = self._holdings[symbol]
        mv = pos.shares * current_prices.get(symbol, pos.cost_price)
        return mv / total

    def update_highest_price(self, current_prices: Dict[str, float]):
        """更新所有持仓的 highest_price。"""
        for sym, pos in self._holdings.items():
            price = current_prices.get(sym)
            if price is not None and price > pos.highest_price:
                pos.highest_price = price

    def apply_buy(
        self,
        symbol: str,
        shares: int,
        price: float,
        cost: float,
        date: str,
    ):
        """买入：扣现金、更新持仓。cost = 成交金额 + 佣金。"""
        if cost > self._cash + 0.01:
            raise ValueError(
                f"资金不足: 需 {cost:.2f}, 可用 {self._cash:.2f}"
            )
        self._cash -= cost

        if symbol in self._holdings:
            pos = self._holdings[symbol]
            old_total = pos.cost_price * pos.shares
            new_total = price * shares
            pos.cost_price = (old_total + new_total) / (pos.shares + shares)
            pos.shares += shares
            pos.last_buy_date = date
            if price > pos.highest_price:
                pos.highest_price = price
        else:
            self._holdings[symbol] = Position(
                symbol=symbol,
                shares=shares,
                cost_price=price,
                entry_date=date,
                highest_price=price,
                last_buy_date=date,
            )

    def apply_sell(
        self,
        symbol: str,
        shares: int,
        price: float,
        proceeds: float,
        date: str,
    ):
        """卖出：加现金、减持仓。proceeds = 成交金额 - 佣金 - 印花税。"""
        if symbol not in self._holdings:
            raise ValueError(f"未持有 {symbol}")
        pos = self._holdings[symbol]
        if shares > pos.shares:
            raise ValueError(
                f"{symbol} 持仓 {pos.shares} 股，卖出 {shares} 股超出"
            )
        self._cash += proceeds
        pos.shares -= shares
        if pos.shares == 0:
            del self._holdings[symbol]

    def can_sell(self, symbol: str, exec_date: str) -> bool:
        """检查 T+1 限制。"""
        if symbol not in self._holdings:
            return False
        pos = self._holdings[symbol]
        return pos.last_buy_date < exec_date

    def get_trade_history(self) -> List[TradeRecord]:
        return list(self._trade_history)

    def add_trade_record(self, record: TradeRecord):
        self._trade_history.append(record)

    def set_prev_total_value(self, value: float):
        """设置前一日总资产（用于计算日收益率）。"""
        self._prev_total_value = value

    def get_prev_total_value(self) -> float:
        return self._prev_total_value


# ---------------------------------------------------------------------------
# OrderGenerator — 订单生成器
# ---------------------------------------------------------------------------


class OrderGenerator:
    """根据融合信号和当前持仓生成买卖订单。"""

    def __init__(
        self,
        max_positions: int = 20,
        max_single_weight: float = 0.1,
        target_buy_count: int = 3,
        target_sell_count: int = 3,
    ):
        self.max_positions = max_positions
        self.max_single_weight = max_single_weight
        self.target_buy_count = target_buy_count
        self.target_sell_count = target_sell_count

    def generate_sell_orders(
        self,
        signal: pd.Series,
        holdings: Dict[str, Position],
        current_prices: Dict[str, float],
        exec_date: str,
        account: "PortfolioAccount" = None,
    ) -> List[TradeOrder]:
        """生成信号驱动的卖出订单。"""
        if signal.empty or not holdings:
            return []

        median_score = signal.median()
        q60 = signal.quantile(0.6)

        candidates = []
        for sym, pos in holdings.items():
            # T+1 限制
            if account and not account.can_sell(sym, exec_date):
                continue

            score = signal.get(sym, 0.5)

            # 条件1: 信号低于中位数
            is_weak = score < median_score

            # 条件2: 持仓超 60 天且信号低于 60 分位
            hold_days = (pd.Timestamp(exec_date) - pd.Timestamp(pos.entry_date)).days
            is_stale = hold_days > 60 and score < q60

            if is_weak or is_stale:
                candidates.append((sym, score, pos))

        # 按 signal_score 升序（最差优先卖）
        candidates.sort(key=lambda x: x[1])

        orders = []
        for sym, score, pos in candidates[:self.target_sell_count]:
            price = current_prices.get(sym, 0)
            if price <= 0:
                continue
            orders.append(TradeOrder(
                symbol=sym,
                direction="sell",
                target_price=price * 0.95,
                shares=pos.shares,
                signal_score=score,
                reason="signal",
                priority=4,
            ))

        return orders

    def generate_buy_orders(
        self,
        signal: pd.Series,
        holdings: Dict[str, Position],
        current_prices: Dict[str, float],
        industry_map: Optional[Dict[str, str]] = None,
        total_value: float = 0,
        available_cash: float = 0,
    ) -> List[TradeOrder]:
        """生成信号驱动的买入订单。"""
        if signal.empty or available_cash <= 0:
            return []

        held_symbols = set(holdings.keys())
        current_count = len(held_symbols)

        # 可用仓位数
        slots = self.max_positions - current_count
        if slots <= 0:
            return []

        # 行业约束
        max_industry_count = max(1, self.max_positions // 5)
        industry_counts: Dict[str, int] = {}
        if industry_map:
            for sym in held_symbols:
                ind = industry_map.get(sym, "unknown")
                industry_counts[ind] = industry_counts.get(ind, 0) + 1

        # 候选：按信号降序排列的非持仓股
        sorted_symbols = signal.sort_values(ascending=False).index
        candidates = [s for s in sorted_symbols if s not in held_symbols]

        # 单票分配资金
        buy_count = min(self.target_buy_count, slots)
        if buy_count <= 0:
            return []
        per_stock_cash = min(
            available_cash / buy_count,
            total_value * self.max_single_weight if total_value > 0 else available_cash,
        )

        orders = []
        for sym in candidates:
            if len(orders) >= buy_count:
                break

            price = current_prices.get(sym, 0)
            if price <= 0:
                continue

            # 行业约束
            if industry_map:
                ind = industry_map.get(sym, "unknown")
                if industry_counts.get(ind, 0) >= max_industry_count:
                    continue

            target_price = price * 1.02  # 允许 2% 溢价
            shares = math.floor(per_stock_cash / target_price / LOT_SIZE) * LOT_SIZE
            if shares < LOT_SIZE:
                continue

            score = signal.get(sym, 0)
            orders.append(TradeOrder(
                symbol=sym,
                direction="buy",
                target_price=target_price,
                shares=shares,
                signal_score=score,
                reason="signal",
                priority=5,
            ))

            # 更新行业计数
            if industry_map:
                ind = industry_map.get(sym, "unknown")
                industry_counts[ind] = industry_counts.get(ind, 0) + 1

        return orders

    def generate_force_sell_orders(
        self,
        symbols: List[str],
        current_prices: Dict[str, float],
        holdings: Dict[str, Position],
        reason: str = "stop_loss",
    ) -> List[TradeOrder]:
        """生成风控强卖订单。"""
        priority_map = {
            "stop_loss": 2,
            "industry_limit": 3,
            "circuit_breaker": 1,
        }
        priority = priority_map.get(reason, 2)

        orders = []
        for sym in symbols:
            pos = holdings.get(sym)
            if pos is None or pos.shares <= 0:
                continue
            price = current_prices.get(sym, 0)
            if price <= 0:
                continue
            orders.append(TradeOrder(
                symbol=sym,
                direction="sell",
                target_price=price * 0.95,
                shares=pos.shares,
                signal_score=0.0,
                reason=reason,
                priority=priority,
            ))
        return orders

    def generate_liquidation_orders(
        self,
        holdings: Dict[str, Position],
        current_prices: Dict[str, float],
    ) -> List[TradeOrder]:
        """生成熔断全部清仓订单。"""
        return self.generate_force_sell_orders(
            list(holdings.keys()),
            current_prices,
            holdings,
            reason="circuit_breaker",
        )


# ---------------------------------------------------------------------------
# OrderExecutor — 成交判定引擎
# ---------------------------------------------------------------------------


class OrderExecutor:
    """以 T+1 开盘价模拟真实成交。"""

    def execute(
        self,
        orders: List[TradeOrder],
        open_prices: Dict[str, float],
        limit_up: Dict[str, float],
        limit_down: Dict[str, float],
        exec_date: str,
        account: PortfolioAccount,
    ) -> List[TradeRecord]:
        """按优先级执行全部订单。"""
        if not orders:
            return []

        # 按 priority 排序
        sorted_orders = sorted(orders, key=lambda o: o.priority)

        records = []
        for order in sorted_orders:
            if order.direction == "sell":
                record = self._execute_sell(
                    order, open_prices, limit_down, exec_date, account
                )
            else:
                record = self._execute_buy(
                    order, open_prices, limit_up, exec_date, account
                )
            records.append(record)
            account.add_trade_record(record)

        return records

    def _execute_sell(
        self,
        order: TradeOrder,
        open_prices: Dict[str, float],
        limit_down: Dict[str, float],
        exec_date: str,
        account: PortfolioAccount,
    ) -> TradeRecord:
        """卖出订单成交判定。"""
        sym = order.symbol
        open_price = open_prices.get(sym, 0)

        # T+1 检查
        if not account.can_sell(sym, exec_date):
            return TradeRecord(
                date=exec_date, symbol=sym, direction="sell",
                status="failed_t1_restriction",
                order_price=order.target_price,
                shares=order.shares, reason=order.reason,
            )

        # 跌停检查
        ld = limit_down.get(sym, 0)
        if open_price > 0 and ld > 0 and open_price <= ld:
            return TradeRecord(
                date=exec_date, symbol=sym, direction="sell",
                status="failed_limit_down",
                order_price=order.target_price,
                shares=order.shares, reason=order.reason,
            )

        if open_price <= 0:
            return TradeRecord(
                date=exec_date, symbol=sym, direction="sell",
                status="failed_price_deviation",
                order_price=order.target_price,
                shares=order.shares, reason=order.reason,
            )

        # 成交
        exec_price = open_price * (1 - SLIPPAGE_RATE)
        amount = exec_price * order.shares
        commission = max(amount * COMMISSION_RATE, MIN_COMMISSION)
        stamp_tax = amount * STAMP_TAX_RATE
        total_cost = commission + stamp_tax
        proceeds = amount - total_cost

        account.apply_sell(sym, order.shares, exec_price, proceeds, exec_date)

        return TradeRecord(
            date=exec_date, symbol=sym, direction="sell",
            status="filled",
            order_price=order.target_price,
            exec_price=exec_price,
            shares=order.shares,
            amount=amount,
            commission=commission,
            stamp_tax=stamp_tax,
            total_cost=total_cost,
            reason=order.reason,
        )

    def _execute_buy(
        self,
        order: TradeOrder,
        open_prices: Dict[str, float],
        limit_up: Dict[str, float],
        exec_date: str,
        account: PortfolioAccount,
    ) -> TradeRecord:
        """买入订单成交判定。"""
        sym = order.symbol
        open_price = open_prices.get(sym, 0)

        if open_price <= 0:
            return TradeRecord(
                date=exec_date, symbol=sym, direction="buy",
                status="failed_price_deviation",
                order_price=order.target_price,
                shares=order.shares, reason=order.reason,
            )

        # 涨停检查
        lu = limit_up.get(sym, float("inf"))
        if open_price >= lu:
            return TradeRecord(
                date=exec_date, symbol=sym, direction="buy",
                status="failed_limit_up",
                order_price=order.target_price,
                shares=order.shares, reason=order.reason,
            )

        # 价格偏离检查
        if open_price > order.target_price:
            return TradeRecord(
                date=exec_date, symbol=sym, direction="buy",
                status="failed_price_deviation",
                order_price=order.target_price,
                exec_price=open_price,
                shares=order.shares, reason=order.reason,
            )

        # 计算成交价和费用
        exec_price = open_price * (1 + SLIPPAGE_RATE)
        shares = order.shares
        amount = exec_price * shares
        commission = max(amount * COMMISSION_RATE, MIN_COMMISSION)
        needed = amount + commission

        status = "filled"

        # 资金检查
        cash = account.get_cash()
        if needed > cash:
            # 尝试减少股数
            affordable = math.floor(cash / (exec_price * (1 + COMMISSION_RATE) + 0.01) / LOT_SIZE) * LOT_SIZE
            if affordable < LOT_SIZE:
                return TradeRecord(
                    date=exec_date, symbol=sym, direction="buy",
                    status="failed_no_cash",
                    order_price=order.target_price,
                    exec_price=exec_price,
                    shares=order.shares, reason=order.reason,
                )
            shares = affordable
            amount = exec_price * shares
            commission = max(amount * COMMISSION_RATE, MIN_COMMISSION)
            needed = amount + commission
            status = "partial_filled"

        cost = needed
        account.apply_buy(sym, shares, exec_price, cost, exec_date)

        return TradeRecord(
            date=exec_date, symbol=sym, direction="buy",
            status=status,
            order_price=order.target_price,
            exec_price=exec_price,
            shares=shares,
            amount=amount,
            commission=commission,
            stamp_tax=0.0,
            total_cost=commission,
            reason=order.reason,
        )


# ---------------------------------------------------------------------------
# ExecutionPipeline — 执行入口
# ---------------------------------------------------------------------------


class ExecutionPipeline:
    """日常回测/实盘执行入口。

    Usage::

        pipeline = ExecutionPipeline(initial_cash=1_000_000)
        orders = pipeline.generate_orders(signal, close_prices, industry_map, T)
        records = pipeline.execute_orders(orders, open_prices, limit_up, limit_down, T1)
        summary = pipeline.get_daily_summary(close_prices_T1, T1)
    """

    def __init__(
        self,
        initial_cash: float = 1_000_000.0,
        max_positions: int = 20,
        max_single_weight: float = 0.1,
        target_buy_count: int = 3,
        target_sell_count: int = 3,
    ):
        self.account = PortfolioAccount(initial_cash)
        self.order_gen = OrderGenerator(
            max_positions=max_positions,
            max_single_weight=max_single_weight,
            target_buy_count=target_buy_count,
            target_sell_count=target_sell_count,
        )
        self.executor = OrderExecutor()
        self._last_total_value: float = initial_cash
        self._daily_records: List[TradeRecord] = []

    def generate_orders(
        self,
        signal: pd.Series,
        current_prices: Dict[str, float],
        industry_map: Optional[Dict[str, str]] = None,
        anchor_date: str = "",
    ) -> List[TradeOrder]:
        """生成信号驱动的买卖订单（T 日收盘后调用）。"""
        holdings = self.account.get_holdings()
        total_value = self.account.get_portfolio_value(current_prices)

        # 卖出订单
        sell_orders = self.order_gen.generate_sell_orders(
            signal, holdings, current_prices, anchor_date, self.account,
        )

        # 预估卖出回款
        estimated_sell_proceeds = 0.0
        for order in sell_orders:
            est_price = order.target_price / 0.95  # 还原估计收盘价
            est_amount = est_price * order.shares * (1 - SLIPPAGE_RATE)
            est_cost = max(est_amount * COMMISSION_RATE, MIN_COMMISSION) + est_amount * STAMP_TAX_RATE
            estimated_sell_proceeds += est_amount - est_cost

        available_cash = self.account.get_cash() + estimated_sell_proceeds

        # 买入订单
        buy_orders = self.order_gen.generate_buy_orders(
            signal, holdings, current_prices, industry_map,
            total_value, available_cash,
        )

        return sell_orders + buy_orders

    def add_force_sell_orders(
        self,
        orders: List[TradeOrder],
        symbols: List[str],
        current_prices: Dict[str, float],
        reason: str = "stop_loss",
    ) -> List[TradeOrder]:
        """追加风控强卖订单到现有订单列表。"""
        holdings = self.account.get_holdings()
        force_orders = self.order_gen.generate_force_sell_orders(
            symbols, current_prices, holdings, reason,
        )
        # 去重：如果已有同 symbol 的卖出订单，取 priority 更高的
        existing_sell_syms = {
            o.symbol for o in orders if o.direction == "sell"
        }
        for fo in force_orders:
            if fo.symbol in existing_sell_syms:
                # 替换为更高优先级
                orders = [
                    o for o in orders
                    if not (o.symbol == fo.symbol and o.direction == "sell")
                ]
            orders.append(fo)
        return orders

    def add_liquidation_orders(
        self,
        orders: List[TradeOrder],
        current_prices: Dict[str, float],
    ) -> List[TradeOrder]:
        """追加熔断清仓订单。"""
        holdings = self.account.get_holdings()
        liq_orders = self.order_gen.generate_liquidation_orders(
            holdings, current_prices,
        )
        # 清除所有买入订单，替换所有卖出订单
        orders = [o for o in orders if o.direction != "buy"]
        sell_syms = {o.symbol for o in orders if o.direction == "sell"}
        for lo in liq_orders:
            if lo.symbol not in sell_syms:
                orders.append(lo)
            else:
                # 替换为清仓优先级
                orders = [
                    o for o in orders
                    if not (o.symbol == lo.symbol and o.direction == "sell")
                ]
                orders.append(lo)
        return orders

    def execute_orders(
        self,
        orders: List[TradeOrder],
        open_prices: Dict[str, float],
        limit_up: Dict[str, float],
        limit_down: Dict[str, float],
        exec_date: str,
    ) -> List[TradeRecord]:
        """T+1 成交判定。"""
        # 更新最高价
        self.account.update_highest_price(open_prices)

        records = self.executor.execute(
            orders, open_prices, limit_up, limit_down, exec_date, self.account,
        )
        self._daily_records = records
        return records

    def get_portfolio_value(self, current_prices: Dict[str, float]) -> float:
        return self.account.get_portfolio_value(current_prices)

    def get_holdings(self) -> Dict[str, Position]:
        return self.account.get_holdings()

    def get_daily_summary(
        self,
        current_prices: Dict[str, float],
        exec_date: str,
    ) -> DailySummary:
        """当日账户摘要。"""
        total_value = self.account.get_portfolio_value(current_prices)
        market_value = self.account.get_market_value(current_prices)
        cash = self.account.get_cash()
        prev_value = self.account.get_prev_total_value()

        daily_pnl = total_value - prev_value
        daily_return = daily_pnl / prev_value if prev_value > 0 else 0.0

        records = self._daily_records
        buy_count = sum(1 for r in records if r.direction == "buy" and r.status in ("filled", "partial_filled"))
        sell_count = sum(1 for r in records if r.direction == "sell" and r.status == "filled")
        total_commission = sum(r.commission for r in records if r.status in ("filled", "partial_filled"))
        total_stamp_tax = sum(r.stamp_tax for r in records if r.status in ("filled", "partial_filled"))

        # 更新前一日价值
        self.account.set_prev_total_value(total_value)

        return DailySummary(
            date=exec_date,
            total_value=total_value,
            cash=cash,
            market_value=market_value,
            position_count=len(self.account.get_holdings()),
            daily_pnl=daily_pnl,
            daily_return=daily_return,
            buy_count=buy_count,
            sell_count=sell_count,
            total_commission=total_commission,
            total_stamp_tax=total_stamp_tax,
        )
