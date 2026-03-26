"""M7 风控模块单元测试。"""

import pytest

from quantlab.execution.execution import Position
from quantlab.risk_control import (
    CircuitBreaker,
    ExposureChecker,
    RiskCheckResult,
    RiskController,
    RiskEventLog,
    StopLossChecker,
)

# ---------------------------------------------------------------------------
# 辅助
# ---------------------------------------------------------------------------

CALENDAR = [f"2024-01-{d:02d}" for d in range(2, 31)]  # 2024-01-02 ~ 01-30


def _pos(sym, shares=1000, cost=10.0, entry="2024-01-02", highest=12.0):
    return Position(
        symbol=sym, shares=shares, cost_price=cost,
        entry_date=entry, highest_price=highest, last_buy_date=entry,
    )


# ---------------------------------------------------------------------------
# StopLossChecker
# ---------------------------------------------------------------------------


class TestStopLossChecker:

    def test_no_trigger_within_threshold(self):
        checker = StopLossChecker(stop_loss_pct=0.08)
        positions = {"A": _pos("A", highest=10.0)}
        prices = {"A": 9.5}  # 回撤 5%
        actions = checker.check(positions, prices, "2024-01-10")
        assert len(actions) == 0

    def test_force_sell_on_drawdown(self):
        checker = StopLossChecker(stop_loss_pct=0.08)
        positions = {"A": _pos("A", highest=10.0)}
        prices = {"A": 9.0}  # 回撤 10%
        actions = checker.check(positions, prices, "2024-01-10")
        assert len(actions) == 1
        assert actions[0].action == "force_sell"
        assert actions[0].symbol == "A"
        assert actions[0].drawdown == pytest.approx(0.1, abs=0.001)

    def test_exact_threshold_triggers(self):
        checker = StopLossChecker(stop_loss_pct=0.08)
        positions = {"A": _pos("A", highest=100.0)}
        prices = {"A": 92.0}  # 回撤 8%
        actions = checker.check(positions, prices, "2024-01-10")
        assert len(actions) == 1
        assert actions[0].action == "force_sell"

    def test_suggest_sell_expired_loss(self):
        checker = StopLossChecker(stop_loss_pct=0.08, max_hold_days=60)
        positions = {"A": _pos("A", cost=10.0, entry="2023-10-01", highest=10.5)}
        prices = {"A": 9.8}  # 回撤 ~6.7%（不触发止损），但亏损且超期
        actions = checker.check(positions, prices, "2024-01-10")
        assert len(actions) == 1
        assert actions[0].action == "suggest_sell"
        assert actions[0].reason == "hold_expired_loss"

    def test_no_suggest_sell_if_profitable(self):
        checker = StopLossChecker(stop_loss_pct=0.08, max_hold_days=60)
        positions = {"A": _pos("A", cost=10.0, entry="2023-10-01", highest=11.0)}
        prices = {"A": 10.5}  # 盈利，超期不触发
        actions = checker.check(positions, prices, "2024-01-10")
        assert len(actions) == 0

    def test_multiple_stocks(self):
        checker = StopLossChecker(stop_loss_pct=0.08)
        positions = {
            "A": _pos("A", highest=10.0),
            "B": _pos("B", highest=20.0),
            "C": _pos("C", highest=15.0),
        }
        prices = {"A": 9.0, "B": 19.0, "C": 13.0}  # A: 10%, B: 5%, C: 13.3%
        actions = checker.check(positions, prices, "2024-01-10")
        triggered = {a.symbol for a in actions if a.action == "force_sell"}
        assert triggered == {"A", "C"}

    def test_missing_price_skipped(self):
        checker = StopLossChecker(stop_loss_pct=0.08)
        positions = {"A": _pos("A", highest=10.0)}
        actions = checker.check(positions, {}, "2024-01-10")
        assert len(actions) == 0


# ---------------------------------------------------------------------------
# ExposureChecker
# ---------------------------------------------------------------------------


class TestExposureChecker:

    def test_industry_within_limit(self):
        checker = ExposureChecker(max_industry_pct=0.30)
        positions = {
            "A": _pos("A", shares=100),
            "B": _pos("B", shares=100),
        }
        prices = {"A": 10.0, "B": 10.0}
        industry_map = {"A": "银行", "B": "科技"}
        actions = checker.check_industry(positions, prices, industry_map, 10000.0)
        assert len(actions) == 0

    def test_industry_over_limit(self):
        checker = ExposureChecker(max_industry_pct=0.30)
        positions = {
            "A": _pos("A", shares=200),
            "B": _pos("B", shares=200),
            "C": _pos("C", shares=100),
        }
        prices = {"A": 10.0, "B": 10.0, "C": 10.0}
        industry_map = {"A": "银行", "B": "银行", "C": "科技"}
        # 总市值 5000, 银行 4000 = 80% > 30%
        actions = checker.check_industry(positions, prices, industry_map, 5000.0)
        assert len(actions) > 0
        assert all(a.action == "reduce" for a in actions)
        # 减持的都是银行股
        syms = {a.symbol for a in actions}
        assert syms <= {"A", "B"}

    def test_concentration_within_limit(self):
        checker = ExposureChecker(max_single_pct=0.20)
        positions = {"A": _pos("A", shares=100)}
        prices = {"A": 10.0}
        actions = checker.check_concentration(positions, prices, 10000.0)
        assert len(actions) == 0

    def test_concentration_over_limit(self):
        checker = ExposureChecker(max_single_pct=0.20)
        positions = {"A": _pos("A", shares=300)}
        prices = {"A": 10.0}
        # A 市值 3000, 总值 10000 → 30% > 20%
        actions = checker.check_concentration(positions, prices, 10000.0)
        assert len(actions) == 1
        assert actions[0].symbol == "A"
        assert actions[0].excess_value == pytest.approx(1000.0)

    def test_empty_positions(self):
        checker = ExposureChecker()
        assert checker.check_industry({}, {}, {}, 10000.0) == []
        assert checker.check_concentration({}, {}, 10000.0) == []


# ---------------------------------------------------------------------------
# CircuitBreaker
# ---------------------------------------------------------------------------


class TestCircuitBreaker:

    def test_no_trigger_below_threshold(self):
        cb = CircuitBreaker(drawdown_pct=0.10)
        cb.update_high_watermark(1_000_000)
        assert cb.check(950_000) is False  # 5% 回撤

    def test_trigger_at_threshold(self):
        cb = CircuitBreaker(drawdown_pct=0.10)
        cb.update_high_watermark(1_000_000)
        assert cb.check(900_000) is True  # 10% 回撤

    def test_trigger_above_threshold(self):
        cb = CircuitBreaker(drawdown_pct=0.10)
        cb.update_high_watermark(1_000_000)
        assert cb.check(850_000) is True  # 15% 回撤

    def test_pause_and_recovery_lifecycle(self):
        cb = CircuitBreaker(drawdown_pct=0.10, pause_days=5, recovery_days=3)
        cb.update_high_watermark(1_000_000)

        # 触发
        cb.trigger("2024-01-05", CALENDAR)
        # 2024-01-05 + 5 = 2024-01-10 暂停截止
        assert cb.is_paused("2024-01-05")
        assert cb.is_paused("2024-01-10")
        assert not cb.is_paused("2024-01-11")

        # 恢复期
        assert cb.is_recovery_mode("2024-01-11")
        assert cb.is_recovery_mode("2024-01-13")
        assert not cb.is_recovery_mode("2024-01-14")

    def test_position_limit_values(self):
        cb = CircuitBreaker(drawdown_pct=0.10, pause_days=3, recovery_days=2)
        cb.update_high_watermark(1_000_000)
        cb.trigger("2024-01-05", CALENDAR)

        # 暂停期
        assert cb.get_position_limit("2024-01-05") == 0.0
        assert cb.get_position_limit("2024-01-08") == 0.0

        # 恢复期
        assert cb.get_position_limit("2024-01-09") == 0.5
        assert cb.get_position_limit("2024-01-10") == 0.5

        # 正常
        assert cb.get_position_limit("2024-01-11") == 1.0

    def test_hwm_not_updated_during_pause(self):
        cb = CircuitBreaker(drawdown_pct=0.10, pause_days=3)
        cb.update_high_watermark(1_000_000)
        cb.trigger("2024-01-05", CALENDAR)

        # 暂停期间尝试更新
        cb.update_high_watermark(1_100_000)
        assert cb.high_watermark == 1_000_000  # 未更新

    def test_trigger_count(self):
        cb = CircuitBreaker(drawdown_pct=0.10, pause_days=2, recovery_days=1)
        cb.update_high_watermark(1_000_000)

        cb.trigger("2024-01-05", CALENDAR)
        assert cb.trigger_count == 1

        # 第二次熔断（恢复后）
        cb.pause_until = None
        cb.recovery_until = None
        cb.update_high_watermark(900_000)
        cb.trigger("2024-01-15", CALENDAR)
        assert cb.trigger_count == 2
        assert len(cb.trigger_history) == 2


# ---------------------------------------------------------------------------
# RiskEventLog
# ---------------------------------------------------------------------------


class TestRiskEventLog:

    def test_log_and_query(self):
        log = RiskEventLog()
        log.log("2024-01-05", "stop_loss", "A", {"drawdown": 0.1})
        log.log("2024-01-06", "circuit_breaker", details={"hwm": 1_000_000})
        log.log("2024-01-07", "stop_loss", "B", {"drawdown": 0.09})

        assert len(log.events) == 3

        sl = log.get_events("stop_loss")
        assert len(sl) == 2

        after = log.get_events(start_date="2024-01-06")
        assert len(after) == 2

        between = log.get_events(start_date="2024-01-05", end_date="2024-01-06")
        assert len(between) == 2

    def test_summary(self):
        log = RiskEventLog()
        log.log("2024-01-05", "stop_loss", "A")
        log.log("2024-01-06", "stop_loss", "B")
        log.log("2024-01-07", "circuit_breaker")

        s = log.summary()
        assert s == {"stop_loss": 2, "circuit_breaker": 1}


# ---------------------------------------------------------------------------
# RiskController（集成测试）
# ---------------------------------------------------------------------------


class TestRiskController:

    def test_normal_no_risk(self):
        """无风险时返回空结果。"""
        rc = RiskController()
        positions = {"A": _pos("A", shares=100, cost=10.0, highest=10.0)}
        prices = {"A": 10.0}
        result = rc.daily_check(
            positions, prices, {"A": "银行"}, 100_000.0, "2024-01-10", CALENDAR,
        )
        assert result.force_sell_symbols == []
        assert result.circuit_breaker_triggered is False
        assert result.position_limit == 1.0

    def test_stop_loss_generates_force_sell(self):
        """个股止损触发强卖。"""
        rc = RiskController(stop_loss_pct=0.08)
        positions = {"A": _pos("A", shares=100, cost=10.0, highest=10.0)}
        prices = {"A": 9.0}  # 回撤 10%

        result = rc.daily_check(
            positions, prices, {"A": "银行"}, 100_000.0, "2024-01-10", CALENDAR,
        )
        assert "A" in result.force_sell_symbols
        assert "A" in result.force_sell_reasons

    def test_circuit_breaker_full_flow(self):
        """熔断完整流程：触发 → 暂停 → 恢复 → 正常。"""
        rc = RiskController(circuit_breaker_pct=0.10, pause_days=3, recovery_days=2)

        # 建立高水位
        result = rc.daily_check(
            {}, {}, {}, 1_000_000.0, "2024-01-03", CALENDAR,
        )
        assert result.position_limit == 1.0

        # 触发熔断（回撤 12%）
        result = rc.daily_check(
            {}, {}, {}, 880_000.0, "2024-01-05", CALENDAR,
        )
        assert result.circuit_breaker_triggered is True

        # 暂停期
        result = rc.daily_check(
            {}, {}, {}, 880_000.0, "2024-01-06", CALENDAR,
        )
        assert result.position_limit == 0.0

        # 恢复期（暂停3天: 01-05~01-08, 恢复: 01-09~01-10）
        result = rc.daily_check(
            {}, {}, {}, 900_000.0, "2024-01-09", CALENDAR,
        )
        assert result.position_limit == 0.5

        # 正常恢复
        result = rc.daily_check(
            {}, {}, {}, 920_000.0, "2024-01-11", CALENDAR,
        )
        assert result.position_limit == 1.0

    def test_industry_reduce_force_sell(self):
        """行业超限触发减持。"""
        rc = RiskController(max_industry_pct=0.30)
        positions = {
            "A": _pos("A", shares=400, cost=10.0, highest=10.0),
            "B": _pos("B", shares=100, cost=10.0, highest=10.0),
        }
        prices = {"A": 10.0, "B": 10.0}
        industry_map = {"A": "银行", "B": "银行"}
        # 银行占 5000/10000 = 50% > 30%
        result = rc.daily_check(
            positions, prices, industry_map, 10_000.0, "2024-01-10", CALENDAR,
        )
        assert len(result.force_sell_symbols) > 0

    def test_concentration_reduce_force_sell(self):
        """个股集中度超限触发减持。"""
        rc = RiskController(max_single_pct=0.20)
        positions = {
            "A": _pos("A", shares=300, cost=10.0, highest=10.0),
        }
        prices = {"A": 10.0}
        # A 占 3000/10000 = 30% > 20%
        result = rc.daily_check(
            positions, prices, {"A": "银行"}, 10_000.0, "2024-01-10", CALENDAR,
        )
        assert "A" in result.force_sell_symbols

    def test_is_paused_delegates(self):
        rc = RiskController(circuit_breaker_pct=0.10, pause_days=3)
        rc.daily_check({}, {}, {}, 1_000_000.0, "2024-01-03", CALENDAR)
        rc.daily_check({}, {}, {}, 880_000.0, "2024-01-05", CALENDAR)
        assert rc.is_paused("2024-01-06") is True
        assert rc.is_paused("2024-01-15") is False

    def test_get_event_log(self):
        rc = RiskController(stop_loss_pct=0.08)
        positions = {"A": _pos("A", shares=100, cost=10.0, highest=10.0)}
        prices = {"A": 9.0}
        rc.daily_check(positions, prices, {"A": "银行"}, 100_000.0, "2024-01-10", CALENDAR)

        log = rc.get_event_log()
        assert log.summary().get("stop_loss", 0) >= 1

    def test_cascade_stop_loss_and_circuit_breaker(self):
        """级联测试：个股止损 + 组合熔断同时触发。"""
        rc = RiskController(stop_loss_pct=0.08, circuit_breaker_pct=0.10, pause_days=3)

        # 建立高水位
        rc.daily_check({}, {}, {}, 1_000_000.0, "2024-01-03", CALENDAR)

        # 同时触发止损和熔断
        positions = {"A": _pos("A", shares=100, cost=10.0, highest=10.0)}
        prices = {"A": 9.0}  # 止损
        result = rc.daily_check(
            positions, prices, {"A": "银行"}, 880_000.0, "2024-01-05", CALENDAR,
        )
        assert "A" in result.force_sell_symbols
        assert result.circuit_breaker_triggered is True
