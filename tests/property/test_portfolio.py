from paper_trading.portfolio_engine import PortfolioEngine
from paper_trading.account import buy_commission, sell_cost


def _bar(prices, limit_up=(), limit_down=(), untradable=()):
    return {c: {"open": p, "close": p,
                "limit_up": c in limit_up, "limit_down": c in limit_down,
                "tradable": c not in untradable}
            for c, p in prices.items()}


def test_nav_identity_holds():
    # 任意步后：记录的 nav == cash + Σ shares*close
    eng = PortfolioEngine(initial_capital=1_000_000.0)
    eng.step("2024-01-02", "2024-01-03", ["A", "B"],
             _bar({"A": 10.0, "B": 20.0}))
    close = {"A": 10.0, "B": 20.0}
    nav = eng.account.cash + sum(h.shares * close[c] for c, h in eng.account.holdings.items())
    assert abs(eng.account.nav_history[-1][1] - nav) < 1e-6


def test_t_plus_1_not_violated():
    # 当日买入不可当日卖出
    eng = PortfolioEngine(initial_capital=1_000_000.0)
    eng.step("2024-01-02", "2024-01-03", ["A"], _bar({"A": 10.0}))
    h = eng.account.holdings["A"]
    assert h.buy_date == "2024-01-03"          # T+1 开盘成交, 买入日=成交日
    assert h.sellable_date == h.buy_date       # 存买入日, 卖出判定用严格小于


def test_limit_up_blocks_buy():
    # 封涨停 → 买不进, 现金不变, 无持仓
    eng = PortfolioEngine(initial_capital=1_000_000.0)
    before = eng.account.cash
    eng.step("2024-01-02", "2024-01-03", ["A"],
             _bar({"A": 10.0}, limit_up=["A"]))
    assert "A" not in eng.account.holdings
    assert eng.account.cash == before


def test_limit_down_blocks_sell():
    # 持有 A, 次日跌停封板 → 卖不出, 仍持有
    eng = PortfolioEngine(initial_capital=1_000_000.0)
    eng.step("2024-01-02", "2024-01-03", ["A"], _bar({"A": 10.0}))
    # 次日 target 清空且 A 跌停
    eng.step("2024-01-03", "2024-01-04", [],
             _bar({"A": 9.0}, limit_down=["A"]))
    assert "A" in eng.account.holdings         # 跌停卖不出


def test_cash_never_negative():
    eng = PortfolioEngine(initial_capital=100_000.0)
    eng.step("2024-01-02", "2024-01-03", ["A", "B", "C"],
             _bar({"A": 10.0, "B": 20.0, "C": 30.0}))
    assert eng.account.cash >= 0
