from paper_trading.account import (
    Account, Holding, buy_commission, sell_cost, max_buyable_shares,
)


def test_buy_commission_min_floor():
    # 成交额小 → 命中最低 5 元
    assert buy_commission(1000.0) == 5.0
    # 成交额大 → 按万2.5
    assert abs(buy_commission(100_000.0) - 25.0) < 1e-9


def test_sell_cost_includes_stamp_tax():
    # 卖出 = 佣金(万2.5,最低5) + 印花税(千1)
    turnover = 100_000.0
    expected = max(turnover * 0.00025, 5.0) + turnover * 0.001
    assert abs(sell_cost(turnover) - expected) < 1e-9


def test_max_buyable_shares_lot_rounding():
    # 现金 10000, 价格 9.9 → 理论 1010 股, 向下取整到 100 → 1000 股
    # 但需预留佣金, 精确到 100 股整数倍且总花费<=现金
    shares = max_buyable_shares(cash=10_000.0, price=9.9)
    assert shares % 100 == 0
    cost = shares * 9.9 + buy_commission(shares * 9.9)
    assert cost <= 10_000.0
    # 再多买 100 股就会超预算
    over = (shares + 100) * 9.9 + buy_commission((shares + 100) * 9.9)
    assert over > 10_000.0


def test_account_nav():
    acc = Account(cash=100_000.0)
    acc.holdings["600000"] = Holding(
        code="600000", shares=1000, cost_price=10.0,
        buy_date="2024-01-02", sellable_date="2024-01-03")
    nav = acc.nav({"600000": 11.0})     # 收盘价 11
    assert abs(nav - (100_000.0 + 1000 * 11.0)) < 1e-9
