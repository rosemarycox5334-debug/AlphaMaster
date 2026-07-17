"""
paper_trading/account.py — 账户数据结构与成交/成本原语

Holding: 单只持仓（含 T+1 sellable_date）。
Account: 现金 + 持仓 + 净值/流水记录。
成本原语: 买入佣金、卖出成本（佣金+印花税）、可买股数（100股整数倍且不透支）。
"""
from __future__ import annotations

from dataclasses import dataclass, field

from paper_trading.config import (
    COMMISSION_RATE, LOT_SIZE, MIN_COMMISSION, STAMP_TAX,
)


def buy_commission(turnover: float) -> float:
    """买入佣金：max(成交额 × 费率, 最低佣金)。"""
    return max(turnover * COMMISSION_RATE, MIN_COMMISSION)


def sell_cost(turnover: float) -> float:
    """卖出总成本：佣金 + 印花税。"""
    return max(turnover * COMMISSION_RATE, MIN_COMMISSION) + turnover * STAMP_TAX


def max_buyable_shares(cash: float, price: float) -> int:
    """在不透支（含买入佣金）前提下，最多可买多少股（100 股整数倍）。"""
    if price <= 0 or cash <= 0:
        return 0
    lots = int(cash // (price * LOT_SIZE))     # 先粗估手数上限
    while lots > 0:
        shares = lots * LOT_SIZE
        turnover = shares * price
        if turnover + buy_commission(turnover) <= cash:
            return shares
        lots -= 1
    return 0


@dataclass
class Holding:
    code: str
    shares: int
    cost_price: float
    buy_date: str
    sellable_date: str


@dataclass
class Account:
    cash: float
    holdings: dict[str, Holding] = field(default_factory=dict)
    nav_history: list[tuple[str, float]] = field(default_factory=list)
    trades: list[dict] = field(default_factory=list)

    def nav(self, close_prices: dict[str, float]) -> float:
        """总净值 = 现金 + Σ(持股数 × 当日收盘价)。停牌股用传入的最后有效价。"""
        mkt = sum(h.shares * close_prices.get(c, h.cost_price)
                  for c, h in self.holdings.items())
        return self.cash + mkt
