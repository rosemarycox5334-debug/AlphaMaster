"""
paper_trading/portfolio_engine.py — T+1 逐日组合账户引擎

step(date, next_date, target_codes, bar): T日收盘决策 → T+1(next_date)开盘成交。
处理 T+1 交收、涨跌停封板不成交、真实成本、等权 Top-K。
"""
from __future__ import annotations

from paper_trading.account import (
    Account, Holding, buy_commission, max_buyable_shares, sell_cost,
)
from paper_trading.config import INITIAL_CAPITAL, LOT_SIZE, TOP_K


class PortfolioEngine:
    def __init__(self, initial_capital: float = INITIAL_CAPITAL) -> None:
        self.account = Account(cash=float(initial_capital))

    def step(self, date: str, next_date: str,
             target_codes: list[str], bar: dict) -> None:
        """在 next_date 开盘按 bar 成交，收盘估值记 nav。

        Args:
            date:         决策日（T日，已收盘）。
            next_date:    成交日（T+1，用其开盘价成交、收盘价估值）。
            target_codes: 目标持仓代码列表（因子选出的 Top-K）。
            bar:          {code: {open, close, limit_up, limit_down, tradable}}。
        """
        acc = self.account
        target = set(target_codes)

        # ── 1. 卖出：不在 target 且已过 T+1 ────────────────────────
        # T+1 口径：holding.buy_date 存买入日；只有买入日 < 成交日（next_date）
        # 才可卖出（严格小于 = 至少隔一个交易日）。
        for code in list(acc.holdings.keys()):
            if code in target:
                continue
            h = acc.holdings[code]
            if h.buy_date >= next_date:
                continue                        # 当日买入，T+1 未到，不可卖
            b = bar.get(code)
            if b is None or not b["tradable"] or b["limit_down"]:
                continue                        # 停牌/跌停封板，卖不出
            price = b["open"]
            turnover = h.shares * price
            acc.cash += turnover - sell_cost(turnover)
            acc.trades.append({
                "date": next_date, "code": code, "side": "SELL",
                "price": price, "shares": h.shares, "cost": sell_cost(turnover),
            })
            del acc.holdings[code]

        # ── 2. 买入：target 中未持有的，等权分配 ─────────────────────
        to_buy = [c for c in target_codes if c not in acc.holdings]
        buyable = [c for c in to_buy
                   if bar.get(c) and bar[c]["tradable"] and not bar[c]["limit_up"]]
        if buyable:
            # 等权目标：每只用 (总资产/TOP_K)，但不超过当前可用现金
            nav_now = acc.nav({c: b["open"] for c, b in bar.items()})
            per_budget = nav_now / TOP_K
            for code in buyable:
                b = bar[code]
                price = b["open"]
                budget = min(per_budget, acc.cash)
                shares = max_buyable_shares(cash=budget, price=price)
                if shares < LOT_SIZE:
                    continue
                turnover = shares * price
                fee = buy_commission(turnover)
                if turnover + fee > acc.cash:
                    continue
                acc.cash -= turnover + fee
                acc.holdings[code] = Holding(
                    code=code, shares=shares, cost_price=price,
                    buy_date=next_date, sellable_date=next_date,
                )
                acc.trades.append({
                    "date": next_date, "code": code, "side": "BUY",
                    "price": price, "shares": shares, "cost": fee,
                })

        # ── 3. 收盘估值 ────────────────────────────────────────────
        close_prices = {c: b["close"] for c, b in bar.items()}
        acc.nav_history.append((next_date, acc.nav(close_prices)))
