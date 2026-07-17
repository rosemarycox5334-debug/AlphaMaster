"""paper_trading/state_store.py — 账户状态 JSON 持久化（实盘续跑用）。"""
from __future__ import annotations

import json
from pathlib import Path

from paper_trading.account import Account, Holding


def save_account(acc: Account, path: str) -> None:
    data = {
        "cash": acc.cash,
        "holdings": {c: vars(h) for c, h in acc.holdings.items()},
        "nav_history": [list(x) for x in acc.nav_history],
        "trades": acc.trades,
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2),
                          encoding="utf-8")


def load_account(path: str, initial_capital: float = 1_000_000.0) -> Account:
    p = Path(path)
    if not p.exists():
        return Account(cash=float(initial_capital))
    data = json.loads(p.read_text(encoding="utf-8"))
    acc = Account(cash=float(data["cash"]))
    for c, hd in data.get("holdings", {}).items():
        acc.holdings[c] = Holding(**hd)
    acc.nav_history = [tuple(x) for x in data.get("nav_history", [])]
    acc.trades = data.get("trades", [])
    return acc
