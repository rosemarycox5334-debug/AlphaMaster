from paper_trading.account import Account, Holding
from paper_trading.state_store import save_account, load_account


def test_roundtrip(tmp_path):
    acc = Account(cash=500_000.0)
    acc.holdings["600000"] = Holding("600000", 1000, 10.0, "2026-07-15", "2026-07-15")
    acc.nav_history.append(("2026-07-15", 510_000.0))
    acc.trades.append({"date": "2026-07-15", "code": "600000", "side": "BUY",
                       "price": 10.0, "shares": 1000, "cost": 5.0})
    path = tmp_path / "account.json"
    save_account(acc, str(path))
    loaded = load_account(str(path))
    assert loaded.cash == 500_000.0
    assert loaded.holdings["600000"].shares == 1000
    assert loaded.nav_history[-1] == ("2026-07-15", 510_000.0)
    assert loaded.trades[0]["code"] == "600000"


def test_load_missing_returns_fresh(tmp_path):
    loaded = load_account(str(tmp_path / "none.json"), initial_capital=1_000_000.0)
    assert loaded.cash == 1_000_000.0
    assert loaded.holdings == {}
