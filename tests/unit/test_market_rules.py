import torch

from strategy_manager.market_rules import apply_market_constraints


def _raw(days, opens, highs=None, lows=None, closes=None):
    opens = torch.tensor([opens], dtype=torch.float32)
    return {
        "time": torch.tensor([[d * 86_400 for d in days]], dtype=torch.int64),
        "open": opens,
        "high": torch.tensor([highs or opens[0].tolist()], dtype=torch.float32),
        "low": torch.tensor([lows or opens[0].tolist()], dtype=torch.float32),
        "close": torch.tensor([closes or opens[0].tolist()], dtype=torch.float32),
    }


def test_a_share_is_long_only():
    desired = torch.tensor([[-0.8, 0.4, -0.2, -0.2]])
    raw = _raw([1, 2, 3, 4], [10, 10, 10, 10])
    actual = apply_market_constraints(desired, raw, market="a_share", symbols=["002192"])
    assert torch.all(actual >= 0)
    assert actual.tolist() == [[0.0, 0.4000000059604645, 0.0, 0.0]]


def test_t_plus_one_blocks_same_day_sale_for_intraday_bars():
    desired = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    # signal 0 buys on bar 1; signal 1 also executes on the same trading day.
    raw = _raw([1, 2, 2, 3], [10, 10, 10, 10])
    actual = apply_market_constraints(desired, raw, market="a_share", symbols=["002192"])
    assert actual.tolist() == [[1.0, 1.0, 0.0, 0.0]]


def test_one_price_limit_up_blocks_buy():
    desired = torch.tensor([[1.0, 1.0, 1.0]])
    raw = _raw(
        [1, 2, 3], [10.0, 11.0, 11.0],
        highs=[10.0, 11.0, 11.2], lows=[10.0, 11.0, 10.8], closes=[10.0, 11.0, 11.0],
    )
    actual = apply_market_constraints(desired, raw, market="a_share", symbols=["002192"])
    assert actual[0, 0].item() == 0.0
    assert actual[0, 1].item() == 1.0


def test_one_price_limit_down_blocks_sale():
    desired = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    raw = _raw(
        [1, 2, 3, 4], [10.0, 10.0, 9.0, 9.1],
        highs=[10.0, 10.2, 9.0, 9.2], lows=[10.0, 9.8, 9.0, 9.0],
        closes=[10.0, 10.0, 9.0, 9.1],
    )
    actual = apply_market_constraints(desired, raw, market="a_share", symbols=["002192"])
    assert actual.tolist() == [[1.0, 1.0, 0.0, 0.0]]
