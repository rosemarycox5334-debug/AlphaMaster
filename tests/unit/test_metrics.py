import math
from paper_trading.metrics import compute_metrics


def test_metrics_basic():
    # 净值 100万→110万，单调上升无回撤
    nav = [("2024-01-02", 1_000_000.0),
           ("2024-01-03", 1_050_000.0),
           ("2024-01-04", 1_100_000.0)]
    m = compute_metrics(nav, initial_capital=1_000_000.0)
    assert abs(m["total_return"] - 0.10) < 1e-9
    assert m["max_drawdown"] == 0.0
    assert m["final_nav"] == 1_100_000.0


def test_metrics_drawdown():
    # 100万→120万→90万：峰值120万, 谷底90万, 回撤 = (120-90)/120 = 0.25
    nav = [("d1", 1_000_000.0), ("d2", 1_200_000.0), ("d3", 900_000.0)]
    m = compute_metrics(nav, initial_capital=1_000_000.0)
    assert abs(m["max_drawdown"] - 0.25) < 1e-9


def test_metrics_empty():
    m = compute_metrics([], initial_capital=1_000_000.0)
    assert m["total_return"] == 0.0
    assert m["final_nav"] == 1_000_000.0
