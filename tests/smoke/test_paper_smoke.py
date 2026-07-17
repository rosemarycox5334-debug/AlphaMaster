import pandas as pd
from paper_trading.data_feed import ReplayFeed
from paper_trading.portfolio_engine import PortfolioEngine
from paper_trading.factor_ranker import FactorRanker
from paper_trading.metrics import compute_metrics
from data_pipeline.ashare_manager import AShareDataManager


class TinyFetcher:
    def universe_codes(self):
        return [f"C{i:03d}" for i in range(20)]

    def trade_calendar(self, start, end):
        return pd.bdate_range("2024-01-01", periods=80).strftime("%Y-%m-%d").tolist()

    def daily(self, code, start, end):
        import numpy as np
        dates = self.trade_calendar(start, end)
        rng = np.random.default_rng(int(code[1:]))
        price = abs(10 + np.cumsum(rng.normal(0, 0.15, len(dates)))) + 1
        return pd.DataFrame({
            "date": dates, "open": price, "high": price * 1.02,
            "low": price * 0.98, "close": price,
            "volume": rng.integers(1e5, 1e6, len(dates)),
        })


def test_replay_end_to_end():
    mgr = AShareDataManager(fetcher=TinyFetcher())
    mgr.load(start="2024-01-01", end="2024-05-01")
    feed = ReplayFeed(mgr)
    ranker = FactorRanker(formula=[0])          # 用特征0当因子
    engine = PortfolioEngine(initial_capital=1_000_000.0)

    dates = feed.trade_dates()
    # 从第40天开始模拟（前面留给特征warm-up），逐日推进
    for i in range(40, len(dates) - 1):
        d, nd = dates[i], dates[i + 1]
        feat_slice, valid = feed.slice_until(d)
        picks = ranker.rank(feat_slice, valid, mgr.symbols, top_k=10)
        bar = feed.bar_at(nd)
        engine.step(d, nd, picks, bar)

    assert len(engine.account.nav_history) > 0
    m = compute_metrics(engine.account.nav_history, 1_000_000.0)
    assert m["final_nav"] > 0
    assert "total_return" in m
    # 账户恒等式最终校验
    last_close = feed.bar_at(dates[-1])
    nav = engine.account.cash + sum(
        h.shares * last_close[c]["close"]
        for c, h in engine.account.holdings.items() if c in last_close)
    assert nav > 0
