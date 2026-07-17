"""
run_paper_live.py — A股实盘模拟：推进一个交易日

每个交易日收盘后运行一次（手动或定时）：拉当日数据 → 决策 → 成交 → 落盘账户。
状态存 paper_trading/state/account.json，进程重启可续跑。

用法（每交易日收盘后）:
    python run_paper_live.py --today 2026-07-16
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from data_pipeline.ashare_manager import AShareDataManager
from model_core.vocab import FORMULA_VOCAB
from paper_trading.config import INITIAL_CAPITAL, TOP_K
from paper_trading.data_feed import LiveFeed
from paper_trading.factor_ranker import FactorRanker
from paper_trading.portfolio_engine import PortfolioEngine
from paper_trading.state_store import load_account, save_account

STATE_PATH = "paper_trading/state/account.json"


def run_live_step(today: str, strategy_file: str, start: str = "2023-01-01") -> None:
    data = json.loads(Path(strategy_file).read_text(encoding="utf-8"))
    if data.get("vocab_version"):
        FORMULA_VOCAB.verify(data["vocab_version"])
    formula = data["formula"]

    mgr = AShareDataManager()
    feed = LiveFeed(mgr, start=start)
    feed.advance(today, end=today)
    dates = feed.trade_dates()
    if today not in dates:
        print(f"[{today}] 非交易日，跳过")
        return

    engine = PortfolioEngine(initial_capital=INITIAL_CAPITAL)
    engine.account = load_account(STATE_PATH, INITIAL_CAPITAL)

    # 用「上一交易日」的因子选股，在 today 开盘成交（T日决策→T+1成交口径）。
    t = dates.index(today)
    if t == 0:
        print("数据不足，无法决策")
        return
    prev = dates[t - 1]
    feat_slice, valid = feed.slice_until(prev)
    picks = ranker_picks(formula, feat_slice, valid, mgr.symbols)
    engine.step(prev, today, picks, feed.bar_at(today))

    save_account(engine.account, STATE_PATH)
    nav = engine.account.nav_history[-1][1] if engine.account.nav_history else INITIAL_CAPITAL
    print(f"[{today}] nav={nav:,.0f} 持仓={len(engine.account.holdings)} 已存档")


def ranker_picks(formula, feat_slice, valid, codes):
    return FactorRanker(formula=formula).rank(feat_slice, valid, codes, top_k=TOP_K)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--today", required=True)
    ap.add_argument("--strategy", default="strategies/best_ashare_universe.json")
    args = ap.parse_args()
    run_live_step(args.today, args.strategy)
