"""
run_paper_replay.py — A股历史回放 CLI（Web subprocess 调用）

流程：加载策略公式 → 加载A股数据 → 逐日回放 → 落盘净值/流水/指标。
产物：paper_trading/output/{equity.json, trades.json, metrics.json}

用法:
    python run_paper_replay.py --strategy strategies/best_ashare_universe.json \
        --start 2023-01-01 --end 2026-06-30 --sim-start 2026-06-01
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
from paper_trading.data_feed import ReplayFeed
from paper_trading.factor_ranker import FactorRanker
from paper_trading.metrics import compute_metrics
from paper_trading.portfolio_engine import PortfolioEngine

OUT_DIR = Path("paper_trading") / "output"


def run_replay(strategy_file: str, start: str, end: str,
               sim_start: str, warmup: int = 60) -> dict:
    data = json.loads(Path(strategy_file).read_text(encoding="utf-8"))
    # vocab 校验：不匹配抛 VocabVersionMismatchError（避免旧公式对不上特征维）
    saved_ver = data.get("vocab_version")
    if saved_ver:
        FORMULA_VOCAB.verify(saved_ver)     # 不匹配则抛异常
    formula = data["formula"]

    mgr = AShareDataManager()
    mgr.load(start=start, end=end)
    feed = ReplayFeed(mgr)
    ranker = FactorRanker(formula=formula)
    engine = PortfolioEngine(initial_capital=INITIAL_CAPITAL)

    dates = feed.trade_dates()
    if sim_start in dates:
        start_i = dates.index(sim_start)
    else:
        start_i = max(warmup, 0)
    start_i = max(start_i, warmup)

    for i in range(start_i, len(dates) - 1):
        d, nd = dates[i], dates[i + 1]
        feat_slice, valid = feed.slice_until(d)
        picks = ranker.rank(feat_slice, valid, mgr.symbols, top_k=TOP_K)
        engine.step(d, nd, picks, feed.bar_at(nd))
        print(f"[{nd}] nav={engine.account.nav_history[-1][1]:,.0f} "
              f"持仓={len(engine.account.holdings)}")

    metrics = compute_metrics(engine.account.nav_history, INITIAL_CAPITAL)

    # 基准对比：同期沪深300收益 + 超额 alpha
    from data_pipeline.ashare_fetcher import AShareFetcher
    bench = AShareFetcher().benchmark(dates[start_i], dates[-1])
    bench_vals = [bench[d] for d in dates[start_i:] if d in bench]
    if len(bench_vals) >= 2:
        bench_ret = bench_vals[-1] / bench_vals[0] - 1.0
        metrics["benchmark_return"] = bench_ret
        metrics["excess_return"] = metrics["total_return"] - bench_ret
    else:
        metrics["benchmark_return"] = None
        metrics["excess_return"] = None

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "equity.json").write_text(
        json.dumps(engine.account.nav_history, ensure_ascii=False), encoding="utf-8")
    (OUT_DIR / "trades.json").write_text(
        json.dumps(engine.account.trades, ensure_ascii=False), encoding="utf-8")
    (OUT_DIR / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[回放完成] 总收益={metrics['total_return']:.2%} "
          f"最大回撤={metrics['max_drawdown']:.2%} 夏普={metrics['sharpe']:.2f}")
    return metrics


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", default="strategies/best_ashare_universe.json")
    ap.add_argument("--start", default="2023-01-01")
    ap.add_argument("--end", default="2026-06-30")
    ap.add_argument("--sim-start", default="2026-06-01")
    args = ap.parse_args()
    run_replay(args.strategy, args.start, args.end, args.sim_start)
