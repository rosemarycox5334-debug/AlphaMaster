"""
train_ashare.py — A股截面因子训练入口

用 AShareDataManager 喂 AlphaEngine 多品种截面模式（target_symbol=None），
REWARD_MODE="ashare"，产出 strategies/best_ashare_universe.json
（可解释 token 公式 + vocab 校验）。

用法:
    python train_ashare.py --start 2023-01-01 --end 2026-05-31
    python train_ashare.py --steps 100          # 短跑调试
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from data_pipeline.ashare_manager import AShareDataManager
from model_core.config import ModelConfig
from model_core.engine import AlphaEngine
from model_core.vocab import VOCAB_VERSION

_UNIVERSE_TAG = "ashare_universe"


def train_ashare(fetcher=None, start="2023-01-01", end="2026-05-31",
                 steps: int | None = None) -> AlphaEngine | None:
    """加载 A股宇宙数据并训练截面因子模型。

    Args:
        fetcher: 数据源（默认 AShareFetcher）。测试可注入 TinyFetcher。
        start/end: 训练区间（交易日历由 fetcher.trade_calendar 决定）。
        steps: 覆盖 ModelConfig.TRAIN_STEPS（None=用配置默认）。

    Returns:
        训练完成的 AlphaEngine（best_formula 可能为 None，若 steps 内无有效公式）。
    """
    ModelConfig.REWARD_MODE = "ashare"
    if steps is not None:
        ModelConfig.TRAIN_STEPS = steps

    mgr = AShareDataManager(fetcher=fetcher)
    mgr.load(start=start, end=end)
    print(f"[A股训练] {len(mgr.symbols)} 只 × {len(mgr.trade_dates)} 交易日")

    engine = AlphaEngine(data_manager=mgr, target_symbol=None)
    engine.mode = "ashare_universe"
    engine.train(start_step=0)
    _save_strategy(engine)
    return engine


def _save_strategy(engine: AlphaEngine) -> None:
    path = pathlib.Path("strategies") / f"best_{_UNIVERSE_TAG}.json"
    path.parent.mkdir(exist_ok=True)
    data = {
        "vocab_version": VOCAB_VERSION,
        "symbol": _UNIVERSE_TAG,
        "mode": "ashare_universe",
        "formula": engine.best_formula,
        "formula_decoded": engine._decode_formula(engine.best_formula)
        if engine.best_formula else None,
        "best_score": engine.best_score,
        "train_steps": ModelConfig.TRAIN_STEPS,
    }
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[A股训练] 策略已保存: {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2023-01-01")
    ap.add_argument("--end", default="2026-05-31")
    ap.add_argument("--steps", type=int, default=None)
    args = ap.parse_args()
    eng = train_ashare(start=args.start, end=args.end, steps=args.steps)
    if eng and eng.best_formula:
        print(f"最优公式: {eng._decode_formula(eng.best_formula)}  分数={eng.best_score:.4f}")
