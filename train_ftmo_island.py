"""
train_ftmo_island.py — FTMO 专属 + Island 多起点并行训练

训练 index 组，使用 IslandAlphaEngine：
  - 3 个独立模型（islands）
  - 每 200 步迁移一次 elite
  - 自适应噪声、部分参数重置、elite decay 已启用
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from config import Config
from data_pipeline.data_manager import MT5DataManager
from data_pipeline.fetcher import MT5DataFetcher
from model_core.config import ModelConfig
from model_core.island_engine import IslandAlphaEngine


def main():
    offline = "--offline" in sys.argv
    t0 = time.time()

    print(f"\n{'='*60}")
    print(f"  FTMO + Island 多起点并行训练 — index 组")
    print(f"{'='*60}")
    print(f"  奖励模式: {ModelConfig.REWARD_MODE}")
    print(f"  公式长度: {ModelConfig.MAX_FORMULA_LEN}")
    print(f"  Islands : {ModelConfig.N_ISLANDS}")
    print(f"  Migration: every {ModelConfig.MIGRATION_INTERVAL} steps, Top-{ModelConfig.MIGRATION_TOP_K}")
    print(f"  Adaptive noise: {ModelConfig.ADAPTIVE_NOISE}")
    print(f"  Partial reset: {ModelConfig.PARTIAL_RESET}")
    print(f"  Elite decay: {ModelConfig.ELITE_DECAY}")
    print(f"  品种: {Config.SYMBOL_GROUPS['index']}")
    print(f"{'='*60}\n")

    with MT5DataFetcher(offline=offline) as fetcher:
        # 临时切换 SYMBOLS 到 index 组
        original_symbols = Config.SYMBOLS[:]
        Config.SYMBOLS = Config.SYMBOL_GROUPS["index"]
        try:
            mgr = MT5DataManager(fetcher)
            mgr.load()
            engine = IslandAlphaEngine(mgr)
            engine.train()

            formula, score = engine.get_global_best()
            print(f"\n最终全局最优: score={score:.4f}")
            print(f"公式: {formula}")
        finally:
            Config.SYMBOLS = original_symbols

    elapsed = time.time() - t0
    print(f"\n总耗时 {elapsed/3600:.2f}h")


if __name__ == "__main__":
    main()
