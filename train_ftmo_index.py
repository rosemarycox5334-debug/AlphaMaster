"""
train_ftmo_index.py — FTMO 专属训练，仅训练 index 组
（跳过 forex 和 metals_comm，直接训练 index）
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from config import Config
from data_pipeline.fetcher import MT5DataFetcher
from model_core.config import ModelConfig
from main import train_group


def main():
    offline = "--offline" in sys.argv
    t0 = time.time()

    print(f"\n{'='*60}")
    print(f"  FTMO 专属因子训练 — index 组")
    print(f"{'='*60}")
    print(f"  奖励模式: REWARD_MODE = '{ModelConfig.REWARD_MODE}'")
    print(f"  训练组 : index")
    print(f"  品种   : {Config.SYMBOL_GROUPS['index']}")
    print(f"  offline={offline}")
    print(f"{'='*60}")

    with MT5DataFetcher(offline=offline) as fetcher:
        gsyms = Config.SYMBOL_GROUPS["index"]
        eng = train_group(fetcher, "index", gsyms, offline)
        if eng is not None:
            print(f"\n<<< [index] 完成: score={eng.best_score:.4f}")
            print(f"    {eng._decode_formula(eng.best_formula)}")
        else:
            print("\n<<< [index] 失败")

    elapsed = time.time() - t0
    print(f"\n耗时 {elapsed/3600:.2f}h")


if __name__ == "__main__":
    main()
