"""benchmark_speed.py — 诊断训练速度瓶颈"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from config import Config
from data_pipeline.data_manager import MT5DataManager
from data_pipeline.fetcher import MT5DataFetcher
from model_core.config import ModelConfig
from model_core.engine import AlphaEngine


def run_benchmark(name, max_steps=10):
    offline = True
    original_symbols = Config.SYMBOLS[:]
    Config.SYMBOLS = Config.SYMBOL_GROUPS["index"]
    try:
        with MT5DataFetcher(offline=offline) as fetcher:
            mgr = MT5DataManager(fetcher)
            mgr.load()
            engine = AlphaEngine(data_manager=mgr)
            t0 = time.time()
            engine.train(start_step=0, end_step=max_steps, verbose_header=False)
            elapsed = time.time() - t0
            print(f"\n[{name}] {max_steps} steps in {elapsed:.1f}s = {elapsed/max_steps:.2f}s/it")
    finally:
        Config.SYMBOLS = original_symbols


if __name__ == "__main__":
    print(f"Benchmark: formula_len={ModelConfig.MAX_FORMULA_LEN}, single model, 10 steps")
    run_benchmark("speed", max_steps=10)
