"""Check data availability per group - how many bars, what time range"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from config import Config
from data_pipeline.data_manager import MT5DataManager
from data_pipeline.fetcher import MT5DataFetcher

groups = {
    "forex": ["EURUSD", "USDJPY"],
    "metals_comm": ["XAUUSD", "AAVUSD", "COCOA.c"],
    "index": ["US30.cash", "US100.cash", "US500.cash", "US2000.cash", "JP225.cash"],
}

with MT5DataFetcher(offline=True) as fetcher:
    for gname, symbols in groups.items():
        Config.SYMBOLS = symbols
        mgr = MT5DataManager(fetcher)
        mgr.load()
        T = mgr.raw_dict["open"].shape[1]
        years = T / 6240
        print(f"{gname:14s}: {mgr.symbols}  T={T}  years={years:.2f}")
