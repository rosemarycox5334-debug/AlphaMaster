"""Smoke test: verify training loop starts with new constraints"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from config import Config
from data_pipeline.data_manager import MT5DataManager
from data_pipeline.fetcher import MT5DataFetcher
from model_core.engine import AlphaEngine
from model_core.config import ModelConfig

Config.SYMBOLS = ["EURUSD", "USDJPY"]

with MT5DataFetcher(offline=True) as fetcher:
    mgr = MT5DataManager(fetcher)
    mgr.load()
    
    engine = AlphaEngine(data_manager=mgr, target_symbol="forex")
    
    # Run 3 steps to verify the training loop works
    print("Starting 3-step smoke test...")
    engine.train(start_step=0, end_step=3, verbose_header=True)
    
    print(f"\nSmoke test complete!")
    print(f"  best_score = {engine.best_score:.4f}")
    print(f"  best_formula = {engine.best_formula}")
    if engine.best_formula:
        from model_core.vocab import FORMULA_VOCAB
        names = FORMULA_VOCAB.token_names
        decoded = ' -> '.join(names[t] for t in engine.best_formula)
        print(f"  decoded = {decoded}")
        
        # Check infection status
        from model_core.vm import validate_formula_structure
        violations = validate_formula_structure(engine.best_formula, names)
        if violations:
            print(f"  ⚠ Structure violations: {len(violations)}")
            for v in violations:
                print(f"    - {v}")
        else:
            print(f"  ✅ No structure violations")
