"""Deep dive: why index factor is always positive"""
import sys, json, math
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from config import Config
from data_pipeline.data_manager import MT5DataManager
from data_pipeline.fetcher import MT5DataFetcher
from model_core.vocab import FORMULA_VOCAB, VOCAB_VERSION
from model_core.vm import StackVM
from model_core.features import MT5FeatureEngineer
from strategy_manager.signal import compute_target_positions_stateless

# Load index strategy
data = json.load(open("strategies/best_index.json"))
formula = data["formula"]
names = FORMULA_VOCAB.token_names
print(f"Formula tokens: {formula}")
print(f"Decode: {' -> '.join(names[t] for t in formula)}")
print()

# Load index group data
Config.SYMBOLS = ["US30.cash", "US100.cash", "US500.cash", "US2000.cash", "JP225.cash"]
with MT5DataFetcher(offline=True) as fetcher:
    mgr = MT5DataManager(fetcher)
    mgr.load()
    
    raw_dict = mgr.raw_dict
    syms = mgr.symbols
    T = raw_dict["open"].shape[1]
    feat = MT5FeatureEngineer.compute_features(raw_dict)
    target_ret = mgr.target_ret
    
    # Execute formula step by step
    vm = StackVM()
    
    # Manually trace each step
    stack = []
    print(f"Feature tensor shape: {feat.shape}  (N={feat.shape[0]}, F={feat.shape[1]}, T={feat.shape[2]})")
    print()
    
    for i, token in enumerate(formula):
        token = int(token)
        if token < vm.feat_offset:
            fname = names[token] if token < len(names) else f"feat_{token}"
            val = feat[:, token, :]
            stack.append(val)
            print(f"  Step {i}: PUSH {fname} (token={token})  shape={val.shape}  mean={val.float().mean():.4f}  std={val.float().std():.4f}  min={val.float().min():.4f}  max={val.float().max():.4f}")
        elif token in vm.op_map:
            oname = names[token] if token < len(names) else f"op_{token}"
            arity = vm.arity_map[token]
            args = []
            for _ in range(arity):
                args.append(stack.pop())
            args.reverse()
            
            # Stats before
            for j, a in enumerate(args):
                print(f"  Step {i}: arg{j} for {oname}: mean={a.float().mean():.4f}  std={a.float().std():.4f}  min={a.float().min():.4f}  max={a.float().max():.4f}  neg%={float((a<0).float().mean())*100:.1f}%")
            
            res = vm.op_map[token](*args)
            # Check for nans
            nan_pct = float(torch.isnan(res).float().mean()) * 100
            print(f"  Step {i}: {oname} -> mean={res.float().mean():.4f}  std={res.float().std():.4f}  min={res.float().min():.4f}  max={res.float().max():.4f}  neg%={float((res<0).float().mean())*100:.1f}%  nan%={nan_pct:.1f}%")
            stack.append(res)
        else:
            print(f"  Step {i}: UNKNOWN token {token}")
    
    if len(stack) == 1:
        raw_factor = stack[0]
        print(f"\n=== RAW FACTOR (before normalize) ===")
        print(f"  mean={raw_factor.float().mean():.4f}  std={raw_factor.float().std():.4f}")
        print(f"  min={raw_factor.float().min():.4f}  max={raw_factor.float().max():.4f}")
        print(f"  neg%={float((raw_factor<0).float().mean())*100:.1f}%  pos%={float((raw_factor>0).float().mean())*100:.1f}%")
        
        normalized = vm._normalize_output(raw_factor)
        print(f"\n=== AFTER NORMALIZE ===")
        print(f"  mean={normalized.float().mean():.4f}  std={normalized.float().std():.4f}")
        print(f"  min={normalized.float().min():.4f}  max={normalized.float().max():.4f}")
        print(f"  neg%={float((normalized<0).float().mean())*100:.1f}%  pos%={float((normalized>0).float().mean())*100:.1f}%")
        
        pos = compute_target_positions_stateless(normalized)
        print(f"\n=== AFTER tanh (positions) ===")
        print(f"  mean={pos.float().mean():.4f}  std={pos.float().std():.4f}")
        print(f"  min={pos.float().min():.4f}  max={pos.float().max():.4f}")
        print(f"  long%={float((pos>0.05).float().mean())*100:.1f}%  short%={float((pos<-0.05).float().mean())*100:.1f}%  flat%={float((pos.abs()<0.05).float().mean())*100:.1f}%")
        print(f"  avg_abs_pos={float(pos.abs().mean()):.4f}")
        
        # Check per-sym
        print(f"\n=== PER-SYMBOL POSITION DISTRIBUTION ===")
        for i, sym in enumerate(syms):
            p = pos[i]
            long_pct = float((p > 0.05).float().mean()) * 100
            short_pct = float((p < -0.05).float().mean()) * 100
            flat_pct = float((p.abs() < 0.05).float().mean()) * 100
            print(f"  {sym:<14s}: long={long_pct:.1f}%  short={short_pct:.1f}%  flat={flat_pct:.1f}%  mean={p.float().mean():+.4f}")
