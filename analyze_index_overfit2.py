"""Analyze index group overfitting: check time-period breakdown of the factor"""
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

_H1_PER_YEAR = 6240

# Load index strategy
data = json.load(open("strategies/best_index.json"))
formula = data["formula"]
print(f"Formula: {formula}")
print(f"Decode: ", end="")
names = FORMULA_VOCAB.token_names
print(" -> ".join(names[t] if 0 <= t < len(names) else f"?{t}" for t in formula))
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
    
    print(f"Symbols: {syms}")
    print(f"T={T} bars ({T/_H1_PER_YEAR:.2f} years)")
    
    # Get bar times - just use bar index for simplicity
    bar_times = None
    
    # Run factor
    vm = StackVM()
    factor = vm.execute(formula, feat)
    factor_np = factor.detach().numpy()
    
    # Compute positions and PnL
    pos = compute_target_positions_stateless(factor)
    pos_np = pos.detach().numpy()
    target_np = target_ret.detach().numpy()
    
    cost_rate = 0.0003
    prev = np.zeros_like(pos_np)
    prev[:, 1:] = pos_np[:, :-1]
    turnover = np.abs(pos_np - prev)
    pnl = pos_np * target_np - turnover * cost_rate
    
    port_pnl = pnl.mean(axis=0)
    port_cum = np.cumsum(port_pnl)
    
    # Split into 8 equal segments
    n_seg = 8
    seg_len = T // n_seg
    print(f"\n=== {n_seg}-SEGMENT BREAKDOWN ===")
    print(f"{'Seg':>4s} {'Bars':>6s} {'Period':>22s} {'Return':>9s} {'Sharpe':>8s} {'MDD':>8s} {'WinRate':>8s} {'AvgPos':>7s}")
    
    for seg in range(n_seg):
        s = seg * seg_len
        e = min((seg + 1) * seg_len, T)
        seg_pnl = port_pnl[s:e]
        seg_cum = np.cumsum(seg_pnl)
        
        t_start = f"bar{s}"
        t_end = f"bar{e}"
        
        seg_ret = float(seg_cum[-1])
        m, sd = seg_pnl.mean(), seg_pnl.std()
        seg_sharpe = m / (sd + 1e-10) * math.sqrt(_H1_PER_YEAR)
        seg_mdd = float((np.maximum.accumulate(seg_cum) - seg_cum).max())
        
        # Win rate per bar
        win_rate = float((seg_pnl > 0).mean())
        
        # Avg position
        seg_pos = pos_np[:, s:e]
        avg_pos = float(np.abs(seg_pos).mean())
        
        print(f"{seg+1:>4d} {e-s:>6d} {t_start+' ~ '+t_end:>22s} {seg_ret:>+9.4f} {seg_sharpe:>+8.3f} {seg_mdd:>8.4f} {win_rate*100:>7.1f}% {avg_pos:>7.3f}")
    
    # Per-year analysis
    print(f"\n=== PER-YEAR BREAKDOWN ===")
    n_years = T // _H1_PER_YEAR
    print(f"{'Year':>6s} {'Bars':>6s} {'Return':>9s} {'Sharpe':>8s} {'MDD':>8s} {'WinRate':>8s}")
    
    for yr in range(n_years + 1):
        s = yr * _H1_PER_YEAR
        e = min((yr + 1) * _H1_PER_YEAR, T)
        if s >= T:
            break
        seg_pnl = port_pnl[s:e]
        seg_cum = np.cumsum(seg_pnl)
        seg_ret = float(seg_cum[-1])
        m, sd = seg_pnl.mean(), seg_pnl.std()
        seg_sharpe = m / (sd + 1e-10) * math.sqrt(_H1_PER_YEAR)
        seg_mdd = float((np.maximum.accumulate(seg_cum) - seg_cum).max())
        win_rate = float((seg_pnl > 0).mean())
        
        t_start = f"bar{s}"
        t_end = f"bar{e}"
        
        print(f"  Y{yr+1:>2d}  {e-s:>6d} {seg_ret:>+9.4f} {seg_sharpe:>+8.3f} {seg_mdd:>8.4f} {win_rate*100:>7.1f}%  ({t_start} ~ {t_end})")
    
    # Per-symbol per-half analysis
    print(f"\n=== PER-SYMBOL PER-HALF ANALYSIS ===")
    split = T // 2
    print(f"{'Symbol':<14s} {'H1_Ret':>9s} {'H2_Ret':>9s} {'H1_Sharpe':>10s} {'H2_Sharpe':>10s} {'H1_MDD':>8s} {'H2_MDD':>8s}")
    for i, sym in enumerate(syms):
        p1 = pnl[i, :split]
        p2 = pnl[i, split:]
        c1 = np.cumsum(p1)
        c2 = np.cumsum(p2)
        r1 = float(c1[-1])
        r2 = float(c2[-1])
        s1 = p1.mean() / (p1.std() + 1e-10) * math.sqrt(_H1_PER_YEAR)
        s2 = p2.mean() / (p2.std() + 1e-10) * math.sqrt(_H1_PER_YEAR)
        mdd1 = float((np.maximum.accumulate(c1) - c1).max())
        mdd2 = float((np.maximum.accumulate(c2) - c2).max())
        print(f"{sym:<14s} {r1:>+9.4f} {r2:>+9.4f} {s1:>+10.3f} {s2:>+10.3f} {mdd1:>8.4f} {mdd2:>8.4f}")
    
    # Factor value distribution over time
    print(f"\n=== FACTOR VALUE STATISTICS OVER TIME ===")
    for seg in range(n_seg):
        s = seg * seg_len
        e = min((seg + 1) * seg_len, T)
        seg_factor = factor_np[:, s:e]
        seg_pos = pos_np[:, s:e]
        print(f"Seg {seg+1}: factor mean={seg_factor.mean():+.4f} std={seg_factor.std():.4f}  "
              f"pos_mean={seg_pos.mean():+.4f} pos_abs={np.abs(seg_pos).mean():.4f}  "
              f"long%={float((seg_pos>0.01).mean())*100:.1f}% short%={float((seg_pos<-0.01).mean())*100:.1f}% flat%={float((np.abs(seg_pos)<0.01).mean())*100:.1f}%")
    
    # Check if factor is basically "always long"
    print(f"\n=== POSITION DISTRIBUTION (overall) ===")
    for i, sym in enumerate(syms):
        p = pos_np[i]
        long_pct = float((p > 0.01).mean()) * 100
        short_pct = float((p < -0.01).mean()) * 100
        flat_pct = float((np.abs(p) < 0.01).mean()) * 100
        print(f"  {sym:<14s}: long={long_pct:.1f}%  short={short_pct:.1f}%  flat={flat_pct:.1f}%  avg_abs_pos={np.abs(p).mean():.3f}")
