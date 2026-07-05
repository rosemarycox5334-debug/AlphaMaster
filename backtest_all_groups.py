"""
backtest_all_groups.py — 对所有组的当前最优因子做完整回测验证

用法：
    python backtest_all_groups.py --offline

分别加载 forex / metals_comm / index 三组的最优因子，
在全量历史数据上回测，输出：回测摘要、品种级详情、前后半段一致性、资金曲线图。
"""
import sys, json, math
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, str(Path(__file__).parent))

from config import Config
from data_pipeline.data_manager import MT5DataManager
from data_pipeline.fetcher import MT5DataFetcher
from model_core.vocab import FORMULA_VOCAB, VOCAB_VERSION
from model_core.vm import StackVM
from model_core.features import MT5FeatureEngineer
from strategy_manager.signal import compute_target_positions_stateless

_H1_PER_YEAR = 6240
OUTPUT_DIR = "backtest_output"

GROUP_CONFIG = {
    "forex":       {"symbols": Config.SYMBOL_GROUPS["forex"],       "cost_rate": 0.00015},
    "metals_comm": {"symbols": Config.SYMBOL_GROUPS["metals_comm"], "cost_rate": 0.00025},
    "index":       {"symbols": Config.SYMBOL_GROUPS["index"],       "cost_rate": 0.00030},
}


def decode(toks):
    names = FORMULA_VOCAB.token_names
    return " -> ".join(names[t] if 0 <= t < len(names) else f"?{t}" for t in toks)


def calc_sharpe(pnl):
    m, s = pnl.mean(), pnl.std()
    return float(m / (s + 1e-10) * math.sqrt(_H1_PER_YEAR))


def calc_sortino(pnl):
    m = pnl.mean()
    down = pnl[pnl < 0]
    ds = down.std() if len(down) > 0 else 1e-10
    ds = max(ds, abs(m), 1e-10)
    return float(np.clip(m / ds * math.sqrt(_H1_PER_YEAR), -20, 20))


def calc_mdd(cum_pnl):
    peak = np.maximum.accumulate(cum_pnl)
    return float((peak - cum_pnl).max())


def calc_ic(factor, target_ret):
    N, T = factor.shape
    ic_list = []
    for n in range(N):
        x = factor[n, :-1]
        y = target_ret[n, 1:]
        xm = x - x.mean()
        ym = y - y.mean()
        sx = np.sqrt((xm**2).mean())
        sy = np.sqrt((ym**2).mean())
        if sx < 1e-6 or sy < 1e-6:
            continue
        ic = (xm * ym).mean() / (sx * sy + 1e-8)
        ic_list.append(float(ic))
    return float(np.mean(ic_list)) if ic_list else 0.0


def calc_annual_return(total_ret, T):
    years = T / _H1_PER_YEAR
    return float(total_ret / years) if years > 0 else 0.0


def calc_calmar(total_ret, mdd, T):
    ann = calc_annual_return(total_ret, T)
    return float(ann / mdd) if mdd > 1e-8 else 0.0


def _max_consecutive_loss(pnl):
    max_streak = 0
    current = 0
    for p in pnl:
        if p < 0:
            current += 1
            max_streak = max(max_streak, current)
        else:
            current = 0
    return max_streak


def backtest_one(formula, feat, target_ret, symbols, cost_rate):
    vm = StackVM()
    factor = vm.execute(formula, feat)
    if factor is None:
        return None

    N, T = factor.shape
    factor_np = factor.detach().numpy()
    target_np = target_ret.detach().numpy()

    pos = compute_target_positions_stateless(factor)
    pos_np = pos.detach().numpy()

    prev = np.zeros_like(pos_np)
    prev[:, 1:] = pos_np[:, :-1]
    turnover = np.abs(pos_np - prev)

    pnl = pos_np * target_np - turnover * cost_rate

    per_sym = {}
    for i, sym in enumerate(symbols[:N]):
        p = pnl[i]
        cum = np.cumsum(p)
        n_trades = int((np.abs(np.diff(pos_np[i])) > 0.5).sum())
        per_sym[sym] = {
            "pnl":       p,
            "cum":       cum,
            "total_ret": float(cum[-1]),
            "ann_ret":   calc_annual_return(float(cum[-1]), T),
            "sharpe":    calc_sharpe(p),
            "sortino":   calc_sortino(p),
            "mdd":       calc_mdd(cum),
            "n_trades":  n_trades,
            "avg_hold":  float(T / max(1, n_trades)) if n_trades > 0 else float(T),
            "ic":        calc_ic(factor_np[i:i+1], target_np[i:i+1]),
        }

    port_pnl = pnl.mean(axis=0)
    port_cum = np.cumsum(port_pnl)
    ic = calc_ic(factor_np, target_np)

    split = T // 2
    p1 = port_pnl[:split]
    p2 = port_pnl[split:]

    port_total_ret = float(port_cum[-1])
    port_mdd = calc_mdd(port_cum)
    port_ann_ret = calc_annual_return(port_total_ret, T)
    port_calmar = calc_calmar(port_total_ret, port_mdd, T)

    return {
        "formula":        formula,
        "readable":       decode(formula),
        "per_sym":        per_sym,
        "port_pnl":       port_pnl,
        "port_cum":       port_cum,
        "port_total_ret": port_total_ret,
        "port_ann_ret":   port_ann_ret,
        "port_sharpe":    calc_sharpe(port_pnl),
        "port_sortino":   calc_sortino(port_pnl),
        "port_mdd":       port_mdd,
        "port_calmar":    port_calmar,
        "ic":             ic,
        "n_pos_syms":     sum(1 for d in per_sym.values() if d["total_ret"] > 0),
        "n_syms":         N,
        "half1_sharpe":   calc_sharpe(p1),
        "half2_sharpe":   calc_sharpe(p2),
        "half1_sortino":  calc_sortino(p1),
        "half2_sortino":  calc_sortino(p2),
        "half1_ret":      float(np.cumsum(p1)[-1]),
        "half2_ret":      float(np.cumsum(p2)[-1]),
        "avg_turnover":   float(turnover.mean()),
        "avg_hold_h":     float(np.mean([d["avg_hold"] for d in per_sym.values()])),
        "max_consec_loss": _max_consecutive_loss(port_pnl),
    }


def plot_group(result, symbols, group_name, output_dir, cost_rate):
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(18, 12), dpi=110)
    gs = gridspec.GridSpec(3, 1, height_ratios=[3, 1, 2], hspace=0.15)

    ax_eq = fig.add_subplot(gs[0])
    ax_dd = fig.add_subplot(gs[1], sharex=ax_eq)
    ax_sym = fig.add_subplot(gs[2], sharex=ax_eq)

    T = len(result["port_pnl"])
    x = np.arange(T)

    port_cum = result["port_cum"]
    ax_eq.plot(x, port_cum, linewidth=2.0, color="#1565c0",
               label=f"Portfolio (AnnRet={result['port_ann_ret']:+.4f}, Sortino={result['port_sortino']:+.2f}, MDD={result['port_mdd']:.3f})")
    ax_eq.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax_eq.set_ylabel("Cumulative Return", fontsize=10)
    ax_eq.legend(loc="upper left", fontsize=9, framealpha=0.85)
    ax_eq.grid(alpha=0.25)
    ax_eq.set_title(f"{group_name} group backtest | {' + '.join(symbols)} equal weight\n"
                    f"Formula: {result['readable']}\n"
                    f"Cost: {cost_rate}  T={T} bars H1", fontsize=11, pad=8)

    peak = np.maximum.accumulate(port_cum)
    dd = port_cum - peak
    ax_dd.fill_between(x, dd, 0, alpha=0.4, color="#1565c0", label="Drawdown")
    ax_dd.axhline(0, color="gray", linewidth=0.5)
    ax_dd.set_ylabel("Drawdown", fontsize=9)
    ax_dd.grid(alpha=0.2)
    ax_dd.legend(loc="lower left", fontsize=8)

    colors = ["#e65100", "#00897b", "#6a1b9a", "#b71c1c", "#26418f"]
    for i, sym in enumerate(symbols):
        if sym in result["per_sym"]:
            cum = result["per_sym"][sym]["cum"]
            c = colors[i % len(colors)]
            ax_sym.plot(x, cum, linewidth=1.2, color=c, alpha=0.8,
                        label=f"{sym} (Ret={result['per_sym'][sym]['total_ret']:+.3f})")
    ax_sym.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax_sym.set_ylabel("Per-Symbol Cum Return", fontsize=9)
    ax_sym.set_xlabel("Bar Index (H1)", fontsize=9)
    ax_sym.legend(loc="upper left", fontsize=7, ncol=3, framealpha=0.85)
    ax_sym.grid(alpha=0.2)

    plt.tight_layout()
    plot_path = Path(output_dir) / f"{group_name}_backtest.png"
    fig.savefig(plot_path, bbox_inches="tight")
    plt.close(fig)
    return str(plot_path)


def run_group_backtest(group_name, group_cfg, fetcher, multi_mgr):
    symbols = group_cfg["symbols"]
    cost_rate = group_cfg["cost_rate"]
    strategy_path = Path(f"strategies/best_{group_name}.json")

    print(f"\n{'='*70}")
    print(f"  [{group_name}] group backtest")
    print(f"{'='*70}")

    if not strategy_path.exists():
        print(f"  [SKIP] No strategy file: {strategy_path}")
        return None

    data = json.load(open(strategy_path))
    if data.get("vocab_version", "unknown") != VOCAB_VERSION:
        print(f"  [SKIP] Vocab mismatch: {data.get('vocab_version')} != {VOCAB_VERSION}")
        return None

    formula = data["formula"]
    best_score = data.get("best_score", 0.0)
    print(f"  Formula tokens : {formula}")
    print(f"  Readable       : {decode(formula)}")
    print(f"  Train score    : {best_score:.4f}")
    print(f"  Symbols        : {symbols}")
    print(f"  Cost rate      : {cost_rate}")

    # Load group data independently
    Config.SYMBOLS = list(symbols)
    group_mgr = MT5DataManager(fetcher)
    group_mgr.load()
    Config.SYMBOLS = list(Config.SYMBOL_GROUPS) and []  # reset
    # Actually restore from the full config
    from config import Config as _Cfg
    _Cfg.SYMBOLS = ["EURUSD", "USDJPY", "XAUUSD", "AAVUSD", "COCOA.c",
                    "US30.cash", "US100.cash", "US500.cash", "US2000.cash", "JP225.cash"]

    raw_dict = group_mgr.raw_dict
    syms_loaded = group_mgr.symbols
    T = raw_dict["open"].shape[1]
    print(f"  Loaded: {syms_loaded}  T={T} bars")

    feat = MT5FeatureEngineer.compute_features(raw_dict)
    target_ret = group_mgr.target_ret

    result = backtest_one(formula, feat, target_ret, syms_loaded, cost_rate)
    if result is None:
        print(f"  [ERROR] Factor execution failed!")
        return None

    # Print summary
    print(f"\n  --- Portfolio Summary ({T} bars H1, {len(syms_loaded)} symbols) ---")
    print(f"  Total Return : {result['port_total_ret']:+.4f}")
    print(f"  Annual Return: {result['port_ann_ret']:+.4f}")
    print(f"  Sharpe       : {result['port_sharpe']:+.4f}")
    print(f"  Sortino      : {result['port_sortino']:+.4f}")
    print(f"  Max DD       : {result['port_mdd']:.4f}")
    print(f"  Calmar       : {result['port_calmar']:+.4f}")
    print(f"  IC           : {result['ic']:.5f}")
    print(f"  Avg Turnover : {result['avg_turnover']:.6f}")
    print(f"  Avg Hold     : {result['avg_hold_h']:.1f}h")
    print(f"  Max Consec Loss: {result['max_consec_loss']} bars")
    print(f"  Profitable   : {result['n_pos_syms']}/{result['n_syms']}")

    print(f"\n  --- Half-split Consistency ---")
    print(f"  H1: Sharpe={result['half1_sharpe']:+.4f}  Ret={result['half1_ret']:+.4f}")
    print(f"  H2: Sharpe={result['half2_sharpe']:+.4f}  Ret={result['half2_ret']:+.4f}")
    h1, h2 = result["half1_sharpe"], result["half2_sharpe"]
    if h1 > 0 and h2 > 0:
        print(f"  [OK] Both halves positive, good consistency")
    elif h1 * h2 > 0:
        print(f"  [WARN] Same sign but negative")
    else:
        print(f"  [FAIL] Opposite signs, overfitting suspected")

    print(f"\n  --- Per-Symbol Detail ---")
    for sym, d in result["per_sym"].items():
        sig = "[OK]" if d["total_ret"] > 0 else "[X]"
        print(f"  {sym:14s}: Ret={d['total_ret']:+.4f}  Ann={d['ann_ret']:+.4f}  "
              f"Sharpe={d['sharpe']:+.4f}  Sortino={d['sortino']:+.4f}  "
              f"MDD={d['mdd']:.4f}  IC={d['ic']:.5f}  "
              f"Trades={d['n_trades']}  Hold={d['avg_hold']:.0f}h  {sig}")

    # FTMO check
    print(f"\n  --- FTMO Assessment ---")
    issues = []
    if result["port_mdd"] > 0.10:
        issues.append(f"[WARN] MDD={result['port_mdd']:.4f} > 10% FTMO Max Loss")
    else:
        print(f"  [OK] MDD={result['port_mdd']:.4f} < 10%")
    if result["port_ann_ret"] > 0:
        print(f"  [OK] AnnRet={result['port_ann_ret']:+.4f} positive")
    else:
        issues.append(f"[WARN] Negative annual return")
    if result["max_consec_loss"] > 500:
        issues.append(f"[WARN] Max consec loss {result['max_consec_loss']} bars too long")
    if result["avg_hold_h"] < 2:
        issues.append(f"[WARN] Avg hold {result['avg_hold_h']:.1f}h too short")
    if result["n_pos_syms"] < result["n_syms"] // 2:
        issues.append(f"[WARN] Only {result['n_pos_syms']}/{result['n_syms']} profitable")
    for iss in issues:
        print(f"    {iss}")
    if not issues:
        print(f"  [OK] No obvious FTMO violations")

    # Plot
    plot_path = plot_group(result, syms_loaded, group_name, OUTPUT_DIR, cost_rate)
    print(f"\n  Plot saved -> {plot_path}")

    # Save JSON report
    report = {
        "group":         group_name,
        "formula":       formula,
        "readable":      result["readable"],
        "best_score":    best_score,
        "vocab_version": VOCAB_VERSION,
        "symbols":       syms_loaded,
        "T":             T,
        "cost_rate":     cost_rate,
        "port_total_ret": round(result["port_total_ret"], 6),
        "port_ann_ret":   round(result["port_ann_ret"], 6),
        "port_sharpe":    round(result["port_sharpe"], 4),
        "port_sortino":   round(result["port_sortino"], 4),
        "port_mdd":       round(result["port_mdd"], 6),
        "port_calmar":    round(result["port_calmar"], 4),
        "ic":             round(result["ic"], 6),
        "avg_turnover":   round(result["avg_turnover"], 8),
        "avg_hold_h":     round(result["avg_hold_h"], 1),
        "max_consec_loss": result["max_consec_loss"],
        "n_pos_syms":     result["n_pos_syms"],
        "n_syms":         result["n_syms"],
        "half1_sharpe":   round(result["half1_sharpe"], 4),
        "half2_sharpe":   round(result["half2_sharpe"], 4),
        "half1_ret":      round(result["half1_ret"], 6),
        "half2_ret":      round(result["half2_ret"], 6),
        "per_sym": {
            sym: {
                "total_ret": round(d["total_ret"], 6),
                "ann_ret":   round(d["ann_ret"], 6),
                "sharpe":    round(d["sharpe"], 4),
                "sortino":   round(d["sortino"], 4),
                "mdd":       round(d["mdd"], 6),
                "ic":        round(d["ic"], 6),
                "n_trades":  d["n_trades"],
                "avg_hold":  round(d["avg_hold"], 1),
            }
            for sym, d in result["per_sym"].items()
        },
    }
    report_path = Path(OUTPUT_DIR) / f"{group_name}_backtest.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"  Report saved -> {report_path}")

    return result


def main():
    offline = "--offline" in sys.argv

    # First, save metals_comm current best from latest checkpoint
    print("Checking for latest metals_comm checkpoint to save current best...")
    import glob
    ckpt_files = sorted(glob.glob(r"D:\cl\MT5_AlphaGPT\checkpoints\ckpt_metals_comm_step_*.pt"))
    if ckpt_files:
        import torch
        latest_ckpt = ckpt_files[-1]
        ckpt = torch.load(latest_ckpt, map_location='cpu', weights_only=False)
        best_formula = ckpt.get('best_formula')
        best_score = ckpt.get('best_score', 0)
        step = ckpt.get('step', 0)
        if best_formula and best_score > 0:
            # Update strategy file with latest best
            strategy_path = Path("strategies/best_metals_comm.json")
            strategy_path.parent.mkdir(exist_ok=True)
            strategy_path.write_text(json.dumps({
                "vocab_version": VOCAB_VERSION,
                "symbol": "metals_comm",
                "formula": best_formula,
                "best_score": best_score,
                "step": step,
            }, indent=2))
            print(f"  Updated best_metals_comm.json: score={best_score:.4f} step={step} formula={best_formula}")

    print(f"\n{'='*70}")
    print(f"  All-Groups Backtest Analysis")
    print(f"{'='*70}")

    with MT5DataFetcher(offline=offline) as fetcher:
        all_results = {}
        for gname, gcfg in GROUP_CONFIG.items():
            result = run_group_backtest(gname, gcfg, fetcher, None)
            if result:
                all_results[gname] = result

        # Cross-group summary
        if all_results:
            print(f"\n{'='*70}")
            print(f"  Cross-Group Comparison")
            print(f"{'='*70}")
            print(f"  {'Group':<14s} {'Score':>8s} {'AnnRet':>9s} {'Sharpe':>8s} {'Sortino':>8s} {'MDD':>8s} {'Calmar':>8s} {'IC':>8s} {'Pos/N':>6s}")
            print(f"  {'-'*14} {'-'*8} {'-'*9} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*6}")
            for gname, r in all_results.items():
                print(f"  {gname:<14s} "
                      f"{r.get('best_score',0):>8.4f} "
                      f"{r['port_ann_ret']:>+9.4f} "
                      f"{r['port_sharpe']:>+8.4f} "
                      f"{r['port_sortino']:>+8.4f} "
                      f"{r['port_mdd']:>8.4f} "
                      f"{r['port_calmar']:>+8.4f} "
                      f"{r['ic']:>8.5f} "
                      f"{r['n_pos_syms']:>2d}/{r['n_syms']:<2d}")

            # Combined portfolio (equal weight across groups)
            print(f"\n  --- Combined Portfolio (equal weight across groups) ---")
            # Align all groups to min T
            min_T = min(len(r["port_pnl"]) for r in all_results.values())
            combined_pnl = np.zeros(min_T)
            for r in all_results.values():
                combined_pnl += r["port_pnl"][:min_T]
            combined_pnl /= len(all_results)

            combined_cum = np.cumsum(combined_pnl)
            combined_mdd = calc_mdd(combined_cum)
            combined_ann = calc_annual_return(float(combined_cum[-1]), min_T)
            combined_sharpe = calc_sharpe(combined_pnl)
            combined_sortino = calc_sortino(combined_pnl)
            combined_calmar = calc_calmar(float(combined_cum[-1]), combined_mdd, min_T)

            print(f"  Total Return : {float(combined_cum[-1]):+.4f}")
            print(f"  Annual Return: {combined_ann:+.4f}")
            print(f"  Sharpe       : {combined_sharpe:+.4f}")
            print(f"  Sortino      : {combined_sortino:+.4f}")
            print(f"  Max DD       : {combined_mdd:.4f}")
            print(f"  Calmar       : {combined_calmar:+.4f}")

            # Plot combined
            fig, axes = plt.subplots(2, 1, figsize=(16, 8), dpi=110, gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.15})
            x = np.arange(min_T)
            colors_g = {"forex": "#2e7d32", "metals_comm": "#e65100", "index": "#1565c0"}
            for gname, r in all_results.items():
                axes[0].plot(x, np.cumsum(r["port_pnl"][:min_T]), linewidth=1.2, alpha=0.7,
                             color=colors_g.get(gname, "gray"), label=f"{gname}")
            axes[0].plot(x, combined_cum, linewidth=2.5, color="#d32f2f",
                         label=f"Combined (Ann={combined_ann:+.4f}, Sharpe={combined_sharpe:+.2f}, MDD={combined_mdd:.3f})")
            axes[0].axhline(0, color="gray", linewidth=0.5, linestyle="--")
            axes[0].legend(fontsize=9)
            axes[0].set_title("All Groups Combined Portfolio", fontsize=12)
            axes[0].grid(alpha=0.25)

            peak = np.maximum.accumulate(combined_cum)
            dd = combined_cum - peak
            axes[1].fill_between(x, dd, 0, alpha=0.4, color="#d32f2f")
            axes[1].axhline(0, color="gray", linewidth=0.5)
            axes[1].set_ylabel("Drawdown", fontsize=9)
            axes[1].set_xlabel("Bar Index (H1)", fontsize=9)
            axes[1].grid(alpha=0.2)

            plt.tight_layout()
            combined_path = Path(OUTPUT_DIR) / "all_groups_combined.png"
            fig.savefig(combined_path, bbox_inches="tight")
            plt.close(fig)
            print(f"\n  Combined plot saved -> {combined_path}")

            # Save combined report
            combined_report = {
                "groups": list(all_results.keys()),
                "T": min_T,
                "total_ret": round(float(combined_cum[-1]), 6),
                "ann_ret": round(combined_ann, 6),
                "sharpe": round(combined_sharpe, 4),
                "sortino": round(combined_sortino, 4),
                "mdd": round(combined_mdd, 6),
                "calmar": round(combined_calmar, 4),
            }
            combined_report_path = Path(OUTPUT_DIR) / "all_groups_combined.json"
            with open(combined_report_path, "w") as f:
                json.dump(combined_report, f, indent=2, ensure_ascii=False)
            print(f"  Combined report saved -> {combined_report_path}")

    print(f"\n{'='*70}")
    print(f"  All-Groups Backtest Complete")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
