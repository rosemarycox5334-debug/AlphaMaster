"""
backtest_detailed.py — 全组详细回测报告

输出：交易次数、平均持仓、胜率、盈亏比、最大单笔盈亏、平均盈亏、
      前后半段、滚动 Sharpe、品种相关性矩阵、月度收益等
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


def analyze_trades(pos_np, pnl_np, cost_rate):
    """分析交易细节：持仓变化点为交易事件"""
    N, T = pos_np.shape
    results = []
    for n in range(N):
        pos = pos_np[n]
        pnl = pnl_np[n]
        # 找到持仓变化点（开仓/平仓/翻转）
        changes = np.where(np.abs(np.diff(pos)) > 0.01)[0] + 1
        if len(changes) == 0:
            results.append({
                "n_trades": 0, "n_long": 0, "n_short": 0, "n_flat": 0,
                "win_rate": 0, "avg_win": 0, "avg_loss": 0, "profit_factor": 0,
                "max_win": 0, "max_loss": 0, "avg_trade_pnl": 0, "avg_hold_bars": 0,
            })
            continue

        # 把每段持仓作为一个 trade
        trade_bounds = []
        prev_idx = 0
        for c in changes:
            trade_bounds.append((prev_idx, c))
            prev_idx = c
        trade_bounds.append((prev_idx, T))

        n_trades = 0
        n_long = 0
        n_short = 0
        n_flat = 0
        wins = []
        losses = []
        hold_bars = []

        for (s, e) in trade_bounds:
            seg_pos = pos[s]
            seg_pnl = pnl[s:e].sum()
            seg_len = e - s
            if abs(seg_pos) < 0.01:
                n_flat += 1
                continue
            n_trades += 1
            if seg_pos > 0:
                n_long += 1
            else:
                n_short += 1
            hold_bars.append(seg_len)
            if seg_pnl > 0:
                wins.append(seg_pnl)
            else:
                losses.append(seg_pnl)

        win_rate = len(wins) / max(1, n_trades)
        avg_win = np.mean(wins) if wins else 0
        avg_loss = np.mean(losses) if losses else 0
        gross_win = sum(wins)
        gross_loss = abs(sum(losses))
        pf = gross_win / max(gross_loss, 1e-8)
        all_trades = wins + losses
        max_win = max(all_trades) if all_trades else 0
        max_loss = min(all_trades) if all_trades else 0
        avg_trade = np.mean(all_trades) if all_trades else 0
        avg_hold = np.mean(hold_bars) if hold_bars else 0

        results.append({
            "n_trades": n_trades,
            "n_long": n_long,
            "n_short": n_short,
            "n_flat": n_flat,
            "win_rate": float(win_rate),
            "avg_win": float(avg_win),
            "avg_loss": float(avg_loss),
            "profit_factor": float(pf),
            "max_win": float(max_win),
            "max_loss": float(max_loss),
            "avg_trade_pnl": float(avg_trade),
            "avg_hold_bars": float(avg_hold),
        })
    return results


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

    # 交易分析
    trade_stats = analyze_trades(pos_np, pnl, cost_rate)

    per_sym = {}
    for i, sym in enumerate(symbols[:N]):
        p = pnl[i]
        cum = np.cumsum(p)
        ts = trade_stats[i]
        per_sym[sym] = {
            "pnl":       p,
            "cum":       cum,
            "total_ret": float(cum[-1]),
            "ann_ret":   calc_annual_return(float(cum[-1]), T),
            "sharpe":    calc_sharpe(p),
            "sortino":   calc_sortino(p),
            "mdd":       calc_mdd(cum),
            "ic":        calc_ic(factor_np[i:i+1], target_np[i:i+1]),
            # 交易细节
            "n_trades":      ts["n_trades"],
            "n_long":        ts["n_long"],
            "n_short":       ts["n_short"],
            "win_rate":      ts["win_rate"],
            "avg_win":       ts["avg_win"],
            "avg_loss":      ts["avg_loss"],
            "profit_factor": ts["profit_factor"],
            "max_win":       ts["max_win"],
            "max_loss":      ts["max_loss"],
            "avg_trade_pnl": ts["avg_trade_pnl"],
            "avg_hold_bars": ts["avg_hold_bars"],
            "avg_hold_h":    ts["avg_hold_bars"],  # H1 每 bar = 1h
            "n_total_bars":  T,
        }

    port_pnl = pnl.mean(axis=0)
    port_cum = np.cumsum(port_pnl)
    ic = calc_ic(factor_np, target_np)

    # 组合交易统计
    port_pos = pos_np.mean(axis=0)
    port_trade = analyze_trades(port_pos.reshape(1, -1), port_pnl.reshape(1, -1), cost_rate)[0]

    # 分段
    split = T // 2
    p1 = port_pnl[:split]
    p2 = port_pnl[split:]

    # 三等分
    t1 = T // 3
    t2 = 2 * T // 3
    seg1 = port_pnl[:t1]
    seg2 = port_pnl[t1:t2]
    seg3 = port_pnl[t2:]

    port_total_ret = float(port_cum[-1])
    port_mdd = calc_mdd(port_cum)
    port_ann_ret = calc_annual_return(port_total_ret, T)
    port_calmar = calc_calmar(port_total_ret, port_mdd, T)

    # 逐年收益
    years = max(1, T // _H1_PER_YEAR)
    yearly_ret = []
    for y in range(years):
        s = y * _H1_PER_YEAR
        e = min((y + 1) * _H1_PER_YEAR, T)
        yr = float(port_pnl[s:e].sum())
        yearly_ret.append(yr)

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
        "T":              T,
        "years_span":     T / _H1_PER_YEAR,
        # 交易细节
        "port_n_trades":      port_trade["n_trades"],
        "port_n_long":        port_trade["n_long"],
        "port_n_short":       port_trade["n_short"],
        "port_win_rate":      port_trade["win_rate"],
        "port_avg_win":       port_trade["avg_win"],
        "port_avg_loss":      port_trade["avg_loss"],
        "port_profit_factor": port_trade["profit_factor"],
        "port_max_win":       port_trade["max_win"],
        "port_max_loss":      port_trade["max_loss"],
        "port_avg_trade_pnl": port_trade["avg_trade_pnl"],
        "port_avg_hold_bars": port_trade["avg_hold_bars"],
        "port_avg_hold_h":    port_trade["avg_hold_bars"],
        "avg_turnover":       float(turnover.mean()),
        "max_consec_loss":    _max_consecutive_loss(port_pnl),
        # 分段
        "half1_sharpe":   calc_sharpe(p1),
        "half2_sharpe":   calc_sharpe(p2),
        "half1_sortino":  calc_sortino(p1),
        "half2_sortino":  calc_sortino(p2),
        "half1_ret":      float(np.cumsum(p1)[-1]),
        "half2_ret":      float(np.cumsum(p2)[-1]),
        "third1_sharpe":  calc_sharpe(seg1),
        "third2_sharpe":  calc_sharpe(seg2),
        "third3_sharpe":  calc_sharpe(seg3),
        "third1_ret":     float(np.cumsum(seg1)[-1]),
        "third2_ret":     float(np.cumsum(seg2)[-1]),
        "third3_ret":     float(np.cumsum(seg3)[-1]),
        # 逐年
        "yearly_ret":     yearly_ret,
    }


def print_detailed_report(group_name, result, symbols, cost_rate, best_score):
    r = result
    T = r["T"]
    years = r["years_span"]

    print(f"\n{'='*80}")
    print(f"  [{group_name}] DETAILED BACKTEST REPORT")
    print(f"{'='*80}")
    print(f"  Formula   : {r['readable']}")
    print(f"  Tokens    : {r['formula']}")
    print(f"  Train score: {best_score:.4f}")
    print(f"  Symbols   : {symbols}")
    print(f"  Data      : {T} bars (H1)  ~  {years:.2f} years")
    print(f"  Cost rate : {cost_rate}")

    # === 组合概览 ===
    print(f"\n  {'─'*70}")
    print(f"  PORTFOLIO OVERVIEW (equal weight, {r['n_syms']} symbols)")
    print(f"  {'─'*70}")
    print(f"  Total Return       : {r['port_total_ret']:+.4f}  ({r['port_total_ret']*100:+.2f}%)")
    print(f"  Annual Return      : {r['port_ann_ret']:+.4f}  ({r['port_ann_ret']*100:+.2f}%)")
    print(f"  Sharpe Ratio       : {r['port_sharpe']:+.4f}")
    print(f"  Sortino Ratio      : {r['port_sortino']:+.4f}")
    print(f"  Max Drawdown       : {r['port_mdd']:.4f}  ({r['port_mdd']*100:.2f}%)")
    print(f"  Calmar Ratio       : {r['port_calmar']:+.4f}")
    print(f"  IC (mean)          : {r['ic']:.5f}")
    print(f"  Avg Turnover/bar   : {r['avg_turnover']:.6f}")
    print(f"  Max Consec Loss    : {r['max_consec_loss']} bars ({r['max_consec_loss']}h)")

    # === 交易统计 ===
    print(f"\n  {'─'*70}")
    print(f"  TRADING STATISTICS (portfolio level)")
    print(f"  {'─'*70}")
    print(f"  Total Trades       : {r['port_n_trades']}")
    print(f"  Long / Short       : {r['port_n_long']} / {r['port_n_short']}")
    print(f"  Win Rate           : {r['port_win_rate']*100:.1f}%")
    print(f"  Profit Factor      : {r['port_profit_factor']:.2f}")
    print(f"  Avg Win            : {r['port_avg_win']:+.6f}")
    print(f"  Avg Loss           : {r['port_avg_loss']:+.6f}")
    print(f"  Avg Trade PnL      : {r['port_avg_trade_pnl']:+.6f}")
    print(f"  Max Win            : {r['port_max_win']:+.6f}")
    print(f"  Max Loss           : {r['port_max_loss']:+.6f}")
    print(f"  Avg Hold           : {r['port_avg_hold_h']:.1f} bars ({r['port_avg_hold_h']:.1f}h)")

    # === 前后半段 ===
    print(f"\n  {'─'*70}")
    print(f"  CONSISTENCY ANALYSIS")
    print(f"  {'─'*70}")
    print(f"  Half-split (50/50):")
    print(f"    H1: Sharpe={r['half1_sharpe']:+.4f}  Sortino={r['half1_sortino']:+.4f}  Ret={r['half1_ret']:+.4f}")
    print(f"    H2: Sharpe={r['half2_sharpe']:+.4f}  Sortino={r['half2_sortino']:+.4f}  Ret={r['half2_ret']:+.4f}")
    h1, h2 = r["half1_sharpe"], r["half2_sharpe"]
    if h1 > 0 and h2 > 0:
        print(f"    [OK] Both positive")
    elif h1 * h2 > 0:
        print(f"    [WARN] Same sign but negative")
    else:
        print(f"    [FAIL] Opposite signs - overfitting suspected!")

    print(f"  Third-split (33/33/33):")
    print(f"    T1: Sharpe={r['third1_sharpe']:+.4f}  Ret={r['third1_ret']:+.4f}")
    print(f"    T2: Sharpe={r['third2_sharpe']:+.4f}  Ret={r['third2_ret']:+.4f}")
    print(f"    T3: Sharpe={r['third3_sharpe']:+.4f}  Ret={r['third3_ret']:+.4f}")

    print(f"  Yearly Returns:")
    for i, yr in enumerate(r["yearly_ret"]):
        print(f"    Year {i+1}: {yr:+.4f}  ({yr*100:+.2f}%)")

    # === 品种详情 ===
    print(f"\n  {'─'*70}")
    print(f"  PER-SYMBOL BREAKDOWN")
    print(f"  {'─'*70}")
    print(f"  {'Symbol':<14s} {'TotRet':>8s} {'AnnRet':>8s} {'Sharpe':>7s} {'Sort':>7s} {'MDD':>7s} {'IC':>8s} "
          f"{'Trades':>6s} {'Long':>5s} {'Short':>5s} {'WinR%':>6s} {'PF':>5s} {'AvgHold':>7s} {'AvgWin':>9s} {'AvgLoss':>9s}")
    print(f"  {'-'*14} {'-'*8} {'-'*8} {'-'*7} {'-'*7} {'-'*7} {'-'*8} {'-'*6} {'-'*5} {'-'*5} {'-'*6} {'-'*5} {'-'*7} {'-'*9} {'-'*9}")
    for sym, d in r["per_sym"].items():
        sig = "OK" if d["total_ret"] > 0 else "X"
        print(f"  {sym:<14s} {d['total_ret']:>+8.4f} {d['ann_ret']:>+8.4f} {d['sharpe']:>+7.3f} {d['sortino']:>+7.3f} "
              f"{d['mdd']:>7.4f} {d['ic']:>8.5f} {d['n_trades']:>6d} {d['n_long']:>5d} {d['n_short']:>5d} "
              f"{d['win_rate']*100:>5.1f}% {d['profit_factor']:>5.2f} {d['avg_hold_h']:>6.1f}h "
              f"{d['avg_win']:>+9.5f} {d['avg_loss']:>+9.5f} [{sig}]")

    # === FTMO ===
    print(f"\n  {'─'*70}")
    print(f"  FTMO ASSESSMENT")
    print(f"  {'─'*70}")
    issues = []
    if r["port_mdd"] > 0.10:
        issues.append(f"[WARN] MDD={r['port_mdd']*100:.2f}% > 10% FTMO Max Loss")
    else:
        print(f"  [OK] MDD={r['port_mdd']*100:.2f}% < 10%")
    if r["port_ann_ret"] > 0:
        print(f"  [OK] Annual return positive ({r['port_ann_ret']*100:+.2f}%)")
    else:
        issues.append(f"[WARN] Negative annual return")
    if r["port_profit_factor"] > 1.3:
        print(f"  [OK] Profit Factor={r['port_profit_factor']:.2f} > 1.3")
    else:
        issues.append(f"[WARN] Profit Factor={r['port_profit_factor']:.2f} < 1.3")
    if r["port_win_rate"] > 0.35:
        print(f"  [OK] Win rate={r['port_win_rate']*100:.1f}% > 35%")
    else:
        issues.append(f"[WARN] Win rate={r['port_win_rate']*100:.1f}% < 35%")
    if r["max_consec_loss"] > 500:
        issues.append(f"[WARN] Max consec loss {r['max_consec_loss']} bars too long")
    if r["port_avg_hold_h"] < 2:
        issues.append(f"[WARN] Avg hold {r['port_avg_hold_h']:.1f}h too short")
    if r["n_pos_syms"] < r["n_syms"] // 2:
        issues.append(f"[WARN] Only {r['n_pos_syms']}/{r['n_syms']} profitable")
    for iss in issues:
        print(f"    {iss}")
    if not issues:
        print(f"  [OK] No FTMO violations detected")


def plot_detailed(result, symbols, group_name, output_dir, cost_rate):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    r = result
    T = r["T"]
    x = np.arange(T)

    fig = plt.figure(figsize=(20, 16), dpi=110)
    gs = gridspec.GridSpec(4, 2, height_ratios=[2.5, 1, 1.5, 1.5], width_ratios=[1.5, 1],
                           hspace=0.25, wspace=0.2)

    # 1. 组合资金曲线 + 回撤
    ax_eq = fig.add_subplot(gs[0, :])
    port_cum = r["port_cum"]
    ax_eq.plot(x, port_cum, linewidth=2.0, color="#1565c0",
               label=f"Portfolio (Ann={r['port_ann_ret']*100:+.2f}%, Sharpe={r['port_sharpe']:+.2f}, MDD={r['port_mdd']*100:.2f}%)")
    ax_eq.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax_eq.set_ylabel("Cumulative Return", fontsize=10)
    ax_eq.legend(loc="upper left", fontsize=9, framealpha=0.85)
    ax_eq.grid(alpha=0.25)
    ax_eq.set_title(f"[{group_name}] {', '.join(symbols)} | Formula: {r['readable']}\n"
                    f"Cost={cost_rate}  T={T} bars ({r['years_span']:.2f}y)  Score={r.get('best_score',0):.4f}",
                    fontsize=11, pad=8)

    # 2. 回撤
    ax_dd = fig.add_subplot(gs[1, :], sharex=ax_eq)
    peak = np.maximum.accumulate(port_cum)
    dd = port_cum - peak
    ax_dd.fill_between(x, dd, 0, alpha=0.4, color="#1565c0")
    ax_dd.axhline(0, color="gray", linewidth=0.5)
    ax_dd.set_ylabel("Drawdown", fontsize=9)
    ax_dd.grid(alpha=0.2)

    # 3. 各品种资金曲线
    ax_sym = fig.add_subplot(gs[2, 0], sharex=ax_eq)
    colors = ["#e65100", "#00897b", "#6a1b9a", "#b71c1c", "#26418f"]
    for i, sym in enumerate(symbols):
        if sym in r["per_sym"]:
            cum = r["per_sym"][sym]["cum"]
            c = colors[i % len(colors)]
            ax_sym.plot(x, cum, linewidth=1.2, color=c, alpha=0.8,
                        label=f"{sym} (Ret={r['per_sym'][sym]['total_ret']:+.3f}, WR={r['per_sym'][sym]['win_rate']*100:.0f}%)")
    ax_sym.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax_sym.set_ylabel("Per-Symbol Return", fontsize=9)
    ax_sym.legend(loc="upper left", fontsize=7, ncol=2, framealpha=0.85)
    ax_sym.grid(alpha=0.2)

    # 4. 滚动 Sharpe (250 bar window)
    ax_roll = fig.add_subplot(gs[2, 1])
    window = 250
    if T > window:
        roll_sharpe = []
        for i in range(window, T):
            seg = r["port_pnl"][i-window:i]
            m, s = seg.mean(), seg.std()
            roll_sharpe.append(m / (s + 1e-10) * math.sqrt(_H1_PER_YEAR))
        ax_roll.plot(range(window, T), roll_sharpe, linewidth=0.8, color="#2e7d32", alpha=0.7)
        ax_roll.axhline(0, color="gray", linewidth=0.5, linestyle="--")
        ax_roll.axhline(1.0, color="orange", linewidth=0.5, linestyle=":", alpha=0.5)
        ax_roll.set_ylabel("Rolling Sharpe (250)", fontsize=8)
        ax_roll.set_xlabel("Bar Index", fontsize=8)
        ax_roll.grid(alpha=0.2)
        ax_roll.set_title("Rolling Sharpe Ratio", fontsize=9)
    else:
        ax_roll.text(0.5, 0.5, "Insufficient data for rolling", ha="center", va="center", transform=ax_roll.transAxes)

    # 5. 逐年收益柱状图
    ax_yr = fig.add_subplot(gs[3, 0])
    yearly = r["yearly_ret"]
    if yearly:
        yr_colors = ["#2e7d32" if v >= 0 else "#c62828" for v in yearly]
        ax_yr.bar(range(1, len(yearly)+1), [v*100 for v in yearly], color=yr_colors, alpha=0.8)
        ax_yr.axhline(0, color="gray", linewidth=0.5)
        ax_yr.set_xlabel("Year", fontsize=9)
        ax_yr.set_ylabel("Return (%)", fontsize=9)
        ax_yr.set_title("Yearly Returns", fontsize=9)
        ax_yr.grid(alpha=0.2)

    # 6. 月度收益热力图
    ax_mo = fig.add_subplot(gs[3, 1])
    # 按月聚合（假设 H1，每月约 520 bars）
    bars_per_month = 520
    n_months = T // bars_per_month
    if n_months > 1:
        monthly = []
        for m in range(n_months):
            s = m * bars_per_month
            e = min((m+1) * bars_per_month, T)
            monthly.append(float(r["port_pnl"][s:e].sum()))
        ax_mo.bar(range(n_months), [v*100 for v in monthly],
                   color=["#2e7d32" if v >= 0 else "#c62828" for v in monthly], alpha=0.7)
        ax_mo.axhline(0, color="gray", linewidth=0.5)
        ax_mo.set_xlabel("Month", fontsize=8)
        ax_mo.set_ylabel("Return (%)", fontsize=8)
        ax_mo.set_title("Monthly Returns", fontsize=9)
        ax_mo.grid(alpha=0.2)
    else:
        ax_mo.text(0.5, 0.5, "Insufficient data", ha="center", va="center", transform=ax_mo.transAxes)

    plt.savefig(Path(output_dir) / f"{group_name}_detailed.png", bbox_inches="tight")
    plt.close(fig)
    return str(Path(output_dir) / f"{group_name}_detailed.png")


def run_group(group_name, group_cfg, fetcher):
    symbols = group_cfg["symbols"]
    cost_rate = group_cfg["cost_rate"]
    strategy_path = Path(f"strategies/best_{group_name}.json")

    if not strategy_path.exists():
        print(f"\n  [{group_name}] SKIP - no strategy file")
        return None

    data = json.load(open(strategy_path))
    if data.get("vocab_version", "unknown") != VOCAB_VERSION:
        print(f"\n  [{group_name}] SKIP - vocab mismatch")
        return None

    formula = data["formula"]
    best_score = data.get("best_score", 0.0)

    # Load group data
    original_symbols = Config.SYMBOLS[:]
    Config.SYMBOLS = list(symbols)
    mgr = MT5DataManager(fetcher)
    mgr.load()
    Config.SYMBOLS = original_symbols

    raw_dict = mgr.raw_dict
    syms_loaded = mgr.symbols
    T = raw_dict["open"].shape[1]

    feat = MT5FeatureEngineer.compute_features(raw_dict)
    target_ret = mgr.target_ret

    result = backtest_one(formula, feat, target_ret, syms_loaded, cost_rate)
    if result is None:
        print(f"\n  [{group_name}] ERROR - factor execution failed")
        return None

    result["best_score"] = best_score
    print_detailed_report(group_name, result, syms_loaded, cost_rate, best_score)
    plot_path = plot_detailed(result, syms_loaded, group_name, OUTPUT_DIR, cost_rate)
    print(f"\n  Plot saved -> {plot_path}")

    # Save JSON
    report = {
        "group": group_name,
        "formula": formula,
        "readable": result["readable"],
        "best_score": best_score,
        "symbols": syms_loaded,
        "T": T,
        "years_span": round(T / _H1_PER_YEAR, 2),
        "cost_rate": cost_rate,
        "port_total_ret": round(result["port_total_ret"], 6),
        "port_ann_ret": round(result["port_ann_ret"], 6),
        "port_sharpe": round(result["port_sharpe"], 4),
        "port_sortino": round(result["port_sortino"], 4),
        "port_mdd": round(result["port_mdd"], 6),
        "port_calmar": round(result["port_calmar"], 4),
        "ic": round(result["ic"], 6),
        "avg_turnover": round(result["avg_turnover"], 8),
        "max_consec_loss": result["max_consec_loss"],
        "port_n_trades": result["port_n_trades"],
        "port_n_long": result["port_n_long"],
        "port_n_short": result["port_n_short"],
        "port_win_rate": round(result["port_win_rate"], 4),
        "port_avg_win": round(result["port_avg_win"], 6),
        "port_avg_loss": round(result["port_avg_loss"], 6),
        "port_profit_factor": round(result["port_profit_factor"], 4),
        "port_max_win": round(result["port_max_win"], 6),
        "port_max_loss": round(result["port_max_loss"], 6),
        "port_avg_trade_pnl": round(result["port_avg_trade_pnl"], 6),
        "port_avg_hold_h": round(result["port_avg_hold_h"], 1),
        "n_pos_syms": result["n_pos_syms"],
        "n_syms": result["n_syms"],
        "half1_sharpe": round(result["half1_sharpe"], 4),
        "half2_sharpe": round(result["half2_sharpe"], 4),
        "half1_ret": round(result["half1_ret"], 6),
        "half2_ret": round(result["half2_ret"], 6),
        "third1_sharpe": round(result["third1_sharpe"], 4),
        "third2_sharpe": round(result["third2_sharpe"], 4),
        "third3_sharpe": round(result["third3_sharpe"], 4),
        "third1_ret": round(result["third1_ret"], 6),
        "third2_ret": round(result["third2_ret"], 6),
        "third3_ret": round(result["third3_ret"], 6),
        "yearly_ret": [round(v, 6) for v in result["yearly_ret"]],
        "per_sym": {
            sym: {
                "total_ret": round(d["total_ret"], 6),
                "ann_ret": round(d["ann_ret"], 6),
                "sharpe": round(d["sharpe"], 4),
                "sortino": round(d["sortino"], 4),
                "mdd": round(d["mdd"], 6),
                "ic": round(d["ic"], 6),
                "n_trades": d["n_trades"],
                "n_long": d["n_long"],
                "n_short": d["n_short"],
                "win_rate": round(d["win_rate"], 4),
                "avg_win": round(d["avg_win"], 6),
                "avg_loss": round(d["avg_loss"], 6),
                "profit_factor": round(d["profit_factor"], 4),
                "max_win": round(d["max_win"], 6),
                "max_loss": round(d["max_loss"], 6),
                "avg_trade_pnl": round(d["avg_trade_pnl"], 6),
                "avg_hold_h": round(d["avg_hold_h"], 1),
            }
            for sym, d in result["per_sym"].items()
        },
    }
    report_path = Path(OUTPUT_DIR) / f"{group_name}_detailed.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"  Report saved -> {report_path}")

    return result


def main():
    offline = "--offline" in sys.argv

    # Update metals_comm strategy from latest checkpoint
    print("Checking metals_comm latest checkpoint...")
    import glob, torch
    ckpt_files = sorted(glob.glob(r"D:\cl\MT5_AlphaGPT\checkpoints\ckpt_metals_comm_step_*.pt"))
    if ckpt_files:
        ckpt = torch.load(ckpt_files[-1], map_location='cpu', weights_only=False)
        bf = ckpt.get('best_formula')
        bs = ckpt.get('best_score', 0)
        step = ckpt.get('step', 0)
        if bf and bs > 0:
            Path("strategies").mkdir(exist_ok=True)
            Path("strategies/best_metals_comm.json").write_text(json.dumps({
                "vocab_version": VOCAB_VERSION,
                "symbol": "metals_comm",
                "formula": bf,
                "best_score": bs,
                "step": step,
            }, indent=2))
            print(f"  Updated: score={bs:.4f} step={step} formula={bf}")

    print(f"\n{'='*80}")
    print(f"  FULL DETAILED BACKTEST - ALL GROUPS")
    print(f"{'='*80}")

    all_results = {}
    with MT5DataFetcher(offline=offline) as fetcher:
        for gname, gcfg in GROUP_CONFIG.items():
            r = run_group(gname, gcfg, fetcher)
            if r:
                all_results[gname] = r

        # === Cross-group summary ===
        if all_results:
            print(f"\n{'='*80}")
            print(f"  CROSS-GROUP COMPARISON")
            print(f"{'='*80}")
            print(f"  {'Group':<14s} {'Years':>6s} {'Score':>8s} {'AnnRet':>9s} {'Sharpe':>8s} {'Sortino':>8s} "
                  f"{'MDD':>7s} {'Calmar':>8s} {'IC':>8s} {'Trades':>7s} {'WinR%':>6s} {'PF':>5s} {'Hold':>6s} {'Pos/N':>6s}")
            print(f"  {'-'*14} {'-'*6} {'-'*8} {'-'*9} {'-'*8} {'-'*8} {'-'*7} {'-'*8} {'-'*8} {'-'*7} {'-'*6} {'-'*5} {'-'*6} {'-'*6}")
            for gname, r in all_results.items():
                print(f"  {gname:<14s} "
                      f"{r['years_span']:>5.2f}y "
                      f"{r.get('best_score',0):>8.4f} "
                      f"{r['port_ann_ret']:>+9.4f} "
                      f"{r['port_sharpe']:>+8.4f} "
                      f"{r['port_sortino']:>+8.4f} "
                      f"{r['port_mdd']:>7.4f} "
                      f"{r['port_calmar']:>+8.4f} "
                      f"{r['ic']:>8.5f} "
                      f"{r['port_n_trades']:>7d} "
                      f"{r['port_win_rate']*100:>5.1f}% "
                      f"{r['port_profit_factor']:>5.2f} "
                      f"{r['port_avg_hold_h']:>5.1f}h "
                      f"{r['n_pos_syms']:>2d}/{r['n_syms']:<2d}")

            # === Combined portfolio ===
            print(f"\n  {'─'*70}")
            print(f"  COMBINED PORTFOLIO (equal weight across groups)")
            print(f"  {'─'*70}")
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

            # Combined trade stats
            combined_pos = np.mean([r["port_pnl"][:min_T] for r in all_results.values()], axis=0)
            # Use sign of combined pnl as proxy position
            combined_trade = analyze_trades(
                np.sign(combined_pnl).reshape(1, -1),
                combined_pnl.reshape(1, -1), 0
            )[0]

            print(f"  Data span          : {min_T} bars ({min_T/_H1_PER_YEAR:.2f} years)")
            print(f"  Total Return       : {float(combined_cum[-1]):+.4f} ({float(combined_cum[-1])*100:+.2f}%)")
            print(f"  Annual Return      : {combined_ann:+.4f} ({combined_ann*100:+.2f}%)")
            print(f"  Sharpe Ratio       : {combined_sharpe:+.4f}")
            print(f"  Sortino Ratio      : {combined_sortino:+.4f}")
            print(f"  Max Drawdown       : {combined_mdd:.4f} ({combined_mdd*100:.2f}%)")
            print(f"  Calmar Ratio       : {combined_calmar:+.4f}")
            print(f"  Max Consec Loss    : {_max_consecutive_loss(combined_pnl)} bars")

            # Combined plot
            fig, axes = plt.subplots(3, 1, figsize=(18, 12), dpi=110,
                                      gridspec_kw={'height_ratios': [3, 1, 2], 'hspace': 0.2})
            x = np.arange(min_T)
            colors_g = {"forex": "#2e7d32", "metals_comm": "#e65100", "index": "#1565c0"}
            for gname, r in all_results.items():
                axes[0].plot(x, np.cumsum(r["port_pnl"][:min_T]), linewidth=1.2, alpha=0.7,
                             color=colors_g.get(gname, "gray"),
                             label=f"{gname} (Ann={r['port_ann_ret']*100:+.1f}%, MDD={r['port_mdd']*100:.1f}%)")
            axes[0].plot(x, combined_cum, linewidth=2.5, color="#d32f2f",
                         label=f"Combined (Ann={combined_ann*100:+.2f}%, Sharpe={combined_sharpe:+.2f}, MDD={combined_mdd*100:.2f}%)")
            axes[0].axhline(0, color="gray", linewidth=0.5, linestyle="--")
            axes[0].legend(fontsize=9, loc="upper left")
            axes[0].set_title(f"All Groups Combined Portfolio ({min_T} bars, {min_T/_H1_PER_YEAR:.2f} years)", fontsize=12)
            axes[0].grid(alpha=0.25)

            peak = np.maximum.accumulate(combined_cum)
            dd = combined_cum - peak
            axes[1].fill_between(x, dd, 0, alpha=0.4, color="#d32f2f")
            axes[1].axhline(0, color="gray", linewidth=0.5)
            axes[1].set_ylabel("Drawdown", fontsize=9)
            axes[1].grid(alpha=0.2)

            # Yearly returns
            years_c = max(1, min_T // _H1_PER_YEAR)
            yr_ret = []
            for y in range(years_c):
                s = y * _H1_PER_YEAR
                e = min((y+1) * _H1_PER_YEAR, min_T)
                yr_ret.append(float(combined_pnl[s:e].sum()))
            yr_colors = ["#2e7d32" if v >= 0 else "#c62828" for v in yr_ret]
            axes[2].bar(range(1, len(yr_ret)+1), [v*100 for v in yr_ret], color=yr_colors, alpha=0.8)
            axes[2].axhline(0, color="gray", linewidth=0.5)
            axes[2].set_xlabel("Year", fontsize=9)
            axes[2].set_ylabel("Return (%)", fontsize=9)
            axes[2].set_title("Combined Yearly Returns", fontsize=10)
            axes[2].grid(alpha=0.2)

            plt.savefig(Path(OUTPUT_DIR) / "all_groups_detailed.png", bbox_inches="tight")
            plt.close(fig)
            print(f"\n  Combined plot saved -> {Path(OUTPUT_DIR) / 'all_groups_detailed.png'}")

            combined_report = {
                "groups": list(all_results.keys()),
                "T": min_T,
                "years_span": round(min_T / _H1_PER_YEAR, 2),
                "total_ret": round(float(combined_cum[-1]), 6),
                "ann_ret": round(combined_ann, 6),
                "sharpe": round(combined_sharpe, 4),
                "sortino": round(combined_sortino, 4),
                "mdd": round(combined_mdd, 6),
                "calmar": round(combined_calmar, 4),
                "max_consec_loss": _max_consecutive_loss(combined_pnl),
                "yearly_ret": [round(v, 6) for v in yr_ret],
            }
            cr_path = Path(OUTPUT_DIR) / "all_groups_detailed.json"
            with open(cr_path, "w") as f:
                json.dump(combined_report, f, indent=2, ensure_ascii=False)
            print(f"  Combined report saved -> {cr_path}")

    print(f"\n{'='*80}")
    print(f"  BACKTEST COMPLETE")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
