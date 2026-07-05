"""
deep_backtest_metals.py — metals_comm FTMO 最佳因子深入回测分析

对 strategies/best_metals_comm.json 中的因子在 XAUUSD/AAVUSD/COCOA.c 上：
  - 全量历史回测
  - Walk-Forward 分折统计
  - 年度/月度收益分解
  - FTMO 合规性检查
  - 品种相关性、换手率、持仓时间
  - 输出可视化图表和 JSON 报告
"""
import sys, json, math
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))

from config import Config
from data_pipeline.data_manager import MT5DataManager
from data_pipeline.fetcher import MT5DataFetcher
from model_core.vocab import FORMULA_VOCAB, VOCAB_VERSION
from model_core.vm import StackVM
from model_core.features import MT5FeatureEngineer
from strategy_manager.signal import compute_target_positions_stateless

H1_PER_YEAR = 6240
COST_RATE = 0.0001  # 默认点差成本
ACCOUNT = 100_000.0


def sharpe(pnl):
    m = pnl.mean()
    s = pnl.std()
    return float(m / s * math.sqrt(H1_PER_YEAR)) if s > 1e-10 else 0.0


def sortino(pnl):
    m = pnl.mean()
    d = pnl[pnl < 0]
    ds = d.std() if len(d) > 0 else 1e-10
    ds = max(ds, abs(m) * 0.2, 1e-10)
    return float(np.clip(m / ds * math.sqrt(H1_PER_YEAR), -20, 20))


def calmar(pnl):
    c = np.cumsum(pnl)
    peak = np.maximum.accumulate(c)
    dd = peak - c
    max_dd = dd.max()
    ann = pnl.mean() * H1_PER_YEAR
    return float(ann / max_dd) if max_dd > 1e-10 else 0.0


def max_dd(pnl):
    c = np.cumsum(pnl)
    peak = np.maximum.accumulate(c)
    dd = peak - c
    return float(dd.max())


def avg_hold(position):
    """计算平均持仓时间（bars）"""
    pos = np.sign(position)
    if pos.size == 0:
        return 0.0
    runs = []
    cur_dir = 0
    cur_len = 0
    for p in pos:
        if p == 0:
            if cur_len > 0:
                runs.append(cur_len)
            cur_dir, cur_len = 0, 0
        elif p == cur_dir:
            cur_len += 1
        else:
            if cur_len > 0:
                runs.append(cur_len)
            cur_dir, cur_len = p, 1
    if cur_len > 0:
        runs.append(cur_len)
    return float(sum(runs) / len(runs)) if runs else 0.0


def daily_pnl(pnl_hourly):
    """H1 PnL -> daily PnL"""
    n = len(pnl_hourly) // 24
    return pnl_hourly[:n * 24].reshape(n, 24).sum(axis=1)


def ftmo_stats(daily, max_daily_loss=0.05, max_overall_loss=0.10, profit_target=0.10):
    """FTMO 2-Step 统计"""
    scaled = daily * ACCOUNT
    balance = ACCOUNT
    peak = ACCOUNT
    cum_pnl = 0.0
    max_daily = 0.0
    trading_days = 0
    violations = 0
    target_day = None
    for d in scaled:
        if abs(d) > 1e-6:
            trading_days += 1
        balance += d
        cum_pnl += d
        peak = max(peak, balance - d)
        if d < 0:
            max_daily = max(max_daily, abs(d) / ACCOUNT)
        if balance < peak - max_overall_loss * ACCOUNT:
            violations += 1
        if target_day is None and cum_pnl >= profit_target * ACCOUNT:
            target_day = trading_days
    return {
        "max_daily_loss": max_daily,
        "violations": violations,
        "trading_days": trading_days,
        "target_day": target_day,
    }


def main():
    offline = "--offline" in sys.argv

    # ── 加载因子 ─────────────────────────────────────────────────
    data = json.load(open("strategies/best_metals_comm.json"))
    if data.get("vocab_version") != VOCAB_VERSION:
        print("[ERROR] vocab 版本不符"); return
    formula = data["formula"]
    readable = " -> ".join(FORMULA_VOCAB.token_names[t] for t in formula)
    print(f"\n因子: {readable}")
    print(f"训练 score: {data['best_score']:.4f}  (step {data.get('step', '?')})\n")

    # ── 加载数据 ─────────────────────────────────────────────────
    original_symbols = Config.SYMBOLS[:]
    Config.SYMBOLS = data.get("symbol_group", "metals_comm").split(",") if "," in data.get("symbol_group", "") else Config.SYMBOL_GROUPS["metals_comm"]
    symbols = Config.SYMBOLS

    results = {}
    try:
        with MT5DataFetcher(offline=offline) as fetcher:
            mgr = MT5DataManager(fetcher)
            mgr.load()
            raw_dict = mgr.raw_dict
            target_ret = mgr.target_ret
            T = raw_dict["open"].shape[1]
            print(f"数据: {symbols}  T={T} bars\n")

            feat = MT5FeatureEngineer.compute_features(raw_dict)
            vm = StackVM()
            factor = vm.execute(formula, feat)
            if factor is None:
                print("[ERROR] 因子执行失败"); return

            position = compute_target_positions_stateless(factor)
            prev_pos = torch.roll(position, 1, dims=1)
            prev_pos[:, 0] = 0.0
            turnover = torch.abs(position - prev_pos)
            pnl = position * target_ret - turnover * COST_RATE

    finally:
        Config.SYMBOLS = original_symbols

    # ── 总体统计 ─────────────────────────────────────────────────
    port_pnl = pnl.mean(dim=0).numpy()
    print(f"{'='*70}")
    print(f"  组合总体统计（等权 {len(symbols)} 品种）")
    print(f"{'='*70}")
    print(f"  累计收益:    {port_pnl.sum():+.4f}")
    print(f"  年化收益:    {port_pnl.mean() * H1_PER_YEAR:+.4f}")
    print(f"  Sharpe:      {sharpe(port_pnl):+.4f}")
    print(f"  Sortino:     {sortino(port_pnl):+.4f}")
    print(f"  Calmar:      {calmar(port_pnl):+.4f}")
    print(f"  MaxDD:       {max_dd(port_pnl):.4f}")
    print(f"  交易次数:    {int(np.sum(np.abs(np.diff(np.sign(port_pnl), prepend=0)) > 0))}")
    print(f"  平均持仓:    {avg_hold(port_pnl):.1f}h")
    print()

    # ── 分品种统计 ───────────────────────────────────────────────
    print(f"{'='*70}")
    print(f"  分品种统计")
    print(f"{'='*70}")
    print(f"  {'品种':12s}{'累计':>10s}{'年化':>10s}{'Sharpe':>8s}{'Sortino':>8s}{'MaxDD':>8s}{'持仓(h)':>8s}")
    print(f"  {'─'*70}")
    per_sym = {}
    for i, sym in enumerate(symbols):
        p = pnl[i].numpy()
        per_sym[sym] = {
            "total": float(p.sum()),
            "annual": float(p.mean() * H1_PER_YEAR),
            "sharpe": sharpe(p),
            "sortino": sortino(p),
            "maxdd": max_dd(p),
            "avg_hold_h": avg_hold(position[i].numpy()),
        }
        print(f"  {sym:12s}{p.sum():>+10.4f}{p.mean()*H1_PER_YEAR:>+10.4f}{sharpe(p):>+8.2f}{sortino(p):>+8.2f}{max_dd(p):>8.4f}{avg_hold(position[i].numpy()):>8.1f}")

    # ── Walk-Forward 4 折 ────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  Walk-Forward 4 折（按时间顺序）")
    print(f"{'='*70}")
    fold_size = T // 4
    wf = []
    for f in range(4):
        s, e = f * fold_size, (f + 1) * fold_size if f < 3 else T
        fp = port_pnl[s:e]
        wf.append({
            "fold": f + 1,
            "start_bar": s,
            "end_bar": e,
            "total": float(fp.sum()),
            "sharpe": sharpe(fp),
            "sortino": sortino(fp),
            "maxdd": max_dd(fp),
        })
        print(f"  Fold {f+1} [{s:6d}:{e:6d}]  Tot={fp.sum():+.4f}  Sharpe={sharpe(fp):+.2f}  Sortino={sortino(fp):+.2f}  MDD={max_dd(fp):.4f}")

    # ── 年度/月度分解 ────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  年度收益分解（近似，按 6240 bars/年）")
    print(f"{'='*70}")
    yearly = []
    bars_year = H1_PER_YEAR
    n_years = T // bars_year
    for y in range(n_years):
        yp = port_pnl[y * bars_year:(y + 1) * bars_year]
        yearly.append({"year": y + 1, "total": float(yp.sum()), "sharpe": sharpe(yp)})
        print(f"  Year {y+1}: Tot={yp.sum():+.4f}  Sharpe={sharpe(yp):+.2f}")

    # ── FTMO 合规性 ──────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  FTMO 2-Step 合规性（1.0x 仓位）")
    print(f"{'='*70}")
    dly = daily_pnl(port_pnl)
    ft = ftmo_stats(dly)
    print(f"  最大日亏:    {ft['max_daily_loss']:.2%} (限 5%)")
    print(f"  违规次数:    {ft['violations']}")
    print(f"  交易天数:    {ft['trading_days']}")
    print(f"  达标 10% 日: {ft['target_day']}")

    # ── 不同仓位倍数 ─────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  仓位缩放对 FTMO 的影响")
    print(f"{'='*70}")
    print(f"  {'倍数':>6s}{'累计':>10s}{'最大日亏':>10s}{'达标日':>8s}{'违规':>6s}")
    print(f"  {'─'*46}")
    for scale in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
        sd = daily_pnl(port_pnl * scale)
        ft2 = ftmo_stats(sd)
        print(f"  {scale:>5.1f}x {port_pnl.sum()*scale:>+9.4f} {ft2['max_daily_loss']:>9.2%} {str(ft2['target_day']):>8s} {ft2['violations']:>6d}")

    # ── 保存报告 ─────────────────────────────────────────────────
    def _to_native(obj):
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, dict): return {k: _to_native(v) for k, v in obj.items()}
        if isinstance(obj, list): return [_to_native(v) for v in obj]
        return obj

    report = _to_native({
        "formula": formula,
        "readable": readable,
        "symbols": symbols,
        "T": int(T),
        "portfolio": {
            "total_return": float(port_pnl.sum()),
            "annual_return": float(port_pnl.mean() * H1_PER_YEAR),
            "sharpe": sharpe(port_pnl),
            "sortino": sortino(port_pnl),
            "calmar": calmar(port_pnl),
            "maxdd": max_dd(port_pnl),
            "avg_hold_h": avg_hold(port_pnl),
        },
        "per_symbol": per_sym,
        "walk_forward": wf,
        "yearly": yearly,
        "ftmo_1x": ft,
    })

    out_dir = Path("backtest_output")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "metals_comm_deep_report.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n报告已保存: {out_path}\n")


if __name__ == "__main__":
    main()
