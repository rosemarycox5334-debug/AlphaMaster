"""paper_trading/metrics.py — 组合绩效指标。"""
from __future__ import annotations

import math

_TRADING_DAYS = 244   # A股年化交易日数


def compute_metrics(nav_history: list[tuple[str, float]],
                    initial_capital: float) -> dict:
    """从净值序列计算总收益/最大回撤/年化/夏普。

    Args:
        nav_history: [(date, nav), ...] 逐日总净值。
        initial_capital: 起始资金。
    """
    if not nav_history:
        return {"total_return": 0.0, "max_drawdown": 0.0, "sharpe": 0.0,
                "annual_return": 0.0, "final_nav": initial_capital, "days": 0}

    navs = [v for _, v in nav_history]
    final = navs[-1]
    total_return = final / initial_capital - 1.0

    # 最大回撤
    peak = navs[0]
    max_dd = 0.0
    for v in navs:
        peak = max(peak, v)
        dd = (peak - v) / peak if peak > 0 else 0.0
        max_dd = max(max_dd, dd)

    # 日收益序列
    rets = [navs[i] / navs[i - 1] - 1.0 for i in range(1, len(navs))]
    if rets:
        mean_r = sum(rets) / len(rets)
        var = sum((r - mean_r) ** 2 for r in rets) / len(rets)
        std = math.sqrt(var)
        sharpe = (mean_r / std * math.sqrt(_TRADING_DAYS)) if std > 1e-12 else 0.0
        annual = mean_r * _TRADING_DAYS
    else:
        sharpe = annual = 0.0

    return {
        "total_return": total_return,
        "max_drawdown": max_dd,
        "sharpe": sharpe,
        "annual_return": annual,
        "final_nav": final,
        "days": len(navs),
    }
