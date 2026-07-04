"""
backtest_viz/engine.py — 逐 bar 可视化回测引擎

与训练用 backtest.py 共享相同的信号逻辑（tanh→sign），
但额外记录每笔交易的开平仓细节，供图表标注使用。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch

from model_core.vm import StackVM
from strategy_manager.signal import compute_target_positions

_H1_PERIODS_PER_YEAR = 6240


@dataclass
class Trade:
    """一笔完整交易记录（开仓 → 平仓/反手）"""
    symbol:      str
    direction:   int          # +1 多 / -1 空
    entry_bar:   int          # 开仓 bar 索引（相对于整个序列）
    entry_time:  int          # Unix 秒
    entry_price: float        # 开仓价（用 open 价格）
    exit_bar:    Optional[int]   = None
    exit_time:   Optional[int]   = None
    exit_price:  Optional[float] = None
    pnl:         float           = 0.0   # 本笔税后 PnL（log return - cost）
    cum_pnl:     float           = 0.0   # 截至本笔结束的累计 PnL


@dataclass
class SymbolResult:
    """单个品种的完整回测结果"""
    symbol:       str
    times:        np.ndarray     # Unix 秒，shape [T]
    open:         np.ndarray     # [T]
    high:         np.ndarray     # [T]
    low:          np.ndarray     # [T]
    close:        np.ndarray     # [T]
    volume:       np.ndarray     # [T]
    factor:       np.ndarray     # StackVM 输出，[T]
    signal:       np.ndarray     # tanh(factor)，[T]
    position:     np.ndarray     # sign(signal) ∈ {-1,0,+1}，[T]
    pnl:          np.ndarray     # 逐 bar PnL，[T]
    cum_pnl:      np.ndarray     # 累计 PnL，[T]
    trades:       list[Trade]    = field(default_factory=list)
    sortino:      float          = 0.0
    total_return: float          = 0.0
    n_trades:     int            = 0
    win_rate:     float          = 0.0
    max_drawdown: float          = 0.0
    avg_hold_bars:float          = 0.0


class BacktestEngine:
    """逐 bar 可视化回测引擎。

    用法：
        engine = BacktestEngine(formula=[6,15,8,...])
        results = engine.run(raw_dict, times, symbols)
    """

    def __init__(
        self,
        formula:         list[int],
        cost_rate:       float = 0.0001,
        periods_per_year:int   = _H1_PERIODS_PER_YEAR,
    ):
        self.formula          = formula
        self.cost_rate        = cost_rate
        self.periods_per_year = periods_per_year
        self.vm               = StackVM()

    # ─────────────────────────────────────────────────────────────────────
    # 主入口
    # ─────────────────────────────────────────────────────────────────────

    def run(
        self,
        raw_dict: dict,          # {open/high/low/close/volume/time: Tensor[N,T]}
        feat_tensor: torch.Tensor,  # [N, F, T]
        symbols: list[str],
    ) -> list[SymbolResult]:
        """执行所有品种的回测，返回每个品种的 SymbolResult。"""

        factors_all = self.vm.execute(self.formula, feat_tensor)  # [N, T]
        if factors_all is None:
            raise RuntimeError(
                f"StackVM 无法执行公式 {self.formula}。"
                "请检查公式 token 是否合法。"
            )

        results = []
        N = len(symbols)
        for n in range(N):
            sym = symbols[n]
            sym_result = self._backtest_symbol(
                symbol     = sym,
                raw_dict   = {k: v[n] for k, v in raw_dict.items()},   # [T] 各字段
                factor_1d  = factors_all[n],                            # [T]
            )
            results.append(sym_result)

        return results

    # ─────────────────────────────────────────────────────────────────────
    # 单品种回测
    # ─────────────────────────────────────────────────────────────────────

    def _backtest_symbol(
        self,
        symbol:   str,
        raw_dict: dict,         # 每个值是 [T] 的 Tensor
        factor_1d: torch.Tensor,  # [T]
    ) -> SymbolResult:

        T = factor_1d.shape[0]

        # numpy 转换（便于后续图表处理）
        factor_np   = factor_1d.detach().float().numpy()
        # 连续仓位模式：tanh 直接作为仓位比例，与训练 backtest.py 完全一致
        position_np = np.tanh(factor_np)

        open_np   = raw_dict["open"].float().numpy()
        high_np   = raw_dict["high"].float().numpy()
        low_np    = raw_dict["low"].float().numpy()
        close_np  = raw_dict["close"].float().numpy()
        volume_np = raw_dict["volume"].float().numpy()

        if "time" in raw_dict:
            times_np = raw_dict["time"].long().numpy()
        else:
            times_np = np.arange(T, dtype=np.int64)

        # ── 计算 PnL 序列（与 backtest.py 完全一致）─────────────────
        # target_ret[t] = log(open[t+2] / open[t+1])
        target_ret = np.zeros(T, dtype=np.float32)
        if T >= 3:
            target_ret[: T - 2] = np.log(
                (open_np[2:] + 1e-12) / (open_np[1:-1] + 1e-12)
            )

        prev_pos = np.zeros(T, dtype=np.float32)
        prev_pos[1:] = position_np[:-1]
        turnover = np.abs(position_np - prev_pos)

        pnl_np    = position_np * target_ret - turnover * self.cost_rate
        cum_pnl   = np.cumsum(pnl_np)

        # ── 提取交易记录 ──────────────────────────────────────────────
        trades = self._extract_trades(
            symbol, position_np, open_np, times_np, pnl_np
        )

        # ── 统计指标 ─────────────────────────────────────────────────
        sortino       = self._calc_sortino(pnl_np)
        total_return  = float(cum_pnl[-1]) if len(cum_pnl) else 0.0
        n_trades      = len(trades)
        win_rate      = (
            sum(1 for t in trades if t.pnl > 0) / n_trades
            if n_trades else 0.0
        )
        max_dd        = self._calc_max_drawdown(cum_pnl)
        avg_hold      = (
            sum(
                (t.exit_bar - t.entry_bar)
                for t in trades if t.exit_bar is not None
            ) / n_trades
            if n_trades else 0.0
        )

        return SymbolResult(
            symbol       = symbol,
            times        = times_np,
            open         = open_np,
            high         = high_np,
            low          = low_np,
            close        = close_np,
            volume       = volume_np,
            factor       = factor_np,
            signal       = signal_np,
            position     = position_np,
            pnl          = pnl_np,
            cum_pnl      = cum_pnl,
            trades       = trades,
            sortino      = sortino,
            total_return = total_return,
            n_trades     = n_trades,
            win_rate     = win_rate,
            max_drawdown = max_dd,
            avg_hold_bars= avg_hold,
        )

    # ─────────────────────────────────────────────────────────────────────
    # 交易记录提取
    # ─────────────────────────────────────────────────────────────────────

    def _extract_trades(
        self,
        symbol:      str,
        position:    np.ndarray,   # [T]
        open_prices: np.ndarray,   # [T]
        times:       np.ndarray,   # [T]
        pnl:         np.ndarray,   # [T]
    ) -> list[Trade]:
        """从仓位序列中提取完整交易列表（含开平仓 bar、价格、PnL）。

        执行价对齐逻辑（与 target_ret 计算保持一致）：
          target_ret[t] = log(open[t+2] / open[t+1])
          position[t] 产生的收益对应 open[t+1] → open[t+2]
          因此：信号在 entry_bar 产生 → 实际成交价 = open[entry_bar + 1]
                信号在 exit_bar 翻转 → 实际成交价 = open[exit_bar + 1]

        PnL 计算：把持仓期间的逐 bar pnl 累加作为本笔盈亏。
        """
        T = len(position)
        trades:       list[Trade] = []
        cum_pnl_total = 0.0

        current_dir: int = 0
        entry_bar:   int = 0

        def _exec_price(bar: int) -> float:
            """信号在 bar 产生，执行价为下一根 open（若越界则取最后一根）。"""
            idx = min(bar + 1, T - 1)
            return float(open_prices[idx])

        def _exec_time(bar: int) -> int:
            idx = min(bar + 1, T - 1)
            return int(times[idx])

        for t in range(T):
            new_dir = int(position[t])

            if new_dir != current_dir:
                # 平掉旧仓
                if current_dir != 0:
                    trade_pnl = float(pnl[entry_bar:t].sum())
                    cum_pnl_total += trade_pnl
                    trade = Trade(
                        symbol      = symbol,
                        direction   = current_dir,
                        entry_bar   = entry_bar,
                        entry_time  = _exec_time(entry_bar),
                        entry_price = _exec_price(entry_bar),
                        exit_bar    = t,
                        exit_time   = _exec_time(t),
                        exit_price  = _exec_price(t),
                        pnl         = trade_pnl,
                        cum_pnl     = cum_pnl_total,
                    )
                    trades.append(trade)

                current_dir = new_dir
                entry_bar   = t

        # 序列末尾强平
        if current_dir != 0:
            trade_pnl = float(pnl[entry_bar:].sum())
            cum_pnl_total += trade_pnl
            trades.append(Trade(
                symbol      = symbol,
                direction   = current_dir,
                entry_bar   = entry_bar,
                entry_time  = _exec_time(entry_bar),
                entry_price = _exec_price(entry_bar),
                exit_bar    = T - 1,
                exit_time   = _exec_time(T - 1),
                exit_price  = _exec_price(T - 1),
                pnl         = trade_pnl,
                cum_pnl     = cum_pnl_total,
            ))

        return trades

    # ─────────────────────────────────────────────────────────────────────
    # 统计辅助
    # ─────────────────────────────────────────────────────────────────────

    def _calc_sortino(self, pnl: np.ndarray) -> float:
        mean_pnl = float(np.mean(pnl))
        downside = pnl[pnl < 0]
        if len(downside) == 0:
            return 0.0
        ds_std = float(np.std(downside, ddof=0))
        floor  = max(abs(mean_pnl), 1e-8)
        ds_std = max(ds_std, floor)
        sortino = mean_pnl / ds_std * math.sqrt(self.periods_per_year)
        return float(np.clip(sortino, -20.0, 20.0))

    @staticmethod
    def _calc_max_drawdown(cum_pnl: np.ndarray) -> float:
        if len(cum_pnl) == 0:
            return 0.0
        peak = np.maximum.accumulate(cum_pnl)
        drawdown = peak - cum_pnl
        return float(drawdown.max())
