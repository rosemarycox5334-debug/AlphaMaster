"""
model_core/backtest.py — MT5 回测评估器（组合级多目标 Reward）

评分框架（5品种组合版）：
  final_score =
      0.35 * portfolio_sortino          # 组合整体风险调整收益
    + 0.20 * portfolio_calmar           # 组合整体回撤控制
    + 0.15 * ts_ic_stability            # 时序IC稳定性（比横截面IC更重要）
    + 0.10 * symbol_consistency         # 品种一致性（防止单品种拖累）
    + 0.10 * cost_stress                # 成本压力测试（2x成本下仍盈利）
    + 0.10 * turnover_quality           # 换手率质量（交易频率奖励）
    - complexity_penalty                # 公式长度惩罚
    - correlation_penalty               # 因子相关性惩罚（由 engine 施加）

symbol_consistency 规则：
  - N 个品种中至少 ceil(N*0.6) 个 Sortino > 0 → 正分
  - 任何品种 Sortino < -2.0 → 重惩罚
  - 全部品种 Sortino > 0 → 额外奖励
"""
import math
import torch
from torch import Tensor

from strategy_manager.signal import compute_target_positions_stateless
from strategy_manager.market_rules import apply_market_constraints, normalize_market
from .config import ModelConfig

_H1_PERIODS_PER_YEAR = 6240
_SORTINO_CLIP        = 20.0


class MT5Backtest:
    """MT5 组合级回测评估器。"""

    def __init__(
        self,
        cost_rate:        float = 0.0001,
        periods_per_year: int   = _H1_PERIODS_PER_YEAR,
        market:           str   = "generic",
        raw_dict:         dict[str, Tensor] | None = None,
        symbols:          list[str] | None = None,
    ):
        self.cost_rate        = cost_rate
        self.periods_per_year = periods_per_year
        self.market = normalize_market(market)
        self.raw_dict = raw_dict
        self.symbols = symbols or []

    def _positions(self, factors: Tensor) -> Tensor:
        desired = compute_target_positions_stateless(factors)
        return apply_market_constraints(
            desired, self.raw_dict, market=self.market, symbols=self.symbols
        )

    # ──────────────────────────────────────────────────────────────────────
    # 基础统计
    # ──────────────────────────────────────────────────────────────────────

    def _sortino(self, pnl: Tensor, eps: float = 1e-8) -> Tensor:
        flat     = pnl.reshape(-1)
        mean_pnl = flat.mean()
        downside = flat[flat < 0]
        raw_std  = downside.std(unbiased=False) if downside.numel() > 0 \
                   else torch.tensor(0.0, dtype=flat.dtype, device=flat.device)
        # P0b 修复：下行标准差地板改为全序列 std 的 20%，防止稀疏 PnL 靠极小分母刷高分。
        # 原来 floor=|mean_pnl| 对稀疏序列趋近于零，导致 Sortino 爆炸。
        full_std       = flat.std(unbiased=False).clamp(min=eps)
        floor          = torch.clamp(full_std * 0.2, min=eps)
        downside_std   = torch.clamp(raw_std, min=floor)
        sortino        = mean_pnl / downside_std * math.sqrt(self.periods_per_year)
        return torch.clamp(sortino, -_SORTINO_CLIP, _SORTINO_CLIP)

    def _calmar(self, pnl: Tensor, eps: float = 1e-8) -> Tensor:
        """Calmar = annualized_return / max_drawdown（截断到 [-10, 10]）。"""
        flat      = pnl.reshape(-1)
        ann_ret   = flat.mean() * self.periods_per_year
        cum       = torch.cumsum(flat, dim=0)
        peak      = torch.cummax(cum, dim=0).values
        drawdown  = (peak - cum).max()
        drawdown  = torch.clamp(drawdown, min=eps)
        calmar    = ann_ret / drawdown
        return torch.clamp(calmar, -10.0, 10.0)

    # ──────────────────────────────────────────────────────────────────────
    # 组合级评分组件
    # ──────────────────────────────────────────────────────────────────────

    def _ts_ic_stability(self, factors: Tensor, target_ret: Tensor) -> float:
        """时序 IC 稳定性：每个品种内部 factor[t] 与 ret[t+1] 的相关性均值。

        比横截面 IC 更适合 5 品种宇宙（横截面 N=5 统计意义弱）。

        Returns:
            float，约 [-1, 1]，正值代表因子有预测力。
        """
        N, T = factors.shape
        if T < 10:
            return 0.0

        ic_list = []
        for n in range(N):
            x = factors[n, :-1]
            y = target_ret[n, 1:]
            xm = x - x.mean()
            ym = y - y.mean()
            sx = (xm ** 2).mean().sqrt()
            sy = (ym ** 2).mean().sqrt()
            if sx < 1e-6 or sy < 1e-6:
                continue
            ic = (xm * ym).mean() / (sx * sy + 1e-8)
            ic_list.append(ic.item())

        if not ic_list:
            return 0.0

        ic_mean = sum(ic_list) / len(ic_list)
        ic_std  = (sum((v - ic_mean) ** 2 for v in ic_list) / len(ic_list)) ** 0.5
        # 稳定性 = IC均值 / IC标准差（IR，截断到 [-3, 3]）
        stability = ic_mean / (ic_std + 1e-6)
        return float(max(-3.0, min(3.0, stability)))

    def _symbol_consistency(
        self,
        per_symbol_sortino: list[float],
        per_symbol_trade_count: list[int] | None = None,
        eval_bars: int = 0,
    ) -> float:
        """品种一致性惩罚/奖励。

        规则（优先级从高到低）：
        1. 无交易品种超过 40%：重惩罚 -3.0
        2. P0a 新增：有交易的品种中，交易笔数 < eval_bars/100 (约每100bar少于1笔)
           视为"稀疏有效"，等同无效。防止3~6笔偶发交易刷高 Sortino。
        3. 任何品种 Sortino < -2.0：重惩罚 -2.0
        4. 有效品种中正收益比例决定奖惩
        """
        N = len(per_symbol_sortino)
        if N == 0:
            return 0.0

        # 最小有效交易数：每 100 bar 至少 1 笔，下限 5 笔
        min_trades = max(5, eval_bars // 100) if eval_bars > 0 else 5

        # 重新判定"活跃"品种（必须交易数 >= min_trades）
        if per_symbol_trade_count is not None:
            n_inactive = sum(1 for c in per_symbol_trade_count if c < min_trades)
            inactive_ratio = n_inactive / N
            if inactive_ratio > 0.4:
                return -3.0
        else:
            n_inactive = 0
            inactive_ratio = 0.0

        if any(s < -2.0 for s in per_symbol_sortino):
            return -2.0

        if per_symbol_trade_count is not None:
            active_sortinos = [
                s for s, c in zip(per_symbol_sortino, per_symbol_trade_count)
                if c >= min_trades
            ]
        else:
            active_sortinos = per_symbol_sortino

        if not active_sortinos:
            return -3.0

        n_positive = sum(1 for s in active_sortinos if s > 0)
        ratio = n_positive / len(active_sortinos)

        if ratio < 0.6:
            score = (ratio - 0.6) / 0.6 * 1.0
        else:
            score = (ratio - 0.6) / 0.4 * 1.0

        if ratio == 1.0:
            score += 0.5

        return float(score)

    def _cost_stress(
        self,
        position:   Tensor,
        target_ret: Tensor,
        stress_mult: float = 2.0,
    ) -> float:
        """成本压力测试：2 倍成本下的 Sortino 是否还 > 0。

        Returns:
            float，压力测试 Sortino（截断到 [-5, 5]）。
        """
        prev_pos = torch.roll(position, 1, dims=1)
        prev_pos[:, 0] = 0.0
        turnover = torch.abs(position - prev_pos)
        stressed_pnl = position * target_ret - turnover * self.cost_rate * stress_mult
        sortino = self._sortino(stressed_pnl)
        return float(torch.clamp(sortino, -5.0, 5.0))

    def _turnover_quality(self, position: Tensor) -> float:
        """交易频率质量奖励（每天约 1 笔为最优）。

        目标：每 12 bar 一笔（H1 每天约一笔）。
        """
        N, T = position.shape
        pos_2d = position.tolist()
        all_runs, total_trades = [], 0

        for n in range(N):
            runs, cur_len, cur_dir = [], 0, 0
            for p in pos_2d[n]:
                pi = int(p)
                if pi != 0:
                    if pi == cur_dir:
                        cur_len += 1
                    else:
                        if cur_len > 0: runs.append(cur_len)
                        cur_dir, cur_len = pi, 1
                else:
                    if cur_len > 0: runs.append(cur_len)
                    cur_dir, cur_len = 0, 0
            if cur_len > 0: runs.append(cur_len)
            all_runs.extend(runs)
            total_trades += len(runs)

        total_bars    = N * T
        target_trades = total_bars / 12.0
        actual_ratio  = total_trades / max(target_trades, 1.0)

        if actual_ratio <= 0:
            freq_score = -2.0
        elif actual_ratio < 0.05:
            freq_score = -2.0 + actual_ratio / 0.05
        elif actual_ratio < 0.5:
            freq_score = -1.0 + (actual_ratio - 0.05) / 0.45
        elif actual_ratio <= 2.0:
            log_r = math.log(actual_ratio) / math.log(2.0)
            freq_score = 1.0 * math.exp(-0.5 * log_r ** 2)
        elif actual_ratio <= 8.0:
            freq_score = 0.5 - (actual_ratio - 2.0) / 6.0 * 1.5
        else:
            freq_score = -2.0

        hold_bonus = 0.0
        if all_runs:
            avg_hold = sum(all_runs) / len(all_runs)
            hold_bonus = min(0.3, math.log(max(avg_hold, 1.0)) / math.log(30.0) * 0.3)

        return float(freq_score + hold_bonus)

    def _beta_neutral_penalty(self, position: Tensor) -> float:
        """Beta 中性惩罚：多空比例严重失衡时扣分。

        因子输出 >85% 同方向时，说明不是 alpha 因子而是 beta 因子
        （如 index 组的 TS_RANK 连续使用导致恒正输出）。

        Returns:
            float，惩罚值（负数或零）
        """
        flat = position.reshape(-1)
        long_ratio = (flat > 0.05).float().mean().item()
        short_ratio = (flat < -0.05).float().mean().item()
        max_ratio = max(long_ratio, short_ratio)
        if max_ratio > 0.85:
            # 超过 85% 同方向，重罚
            excess = (max_ratio - 0.85) / 0.15  # 0~1
            return -2.0 * excess  # 最多 -2.0
        elif max_ratio > 0.70:
            # 70-85% 轻度失衡，轻罚
            excess = (max_ratio - 0.70) / 0.15  # 0~1
            return -0.5 * excess  # 最多 -0.5
        return 0.0

    def _half_consistency_bonus(self, pnl: Tensor) -> float:
        """前后一致性奖励：前半段和后半段 Sortino 同号时加分。

        防止因子只在某一段市场环境（如牛市）有效。

        Returns:
            float，奖励/惩罚值
        """
        T = pnl.shape[1]
        if T < 20:
            return 0.0
        half = T // 2
        s1 = self._sortino(pnl[:, :half]).item()
        s2 = self._sortino(pnl[:, half:]).item()
        if s1 > 0 and s2 > 0:
            return 0.5  # 前后都赚钱，奖励
        elif s1 * s2 < 0:
            return -1.0  # 前后相反，重罚（如 index 组的 beta 因子）
        return 0.0  # 一正一零或两零，不奖不罚

    def _exposure_penalty(self, position: Tensor) -> float:
        """在场时间惩罚（仅下限，无上限）：收益优先模式。

        只惩罚极稀疏交易（<10%在场），不惩罚高在场时间。
        高在场时间（满仓趋势跟踪）是外汇市场最赚钱的形态之一，不应受罚。
        """
        flat = position.reshape(-1).abs()
        exposure = flat.mean().item()   # 连续仓位：均值即平均持仓量
        if exposure < 0.10:
            # 极稀疏：平均持仓 < 10% → 线性惩罚 [-2, 0)
            return float((exposure / 0.10 - 1.0) * 2.0)
        return 0.0

    def _turnover_penalty(self, turnover: Tensor) -> Tensor:
        """梯度式换手率惩罚。"""
        mean_to = turnover.mean()
        penalty = torch.clamp(
            (mean_to - 0.2) * 3.0,
            min=0.0,
            max=3.0,
        )
        return -penalty

    # ──────────────────────────────────────────────────────────────────────
    # Walk-Forward 辅助接口
    # ──────────────────────────────────────────────────────────────────────

    def evaluate_fold(
        self,
        factors:     Tensor,
        target_ret:  Tensor,
        train_start: int,
        train_end:   int,
        val_start:   int,
        val_end:     int,
    ) -> tuple[Tensor, Tensor]:
        """在指定训练/验证切片上计算组合多目标得分。

        train_score：用于 REINFORCE 梯度更新（in-sample 多目标）。
        val_score：用于选冠军，加入 OOS Sortino 门控：
          - OOS Sortino <= 0：乘以 0.1~0.5 惩罚，强制冠军必须在验证段盈利
          - OOS Sortino > 0：乘以最多 1.2 奖励
        """
        position = self._positions(factors)

        prev_pos = torch.roll(position, 1, dims=1)
        prev_pos[:, 0] = 0.0
        turnover = torch.abs(position - prev_pos)
        pnl      = position * target_ret - turnover * self.cost_rate

        pnl_train = pnl[:, train_start:train_end]
        pnl_val   = pnl[:, val_start:val_end]

        # 训练段：多目标 + 换手率惩罚
        train_bars = train_end - train_start
        train_score = self._multi_objective(
            factors[:, train_start:train_end],
            target_ret[:, train_start:train_end],
            pnl_train,
            position[:, train_start:train_end],
            eval_bars=train_bars,
        ) + self._turnover_penalty(turnover[:, train_start:train_end])

        # 验证段：多目标 × OOS Sortino 门控
        val_bars = val_end - val_start
        base_val    = self._multi_objective(
            factors[:, val_start:val_end],
            target_ret[:, val_start:val_end],
            pnl_val,
            position[:, val_start:val_end],
            eval_bars=val_bars,
        )
        oos_sor = self._sortino(pnl_val).item()
        if oos_sor <= 0:
            # OOS亏损：重惩罚（Sortino=-1 → mult=0.1；Sortino=0 → mult=0.5）
            mult = max(0.1, 0.5 + oos_sor * 0.4)
        else:
            # OOS盈利：轻奖励（最多+20%）
            mult = min(1.2, 1.0 + oos_sor * 0.1)
        val_score = base_val * mult

        return train_score, val_score

    def _reversal_bonus(self, factors: Tensor) -> Tensor:
        """反转奖励：鼓励因子有低/负自相关（均值回归特征）。

        计算每个品种的 lag-1 自相关系数，越接近 0 或负值 = 越好。
        高度正自相关（>0.5）= 趋势跟踪，减分。

        单品种模式：直接返回标量。
        """
        N = factors.shape[0]
        scores = []
        for n in range(N):
            x = factors[n, :-1]     # t=0..T-2
            y = factors[n, 1:]      # t=1..T-1
            xm = x - x.mean(); ym = y - y.mean()
            sx = (xm**2).mean().sqrt(); sy = (ym**2).mean().sqrt()
            ac1 = (xm*ym).mean() / (sx*sy + 1e-8) if sx > 1e-6 and sy > 1e-6 else torch.tensor(0.0)
            # 奖励低自相关：bonus = 1 - |ac1|, 负自相关额外加分
            bonus = 1.0 - torch.abs(ac1)
            if ac1 < 0:
                bonus = bonus + 0.5  # 负自相关（真正反转）额外加分
            bonus = torch.clamp(bonus, -1.0, 2.0)
            scores.append(bonus)
        return torch.stack(scores).mean()

    def _symmetry_check(self, position: Tensor) -> Tensor:
        """多空对称性检查：奖励 50/50 多空分布。

        均值回归策略应该在多空之间大致平衡，
        过度偏向某一侧 = 趋势跟踪特征，应惩罚。
        """
        long_ratio  = (position > 0).float().mean()
        short_ratio = (position < 0).float().mean()
        # 理想值：long_ratio ≈ 0.5, short_ratio ≈ 0.5
        # 偏差：|long_ratio - 0.5| + |short_ratio - 0.5|
        deviation = torch.abs(long_ratio - 0.5) + torch.abs(short_ratio - 0.5)
        # 偏差 0 → 奖励 1.0; 偏差 1.0 → 奖励 -1.0
        bonus = 1.0 - 2.0 * deviation
        return torch.clamp(bonus, -1.0, 1.0)

    def _multi_objective(
        self,
        factors:    Tensor,
        target_ret: Tensor,
        pnl:        Tensor,
        position:   Tensor,
        eval_bars:  int = 0,
    ) -> Tensor:
        """收益优先的多目标评分（2026-07-04 重构）。

        核心改变：加入年化绝对收益项（权重 0.40），这是最主要的优化目标。
        Sortino/Calmar 权重大幅下调，仅作为风险调整辅助。
        clamp 上限放开（Sortino 40→20 保持，收益无上限）。

        N=1 单品种模式权重略有不同（无 symbol_consistency/cost_stress）。

        2026-07-08: 新增 forex 模式 — 偏向均值回归策略。
        """
        N = pnl.shape[0]

        # ── 绝对收益（年化 log return）──────────────────────────────────
        # 连续仓位 pnl = position * target_ret - turnover * cost。
        # pnl.mean() 已是单 bar 平均收益，因此年化只乘每年 bar 数；不能再除以样本长度。
        ann_ret = pnl.mean() * self.periods_per_year   # 标量张量，无截断

        port_sortino = self._sortino(pnl)
        port_calmar  = self._calmar(pnl)
        ts_ic        = self._ts_ic_stability(factors, target_ret)
        tq           = self._turnover_quality(position)
        exp_pen      = self._exposure_penalty(position)

        if N == 1:
            beta_pen = 0.0 if self.market == "a_share" else self._beta_neutral_penalty(position)
            consist = self._half_consistency_bonus(pnl)

            if ModelConfig.REWARD_MODE == "forex":
                # Forex 均值回归模式：
                #   - 降年化收益权重 (0.80→0.25)：外汇趋势弱，避免奖励虚假趋势
                #   - 提 IC 权重 (0.03→0.25)：信号质量是核心
                #   - 新增反转奖励 (0.20)：奖励低/负因子自相关
                #   - 新增对称检查 (0.15)：奖励 50/50 多空平衡
                rev_bonus = self._reversal_bonus(factors)
                sym_bonus = self._symmetry_check(position)
                return (
                    0.25 * ann_ret           # 年化收益（降权，外汇趋势噪声大）
                    + 0.05 * port_sortino    # 风险调整辅助
                    + 0.05 * port_calmar     # 回撤控制
                    + 0.25 * ts_ic           # 信号质量（大幅提权）
                    + 0.20 * rev_bonus       # 反转奖励（核心：反趋势）
                    + 0.15 * sym_bonus       # 多空对称（均值回归特征）
                    + 0.05 * tq              # 交易频率质量
                    + exp_pen                # 稀疏惩罚
                    + beta_pen               # Beta 中性惩罚
                    + consist                # 前后一致性奖惩
                )

            if ModelConfig.REWARD_MODE == "ftmo":
                # FTMO 专属：年化收益 0.80，Calmar 0.10（控制 MDD 贴近 10% 上限）
                return (
                    0.80 * ann_ret           # 主目标：年化绝对收益（FTMO 加权）
                    + 0.05 * port_sortino    # 风险调整辅助（降权）
                    + 0.10 * port_calmar     # 回撤控制（保持，对齐 10% Max Loss）
                    + 0.03 * ts_ic           # IC 预测方向（降权）
                    + 0.02 * tq              # 交易频率质量（降权）
                    + exp_pen                # 稀疏惩罚
                    + beta_pen               # Beta 中性惩罚
                    + consist                # 前后一致性奖惩
                )
            return (
                0.60 * ann_ret           # 主目标：年化绝对收益
                + 0.15 * port_sortino    # 风险调整辅助
                + 0.10 * port_calmar     # 回撤控制辅助
                + 0.10 * ts_ic           # IC 预测方向
                + 0.05 * tq              # 交易频率质量
                + exp_pen                # 稀疏惩罚
                + beta_pen               # Beta 中性惩罚
                + consist                # 前后一致性奖惩
            )

        per_sym_sortino     = []
        per_sym_trade_count = []
        for n in range(N):
            per_sym_sortino.append(self._sortino(pnl[n]).item())
            # 连续仓位下，用 |position| 变化来估算交易次数
            pos_n = position[n].abs()
            # 视 tanh 输出均值作为持仓量，换手次数用前后差异估计
            diff = (pos_n[1:] - pos_n[:-1]).abs()
            trades = int((diff > 0.1).sum().item())
            per_sym_trade_count.append(trades)

        sym_cons = self._symbol_consistency(
            per_sym_sortino, per_sym_trade_count, eval_bars=eval_bars
        )
        cost_s   = self._cost_stress(position, target_ret)
        beta_pen = 0.0 if self.market == "a_share" else self._beta_neutral_penalty(position)
        consist  = self._half_consistency_bonus(pnl)

        if ModelConfig.REWARD_MODE == "ftmo":
            # FTMO 专属：年化收益 0.75（提权），Calmar 0.10（对齐 10% Max Loss）
            return (
                0.75 * ann_ret               # 主目标：年化绝对收益（FTMO 加权）
                + 0.05 * port_sortino        # 风险调整辅助（降权）
                + 0.10 * port_calmar         # 回撤控制（提权，控制 MDD）
                + 0.02 * ts_ic               # IC 预测方向（降权）
                + 0.03 * sym_cons            # 品种一致性（降权）
                + 0.02 * cost_s              # 成本压力测试（降权）
                + 0.03 * tq                  # 交易频率质量（降权）
                + exp_pen                    # 稀疏惩罚
                + beta_pen                   # Beta 中性惩罚
                + consist                    # 前后一致性奖惩
            )

        return (
            0.60 * ann_ret               # 主目标：年化绝对收益
            + 0.10 * port_sortino        # 风险调整辅助
            + 0.05 * port_calmar         # 回撤控制辅助
            + 0.10 * ts_ic               # IC 预测方向
            + 0.05 * sym_cons            # 品种一致性
            + 0.05 * cost_s              # 成本压力测试
            + 0.05 * tq                  # 交易频率质量
            + exp_pen                    # 稀疏惩罚
            + beta_pen                   # Beta 中性惩罚
            + consist                    # 前后一致性奖惩
        )

    # ──────────────────────────────────────────────────────────────────────
    # 公开接口（非 Walk-Forward 模式）
    # ──────────────────────────────────────────────────────────────────────

    def evaluate(
        self,
        factors:    Tensor,
        raw_dict:   dict,
        target_ret: Tensor,
    ) -> tuple[Tensor, float]:
        """评估一组 Alpha 因子（含 OOS 80/20 门控）。"""
        position = self._positions(factors)

        prev_pos = torch.roll(position, 1, dims=1)
        prev_pos[:, 0] = 0.0
        turnover = torch.abs(position - prev_pos)
        pnl      = position * target_ret - turnover * self.cost_rate

        T     = factors.shape[1]
        split = int(math.floor(T * 0.8))

        score = self._multi_objective(
            factors[:, :split], target_ret[:, :split],
            pnl[:, :split], position[:, :split],
            eval_bars=split,
        ) + self._turnover_penalty(turnover[:, :split])

        # OOS 门控（最后 20%）
        pnl_oos = pnl[:, split:]
        oos_sor = self._sortino(pnl_oos).item()
        if oos_sor <= 0:
            mult = max(0.1, 0.5 + oos_sor * 0.4)
            score = score * mult
        else:
            score = score * min(1.2, 1.0 + oos_sor * 0.1)

        mean_oos = pnl_oos.mean().item()
        return score, mean_oos
