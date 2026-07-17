"""
paper_trading/factor_ranker.py — 因子公式 → 当日 Top-K 选股

用 StackVM 执行 token 公式得到截面因子值，剔除无效股，降序取 Top-K。
不用 tanh：纯多头排序信号（A股只能做多）。
"""
from __future__ import annotations

import torch

from model_core.vm import StackVM


class FactorRanker:
    def __init__(self, formula: list[int]) -> None:
        self.formula = [int(t) for t in formula]
        self.vm = StackVM()

    def rank(self, feat_slice: torch.Tensor, valid_mask: torch.Tensor,
             codes: list[str], top_k: int = 10) -> list[str]:
        """对当日有效股票按因子值降序取 Top-K 代码。

        Args:
            feat_slice: [N, F, T] 截至当日的特征张量（用最后时间步作当日截面）。
            valid_mask: [N] bool，当日是否可交易（停牌/未上市/封板剔除）。
            codes:      [N] 股票代码，与张量 N 维一一对应。
            top_k:      取前 K 只。
        """
        factor = self.vm.execute(self.formula, feat_slice)   # [N, T] 或 None
        if factor is None:
            return []
        vals = factor[:, -1]                                 # 当日截面因子值 [N]
        vals = torch.nan_to_num(vals, nan=float("-inf"))
        # 无效股设为 -inf，排序自然沉底
        vals = torch.where(valid_mask, vals, torch.full_like(vals, float("-inf")))
        n_valid = int(valid_mask.sum().item())
        k = min(top_k, n_valid)
        if k <= 0:
            return []
        order = torch.argsort(vals, descending=True)[:k]
        return [codes[i] for i in order.tolist()]
