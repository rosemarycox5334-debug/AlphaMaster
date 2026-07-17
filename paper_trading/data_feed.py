"""
paper_trading/data_feed.py — 数据喂法

ReplayFeed: 历史回放，持有已加载 AShareDataManager，按日切片。
LiveFeed:   实盘（Task 12 实现）。
两者共用 PortfolioEngine.step。
"""
from __future__ import annotations

import torch

from paper_trading.config import LIMIT_PCT


class ReplayFeed:
    def __init__(self, manager) -> None:
        self.mgr = manager
        self._feat = manager.feat_tensor          # [N, F, T] 一次算好
        self._dates = manager.trade_dates
        self._date_idx = {d: i for i, d in enumerate(self._dates)}
        self._close = manager.raw_dict["close"]   # [N, T]
        self._open = manager.raw_dict["open"]     # [N, T]
        self._valid = manager.valid_mask          # [N, T]

    def trade_dates(self) -> list[str]:
        return list(self._dates)

    def slice_until(self, date: str) -> tuple[torch.Tensor, torch.Tensor]:
        """返回截至 date（含）的特征张量 [N,F,t+1] 与当日 valid_mask [N]。"""
        t = self._date_idx[date]
        feat_slice = self._feat[:, :, : t + 1]
        valid = self._valid[:, t]
        return feat_slice, valid

    def bar_at(self, date: str) -> dict:
        """返回该日各股票的成交 bar：{code:{open,close,limit_up,limit_down,tradable}}。

        涨跌停判定：当日 open 相对前一交易日 close 涨/跌达 ±LIMIT_PCT（留 0.1% 容差）
        视为封板。open >= prev_close*(1+LIMIT_PCT)*0.999 → 封涨停；对称判跌停。
        """
        t = self._date_idx[date]
        codes = self.mgr.symbols
        bar = {}
        for n, code in enumerate(codes):
            o = float(self._open[n, t])
            c = float(self._close[n, t])
            tradable = bool(self._valid[n, t])
            limit_up = limit_down = False
            if t > 0:
                pc = float(self._close[n, t - 1])
                if pc > 0:
                    if o >= pc * (1 + LIMIT_PCT) * 0.999:
                        limit_up = True
                    elif o <= pc * (1 - LIMIT_PCT) * 1.001:
                        limit_down = True
            bar[code] = {"open": o, "close": c, "limit_up": limit_up,
                         "limit_down": limit_down, "tradable": tradable}
        return bar


class LiveFeed(ReplayFeed):
    """实盘：每次 advance() 拉当日增量数据，重建底层管理器后复用切片逻辑。

    与 ReplayFeed 唯一差异是数据来源随时间增长；切片/bar 计算完全复用父类。
    """
    def __init__(self, manager, start: str) -> None:
        super().__init__(manager)
        self._start = start

    def advance(self, today: str, end: str) -> None:
        """拉取截至 today 的最新数据并重建内部张量（含 today 当日已收盘 K线）。"""
        self.mgr.load(start=self._start, end=end)
        # 重建父类缓存的张量视图
        self._feat = self.mgr.feat_tensor
        self._dates = self.mgr.trade_dates
        self._date_idx = {d: i for i, d in enumerate(self._dates)}
        self._close = self.mgr.raw_dict["close"]
        self._open = self.mgr.raw_dict["open"]
        self._valid = self.mgr.valid_mask
