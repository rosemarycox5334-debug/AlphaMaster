# -*- coding: utf-8 -*-
"""A 股数据源：东方财富首选 + 腾讯 / zzshare / 新浪兜底。

取代原先的 TongdaxinSource（其实际底层也是新浪单一源）。
"""
from __future__ import annotations

import logging

from web.data_sources.ashare_quotes import (
    SUPPORTED_TIMEFRAMES,
    _PRESET_SYMBOLS,
    fetch_ashare_bars,
)
from web.data_sources.base import Bar, DataSource, DataSourceUnavailable

logger = logging.getLogger(__name__)


class AShareSource(DataSource):
    """A 股 / 指数行情，多源自动降级。"""

    kind = "ashare"
    label = "A股/指数"

    def available(self) -> tuple[bool, str]:
        try:
            import requests  # noqa: F401
        except ImportError:
            return (False, "未安装 requests：pip install requests")
        return (True, "东方财富首选 + 腾讯/zzshare/新浪兜底")

    def supported_timeframes(self) -> list[str]:
        return list(SUPPORTED_TIMEFRAMES)

    def preset_symbols(self) -> list[str]:
        return list(_PRESET_SYMBOLS)

    def fetch_bars(
        self, symbol: str, timeframe: str, n: int, drop_forming: bool = True
    ) -> list[Bar]:
        if timeframe not in SUPPORTED_TIMEFRAMES:
            raise DataSourceUnavailable(
                f"A股数据源不支持周期 {timeframe}，支持 {list(SUPPORTED_TIMEFRAMES)}"
            )
        try:
            rows = fetch_ashare_bars(symbol, timeframe, n, drop_forming=drop_forming)
        except Exception as exc:
            raise DataSourceUnavailable(str(exc)) from exc
        bars = [
            Bar(
                ts=int(r["ts"]),
                open=float(r["open"]),
                high=float(r["high"]),
                low=float(r["low"]),
                close=float(r["close"]),
                volume=float(r["volume"]),
            )
            for r in rows
        ]
        if not bars:
            raise DataSourceUnavailable(f"A股无数据：{symbol} {timeframe}")
        return bars
