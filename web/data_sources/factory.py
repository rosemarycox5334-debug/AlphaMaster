"""数据源工厂 + 单例注册（复用连接）。"""
from __future__ import annotations

import threading

from web.data_sources.base import DataSource

# 前端下拉可见的数据源（国内期货默认隐藏，如需启用加回列表即可）
SOURCE_KINDS: tuple[tuple[str, str], ...] = (
    ("mt5", "MT5"),
    ("tradingview", "TradingView"),
)

_INSTANCES: dict[str, DataSource] = {}
_LOCK = threading.Lock()


def _build(kind: str) -> DataSource:
    if kind == "mt5":
        from web.data_sources.mt5_source import MT5Source
        return MT5Source()
    if kind == "tradingview":
        from web.data_sources.tradingview_source import TradingViewSource
        return TradingViewSource()
    if kind == "okx":
        from web.data_sources.okx_source import OKXSource
        return OKXSource()
    if kind == "tongdaxin":
        from web.data_sources.tongdaxin_source import TongdaxinSource
        return TongdaxinSource()
    if kind == "domestic_futures":
        from web.data_sources.domestic_futures_source import DomesticFuturesSource
        return DomesticFuturesSource()
    raise ValueError(f"未知数据源: {kind}")


def get_source(kind: str) -> DataSource:
    """返回该 kind 的单例数据源（懒创建，复用连接）。"""
    with _LOCK:
        inst = _INSTANCES.get(kind)
        if inst is None:
            inst = _build(kind)
            _INSTANCES[kind] = inst
        return inst


def list_sources() -> list[dict]:
    """列出所有数据源及其可用状态（供前端灰显/引导）。"""
    out = []
    for kind, label in SOURCE_KINDS:
        try:
            src = get_source(kind)
            ok, hint = src.available()
            tfs = src.supported_timeframes()
            presets = src.preset_symbols()
        except Exception as exc:  # noqa: BLE001
            ok, hint, tfs, presets = False, str(exc), [], []
        out.append(
            {
                "id": kind,
                "label": label,
                "available": ok,
                "hint": hint,
                "timeframes": tfs,
                "presets": presets,
            }
        )
    return out
