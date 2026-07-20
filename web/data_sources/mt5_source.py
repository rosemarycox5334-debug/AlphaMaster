"""MT5 数据源（MetaTrader5 终端，需已登录运行）。

NOTE: MT5 copy_rates_from_pos 返回的 'time' 字段使用经纪商服务器本地时间
（如 UTC+2/+3），而非真正的 Unix 时间戳。本模块在返回 Bar 之前会自动检测
并修正该偏移，将所有 ts 统一为真实 UTC epoch 秒，确保与 time.time() 一致。
"""
from __future__ import annotations

import threading
import time as _time

from web.data_sources.base import Bar, DataSource, DataSourceUnavailable

_TF = {
    "1m": "TIMEFRAME_M1",
    "5m": "TIMEFRAME_M5",
    "15m": "TIMEFRAME_M15",
    "30m": "TIMEFRAME_M30",
    "1h": "TIMEFRAME_H1",
    "4h": "TIMEFRAME_H4",
    "1d": "TIMEFRAME_D1",
    "1w": "TIMEFRAME_W1",
    "1M": "TIMEFRAME_MN1",
}

_PRESETS = [
    "XAUUSD", "XAGUSD", "EURUSD", "USDJPY", "GBPUSD",
    "US30.cash", "US100.cash", "US500.cash", "US2000.cash", "JP225.cash",
]


class MT5Source(DataSource):
    kind = "mt5"
    label = "MT5"

    def __init__(self) -> None:
        self._connected = False
        self._lock = threading.Lock()
        self._server_offset: int | None = None  # broker_ts - real_utc (秒)

    def available(self) -> tuple[bool, str]:
        try:
            import MetaTrader5  # noqa: F401
        except ImportError:
            return (False, "未安装 MetaTrader5：pip install MetaTrader5")
        return (True, "需 MT5 终端已登录运行")

    def supported_timeframes(self) -> list[str]:
        return list(_TF.keys())

    def preset_symbols(self) -> list[str]:
        return list(_PRESETS)

    def connect(self) -> None:
        try:
            import MetaTrader5 as mt5
        except ImportError as exc:
            raise DataSourceUnavailable("未安装 MetaTrader5") from exc
        if self._connected:
            return
        if not mt5.initialize():
            raise DataSourceUnavailable(
                f"MT5 初始化失败 {mt5.last_error()}；请确认终端已打开并登录"
            )
        self._connected = True
        self._detect_server_offset(mt5)

    def _detect_server_offset(self, mt5) -> None:
        """检测 MT5 经纪商服务器时间与真实 UTC 的偏移量。

        MT5 的 copy_rates / symbol_info_tick 返回的时间戳使用经纪商服务器
        本地时钟（常见 UTC+2 或 UTC+3），但以 Unix 时间戳格式呈现。
        通过比较 tick.time 与本机 time.time() 推算偏移。
        """
        try:
            tick = mt5.symbol_info_tick("XAUUSD") or mt5.symbol_info_tick("EURUSD")
            if tick and tick.time:
                real_now = int(_time.time())
                # 偏移 = broker_ts - real_utc，四舍五入到最近整小时
                raw_offset = int(tick.time) - real_now
                # 容错：±30 分钟内的抖动忽略，round 到整小时
                self._server_offset = round(raw_offset / 3600) * 3600
            else:
                self._server_offset = 0
        except Exception:
            self._server_offset = 0

    def disconnect(self) -> None:
        if self._connected:
            try:
                import MetaTrader5 as mt5
                mt5.shutdown()
            except Exception:
                pass
        self._connected = False

    def fetch_bars(
        self, symbol: str, timeframe: str, n: int, drop_forming: bool = True
    ) -> list[Bar]:
        if timeframe not in _TF:
            raise DataSourceUnavailable(f"MT5 不支持周期 {timeframe}")
        try:
            import MetaTrader5 as mt5
        except ImportError as exc:
            raise DataSourceUnavailable("未安装 MetaTrader5") from exc

        with self._lock:
            self.connect()
            tf_const = getattr(mt5, _TF[timeframe])
            try:
                mt5.symbol_select(symbol, True)
            except Exception:
                pass
            fetch_n = n + 1 if drop_forming else n
            rates = mt5.copy_rates_from_pos(symbol, tf_const, 0, fetch_n)

        if rates is None or len(rates) == 0:
            raise DataSourceUnavailable(
                f"MT5 无法获取 {symbol} {timeframe} 数据 {mt5.last_error()}"
            )

        # copy_rates_from_pos 返回升序，最后一根为正在形成的 bar
        rows = list(rates)
        if drop_forming and len(rows) > 1:
            rows = rows[:-1]

        bars: list[Bar] = []
        offset = self._server_offset or 0
        for r in rows:
            try:
                vol = float(r["tick_volume"])
            except Exception:
                vol = float(r["real_volume"]) if "real_volume" in r.dtype.names else 0.0
            bars.append(
                Bar(
                    ts=int(r["time"]) - offset,  # 修正为真实 UTC epoch
                    open=float(r["open"]),
                    high=float(r["high"]),
                    low=float(r["low"]),
                    close=float(r["close"]),
                    volume=vol,
                )
            )
        return bars
