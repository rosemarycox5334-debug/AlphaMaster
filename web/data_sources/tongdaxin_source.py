"""通达信数据源（新浪财经 HTTP API，A 股 / 指数）。

pytdx 1.72 已停止维护，与当前通达信服务器协议不兼容（返回 ret_count=800
但响应体只有 2 字节，导致 unpack 报错）。改用新浪财经公开 K 线接口，
无需登录、稳定、支持 5m/15m/30m/1h/1d，1w/1M 从日线聚合，1m 自动降级到 5m。
"""
from __future__ import annotations

import calendar
import json
import threading
import time
from datetime import date, datetime, timedelta, timezone

import requests

from web.data_sources.base import Bar, DataSource, DataSourceUnavailable

_SINA_URL = (
    "https://money.finance.sina.com.cn/quotes_service/api/json_v2.php/"
    "CN_MarketData.getKLineData"
)
_CST = timezone(timedelta(hours=8))

# 项目周期 -> 新浪 scale（分钟）。1m 降级到 5m；1w/1M 从日线聚合。
_SCALE = {
    "1m": 5,
    "5m": 5,
    "15m": 15,
    "30m": 30,
    "1h": 60,
    "1d": 240,
    "1w": 240,
    "1M": 240,
}
_AGGREGATE = {"1w", "1M"}

# 新浪单次返回上限（实测 5000 可用）
_MAX_DATALEN = 5000

_PRESETS = ["600519", "000001", "300750", "601318", "000858", "sh000001", "sz399006"]

_HTTP_TIMEOUT = 15.0
_HTTP_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
    ),
    "Referer": "https://finance.sina.com.cn/",
}


def _parse_market(code: str) -> tuple[int, str]:
    """返回 (market, pure_code)。1=上海, 0=深圳。规则与旧实现一致。

    优先按代码本身判定（002/003/300/301=深市，60/688/689=沪市主板/科创板，
    5/9/11/13=沪市基金/B股/转债，399=深证指数）；仅 000xxx 有歧义时才以
    sh/sz 前缀为准（000001 无前缀 → 深市平安银行；sh000001 → 上证指数）。
    """
    c = code.strip().upper()
    has_prefix = c.startswith("SH") or c.startswith("SZ")
    prefix_market = 1 if c.startswith("SH") else 0
    pure = c[2:] if has_prefix else c

    if pure.startswith(("002", "003", "300", "301")):
        return 0, pure
    if pure.startswith(("60", "688", "689")):
        return 1, pure
    if pure[:1] in ("5", "9") or pure.startswith(("11", "13")):
        return 1, pure
    if pure.startswith("399"):
        return 0, pure

    if has_prefix:
        return prefix_market, pure
    return 0, pure


def _to_sina_symbol(market: int, code: str) -> str:
    return ("sh" if market == 1 else "sz") + code


def _parse_sina_dt(day: str) -> int:
    """新浪 'day' 字段 -> 该 bar 收盘时刻的 Unix 秒（北京时间 UTC+8）。

    - 分钟级：'2026-10-23 15:00:00'（本身就是 bar 结束时刻）
    - 日级：'2026-10-23'（无时间，按 A 股 15:00 收盘处理）
    """
    s = str(day).strip()
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
        try:
            dt = datetime.strptime(s, fmt)
        except ValueError:
            continue
        if fmt == "%Y-%m-%d":
            dt = dt.replace(hour=15, minute=0, second=0)
        return int(dt.replace(tzinfo=_CST).timestamp())
    return 0


def _week_end_ts(year: int, week: int) -> int:
    """ISO 周的周五 15:00（北京时间）-> Unix 秒；非法周返回 0。"""
    try:
        monday = date.fromisocalendar(year, week, 1)
    except ValueError:
        return 0
    friday = monday + timedelta(days=4)
    dt = datetime(friday.year, friday.month, friday.day, 15, 0, 0, tzinfo=_CST)
    return int(dt.timestamp())


def _month_end_ts(year: int, month: int) -> int:
    """某月最后一天 15:00（北京时间）-> Unix 秒。"""
    last_day = calendar.monthrange(year, month)[1]
    dt = datetime(year, month, last_day, 15, 0, 0, tzinfo=_CST)
    return int(dt.timestamp())


class TongdaxinSource(DataSource):
    kind = "tongdaxin"
    label = "通达信"

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._session: requests.Session | None = None
        self._session_time = 0.0

    def available(self) -> tuple[bool, str]:
        try:
            import requests  # noqa: F401
        except ImportError:
            return (False, "未安装 requests：pip install requests")
        return (True, "新浪财经公开接口 · A 股 / 指数")

    def supported_timeframes(self) -> list[str]:
        return list(_SCALE.keys())

    def preset_symbols(self) -> list[str]:
        return list(_PRESETS)

    def connect(self) -> None:
        # HTTP 无状态，无需建立长连接
        self.disconnect()

    def disconnect(self) -> None:
        if self._session is not None:
            try:
                self._session.close()
            except Exception:
                pass
        self._session = None
        self._session_time = 0.0

    def _get_session(self) -> requests.Session:
        # 15 分钟内复用 session（keep-alive）
        if self._session is None or time.time() - self._session_time > 900:
            s = requests.Session()
            s.headers.update(_HTTP_HEADERS)
            self._session = s
            self._session_time = time.time()
        return self._session

    def _fetch_raw(self, sina_symbol: str, scale: int, datalen: int) -> list[dict]:
        """调用新浪 K 线接口，返回原始 dict 列表（可能为空）。"""
        sess = self._get_session()
        params = {
            "symbol": sina_symbol,
            "scale": scale,
            "ma": "no",
            "datalen": min(int(datalen), _MAX_DATALEN),
        }
        try:
            r = sess.get(_SINA_URL, params=params, timeout=_HTTP_TIMEOUT)
        except Exception as exc:
            raise DataSourceUnavailable(f"新浪财经请求失败：{exc}") from exc
        if r.status_code != 200:
            raise DataSourceUnavailable(f"新浪财经返回 HTTP {r.status_code}")
        text = (r.text or "").strip()
        if not text or text == "null":
            return []
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise DataSourceUnavailable(f"新浪财经返回无法解析：{text[:120]}") from exc
        return data if isinstance(data, list) else []

    def _aggregate(self, daily: list[dict], timeframe: str) -> list[dict]:
        """把日线聚合成周线 / 月线。

        返回项的 day 用「周五 15:00」或「月末最后一天 15:00」表示（理论周期结束），
        以便 drop_forming 依据 ts>now 正确剔除仍在形成的聚合 bar。
        """
        if not daily:
            return []
        groups: dict[tuple[int, int], list[dict]] = {}
        for r in daily:
            ts = _parse_sina_dt(r.get("day", ""))
            if ts <= 0:
                continue
            dt = datetime.fromtimestamp(ts, tz=_CST)
            if timeframe == "1w":
                iso = dt.isocalendar()
                key = (iso[0], iso[1])
            else:  # 1M
                key = (dt.year, dt.month)
            groups.setdefault(key, []).append(r)

        out: list[dict] = []
        for key, rows in groups.items():
            end_ts = _week_end_ts(*key) if timeframe == "1w" else _month_end_ts(*key)
            if end_ts <= 0:
                continue
            try:
                o = float(rows[0]["open"])
                c = float(rows[-1]["close"])
                h = max(float(r["high"]) for r in rows)
                l = min(float(r["low"]) for r in rows)
                v = sum(float(r.get("volume", 0) or 0) for r in rows)
            except (KeyError, ValueError, TypeError):
                continue
            out.append({
                "day": datetime.fromtimestamp(end_ts, tz=_CST).strftime("%Y-%m-%d %H:%M:%S"),
                "open": str(o),
                "high": str(h),
                "low": str(l),
                "close": str(c),
                "volume": str(v),
            })
        out.sort(key=lambda r: _parse_sina_dt(r["day"]))
        return out

    def fetch_bars(
        self, symbol: str, timeframe: str, n: int, drop_forming: bool = True
    ) -> list[Bar]:
        """从新浪财经拉取最近 n 根已收盘 K 线（升序）。

        注意：volume 单位为「股」（Sina 原始单位），与旧 pytdx 的「手」不同。
        """
        if timeframe not in _SCALE:
            raise DataSourceUnavailable(
                f"通达信不支持周期 {timeframe}，支持 {list(_SCALE.keys())}"
            )
        market, code = _parse_market(symbol)
        sina_symbol = _to_sina_symbol(market, code)
        scale = _SCALE[timeframe]
        want = max(n + 2, 20)

        # 聚合周期需要更多原始日线
        fetch_n = _MAX_DATALEN if timeframe in _AGGREGATE else want

        with self._lock:
            raw = self._fetch_raw(sina_symbol, scale, fetch_n)

        # 000xxx 无前缀：默认深市股票（000001=平安银行）。
        # 若返回空，回退到沪市指数（000001=上证指数）。
        if not raw and not symbol.strip().upper().startswith(("SH", "SZ")):
            if code.startswith("000"):
                alt_symbol = "sh" + code
                with self._lock:
                    alt_raw = self._fetch_raw(alt_symbol, scale, fetch_n)
                if alt_raw:
                    raw = alt_raw
                    sina_symbol = alt_symbol

        if not raw:
            raise DataSourceUnavailable(
                f"通达信无数据：{symbol}（sina={sina_symbol}, scale={scale}）。"
                f"请确认代码正确；指数需带 sh/sz 前缀（如 sh000001）。"
            )

        if timeframe in _AGGREGATE:
            raw = self._aggregate(raw, timeframe)

        ts_to_bar: dict[int, Bar] = {}
        for r in raw:
            try:
                ts = _parse_sina_dt(r.get("day", ""))
                if ts <= 0:
                    continue
                ts_to_bar[ts] = Bar(
                    ts=ts,
                    open=float(r["open"]),
                    high=float(r["high"]),
                    low=float(r["low"]),
                    close=float(r["close"]),
                    volume=float(r.get("volume", 0.0) or 0.0),
                )
            except (KeyError, ValueError, TypeError):
                continue

        bars = sorted(ts_to_bar.values(), key=lambda b: b.ts)

        # 剔除未收盘 / 未来 bar：ts 为收盘时刻，ts>now 说明仍在形成中
        if drop_forming and bars:
            now = time.time()
            while bars and int(bars[-1].ts) > now:
                bars.pop()
        return bars[-n:]
