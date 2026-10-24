# -*- coding: utf-8 -*-
"""A 股 K 线多源取数：东方财富首选 + 腾讯 / zzshare / 新浪兜底。

完全自包含，仅依赖 requests（zzshare 可选）。不依赖 PA_Agent。

统一返回结构（升序，最旧在前）：
    {"ts": int, "open": float, "high": float, "low": float,
     "close": float, "volume": float}

ts 语义 = bar 结束时刻的 Unix 秒（北京时间），与 tongdaxin_source 保持一致，
便于 realtime_manager 的 drop_forming / _ensure_closed_bars 直接复用。
"""
from __future__ import annotations

import json
import logging
import threading
import time
from datetime import datetime, timedelta, timezone

logger = logging.getLogger(__name__)

_CST = timezone(timedelta(hours=8))
_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)
_HTTP_TIMEOUT = 12.0

_EM_KLT = {
    "1m": "1", "5m": "5", "15m": "15", "30m": "30",
    "1h": "60", "1d": "101", "1w": "102", "1M": "103",
}
_TX_MIN = {"1m": "m1", "5m": "m5", "15m": "m15", "30m": "m30", "1h": "m60"}
_ZZ_FREQ = {"1m": "1min", "5m": "5min", "15m": "15min", "30m": "30min", "1h": "60min"}

SUPPORTED_TIMEFRAMES = ("1m", "5m", "15m", "30m", "1h", "4h", "1d", "1w", "1M")

_PRESET_SYMBOLS = [
    "600519", "000001", "300750", "601318", "000858",
    "sh000001", "sz399006", "sh000300",
]


def _f(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def _digits(symbol: str) -> tuple[str, str]:
    s = symbol.strip().lower()
    digits = "".join(ch for ch in s if ch.isdigit())
    digits = digits[-6:] if len(digits) >= 6 else digits
    prefix = "sh" if s.startswith("sh") else ("sz" if s.startswith("sz") else "")
    return prefix, digits


def _to_em_secid(symbol: str) -> str:
    prefix, digits = _digits(symbol)
    if prefix == "sh":
        return f"1.{digits}"
    if prefix == "sz":
        return f"0.{digits}"
    if digits.startswith("399"):
        return f"0.{digits}"
    if digits[:1] in ("5", "6", "9"):
        return f"1.{digits}"
    return f"0.{digits}"


def _to_tx_code(symbol: str) -> str:
    prefix, digits = _digits(symbol)
    if prefix:
        return prefix + digits
    if digits.startswith(("5", "6", "9")):
        return "sh" + digits
    return "sz" + digits


def _to_zz_code(symbol: str) -> str:
    prefix, digits = _digits(symbol)
    if prefix == "sh":
        return f"{digits}.SH"
    if prefix == "sz":
        return f"{digits}.SZ"
    if digits[:1] in ("5", "6", "9"):
        return f"{digits}.SH"
    if digits in ("000300", "000016", "000905", "000852"):
        return f"{digits}.SH"
    return f"{digits}.SZ"


def _to_sina_symbol(symbol: str) -> str:
    prefix, digits = _digits(symbol)
    if prefix:
        return prefix + digits
    if digits.startswith(("5", "6", "9")):
        return "sh" + digits
    return "sz" + digits




def _parse_time_to_ts(text: str) -> int:
    """把各种时间字符串解析成 bar 结束时刻的 Unix 秒（CST）。"""
    s = str(text).strip()
    if not s:
        return 0
    if s.isdigit():
        if len(s) == 8:
            try:
                return int(datetime.strptime(s, "%Y%m%d").replace(hour=15, tzinfo=_CST).timestamp())
            except ValueError:
                return 0
        if len(s) in (12, 14):
            fmt = "%Y%m%d%H%M" if len(s) == 12 else "%Y%m%d%H%M%S"
            try:
                return int(datetime.strptime(s, fmt).replace(tzinfo=_CST).timestamp())
            except ValueError:
                return 0
        return 0
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"):
        try:
            return int(datetime.strptime(s, fmt).replace(tzinfo=_CST).timestamp())
        except ValueError:
            continue
    try:
        return int(datetime.strptime(s[:10], "%Y-%m-%d").replace(hour=15, tzinfo=_CST).timestamp())
    except ValueError:
        return 0


def _drop_forming(rows: list, drop: bool) -> list:
    if not drop or not rows:
        return rows
    now = int(time.time())
    while rows and int(rows[-1]["ts"]) > now:
        rows.pop()
    return rows


def _aggregate_4h(hourly: list) -> list:
    out = []
    for i in range(0, len(hourly) - 3, 4):
        chunk = hourly[i:i + 4]
        out.append({
            "ts": chunk[-1]["ts"],
            "open": chunk[0]["open"],
            "high": max(r["high"] for r in chunk),
            "low": min(r["low"] for r in chunk),
            "close": chunk[-1]["close"],
            "volume": sum(r["volume"] for r in chunk),
        })
    return out


def _aggregate_daily(daily: list, timeframe: str) -> list:
    groups = {}
    for r in daily:
        dt = datetime.fromtimestamp(r["ts"], tz=_CST)
        if timeframe == "1w":
            iso = dt.isocalendar()
            key = (iso[0], iso[1])
        else:
            key = (dt.year, dt.month)
        groups.setdefault(key, []).append(r)
    out = []
    for key in sorted(groups):
        chunk = groups[key]
        out.append({
            "ts": chunk[-1]["ts"],
            "open": chunk[0]["open"],
            "high": max(x["high"] for x in chunk),
            "low": min(x["low"] for x in chunk),
            "close": chunk[-1]["close"],
            "volume": sum(x["volume"] for x in chunk),
        })
    return out


def _em_fetch(symbol: str, timeframe: str, n: int) -> list:
    import requests

    if timeframe == "4h":
        return _aggregate_4h(_em_fetch(symbol, "1h", n * 4 + 8))[-n:]
    klt = _EM_KLT.get(timeframe)
    if not klt:
        return []
    params = {
        "secid": _to_em_secid(symbol),
        "fields1": "f1,f2,f3,f4,f5,f6",
        "fields2": "f51,f52,f53,f54,f55,f56,f57,f58",
        "klt": klt,
        "fqt": "1",
    }
    if timeframe in ("1d", "1w", "1M"):
        params["end"] = "20500101"
        params["lmt"] = str(max(5, min(n + 8, 1200)))
    else:
        params["beg"] = "0"
        params["end"] = "20500101"
    url = "https://push2his.eastmoney.com/api/qt/stock/kline/get"
    headers = {"User-Agent": _UA, "Referer": "https://quote.eastmoney.com/"}
    resp = None
    last_exc = None
    # 东财检测 TLS 指纹，优先用 curl_cffi 模拟浏览器
    try:
        from curl_cffi import requests as _cffi
        for imp in ("chrome120", "chrome131", "edge101"):
            try:
                resp = _cffi.get(
                    url, params=params, headers=headers,
                    timeout=_HTTP_TIMEOUT, impersonate=imp,
                )
                if resp.status_code == 200:
                    break
            except Exception as exc:
                last_exc = exc
                resp = None
    except ImportError:
        pass
    if resp is None or resp.status_code != 200:
        try:
            resp = requests.get(url, params=params, headers=headers, timeout=_HTTP_TIMEOUT)
        except Exception as exc:
            raise RuntimeError(f"东财请求失败: {exc or last_exc}") from exc
    if resp.status_code != 200:
        raise RuntimeError(f"东财 HTTP {resp.status_code}")
    klines = (resp.json().get("data") or {}).get("klines") or []
    rows = []
    for line in klines:
        parts = str(line).split(",")
        if len(parts) < 6:
            continue
        ts = _parse_time_to_ts(parts[0])
        if ts <= 0:
            continue
        rows.append({
            "ts": ts,
            "open": _f(parts[1]),
            "close": _f(parts[2]),
            "high": _f(parts[3]),
            "low": _f(parts[4]),
            "volume": _f(parts[5]),
        })
    rows.sort(key=lambda r: r["ts"])
    return rows[-n:]




def _tx_fetch(symbol: str, timeframe: str, n: int) -> list:
    import requests

    if timeframe == "4h":
        return _aggregate_4h(_tx_fetch(symbol, "1h", n * 4 + 8))[-n:]
    code = _to_tx_code(symbol)
    headers = {"User-Agent": _UA, "Referer": "https://gu.qq.com/"}
    rows = []
    if timeframe in _TX_MIN:
        p = _TX_MIN[timeframe]
        resp = requests.get(
            "https://ifzq.gtimg.cn/appstock/app/kline/mkline",
            params={"param": f"{code},{p},,{min(n + 8, 320)}"},
            headers=headers,
            timeout=_HTTP_TIMEOUT,
        )
        if resp.status_code != 200:
            raise RuntimeError(f"腾讯分钟 HTTP {resp.status_code}")
        node = (resp.json().get("data") or {}).get(code) or {}
        raw = node.get(p) or node.get("qfq" + p) or []
        for item in raw:
            if not isinstance(item, (list, tuple)) or len(item) < 6:
                continue
            ts = _parse_time_to_ts(item[0])
            if ts <= 0:
                continue
            rows.append({
                "ts": ts,
                "open": _f(item[1]),
                "close": _f(item[2]),
                "high": _f(item[3]),
                "low": _f(item[4]),
                "volume": _f(item[5]),
            })
    elif timeframe in ("1d", "1w", "1M"):
        resp = requests.get(
            "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get",
            params={"param": f"{code},day,,,{min(max(n + 8, 20), 6400)},qfq"},
            headers=headers,
            timeout=_HTTP_TIMEOUT,
        )
        if resp.status_code != 200:
            raise RuntimeError(f"腾讯日线 HTTP {resp.status_code}")
        node = (resp.json().get("data") or {}).get(code) or {}
        raw = node.get("qfqday") or node.get("day") or []
        daily = []
        for item in raw:
            if not isinstance(item, (list, tuple)) or len(item) < 6:
                continue
            ts = _parse_time_to_ts(item[0])
            if ts <= 0:
                continue
            daily.append({
                "ts": ts,
                "open": _f(item[1]),
                "close": _f(item[2]),
                "high": _f(item[3]),
                "low": _f(item[4]),
                "volume": _f(item[5]),
            })
        daily.sort(key=lambda r: r["ts"])
        rows = daily if timeframe == "1d" else _aggregate_daily(daily, timeframe)
    rows.sort(key=lambda r: r["ts"])
    return rows[-n:]


_zz_lock = threading.Lock()
_zz_cooldown_until = 0.0
_ZZ_COOLDOWN_S = 60.0


def _zz_api():
    global _zz_cooldown_until
    if time.time() < _zz_cooldown_until:
        return None
    try:
        import zzshare
    except Exception:
        return None
    try:
        api = zzshare.pro_api()
    except Exception:
        return None

    def _fast_request(url, params=None, max_retries=0):
        global _zz_cooldown_until
        import requests
        res = requests.get(url, params=params, headers=api.headers, timeout=api.timeout)
        if res.status_code == 429:
            _zz_cooldown_until = time.time() + _ZZ_COOLDOWN_S
            logger.info("zzshare 429，冷却 %.0fs", _ZZ_COOLDOWN_S)
        return res

    try:
        api._request_with_retry = _fast_request
    except Exception:
        pass
    return api


def _zz_fetch(symbol: str, timeframe: str, n: int) -> list:
    api = _zz_api()
    if api is None:
        return []
    code = _to_zz_code(symbol)
    rows = []
    if timeframe in ("1d", "1w", "1M"):
        import zzshare
        df = zzshare.daily(ts_code=code)
        if df is None or getattr(df, "empty", True):
            return []
        for _, r in df.iterrows():
            ts = _parse_time_to_ts(str(r["trade_date"]))
            if ts <= 0:
                continue
            rows.append({
                "ts": ts,
                "open": _f(r["open"]),
                "high": _f(r["high"]),
                "low": _f(r["low"]),
                "close": _f(r["close"]),
                "volume": _f(r["vol"]) / 100.0,
            })
        rows.sort(key=lambda x: x["ts"])
        if timeframe == "1d":
            return rows[-n:]
        return _aggregate_daily(rows, timeframe)[-n:]
    if timeframe == "4h":
        return _aggregate_4h(_zz_fetch(symbol, "1h", n * 4 + 8))[-n:]
    freq = _ZZ_FREQ.get(timeframe)
    if not freq:
        return []
    df = api.stk_mins(ts_code=code, freq=freq)
    if df is None or getattr(df, "empty", True):
        return []
    for _, r in df.iterrows():
        ts = _parse_time_to_ts(str(r["trade_time"])[:12])
        if ts <= 0:
            continue
        rows.append({
            "ts": ts,
            "open": _f(r["open"]),
            "high": _f(r["high"]),
            "low": _f(r["low"]),
            "close": _f(r["close"]),
            "volume": _f(r["vol"]) / 100.0,
        })
    rows.sort(key=lambda x: x["ts"])
    return rows[-n:]




_SINA_URL = (
    "https://money.finance.sina.com.cn/quotes_service/api/json_v2.php/"
    "CN_MarketData.getKLineData"
)
_SINA_SCALE = {
    "1m": 5, "5m": 5, "15m": 15, "30m": 30,
    "1h": 60, "1d": 240, "1w": 240, "1M": 240,
}


def _sina_fetch(symbol: str, timeframe: str, n: int) -> list:
    import requests

    if timeframe == "4h":
        return _aggregate_4h(_sina_fetch(symbol, "1h", n * 4 + 8))[-n:]
    scale = _SINA_SCALE.get(timeframe)
    if not scale:
        return []
    code = _to_sina_symbol(symbol)
    params = {
        "symbol": code,
        "scale": scale,
        "ma": "no",
        "datalen": min(max(n + 8, 30), 5000),
    }
    headers = {"User-Agent": _UA, "Referer": "https://finance.sina.com.cn/"}
    resp = requests.get(_SINA_URL, params=params, headers=headers, timeout=_HTTP_TIMEOUT)
    if resp.status_code != 200:
        raise RuntimeError(f"新浪 HTTP {resp.status_code}")
    text = (resp.text or "").strip()
    if not text or text == "null":
        return []
    data = json.loads(text)
    if not isinstance(data, list):
        return []
    rows = []
    for r in data:
        ts = _parse_time_to_ts(r.get("day", ""))
        if ts <= 0:
            continue
        rows.append({
            "ts": ts,
            "open": _f(r.get("open")),
            "high": _f(r.get("high")),
            "low": _f(r.get("low")),
            "close": _f(r.get("close")),
            "volume": _f(r.get("volume")),
        })
    rows.sort(key=lambda x: x["ts"])
    if timeframe in ("1w", "1M"):
        return _aggregate_daily(rows, timeframe)[-n:]
    return rows[-n:]


_SOURCES = (
    ("东财", _em_fetch),
    ("腾讯", _tx_fetch),
    ("zzshare", _zz_fetch),
    ("新浪", _sina_fetch),
)


def fetch_ashare_bars(
    symbol: str, timeframe: str, n: int, drop_forming: bool = True
) -> list:
    """按优先级取数：东方财富 -> 腾讯 -> zzshare -> 新浪。

    任何一源成功（返回非空）即返回；全部失败抛 RuntimeError。
    """
    if timeframe not in SUPPORTED_TIMEFRAMES:
        raise ValueError(
            f"不支持周期 {timeframe}，支持 {list(SUPPORTED_TIMEFRAMES)}"
        )
    # 聚合周期需要更多原始 bar：1 周≈5 根日线，1 月≈21 根日线
    fetch_n = n
    if timeframe == "1w":
        fetch_n = n * 5 + 20
    elif timeframe == "1M":
        fetch_n = n * 21 + 40
    errors = []
    best_name = ""
    best_rows: list = []
    # 逐源尝试；返回根数不够 n 时继续找更多的源，取满足 n 的最小够用者
    for name, fn in _SOURCES:
        try:
            rows = fn(symbol, timeframe, fetch_n)
        except Exception as exc:
            errors.append(f"{name}: {exc}")
            logger.debug("A股取数 %s 失败 %s %s: %s", name, symbol, timeframe, exc)
            continue
        if not rows:
            errors.append(f"{name}: 空")
            continue
        rows = _drop_forming(rows, drop_forming)
        if not rows:
            errors.append(f"{name}: 无已收盘数据")
            continue
        if len(rows) >= n:
            logger.info(
                "A股取数命中 %s: %s %s %d 根(满足 %d)",
                name, symbol, timeframe, len(rows), n,
            )
            return rows[-n:]
        # 不够 n 根：记下当前最多的，继续找更全的源
        if len(rows) > len(best_rows):
            best_name, best_rows = name, rows
        errors.append(f"{name}: 仅 {len(rows)} 根(需 {n})")
    if best_rows:
        logger.info(
            "A股取数部分满足 %s: %s %s %d 根(需 %d)",
            best_name, symbol, timeframe, len(best_rows), n,
        )
        return best_rows[-n:]
    raise RuntimeError(
        f"A股取数全部失败 {symbol} {timeframe}: " + " | ".join(errors)
    )


