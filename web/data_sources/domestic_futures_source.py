"""国内期货数据源（tqsdk 天勤量化，实时行情）。

通过 tqsdk 实时拉取国内期货主连 K 线，支持 60 个品种 × 5 个周期。
账号优先从 web_settings.json 读取，未配置时使用内置默认账号。

特点：
  - 实时拉取最新 K 线（不是本地文件）
  - 前复权 adj_type="F"（自动消除换月跳价）
  - 主连代码 KQ.m@{交易所}.{品种}（tqsdk 自带主力切换）
  - 每次 fetch_bars 建连接 → 拉数据 → 关连接（1-3 秒）
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path

from web.data_sources.base import Bar, DataSource, DataSourceUnavailable

# 周期映射：CANON_TIMEFRAMES → tqsdk duration_seconds
_TF_TO_SECONDS = {
    "5m": 300,
    "15m": 900,
    "30m": 1800,
    "1h": 3600,
    "1d": 86400,
}

# 中文品种 → tqsdk 主连代码（大小写敏感！SHFE/DCE/GFEX 小写，CZCE/CFFEX 大写）
_SYMBOL_MAP = {
    # 上期所 SHFE
    "沪铜主连": "KQ.m@SHFE.cu", "沪铝主连": "KQ.m@SHFE.al", "沪锌主连": "KQ.m@SHFE.zn",
    "沪铅主连": "KQ.m@SHFE.pb", "沪镍主连": "KQ.m@SHFE.ni", "沪锡主连": "KQ.m@SHFE.sn",
    "沪金主连": "KQ.m@SHFE.au", "沪银主连": "KQ.m@SHFE.ag", "螺纹钢主连": "KQ.m@SHFE.rb",
    "热卷主连": "KQ.m@SHFE.hc", "橡胶主连": "KQ.m@SHFE.ru", "燃油主连": "KQ.m@SHFE.fu",
    "沥青主连": "KQ.m@SHFE.bu", "纸浆主连": "KQ.m@SHFE.sp", "不锈钢主连": "KQ.m@SHFE.ss",
    # 大商所 DCE
    "铁矿石主连": "KQ.m@DCE.i", "焦炭主连": "KQ.m@DCE.j", "焦煤主连": "KQ.m@DCE.jm",
    "豆粕主连": "KQ.m@DCE.m", "豆油主连": "KQ.m@DCE.y", "棕榈油主连": "KQ.m@DCE.p",
    "玉米主连": "KQ.m@DCE.c", "淀粉主连": "KQ.m@DCE.cs", "豆一主连": "KQ.m@DCE.a",
    "豆二主连": "KQ.m@DCE.b", "塑料主连": "KQ.m@DCE.l", "PVC主连": "KQ.m@DCE.v",
    "聚丙烯主连": "KQ.m@DCE.pp", "LPG主连": "KQ.m@DCE.pg", "乙二醇主连": "KQ.m@DCE.eg",
    "苯乙烯主连": "KQ.m@DCE.eb",
    # 郑商所 CZCE
    "白糖主连": "KQ.m@CZCE.SR", "棉花主连": "KQ.m@CZCE.CF", "PTA主连": "KQ.m@CZCE.TA",
    "甲醇主连": "KQ.m@CZCE.MA", "菜油主连": "KQ.m@CZCE.OI", "菜粕主连": "KQ.m@CZCE.RM",
    "玻璃主连": "KQ.m@CZCE.FG", "纯碱主连": "KQ.m@CZCE.SA", "硅铁主连": "KQ.m@CZCE.SF",
    "锰硅主连": "KQ.m@CZCE.SM", "苹果主连": "KQ.m@CZCE.AP", "红枣主连": "KQ.m@CZCE.CJ",
    "尿素主连": "KQ.m@CZCE.UR", "烧碱主连": "KQ.m@CZCE.SH",
    # 中金所 CFFEX
    "沪深300主连": "KQ.m@CFFEX.IF", "中证500主连": "KQ.m@CFFEX.IC",
    "上证50主连": "KQ.m@CFFEX.IH", "中证1000主连": "KQ.m@CFFEX.IM",
    "10年国债主连": "KQ.m@CFFEX.T", "5年国债主连": "KQ.m@CFFEX.TF",
    "2年国债主连": "KQ.m@CFFEX.TS", "30年国债主连": "KQ.m@CFFEX.TL",
    # 上期所能源 INE + 广期所 GFEX
    "原油主连": "KQ.m@INE.sc", "低硫燃油主连": "KQ.m@INE.lu",
    "20号胶主连": "KQ.m@INE.nr", "国际铜主连": "KQ.m@INE.bc",
    "欧线集运主连": "KQ.m@INE.ec",
    "工业硅主连": "KQ.m@GFEX.si", "碳酸锂主连": "KQ.m@GFEX.lc",
}

_PRESETS = list(_SYMBOL_MAP.keys())

# 内置默认 tqsdk 账号（web_settings.json 未配置时使用）
_DEFAULT_TQSDK_USER = "七斗居士"
_DEFAULT_TQSDK_PASSWORD = "ghhkphs8"

# web_settings.json 路径
_SETTINGS_PATH = Path(__file__).resolve().parents[2] / "web_settings.json"


def _load_credentials() -> tuple[str, str]:
    """读取 tqsdk 账号密码。优先 web_settings.json，未配置则用内置默认账号。"""
    try:
        data = json.loads(_SETTINGS_PATH.read_text(encoding="utf-8"))
        user = str(data.get("tqsdk_user", "")).strip()
        pwd = str(data.get("tqsdk_password", "")).strip()
        if user and pwd:
            return user, pwd
    except Exception:
        pass
    return _DEFAULT_TQSDK_USER, _DEFAULT_TQSDK_PASSWORD


class DomesticFuturesSource(DataSource):
    """国内期货实时数据源（tqsdk 天勤量化）。

    每次 fetch_bars 建立临时 tqsdk 连接，拉取最新 N 根 K 线后关闭。
    realtime_manager 的轮询间隔 45-300 秒，1-3 秒的连接开销可接受。
    """

    kind = "domestic_futures"
    label = "国内期货"

    def __init__(self) -> None:
        self._lock = threading.Lock()

    def available(self) -> tuple[bool, str]:
        try:
            from tqsdk import TqApi  # noqa: F401
        except ImportError:
            return (False, "未安装 tqsdk：pip install tqsdk")
        user, pwd = _load_credentials()
        if not user or not pwd:
            return (False, "tqsdk 账号配置异常")
        return (
            True,
            f"天勤量化实时行情 · 60 品种 · 5 周期（5m/15m/30m/1h/1d）· 前复权",
        )

    def supported_timeframes(self) -> list[str]:
        return list(_TF_TO_SECONDS.keys())

    def preset_symbols(self) -> list[str]:
        return list(_PRESETS)

    def fetch_bars(
        self, symbol: str, timeframe: str, n: int, drop_forming: bool = True
    ) -> list[Bar]:
        """从 tqsdk 实时拉取最近 n 根 K 线。

        每次调用建立临时连接，拉完即关（1-3 秒）。
        drop_forming=True 时剔除当前正在形成的 bar（ts > now）。
        """
        if timeframe not in _TF_TO_SECONDS:
            raise DataSourceUnavailable(
                f"国内期货不支持周期 {timeframe}，仅支持 {list(_TF_TO_SECONDS.keys())}"
            )

        # 品种名 → tqsdk 主连代码
        tqsdk_symbol = _SYMBOL_MAP.get(symbol.strip())
        if not tqsdk_symbol:
            raise DataSourceUnavailable(
                f"未知品种 {symbol}。支持 {len(_PRESETS)} 个品种，如：螺纹钢主连、沪铜主连、沪深300主连"
            )

        user, pwd = _load_credentials()
        if not user or not pwd:
            raise DataSourceUnavailable("tqsdk 账号配置异常")

        duration = _TF_TO_SECONDS[timeframe]
        want = max(n + 5, 20)  # 多拉几根防止 drop_forming 后不足

        from tqsdk import TqApi, TqAuth, TqSim

        with self._lock:
            api = None
            try:
                api = TqApi(
                    TqSim(),
                    auth=TqAuth(user, pwd),
                    disable_print=True,
                )
                # adj_type="F" 前复权，自动消除换月跳价
                df = api.get_kline_serial(
                    tqsdk_symbol, duration, data_length=want, adj_type="F"
                )
            except Exception as exc:
                if api is not None:
                    try:
                        api.close()
                    except Exception:
                        pass
                raise DataSourceUnavailable(
                    f"tqsdk 连接失败：{exc}。请检查账号密码或网络"
                ) from exc
            finally:
                if api is not None:
                    try:
                        api.close()
                    except Exception:
                        pass

        if df is None or df.empty:
            raise DataSourceUnavailable(f"tqsdk 无数据：{symbol} {timeframe}")

        # tqsdk datetime 是纳秒级 Unix 时间戳，转秒
        bars: list[Bar] = []
        for _, row in df.iterrows():
            ts_ns = row["datetime"]
            if ts_ns == 0 or ts_ns != ts_ns:  # 跳过空值/NaN
                continue
            ts = int(ts_ns // 1_000_000_000)
            vol = float(row.get("volume", 0) or 0)
            if vol <= 0:
                continue  # 过滤非交易时段（volume=0）
            bars.append(
                Bar(
                    ts=ts,
                    open=float(row["open"]),
                    high=float(row["high"]),
                    low=float(row["low"]),
                    close=float(row["close"]),
                    volume=vol,
                )
            )

        bars.sort(key=lambda b: b.ts)  # 升序（最旧在前）

        # 剔除尚未收盘的 bar（ts > now）
        if drop_forming and bars:
            now = time.time()
            while bars and int(bars[-1].ts) > now:
                bars.pop()

        if not bars:
            raise DataSourceUnavailable(
                f"{symbol} {timeframe} 无已收盘 K 线（可能非交易时段）"
            )

        return bars[-n:] if n > 0 else bars
