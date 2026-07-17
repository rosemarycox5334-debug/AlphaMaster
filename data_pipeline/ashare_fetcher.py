"""
data_pipeline/ashare_fetcher.py — A股数据拉取与本地缓存

用 akshare 拉取沪深300+中证500成份股、交易日历、日线行情，
单只股票以 parquet 缓存到 ASHARE_CACHE_DIR，增量更新。
所有网络调用带重试；缓存优先。
"""
from __future__ import annotations

import time
from pathlib import Path

import akshare as ak
import pandas as pd
from loguru import logger

from config import Config

_COL_MAP = {
    "日期": "date", "开盘": "open", "最高": "high",
    "最低": "low", "收盘": "close", "成交量": "volume",
}


def _normalize_daily_df(raw: pd.DataFrame) -> pd.DataFrame:
    """akshare 列名归一化 → 标准英文列，只保留 OHLCV+date。

    兼容两种源：东方财富（中文列名，经 _COL_MAP）与新浪（已是英文列名，
    rename 为空操作）。
    """
    df = raw.rename(columns=_COL_MAP)
    keep = ["date", "open", "high", "low", "close", "volume"]
    df = df[[c for c in keep if c in df.columns]].copy()
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.reset_index(drop=True)


def _sina_symbol(code: str) -> str:
    """6 位代码 → 新浪源符号（带交易所前缀）。

    6xxxxx/9xxxxx → sh（沪市主板/科创板/B股）；0xxxxx/2xxxxx/3xxxxx → sz
    （深市主板/B股/创业板）；4xxxxx/8xxxxx → bj（北交所）。
    """
    c = str(code).zfill(6)
    head = c[0]
    if head in ("6", "9"):
        return f"sh{c}"
    if head in ("4", "8"):
        return f"bj{c}"
    return f"sz{c}"


def _retry(fn, *args, tries: int = 3, delay: float = 1.0, **kwargs):
    """带指数退避的重试包装，全部失败则抛最后一次异常。"""
    last = None
    for i in range(tries):
        try:
            return fn(*args, **kwargs)
        except Exception as e:  # noqa: BLE001 — akshare 抛各种网络异常
            last = e
            logger.warning(f"akshare 调用失败({i+1}/{tries}): {e}")
            time.sleep(delay * (2 ** i))
    raise last


class AShareFetcher:
    def __init__(self, cache_dir: str | None = None) -> None:
        self.cache_dir = Path(cache_dir or Config.ASHARE_CACHE_DIR)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def universe_codes(self) -> list[str]:
        """沪深300 + 中证500 当前成份股代码（6位数字，去重排序）。"""
        hs300 = _retry(ak.index_stock_cons, symbol="000300")
        zz500 = _retry(ak.index_stock_cons, symbol="000905")
        codes = set()
        for df in (hs300, zz500):
            col = "品种代码" if "品种代码" in df.columns else df.columns[0]
            codes.update(str(c).zfill(6) for c in df[col].tolist())
        return sorted(codes)

    def trade_calendar(self, start: str, end: str) -> list[str]:
        """交易日历（YYYY-MM-DD），区间 [start, end]。"""
        cal = _retry(ak.tool_trade_date_hist_sina)
        dates = pd.to_datetime(cal["trade_date"]).dt.strftime("%Y-%m-%d")
        mask = (dates >= start) & (dates <= end)
        return dates[mask].tolist()

    def daily(self, code: str, start: str, end: str) -> pd.DataFrame:
        """单只日线，parquet 缓存增量更新。列 [date,open,high,low,close,volume]。"""
        cache_path = self.cache_dir / f"{code}.parquet"
        cached = None
        if cache_path.exists():
            cached = pd.read_parquet(cache_path)
            if not cached.empty and cached["date"].max() >= end:
                sub = cached[(cached["date"] >= start) & (cached["date"] <= end)]
                if not sub.empty:
                    return sub.reset_index(drop=True)
        raw = self._fetch_daily_raw(code, start, end)
        if raw is None or len(raw) == 0:
            return cached if cached is not None else pd.DataFrame(
                columns=["date", "open", "high", "low", "close", "volume"])
        fresh = _normalize_daily_df(raw)
        merged = fresh if cached is None else pd.concat([cached, fresh])
        merged = merged.drop_duplicates("date", keep="last").sort_values("date")
        merged = merged.reset_index(drop=True)
        merged.to_parquet(cache_path, index=False)
        sub = merged[(merged["date"] >= start) & (merged["date"] <= end)]
        return sub.reset_index(drop=True)

    def _fetch_daily_raw(self, code: str, start: str, end: str) -> pd.DataFrame | None:
        """拉单只日线原始数据：优先新浪源（本网稳定），失败回退东方财富源。

        新浪源 stock_zh_a_daily 返回英文列名、需交易所前缀符号；
        东方财富源 stock_zh_a_hist 返回中文列名、用纯数字代码。
        两者都经 _normalize_daily_df 归一化后由调用方统一处理。
        """
        s, e = start.replace("-", ""), end.replace("-", "")
        # 主源：新浪（sh/sz/bj 前缀）
        try:
            return _retry(ak.stock_zh_a_daily, symbol=_sina_symbol(code),
                          start_date=s, end_date=e, adjust="qfq", tries=2)
        except Exception as sina_err:  # noqa: BLE001
            logger.warning(f"新浪源日线失败({code})，回退东财源: {sina_err}")
        # 回退源：东方财富（纯数字代码）
        try:
            return _retry(ak.stock_zh_a_hist, symbol=str(code).zfill(6),
                          period="daily", start_date=s, end_date=e,
                          adjust="qfq", tries=2)
        except Exception as em_err:  # noqa: BLE001
            logger.error(f"东财源日线也失败({code}): {em_err}")
            return None

    def benchmark(self, start: str, end: str, symbol: str = "sh000300") -> dict[str, float]:
        """沪深300指数日线收盘，返回 {date: close}，用于超额收益对比。"""
        raw = _retry(ak.stock_zh_index_daily, symbol=symbol)
        df = raw.copy()
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
        df = df[(df["date"] >= start) & (df["date"] <= end)]
        return dict(zip(df["date"], df["close"].astype(float)))
