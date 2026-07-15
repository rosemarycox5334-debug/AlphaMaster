"""Download Eastmoney A-share history into AlphaMaster Parquet format."""
from __future__ import annotations

import argparse
import os
import threading
import time as time_module
from datetime import datetime, time
from pathlib import Path
from zoneinfo import ZoneInfo

import efinance as ef
import pandas as pd


_EFINANCE_NETWORK_LOCK = threading.Lock()


def _fetch_tdx_history(symbol: str, begin: str, end: str) -> pd.DataFrame:
    """Page through pytdx daily bars (unadjusted fallback, 800 bars/page)."""
    from pytdx.hq import TdxHq_API

    servers = [
        ("115.238.90.165", 7709), ("180.153.18.170", 7709),
        ("119.147.212.81", 7709), ("14.17.75.71", 7709),
        ("59.173.18.77", 7709),
    ]
    market = 1 if symbol.startswith(("6", "5", "9", "11", "13")) else 0
    api = TdxHq_API(heartbeat=True, auto_retry=True)
    connected = False
    errors: list[str] = []
    for host, port in servers:
        try:
            if api.connect(host, port):
                connected = True
                break
        except Exception as exc:
            errors.append(f"{host}: {exc}")
    if not connected:
        raise RuntimeError("通达信服务器均不可用: " + "；".join(errors))

    rows: list[dict] = []
    seen: set[str] = set()
    try:
        for start in range(0, 80_000, 800):
            batch = api.get_security_bars(9, market, symbol, start, 800)
            if not batch:
                break
            added = 0
            for item in batch:
                dt = str(item.get("datetime") or "")[:10]
                if not dt or dt in seen:
                    continue
                seen.add(dt)
                rows.append(dict(item))
                added += 1
            if added == 0 or len(batch) < 800:
                break
    finally:
        api.disconnect()
    if not rows:
        raise RuntimeError(f"通达信未返回 {symbol} 的日线数据")

    frame = pd.DataFrame(rows)
    frame["日期"] = pd.to_datetime(frame["datetime"]).dt.strftime("%Y-%m-%d")
    begin_dt = pd.to_datetime(begin)
    end_dt = pd.to_datetime(end)
    dates = pd.to_datetime(frame["日期"])
    frame = frame.loc[(dates >= begin_dt) & (dates <= end_dt)].copy()
    return pd.DataFrame({
        "股票名称": symbol,
        "股票代码": symbol,
        "日期": frame["日期"],
        "开盘": frame["open"],
        "收盘": frame["close"],
        "最高": frame["high"],
        "最低": frame["low"],
        "成交量": frame["vol"],
    })


def _fetch_history(symbol: str, begin: str, end: str) -> tuple[pd.DataFrame, str, str]:
    """Eastmoney qfq first; pytdx unadjusted data if that endpoint is blocked."""
    import efinance.utils as ef_utils

    attempts: list[tuple[str, bool]] = [("直连", False)]
    if any(os.getenv(name) for name in ("HTTPS_PROXY", "https_proxy", "ALL_PROXY", "all_proxy")):
        attempts.append(("系统代理", True))
    errors: list[str] = []

    # efinance uses one module-global requests Session. Serialize the temporary
    # trust_env switch so concurrent Web requests cannot change it mid-call.
    with _EFINANCE_NETWORK_LOCK:
        original = ef_utils.session.trust_env
        try:
            for label, trust_env in attempts:
                ef_utils.session.trust_env = trust_env
                for retry_no in range(2):
                    try:
                        frame = ef.stock.get_quote_history(
                            symbol, beg=begin, end=end, klt=101, fqt=1
                        )
                        return frame, "eastmoney", "qfq"
                    except Exception as exc:
                        errors.append(f"{label}第{retry_no + 1}次: {exc}")
                        if retry_no == 0:
                            time_module.sleep(0.8)
        finally:
            ef_utils.session.trust_env = original
    try:
        return _fetch_tdx_history(symbol, begin, end), "pytdx", "none"
    except Exception as exc:
        errors.append(f"通达信备用源: {exc}")
    raise RuntimeError("；".join(errors))


def download_a_stock(
    symbol: str,
    *,
    out_dir: str | Path = "data",
    begin: str = "19900101",
    end: str | None = None,
) -> dict:
    """Download one A-share and return metadata for the Web/CLI callers."""
    symbol = symbol.strip()
    if len(symbol) != 6 or not symbol.isdigit():
        raise ValueError("股票代码必须是六位数字，例如 002192")
    end = end or datetime.now().strftime("%Y%m%d")

    raw, source, adjustment = _fetch_history(symbol, begin, end)
    if raw is None or raw.empty:
        raise RuntimeError(f"行情源未返回 {symbol} 的日线数据")

    required = ["日期", "开盘", "最高", "最低", "收盘", "成交量"]
    missing = [name for name in required if name not in raw.columns]
    if missing:
        raise RuntimeError(f"行情源返回字段缺失: {missing}")

    dates = pd.to_datetime(raw["日期"], errors="coerce")
    shanghai_now = datetime.now(ZoneInfo("Asia/Shanghai"))
    if shanghai_now.time() < time(15, 5):
        # Eastmoney may expose today's still-forming daily bar before close.
        raw = raw.loc[dates.dt.date < shanghai_now.date()].copy()
        dates = pd.to_datetime(raw["日期"], errors="coerce")
    if raw.empty:
        raise RuntimeError(f"行情源未返回 {symbol} 的已收盘日线数据")
    unix_seconds = (dates - pd.Timestamp("1970-01-01")) // pd.Timedelta(seconds=1)
    frame = pd.DataFrame(
        {
            "time": unix_seconds,
            "open": pd.to_numeric(raw["开盘"], errors="coerce"),
            "high": pd.to_numeric(raw["最高"], errors="coerce"),
            "low": pd.to_numeric(raw["最低"], errors="coerce"),
            "close": pd.to_numeric(raw["收盘"], errors="coerce"),
            "tick_volume": pd.to_numeric(raw["成交量"], errors="coerce"),
        }
    )
    frame = frame.dropna().drop_duplicates("time", keep="last").sort_values("time")
    frame = frame.astype(
        {
            "time": "int64",
            "open": "float32",
            "high": "float32",
            "low": "float32",
            "close": "float32",
            "tick_volume": "int64",
        }
    ).reset_index(drop=True)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    output = out_dir / f"{symbol}_D1.parquet"
    temp_output = output.with_suffix(".parquet.tmp")
    frame.to_parquet(temp_output, index=False)
    temp_output.replace(output)
    first = pd.to_datetime(frame["time"].iloc[0], unit="s").date()
    last = pd.to_datetime(frame["time"].iloc[-1], unit="s").date()
    name = str(raw["股票名称"].iloc[-1]) if "股票名称" in raw.columns else symbol
    return {
        "symbol": symbol,
        "name": name,
        "bars": len(frame),
        "date_start": str(first),
        "date_end": str(last),
        "data_file": str(output.resolve()),
        "source": source,
        "price_adjustment": adjustment,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="下载东方财富 A 股前复权历史行情")
    parser.add_argument("symbol", help="六位 A 股代码，例如 002192")
    parser.add_argument("--out", default="data", help="输出目录，默认 data")
    parser.add_argument("--begin", default="19900101", help="开始日期 YYYYMMDD")
    parser.add_argument("--end", default=datetime.now().strftime("%Y%m%d"), help="结束日期 YYYYMMDD")
    args = parser.parse_args()
    try:
        result = download_a_stock(
            args.symbol, out_dir=args.out, begin=args.begin, end=args.end
        )
    except (ValueError, RuntimeError) as exc:
        raise SystemExit(str(exc)) from exc

    name = result["name"]
    symbol = result["symbol"]
    first = result["date_start"]
    last = result["date_end"]
    output = result["data_file"]
    frame_len = result["bars"]
    source = result["source"]
    adjustment = result["price_adjustment"]
    print(
        f"已保存 {name}({symbol}): {frame_len:,} 根日线，{first} ~ {last} "
        f"source={source} adjustment={adjustment}"
    )
    print(output)


if __name__ == "__main__":
    main()
