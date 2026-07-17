import pandas as pd
import pytest
from data_pipeline.ashare_fetcher import AShareFetcher, _normalize_daily_df


def test_normalize_daily_df_columns():
    # akshare 返回中文列名，需归一化为标准英文列
    raw = pd.DataFrame({
        "日期": ["2023-01-03", "2023-01-04"],
        "开盘": [10.0, 10.5], "最高": [10.8, 10.9],
        "最低": [9.9, 10.4], "收盘": [10.5, 10.6],
        "成交量": [1000, 1200],
    })
    out = _normalize_daily_df(raw)
    assert list(out.columns) == ["date", "open", "high", "low", "close", "volume"]
    assert len(out) == 2
    assert out.iloc[0]["close"] == 10.5


def test_daily_uses_cache(tmp_path, monkeypatch):
    # 第二次调用应命中缓存，不再访问网络。
    # 主源是新浪 stock_zh_a_daily（英文列名），故 mock 它。
    f = AShareFetcher(cache_dir=str(tmp_path))
    calls = {"n": 0}

    def fake_daily(*args, **kwargs):
        calls["n"] += 1
        return pd.DataFrame({
            "date": pd.to_datetime(["2023-01-03", "2023-01-04"]),
            "open": [10.0, 10.5], "high": [10.8, 10.9],
            "low": [9.9, 10.4], "close": [10.5, 10.6],
            "volume": [1000, 1200],
        })

    monkeypatch.setattr("data_pipeline.ashare_fetcher.ak.stock_zh_a_daily", fake_daily)
    df1 = f.daily("600000", "2023-01-01", "2023-01-04")
    df2 = f.daily("600000", "2023-01-01", "2023-01-04")
    assert calls["n"] == 1               # 第二次命中缓存
    assert len(df1) == len(df2) == 2
    assert list(df1.columns) == ["date", "open", "high", "low", "close", "volume"]


def test_sina_symbol_prefix():
    from data_pipeline.ashare_fetcher import _sina_symbol
    assert _sina_symbol("600000") == "sh600000"   # 沪市主板
    assert _sina_symbol("000001") == "sz000001"   # 深市主板
    assert _sina_symbol("300750") == "sz300750"   # 创业板
    assert _sina_symbol("688981") == "sh688981"   # 科创板
    assert _sina_symbol("830799") == "bj830799"   # 北交所
