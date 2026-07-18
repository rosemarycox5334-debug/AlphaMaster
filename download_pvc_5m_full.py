"""修复版：下载 PVC 全历史 5分钟 K 线数据（akshare 多合约拼接）。

修复 pandas 3.0 时间戳转换问题，使用显式时区处理。
"""
import akshare as ak
import pandas as pd
from pathlib import Path
from datetime import timezone, timedelta
import time
import sys

print("=" * 60)
print("  PVC 全历史 5分钟 K 线下载（akshare 多合约拼接·修复版）")
print("=" * 60)

# 北京时间 UTC+8
BJ_TZ = timezone(timedelta(hours=8))

# 生成所有 PVC 合约代码: V2101 ~ V2612
contracts = []
for year in range(21, 27):
    for month in range(1, 13):
        contracts.append(f"V{year:02d}{month:02d}")

print(f"待下载合约数: {len(contracts)} 个 (V2101 ~ V2612)")
print()

# 下载所有合约
all_data = []
success = 0
for i, code in enumerate(contracts):
    try:
        df = ak.futures_zh_minute_sina(symbol=code, period="5")
        if df is not None and len(df) > 0:
            df = df.copy()
            df["_contract"] = code
            all_data.append(df)
            success += 1
            if (i + 1) % 12 == 0 or i == len(contracts) - 1:
                print(f"  进度: {i+1}/{len(contracts)} | 成功: {success}")
    except Exception:
        pass
    time.sleep(0.15)

print(f"\n下载完成: 成功 {success} 个合约, 总行数 {sum(len(d) for d in all_data)}")

# 合并
print("\n正在拼接主力合约序列...")
combined = pd.concat(all_data, ignore_index=True)

# 修复时间戳转换：显式按北京时间处理
def to_unix_seconds(dt_str):
    """将 '2020-09-17 21:35:00' 转为 Unix 秒（按北京时间 UTC+8）。"""
    try:
        dt = pd.to_datetime(dt_str)
        # 显式设置时区为北京时间，再转 Unix 时间戳
        return int(dt.replace(tzinfo=BJ_TZ).timestamp())
    except Exception:
        return 0

combined["_ts"] = combined["datetime"].apply(to_unix_seconds)
combined = combined[combined["_ts"] > 0].copy()  # 过滤无效

# 按时间戳+成交量排序，去重时保留成交量最大的（近似主力合约）
print("按成交量选择主力合约（去重）...")
combined = combined.sort_values(["_ts", "volume"], ascending=[True, False])
combined = combined.drop_duplicates(subset=["_ts"], keep="first")
combined = combined.sort_values("_ts").reset_index(drop=True)

print(f"去重后行数: {len(combined):,}")

# 验证时间戳
first_dt = pd.to_datetime(combined["datetime"].iloc[0])
last_dt = pd.to_datetime(combined["datetime"].iloc[-1])
print(f"时间范围: {first_dt} ~ {last_dt}")
span_days = (last_dt - first_dt).days
print(f"时间跨度: {span_days} 天（约 {span_days/365.25:.1f} 年）")

# 验证时间戳正确性
print(f"首行 Unix 时间戳: {combined['_ts'].iloc[0]}")
print(f"首行时间戳验证: {pd.to_datetime(combined['_ts'].iloc[0], unit='s', utc=True).tz_convert(BJ_TZ)}")

# 转换为项目兼容格式
result = pd.DataFrame()
result["time"] = combined["_ts"].astype("int64")
result["open"] = combined["open"].astype("float64")
result["high"] = combined["high"].astype("float64")
result["low"] = combined["low"].astype("float64")
result["close"] = combined["close"].astype("float64")
result["tick_volume"] = combined["volume"].astype("int64")
result = result.drop_duplicates(subset=["time"], keep="last").reset_index(drop=True)

# 保存
out_dir = Path(r"C:\Users\14795\Desktop\国内期货K线数据")
out_dir.mkdir(parents=True, exist_ok=True)
out_file = out_dir / "PVC_5min.parquet"
result.to_parquet(out_file, index=False)

# 统计
print(f"\n{'=' * 60}")
print(f"  下载完成!")
print(f"{'=' * 60}")
print(f"保存路径: {out_file}")
print(f"总行数: {len(result):,}")
print(f"时间范围: {first_dt} ~ {last_dt}")
print(f"时间跨度: {span_days} 天（约 {span_days/365.25:.1f} 年）")

print(f"\n价格统计:")
print(f"  开盘价: {result['open'].min():.0f} ~ {result['open'].max():.0f}")
print(f"  收盘价: {result['close'].min():.0f} ~ {result['close'].max():.0f}")
print(f"  均价: {result['close'].mean():.0f}")

# 按年统计
result["_dt"] = pd.to_datetime(result["time"], unit="s", utc=True).dt.tz_convert(BJ_TZ)
result["_year"] = result["_dt"].dt.year
yearly = result.groupby("_year").size()
print(f"\n各年数据量:")
for yr, cnt in yearly.items():
    print(f"  {yr}年: {cnt:,} 根")

print(f"\n前 3 行:")
print(result[["time","open","high","low","close","tick_volume"]].head(3).to_string())
print(f"\n后 3 行:")
print(result[["time","open","high","low","close","tick_volume"]].tail(3).to_string())
