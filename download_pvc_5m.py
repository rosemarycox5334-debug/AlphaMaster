"""下载 PVC 5分钟 K 线数据到桌面。

使用 tqsdk 拉取 PVC主连(KQ.m@DCE.v) 的 5 分钟 K 线，
保存为项目兼容的 Parquet 格式（time/open/high/low/close/tick_volume）。
"""
import json
import sys
from pathlib import Path

# 读取 tqsdk 凭证（优先 web_settings.json，未配置则用默认账号）
settings_path = Path(__file__).parent / "web_settings.json"
try:
    settings = json.loads(settings_path.read_text(encoding="utf-8"))
    user = str(settings.get("tqsdk_user", "")).strip()
    pwd = str(settings.get("tqsdk_password", "")).strip()
except Exception:
    user, pwd = "", ""

# 内置默认账号
if not user or not pwd:
    user, pwd = "七斗居士", "ghhkphs8"

print(f"tqsdk 账号: {user}")
print("正在连接天勤量化...")

from tqsdk import TqApi, TqAuth, TqSim
import pandas as pd

# PVC主连 → tqsdk 主连代码
TQSDK_SYMBOL = "KQ.m@DCE.v"
DURATION = 300  # 5分钟 = 300秒
DATA_LENGTH = 8000  # 尽量多拉

api = None
try:
    api = TqApi(TqSim(), auth=TqAuth(user, pwd), disable_print=True)
    print("连接成功，正在拉取 PVC 5分钟 K 线...")

    # adj_type="F" 前复权，消除换月跳价
    df = api.get_kline_serial(
        TQSDK_SYMBOL, DURATION, data_length=DATA_LENGTH, adj_type="F"
    )
    print(f"拉取到 {len(df)} 根 K 线")
finally:
    if api is not None:
        try:
            api.close()
        except Exception:
            pass

if df is None or df.empty:
    print("ERROR: 未拉取到数据")
    sys.exit(1)

# 过滤无效行（volume=0 的非交易时段，或 datetime=0）
df = df[df["datetime"] != 0].copy()
df = df[df["volume"] > 0].copy()
print(f"过滤后有效 K 线: {len(df)} 根")

# 转换为项目兼容格式
# tqsdk datetime 是纳秒级 Unix 时间戳 → 转秒
result = pd.DataFrame()
result["time"] = (df["datetime"].astype("int64") // 1_000_000_000).astype("int64")
result["open"] = df["open"].astype("float64")
result["high"] = df["high"].astype("float64")
result["low"] = df["low"].astype("float64")
result["close"] = df["close"].astype("float64")
result["tick_volume"] = df["volume"].astype("int64")

# 按时间升序排列
result = result.sort_values("time").reset_index(drop=True)

# 去除可能重复的时间戳
result = result.drop_duplicates(subset=["time"], keep="last").reset_index(drop=True)

# 输出目录
out_dir = Path(r"C:\Users\14795\Desktop\国内期货K线数据")
out_dir.mkdir(parents=True, exist_ok=True)
out_file = out_dir / "PVC_5min.parquet"

# 保存为 Parquet
result.to_parquet(out_file, index=False)
print(f"\n已保存: {out_file}")
print(f"总行数: {len(result)}")

# 显示数据概览
print(f"\n=== 数据概览 ===")
print(f"时间范围: {pd.to_datetime(result['time'].iloc[0], unit='s')} ~ {pd.to_datetime(result['time'].iloc[-1], unit='s')}")
print(f"开盘价范围: {result['open'].min():.2f} ~ {result['open'].max():.2f}")
print(f"收盘价范围: {result['close'].min():.2f} ~ {result['close'].max():.2f}")
print(f"\n前 5 行:")
print(result.head().to_string())
print(f"\n后 5 行:")
print(result.tail().to_string())
print("\n下载完成!")
