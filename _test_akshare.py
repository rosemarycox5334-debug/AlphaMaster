"""测试 akshare 各函数能获取多少 PVC 5分钟历史数据。"""
import akshare as ak
import pandas as pd
from datetime import datetime

print(f"akshare version: {ak.__version__}")
print()

# 方法1: futures_zh_minute_sina - 新浪连续合约
print("=== 方法1: futures_zh_minute_sina(V0, 5min) ===")
try:
    df1 = ak.futures_zh_minute_sina(symbol="V0", period="5")
    print(f"行数: {len(df1)}")
    print(f"列: {list(df1.columns)}")
    if len(df1) > 0:
        print(f"时间范围: {df1.iloc[0,0]} ~ {df1.iloc[-1,0]}")
        print(df1.head(2))
        print(df1.tail(2))
except Exception as e:
    print(f"ERROR: {e}")

print()

# 方法2: 尝试下载多个具体合约拼接
print("=== 方法2: 下载多个 PVC 合约 ===")
all_dfs = []
# PVC合约代码: V+YYMM, 如 V2401=2024年1月
contracts = []
for year in [23, 24, 25, 26]:
    for month in range(1, 13):
        code = f"V{year:02d}{month:02d}"
        contracts.append(code)

for code in contracts:
    try:
        df = ak.futures_zh_minute_sina(symbol=code, period="5")
        if df is not None and len(df) > 0:
            print(f"  {code}: {len(df)} 行, {df.iloc[0,0]} ~ {df.iloc[-1,0]}")
            all_dfs.append(df)
    except Exception as e:
        pass  # 合约可能已过期，跳过

print(f"\n总计获取 {len(all_dfs)} 个合约, 合计 {sum(len(d) for d in all_dfs)} 行")
