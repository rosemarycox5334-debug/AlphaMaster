# -*- coding: utf-8 -*-
"""验证 tqsdk 账户能否登录并获取行情数据"""
from tqsdk import TqApi, TqAuth
import sys, time

USER = "七斗居士"
PASSWORD = "ghhkphs8"

print(f"正在登录 tqsdk: {USER}")
try:
    api = TqApi(auth=TqAuth(USER, PASSWORD))
    print("[OK] 登录成功!")
    
    # 获取行情
    quote = api.get_quote("SHFE.cu2509")
    # 等待数据到达
    while quote.datetime == "":
        api.wait_update()
    
    print(f"合约代码: {quote.instrument_id}")
    print(f"最新价: {quote.last_price}")
    print(f"买一价: {quote.bid_price1}")
    print(f"卖一价: {quote.ask_price1}")
    print(f"开盘价: {quote.open}")
    print(f"最高价: {quote.high}")
    print(f"最低价: {quote.low}")
    print(f"更新时间: {quote.datetime}")
    print("\n[OK] 账户有效, 行情数据获取成功!")
    
    api.close()
except Exception as e:
    print(f"[FAIL] {type(e).__name__}: {e}")
    sys.exit(1)
