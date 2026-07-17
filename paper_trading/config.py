"""paper_trading/config.py — A股交易规则常量（从根 Config 读取）。"""
from config import Config

INITIAL_CAPITAL = Config.ASHARE_INITIAL_CAPITAL
TOP_K           = Config.ASHARE_TOP_K
COMMISSION_RATE = Config.ASHARE_COMMISSION_RATE
MIN_COMMISSION  = Config.ASHARE_MIN_COMMISSION
STAMP_TAX       = Config.ASHARE_STAMP_TAX
LIMIT_PCT       = Config.ASHARE_LIMIT_PCT
LOT_SIZE        = Config.ASHARE_LOT_SIZE
