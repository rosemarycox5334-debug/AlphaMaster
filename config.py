"""
config.py — 统一配置模块（项目根目录）

所有子模块从此文件导入 Config，废弃各自的 config.py。
MT5 连接凭证通过环境变量或 .env 文件加载。
"""
import os

try:
    import MetaTrader5 as mt5
    _MT5_AVAILABLE = True
except ImportError:
    _MT5_AVAILABLE = False
    # 测试环境无 MT5 时使用整数占位常量（与真实 MT5 值一致）
    class _MT5Stub:
        TIMEFRAME_M1  = 1
        TIMEFRAME_M5  = 5
        TIMEFRAME_M15 = 15
        TIMEFRAME_M30 = 30
        TIMEFRAME_H1  = 16385
        TIMEFRAME_H4  = 16388
        TIMEFRAME_D1  = 16408
        TIMEFRAME_W1  = 32769
        TIMEFRAME_MN1 = 49153
    mt5 = _MT5Stub()

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

from dotenv import load_dotenv

load_dotenv()


class Config:
    # ── MT5 连接 ──────────────────────────────────────────
    MT5_LOGIN    = int(os.getenv("MT5_LOGIN", "0"))
    MT5_PASSWORD = os.getenv("MT5_PASSWORD", "")
    MT5_SERVER   = os.getenv("MT5_SERVER", "")

    # ── 品种与周期 ────────────────────────────────────────
    # TRADE_SYMBOLS：实际交易的品种（新账号，无 m 后缀）
    SYMBOLS   = [
        # 外汇
        "EURUSD", "USDJPY",
        # 贵金属
        "XAUUSD", "XAGUSD",
        # AAVUSD 已移除：实为 Aave 加密货币，非大宗商品。波动特征与贵金属完全不匹配。
        # COCOA.c 已移除：大宗商品期货，日交易~10h，时间对齐后仅8546 bar，
        # 拖累整组数据量，且流动性/交易时段与贵金属不匹配。
        # 美国指数
        "US30.cash", "US100.cash", "US500.cash", "US2000.cash",
        # 其他指数
        "JP225.cash",
    ]

    # ── 训练品种（单品种模式）──────────────────────────────
    # 2026-07-07 从分组模式切换到单品种模式。原因：
    #   1. 截面信息没用上：precious_metals/index 跑出的4个最优公式，没有一个用了 CS 算子
    #   2. 跨品种干扰严重：XAUUSD/XAGUSD 同组时，白银 Kyle Lambda 波动是黄金4倍，
    #      模型被白银主导，黄金信号被淹没
    #   3. 指数组无效探索：5个美股指数相关性>0.85，截面空间狭窄，19次重启后仍是beta
    #   4. 单品种策略更纯粹：因子只针对一种资产特征，实盘也更容易管理
    # 每个品种独立训练，checkpoint 按 ckpt_{symbol}_step_{N}.pt 保存。
    TRAINABLE_SYMBOLS = [
        # 外汇（各8年数据，24h连续交易）
        "EURUSD",
        "USDJPY",
        # 贵金属（8年数据，24h连续交易）
        "XAUUSD",
        # XAGUSD 已移除：白银与黄金高度相关，单品种训练收益有限，
        # 且黄金已有验证策略(Sharpe 2.66)，优先覆盖未挖掘品种
        # 美国指数（各5年数据）
        "US30.cash",
        "US100.cash",
        "US500.cash",
        "US2000.cash",
        # 日本指数
        "JP225.cash",
    ]

    # [deprecated] 相关性分组（已废弃，改用 TRAINABLE_SYMBOLS 单品种训练）
    # 保留供回测参考，新训练不再使用
    SYMBOL_GROUPS = {
        "forex":          ["EURUSD", "USDJPY"],
        "precious_metals":["XAUUSD", "XAGUSD"],
        "index":          ["US30.cash", "US100.cash", "US500.cash", "US2000.cash", "JP225.cash"],
    }

    # FEATURE_SYMBOLS：用于计算截面特征的宽品种集
    # 包含主要外汇、贵金属、大宗商品、主流指数，时间与 SYMBOLS 高度对齐
    # REL_RET5/REL_RET20/REL_VOL 等跨资产特征将基于这 40 个品种计算截面均值
    # 若设为 None，则退化为只用 SYMBOLS（5品种截面）
    FEATURE_SYMBOLS = [
        # 主要外汇（26个）
        "EURUSD", "GBPUSD", "AUDUSD", "NZDUSD", "USDCAD", "USDCHF",
        "USDJPY", "EURJPY", "GBPJPY", "AUDJPY", "EURGBP", "EURAUD",
        "EURCAD", "EURCHF", "GBPAUD", "GBPCAD", "GBPCHF",
        "AUDCAD", "AUDCHF", "AUDNZD", "NZDCAD", "NZDCHF", "NZDJPY",
        "CADCHF", "CADJPY", "CHFJPY",
        # 贵金属（3个）
        "XAUUSD", "XAGUSD", "XPTUSD",
        # 美元指数（1个）
        "DXY.cash",
        # 大宗商品（2个）
        "USOIL.cash", "UKOIL.cash",
        # 主流指数（8个）
        "US30.cash", "US500.cash", "US100.cash", "UK100.cash",
        "DE30.cash", "FR40.cash", "JP225.cash", "AUS200.cash",
    ]

    # ── 数据参数 ──────────────────────────────────────────
    TIMEFRAME             = mt5.TIMEFRAME_H1   # K 线周期
    # 每品种拉取的历史 K 线上限。设为极大值以使用 MT5 全部可用历史；
    # 本地缓存优先：若 D:\K线数据 已有数据，fetcher 会返回本地全部历史（不截断）。
    BARS_COUNT            = 10_000_000
    MIN_BARS              = 1000   # Hard minimum; reject files below this value.
    RECOMMENDED_BARS      = 3000   # Allow fewer bars, but warn about sample size.
    DATA_REFRESH_INTERVAL = 300    # 秒，实盘数据刷新间隔
    KLINE_CACHE_DIR       = os.getenv("KLINE_CACHE_DIR", r"D:\K线数据")  # 本地 K 线缓存目录

    # ── 模型参数（仅供参考，训练实际使用 model_core.config.ModelConfig）────
    # 训练参数的权威来源是 model_core/config.py，这里的值不生效
    INPUT_DIM       = 20           # 特征数（与 MT5FeatureEngineer.INPUT_DIM 一致）
    BATCH_SIZE      = 128          # 参见 ModelConfig.BATCH_SIZE
    TRAIN_STEPS     = 300          # 参见 ModelConfig.TRAIN_STEPS
    MAX_FORMULA_LEN = 8            # 参见 ModelConfig.MAX_FORMULA_LEN
    # DEVICE 同样以 model_core/config.py 为准（已改为 cpu，原因见该文件注释）
    DEVICE          = (
        torch.device("cpu")
        if _TORCH_AVAILABLE
        else "cpu"
    )

    # ── 风控参数 ──────────────────────────────────────────
    RISK_PER_TRADE     = 0.01      # legacy: 保留给旧接口/测试；实盘仓位使用 VOL_TARGET_* 参数
    COST_RATE          = 0.0001    # 单边点差+佣金（forex/metals）
    MAX_OPEN_POSITIONS = 4         # 最多同时持仓品种数
    MAX_LOT_PER_TRADE  = 5.0       # 兜底上限；实际手数由 XAUUSD 0.01 手波动预算决定
    # 永不自动交易的品种（白银合约乘数 5000，2026-07-08 起停用）
    EXCLUDED_TRADE_SYMBOLS = ["XAGUSD"]
    # 手数校准（实盘）：
    # - 以 XAUUSD 0.01 手的一根 ATR 美元波动作为基准
    # - 其它品种按各自 ATR 与 tick value 反推手数，使金额波动接近
    # - 可选 Sharpe 权重：Sharpe 高于基准则略放大，低于基准则收缩
    FIXED_LOT_BY_SYMBOL = {
        "XAUUSD": 0.01,
    }
    VOL_TARGET_REFERENCE_SYMBOL = "XAUUSD"
    VOL_TARGET_REFERENCE_LOT = 0.01
    VOL_TARGET_SHARPE_REFERENCE = 2.447
    VOL_TARGET_SHARPE_EXPONENT = 0.50
    VOL_TARGET_MIN_SHARPE_WEIGHT = 0.50
    VOL_TARGET_MAX_SHARPE_WEIGHT = 1.50
    VOL_TARGET_SHARPE_BY_SYMBOL = {
        "XAUUSD": 2.447,
        "US100.cash": 1.811,
        "US500.cash": 0.959,
        "US2000.cash": 0.575,
        "US30.cash": 0.923,
        "JP225.cash": -0.653,
    }
    MIN_TRADE_EXPOSURE = 0.05      # |tanh(factor)| 小于该值时视为空仓，回测/实盘共用

    # ── 策略参数 ──────────────────────────────────────────
    # SIGNAL_MODE 控制信号→仓位的转换方式：
    #   "backtest_parity": tanh 连续仓位，与 backtest.py 完全一致（推荐）
    #   "threshold":       sigmoid + BUY_THRESHOLD / SELL_THRESHOLD（旧逻辑）
    SIGNAL_MODE = "backtest_parity"

    # EXIT_MODE 控制出场机制：
    #   "signal":  仅靠信号翻转出场，严格对标回测
    #   "risk":    保留止损/止盈/追踪止损
    #   "hybrid":  信号翻转为主，保留紧急熔断（单日最大亏损 / 极端滑点）
    EXIT_MODE = "signal"

    # threshold 模式专用（SIGNAL_MODE="threshold" 时生效）
    BUY_THRESHOLD       = 0.70
    SELL_THRESHOLD      = 0.40

    # risk / hybrid 模式专用（EXIT_MODE != "signal" 时生效）
    STOP_LOSS_PCT       = -0.02   # -2%
    TAKE_PROFIT_PCT     = 0.04    # +4%
    TRAILING_ACTIVATION = 0.03
    TRAILING_DROP       = 0.015

    # 时间对齐
    REBALANCE_ON_BAR_CLOSE = True  # True=仅新 K 线收盘后调仓，对标回测
    EXECUTION_LAG_BARS     = 1     # 与回测 target_ret 的执行延迟对齐

    # 持仓上限：None = 不限制（严格对标回测，各品种独立）
    # 设为整数（如 3）则启用约束（需回测里同步加同样约束才对标）
    MAX_OPEN_POSITIONS: int | None = None

    # ── 文件路径 ──────────────────────────────────────────
    STRATEGY_FILE  = "best_mt5_strategy.json"
    PORTFOLIO_FILE = "portfolio_state.json"
    STOP_SIGNAL    = "STOP_SIGNAL"

    # ── Magic Number ──────────────────────────────────────
    MAGIC_NUMBER = 20250101

    @classmethod
    def get_timeframe(cls, tf_str: str) -> int:
        """将字符串（如 'H1'）映射为 MT5 时间周期常量。

        支持的周期：M1, M5, M15, M30, H1, H4, D1, W1, MN1

        Args:
            tf_str: 时间周期字符串，例如 "H1"

        Returns:
            对应的 MT5 TIMEFRAME_* 整数常量

        Raises:
            ValueError: 若 tf_str 不在支持列表中
        """
        mapping = {
            "M1":  mt5.TIMEFRAME_M1,
            "M5":  mt5.TIMEFRAME_M5,
            "M15": mt5.TIMEFRAME_M15,
            "M30": mt5.TIMEFRAME_M30,
            "H1":  mt5.TIMEFRAME_H1,
            "H4":  mt5.TIMEFRAME_H4,
            "D1":  mt5.TIMEFRAME_D1,
            "W1":  mt5.TIMEFRAME_W1,
            "MN1": mt5.TIMEFRAME_MN1,
        }
        if tf_str not in mapping:
            raise ValueError(
                f"Unknown timeframe: '{tf_str}'. "
                f"Supported values: {list(mapping.keys())}"
            )
        return mapping[tf_str]
