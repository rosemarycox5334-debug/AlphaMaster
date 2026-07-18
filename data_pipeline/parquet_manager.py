"""Load training data from a single Parquet K-line file."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import torch
from loguru import logger

from config import Config
from data_pipeline.data_manager import MT5DataManager
from model_core.features import MT5FeatureEngineer

# Canonical labels used across the project
_TIMEFRAMES = ("M1", "M5", "M15", "M30", "H1", "H4", "D1", "W1", "MN1")

# Filename suffix aliases → canonical (case-insensitive keys)
_TF_ALIASES: dict[str, str] = {
    # M1
    "m1": "M1",
    "1m": "M1",
    "1min": "M1",
    "min1": "M1",
    # M5
    "m5": "M5",
    "5m": "M5",
    "5min": "M5",
    "min5": "M5",
    # M15
    "m15": "M15",
    "15m": "M15",
    "15min": "M15",
    "min15": "M15",
    # M30
    "m30": "M30",
    "30m": "M30",
    "30min": "M30",
    "min30": "M30",
    # H1
    "h1": "H1",
    "1h": "H1",
    "60m": "H1",
    "60min": "H1",
    "min60": "H1",
    "60": "H1",
    # H4
    "h4": "H4",
    "4h": "H4",
    "240m": "H4",
    "240min": "H4",
    "min240": "H4",
    "240": "H4",
    # D1
    "d1": "D1",
    "1d": "D1",
    "day": "D1",
    "daily": "D1",
    "1440m": "D1",
    "1440min": "D1",
    # W1
    "w1": "W1",
    "1w": "W1",
    "week": "W1",
    "weekly": "W1",
    # MN1 (month) — avoid bare "1m" which already maps to M1
    "mn1": "MN1",
    "1mo": "MN1",
    "1mon": "MN1",
    "month": "MN1",
    "monthly": "MN1",
}


def normalize_timeframe_token(token: str) -> str | None:
    """Map a filename timeframe token to canonical M1/M5/.../MN1."""
    raw = (token or "").strip()
    if not raw:
        return None
    key = raw.lower().replace("-", "").replace("_", "")
    if key in _TF_ALIASES:
        return _TF_ALIASES[key]
    upper = raw.upper()
    if upper in _TIMEFRAMES:
        return upper
    return None


def parse_parquet_filename(path: str | Path) -> tuple[str, str]:
    """Parse ``{symbol}_{timeframe}.parquet``.

    Accepts canonical suffixes (``H1``) and common aliases (``60min``, ``1h``, ``5m``…).
    Examples: ``AAPL_H1.parquet``, ``002008_60min.parquet``, ``BTCUSDT_1h.parquet``.
    """
    name = Path(path).name
    if Path(path).suffix.lower() != ".parquet":
        raise ValueError(f"请选择 .parquet 文件；当前: {name}")
    stem = Path(path).stem
    if "_" not in stem:
        raise ValueError(
            f"文件名须为 {{品种}}_{{周期}}.parquet，例如 AAPL_H1.parquet / 002008_60min.parquet；"
            f"当前: {name}"
        )
    symbol, tf_raw = stem.rsplit("_", 1)
    symbol = symbol.strip()
    timeframe = normalize_timeframe_token(tf_raw)
    if not symbol or timeframe is None:
        raise ValueError(
            f"文件名须为 {{品种}}_{{周期}}.parquet，例如 AAPL_H1.parquet / 002008_60min.parquet；"
            f"支持周期别名: H1/60min/1h, M5/5min, D1/1d …；当前: {name}"
        )
    return symbol, timeframe


def inspect_parquet_file(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"文件不存在: {p}")
    if p.suffix.lower() != ".parquet":
        raise ValueError("请选择 .parquet 文件")

    symbol, timeframe = parse_parquet_filename(p)
    df = pd.read_parquet(p)
    bars = len(df)
    if bars < Config.MIN_BARS:
        raise ValueError(
            f"数据不足: {bars} bars（至少需要 {Config.MIN_BARS}）"
        )

    # 从实际时间跨度计算年数（适用于所有周期，比固定公式更准确）
    years = None
    if "time" in df.columns and len(df) > 1:
        try:
            t_min = float(df["time"].min())
            t_max = float(df["time"].max())
            if t_max > t_min and t_max > 1_000_000_000:  # 合法的 Unix 时间戳
                span_seconds = t_max - t_min
                years = round(span_seconds / (365.25 * 24 * 3600), 2)
        except Exception:
            pass
    # 回退：H1 用固定公式（6240 根/年，24h 市场）
    if years is None and timeframe == "H1":
        years = round(bars / 6240, 2)
    return {
        "data_file": str(p.resolve()),
        "filename": p.name,
        "symbol": symbol,
        "timeframe": timeframe,
        "bars": bars,
        "years_h1": years,
        "valid": True,
        "message": "",
    }


class ParquetDataManager:
    """Single-symbol data manager backed by one Parquet file."""

    def __init__(self, file_path: str | Path) -> None:
        self.file_path = Path(file_path)
        self.symbol, self.timeframe = parse_parquet_filename(self.file_path)
        self._raw_dict: dict[str, torch.Tensor] | None = None
        self._target_ret: torch.Tensor | None = None

    def load(self) -> None:
        df = pd.read_parquet(self.file_path)
        if len(df) < Config.MIN_BARS:
            raise ValueError(
                f"数据不足: {len(df)} bars（至少需要 {Config.MIN_BARS}）"
            )

        volume_col = "tick_volume" if "tick_volume" in df.columns else "volume"
        required = ["time", "open", "high", "low", "close", volume_col]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Parquet 缺少列: {missing}")

        sub = df[required].copy().rename(columns={volume_col: "volume"})

        # 兼容性修复：某些 A 股 parquet 导出工具把 Unix 秒时间戳误存为
        # "秒/1000"（数值被缩小 1000 倍，导致日期变成 1970 年）。
        # 若最大时间戳 < 1e7（1970-04-27 之前），则视为被除过 1000，乘回。
        if pd.api.types.is_numeric_dtype(sub["time"]) and sub["time"].max() < 10_000_000:
            sub["time"] = sub["time"] * 1000
            logger.info(
                f"[数据] {self.file_path.name} 时间戳被识别为秒/1000，"
                f"已乘 1000 恢复为 Unix 秒。"
            )

        sub = sub.sort_values("time")
        sub = sub[~sub["time"].duplicated(keep="last")]

        rows = {field: sub[field].values for field in ["open", "high", "low", "close", "volume"]}
        import numpy as np

        raw: dict[str, torch.Tensor] = {
            field: torch.tensor(np.array([rows[field]]), dtype=torch.float32)
            for field in ["open", "high", "low", "close", "volume"]
        }
        raw["time"] = torch.tensor(
            np.array([sub["time"].values.astype("int64")]),
            dtype=torch.int64,
        )

        self._raw_dict = raw
        self._target_ret = MT5DataManager._compute_target_ret(raw["open"])
        logger.info(
            f"[数据] 已加载 {self.symbol} {self.timeframe}，"
            f"共 {raw['open'].shape[1]} 根K线，文件 {self.file_path.name}"
        )

    @property
    def symbols(self) -> list[str]:
        return [self.symbol]

    @property
    def raw_dict(self) -> dict[str, torch.Tensor]:
        if self._raw_dict is None:
            raise RuntimeError("Call load() first")
        return self._raw_dict

    @property
    def feat_tensor(self) -> torch.Tensor:
        return MT5FeatureEngineer.compute_features(self.raw_dict)

    @property
    def target_ret(self) -> torch.Tensor:
        if self._target_ret is None:
            raise RuntimeError("Call load() first")
        return self._target_ret

    @property
    def bar_time(self) -> torch.Tensor:
        raw = self.raw_dict
        if "time" in raw:
            return raw["time"][:, -1].long()
        return torch.zeros(1, dtype=torch.int64)
