"""
data_pipeline/ashare_manager.py — A股多品种数据管理器

产出与 MT5DataManager 相同的接口契约（raw_dict/feat_tensor/target_ret/symbols），
下游 model_core 零改动。额外提供 valid_mask（停牌/未上市剔除）与 trade_dates。

关键：用固定交易日历对齐（不用时间戳交集），停牌日前向填充价格但标记无效。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from loguru import logger

from config import Config
from data_pipeline.ashare_fetcher import AShareFetcher


class AShareDataManager:
    def __init__(self, fetcher=None) -> None:
        self._fetcher = fetcher or AShareFetcher()
        self._symbols: list[str] = []
        self._trade_dates: list[str] = []
        self._raw_dict: dict[str, torch.Tensor] | None = None
        self._target_ret: torch.Tensor | None = None
        self._valid_mask: torch.Tensor | None = None

    def load(self, codes: list[str] | None = None,
             start: str = "2023-01-01", end: str = "2026-06-30") -> None:
        codes = codes or self._fetcher.universe_codes()
        calendar = self._fetcher.trade_calendar(start, end)
        cal_idx = pd.Index(calendar)
        T = len(calendar)

        fields = ["open", "high", "low", "close", "volume"]
        price_rows = {f: [] for f in fields}
        mask_rows = []
        valid_codes = []

        for code in codes:
            df = self._fetcher.daily(code, start, end)
            if df is None or len(df) < 1:
                continue
            df = df.drop_duplicates("date", keep="last").set_index("date")
            df = df.reindex(cal_idx)                       # 对齐到固定日历
            present = df["close"].notna().values           # 该日有真实成交=有效
            if present.sum() < 1:
                continue
            # 停牌/未上市：价格前向+后向填充保证连续，mask 记录真实有效位
            df_filled = df[fields].ffill().bfill()
            for f in fields:
                price_rows[f].append(df_filled[f].values.astype("float32"))
            mask_rows.append(present)
            valid_codes.append(code)

        if not valid_codes:
            raise ValueError("无有效股票数据")

        self._symbols = valid_codes
        self._trade_dates = list(calendar)
        self._raw_dict = {
            f: torch.tensor(np.array(price_rows[f]), dtype=torch.float32)
            for f in fields
        }
        # time 字段：交易日历转 Unix 秒，[N,T]（各股相同）
        # 用 Timedelta 做分辨率无关换算——pandas 3.0 默认 datetime64[us]，
        # 直接 astype(int64)//1e9 会因单位是微秒而错 1000 倍。
        dt = pd.to_datetime(calendar)
        secs = (dt - pd.Timestamp("1970-01-01")) // pd.Timedelta(seconds=1)
        self._raw_dict["time"] = torch.tensor(
            np.tile(secs.values, (len(valid_codes), 1)), dtype=torch.int64)
        self._valid_mask = torch.tensor(np.array(mask_rows), dtype=torch.bool)
        self._target_ret = self._compute_target_ret(self._raw_dict["open"])
        logger.info(f"[A股] 加载 {len(valid_codes)} 只 × {T} 交易日")

    @staticmethod
    def _compute_target_ret(open_tensor: torch.Tensor) -> torch.Tensor:
        """target_ret[n,t] = log(open[t+2]/open[t+1])，末两位补0（与 MT5 口径一致）。"""
        n, t = open_tensor.shape
        target = torch.zeros(n, t, dtype=torch.float32)
        if t >= 3:
            num = open_tensor[:, 2:]
            den = open_tensor[:, 1:-1].clone()
            den[den == 0] = 1.0
            target[:, :t - 2] = torch.log(num / den)
        return target

    @property
    def symbols(self) -> list[str]:
        return list(self._symbols)

    @property
    def trade_dates(self) -> list[str]:
        return list(self._trade_dates)

    @property
    def raw_dict(self) -> dict:
        assert self._raw_dict is not None, "call load() first"
        return self._raw_dict

    @property
    def target_ret(self) -> torch.Tensor:
        assert self._target_ret is not None, "call load() first"
        return self._target_ret

    @property
    def valid_mask(self) -> torch.Tensor:
        assert self._valid_mask is not None, "call load() first"
        return self._valid_mask

    @property
    def feat_tensor(self) -> torch.Tensor:
        from model_core.features import MT5FeatureEngineer
        return MT5FeatureEngineer.compute_features(self.raw_dict)

    @property
    def bar_time(self) -> torch.Tensor:
        return self.raw_dict["time"][:, -1].long()
