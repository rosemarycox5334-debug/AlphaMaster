import pandas as pd
import torch
from data_pipeline.ashare_manager import AShareDataManager


class FakeFetcher:
    """两只股票：A 全区间有数据，B 中间停牌一天、且上市晚一天。"""
    def universe_codes(self):
        return ["AAA", "BBB"]

    def trade_calendar(self, start, end):
        return ["2023-01-03", "2023-01-04", "2023-01-05", "2023-01-06"]

    def daily(self, code, start, end):
        if code == "AAA":
            dates = ["2023-01-03", "2023-01-04", "2023-01-05", "2023-01-06"]
            close = [10.0, 10.5, 11.0, 10.8]
        else:  # BBB：缺 2023-01-03（未上市）与 2023-01-05（停牌）
            dates = ["2023-01-04", "2023-01-06"]
            close = [20.0, 21.0]
        return pd.DataFrame({
            "date": dates, "open": close, "high": close,
            "low": close, "close": close, "volume": [1000] * len(dates),
        })


class BigFetcher:
    """两只股票、30 个交易日的正常规模 fixture，用于验证 feat_tensor 委托。

    feat_tensor 委托 MT5FeatureEngineer.compute_features，后者的滚动窗口最长
    可达 200，只有在 T 达到正常规模（>=20）时才处于其有效工作域；用 T=4 的
    极小 fixture 计算 feat_tensor 是把共享特征引擎推出有效域，测的不是本管理器
    的委托契约。故此处用 30 日 fixture 校验 [N, F, T] 形状契约。
    """
    def universe_codes(self):
        return ["AAA", "BBB"]

    def trade_calendar(self, start, end):
        return [f"2023-03-{d:02d}" for d in range(1, 31)]

    def daily(self, code, start, end):
        dates = [f"2023-03-{d:02d}" for d in range(1, 31)]
        base = 10.0 if code == "AAA" else 20.0
        close = [base + 0.1 * i for i in range(30)]
        return pd.DataFrame({
            "date": dates, "open": close, "high": close,
            "low": close, "close": close, "volume": [1000] * 30,
        })


def test_shapes_and_calendar():
    mgr = AShareDataManager(fetcher=FakeFetcher())
    mgr.load(start="2023-01-03", end="2023-01-06")
    assert mgr.symbols == ["AAA", "BBB"]
    assert mgr.trade_dates == ["2023-01-03", "2023-01-04", "2023-01-05", "2023-01-06"]
    assert mgr.raw_dict["close"].shape == (2, 4)   # [N=2, T=4]
    assert mgr.target_ret.shape == (2, 4)
    # bar_time 必须是 Unix 秒（供实时 runner 检测 K 线收盘，与 MT5 口径一致）
    assert mgr.raw_dict["time"].shape == (2, 4)
    assert mgr.bar_time.tolist() == [1672963200, 1672963200]  # 2023-01-06 UTC 秒


def test_feat_tensor_delegation_shape():
    # feat_tensor 委托契约在正常规模 T 下验证：[N, F, T]
    mgr = AShareDataManager(fetcher=BigFetcher())
    mgr.load(start="2023-03-01", end="2023-03-30")
    ft = mgr.feat_tensor
    assert ft.shape[0] == 2 and ft.shape[2] == 30   # [N=2, F, T=30]


def test_valid_mask_marks_suspension_and_prelisting():
    mgr = AShareDataManager(fetcher=FakeFetcher())
    mgr.load(start="2023-01-03", end="2023-01-06")
    vm = mgr.valid_mask                             # [2, 4] bool
    # AAA 全有效
    assert vm[0].all()
    # BBB: 01-03 未上市无效, 01-05 停牌无效, 01-04/01-06 有效
    assert not vm[1, 0]     # 未上市
    assert vm[1, 1]         # 有效
    assert not vm[1, 2]     # 停牌
    assert vm[1, 3]         # 有效


def test_forward_fill_price_continuity():
    # 停牌日价格前向填充，保证特征计算连续（无 NaN）
    mgr = AShareDataManager(fetcher=FakeFetcher())
    mgr.load(start="2023-01-03", end="2023-01-06")
    close = mgr.raw_dict["close"]
    assert not torch.isnan(close).any()
    # BBB 01-05 停牌 → 用 01-04 的 20.0 填充
    assert close[1, 2].item() == 20.0
    # BBB 01-03 未上市 → 用首个有效值 20.0 回填
    assert close[1, 0].item() == 20.0
