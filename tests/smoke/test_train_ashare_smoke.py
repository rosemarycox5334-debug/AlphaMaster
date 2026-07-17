"""冒烟测试：train_ashare 训练入口（小样本、少步数）。

用合成数据（8 只股 × 120 交易日）跑通 AlphaEngine 多品种截面模式 5 步，
确认能产出 strategies/best_ashare_universe.json。

注意：train_ashare 会全局改写 ModelConfig.REWARD_MODE / TRAIN_STEPS，
本文件用 autouse fixture 在每条用例后恢复，避免污染同会话的其他测试。
"""
import pandas as pd
import pytest


@pytest.fixture(autouse=True)
def _restore_model_config():
    """保存并恢复被 train_ashare 全局改写的 ModelConfig 字段。"""
    import model_core.config as mc
    saved = {
        "REWARD_MODE": mc.ModelConfig.REWARD_MODE,
        "TRAIN_STEPS": mc.ModelConfig.TRAIN_STEPS,
    }
    try:
        yield
    finally:
        for k, v in saved.items():
            setattr(mc.ModelConfig, k, v)


class TinyFetcher:
    def universe_codes(self):
        return [f"C{i:03d}" for i in range(8)]      # 8 只股

    def trade_calendar(self, start, end):
        return pd.bdate_range("2024-01-01", periods=120).strftime("%Y-%m-%d").tolist()

    def daily(self, code, start, end):
        import numpy as np
        dates = self.trade_calendar(start, end)
        seed = int(code[1:])
        rng = np.random.default_rng(seed)
        price = 10 + np.cumsum(rng.normal(0, 0.2, len(dates)))
        price = abs(price) + 1
        return pd.DataFrame({
            "date": dates, "open": price, "high": price * 1.01,
            "low": price * 0.99, "close": price, "volume": rng.integers(1e5, 1e6, len(dates)),
        })


def test_train_ashare_runs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from train_ashare import train_ashare
    engine = train_ashare(fetcher=TinyFetcher(), start="2024-01-01",
                          end="2024-06-30", steps=5)
    assert engine is not None
    out = tmp_path / "strategies" / "best_ashare_universe.json"
    assert out.exists()
