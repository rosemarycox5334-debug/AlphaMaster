import pytest
import torch

import model_core.config as mc
from model_core.backtest import MT5Backtest


@pytest.fixture
def restore_reward_mode():
    old = mc.ModelConfig.REWARD_MODE
    yield
    mc.ModelConfig.REWARD_MODE = old


def test_cross_sectional_ic_perfect_positive():
    # 因子值与次日收益截面完全同序 → Rank-IC ≈ +1
    bt = MT5Backtest()
    # N=4 只股, T=3 期。factor[:,t] 与 target_ret[:,t+1] 同序
    factor = torch.tensor([[1., 1., 1.],
                           [2., 2., 2.],
                           [3., 3., 3.],
                           [4., 4., 4.]])
    target = torch.tensor([[0.0, 0.1, 0.1],
                           [0.0, 0.2, 0.2],
                           [0.0, 0.3, 0.3],
                           [0.0, 0.4, 0.4]])
    ic = bt._cross_sectional_ic(factor, target)
    assert ic > 0.9


def test_cross_sectional_ic_perfect_negative():
    bt = MT5Backtest()
    factor = torch.tensor([[1., 1.], [2., 2.], [3., 3.], [4., 4.]])
    target = torch.tensor([[0.0, 0.4], [0.0, 0.3], [0.0, 0.2], [0.0, 0.1]])
    ic = bt._cross_sectional_ic(factor, target)
    assert ic < -0.9


def test_ashare_reward_prefers_higher_ic(restore_reward_mode):
    mc.ModelConfig.REWARD_MODE = "ashare"
    bt = MT5Backtest()
    good_f = torch.tensor([[1., 1.], [2., 2.], [3., 3.], [4., 4.]])
    good_t = torch.tensor([[0.0, 0.1], [0.0, 0.2], [0.0, 0.3], [0.0, 0.4]])
    pos = torch.tanh(good_f)
    r_good = bt._multi_objective(good_f, good_t, pos * good_t, pos)
    # 打乱因子与收益的截面对应 → IC 崩 → reward 更低
    bad_f = torch.tensor([[4., 4.], [1., 1.], [3., 3.], [2., 2.]])
    r_bad = bt._multi_objective(bad_f, good_t, torch.tanh(bad_f) * good_t, torch.tanh(bad_f))
    assert r_good.item() > r_bad.item()
