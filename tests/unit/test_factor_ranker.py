import torch
from paper_trading.factor_ranker import FactorRanker


def test_rank_picks_top_k_by_factor():
    # 构造一个恒等公式：直接取特征0作为因子值
    # feat_slice [N, F, T]，公式 [0] 表示压入特征0
    ranker = FactorRanker(formula=[0])
    N, F, T = 5, 3, 4
    feat = torch.zeros(N, F, T)
    # 特征0在最后时间步的值：股票越靠后越大
    feat[:, 0, -1] = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    valid = torch.ones(N, dtype=torch.bool)
    codes = ["A", "B", "C", "D", "E"]
    picks = ranker.rank(feat, valid, codes, top_k=2)
    assert picks == ["E", "D"]                 # 因子值最大的两只


def test_rank_excludes_invalid():
    ranker = FactorRanker(formula=[0])
    N, F, T = 5, 3, 4
    feat = torch.zeros(N, F, T)
    feat[:, 0, -1] = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    valid = torch.tensor([True, True, True, True, False])  # E 停牌
    codes = ["A", "B", "C", "D", "E"]
    picks = ranker.rank(feat, valid, codes, top_k=2)
    assert "E" not in picks                    # 无效股被剔除
    assert picks == ["D", "C"]


def test_rank_handles_fewer_valid_than_k():
    ranker = FactorRanker(formula=[0])
    N, F, T = 5, 3, 2
    feat = torch.zeros(N, F, T)
    feat[:, 0, -1] = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    valid = torch.tensor([True, False, False, False, False])
    picks = ranker.rank(feat, valid, ["A", "B", "C", "D", "E"], top_k=3)
    assert picks == ["A"]                      # 只有 1 只有效
