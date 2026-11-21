import math

import pytest
import torch

from model_core.backtest import MT5Backtest
from model_core.config import ModelConfig
from model_core.engine import AlphaEngine, _build_walk_forward_folds, _repetition_penalty


def _turnover_quality_reference(position: torch.Tensor) -> float:
    n_symbols, t = position.shape
    pos_2d = position.tolist()
    all_runs, total_trades = [], 0
    for n in range(n_symbols):
        runs, cur_len, cur_dir = [], 0, 0
        for p in pos_2d[n]:
            pi = int(p)
            if pi != 0:
                if pi == cur_dir:
                    cur_len += 1
                else:
                    if cur_len > 0:
                        runs.append(cur_len)
                    cur_dir, cur_len = pi, 1
            else:
                if cur_len > 0:
                    runs.append(cur_len)
                cur_dir, cur_len = 0, 0
        if cur_len > 0:
            runs.append(cur_len)
        all_runs.extend(runs)
        total_trades += len(runs)

    total_bars = n_symbols * t
    target_trades = total_bars / 12.0
    actual_ratio = total_trades / max(target_trades, 1.0)
    if actual_ratio <= 0:
        freq_score = -2.0
    elif actual_ratio < 0.05:
        freq_score = -2.0 + actual_ratio / 0.05
    elif actual_ratio < 0.5:
        freq_score = -1.0 + (actual_ratio - 0.05) / 0.45
    elif actual_ratio <= 2.0:
        log_r = math.log(actual_ratio) / math.log(2.0)
        freq_score = math.exp(-0.5 * log_r ** 2)
    elif actual_ratio <= 8.0:
        freq_score = 0.5 - (actual_ratio - 2.0) / 6.0 * 1.5
    else:
        freq_score = -2.0

    hold_bonus = 0.0
    if all_runs:
        avg_hold = sum(all_runs) / len(all_runs)
        hold_bonus = min(0.3, math.log(max(avg_hold, 1.0)) / math.log(30.0) * 0.3)
    return float(freq_score + hold_bonus)


@pytest.mark.parametrize("shape", [(1, 97), (3, 113), (5, 19)])
def test_turnover_quality_preserves_legacy_int_semantics(shape):
    torch.manual_seed(1203 + shape[0])
    # Include values both inside and outside (-1, 1), so this tests the generic API,
    # not only the current tanh-position path.
    position = torch.randn(*shape) * 2.5
    bt = MT5Backtest()
    assert bt._turnover_quality(position) == _turnover_quality_reference(position)


def _reference_eval(engine, idx, fml, feat, target_ret, folds, use_wf):
    with torch.no_grad():
        res = engine.vm.execute(fml, feat)
    if res is None:
        return {"idx": idx, "status": "none", "reward": -5.0, "val_score": -5.0, "fml": fml}
    if res.std() < 1e-4:
        return {"idx": idx, "status": "const", "reward": -2.0, "val_score": -2.0, "fml": fml}

    with torch.no_grad():
        if use_wf:
            fold_tr, fold_vl, fold_ic = [], [], []
            for fold in folds:
                tr_sc, vl_sc = engine.bt.evaluate_fold(
                    res,
                    target_ret,
                    fold["train_start"],
                    fold["train_end"],
                    fold["val_start"],
                    fold["val_end"],
                )
                ic_m, _ = AlphaEngine._compute_ic(
                    res[:, fold["train_start"] : fold["train_end"]],
                    target_ret[:, fold["train_start"] : fold["train_end"]],
                )
                tr_adj = AlphaEngine._apply_ic_gate(tr_sc, ic_m)
                fold_tr.append(ModelConfig.REWARD_ALPHA * tr_adj)
                ic_v, _ = AlphaEngine._compute_ic(
                    res[:, fold["val_start"] : fold["val_end"]],
                    target_ret[:, fold["val_start"] : fold["val_end"]],
                )
                vl_adj = AlphaEngine._apply_ic_gate(vl_sc, ic_v)
                fold_vl.append(vl_adj)
                fold_ic.append(ic_m.item())
            train_score = torch.stack(fold_tr).mean()
            val_score = torch.stack(fold_vl).mean()
            ic_i = sum(fold_ic) / len(fold_ic)
        else:
            raise AssertionError("test only covers WF path")
        ic_full, ic_stab_full = AlphaEngine._compute_ic(res, target_ret)

    reward = train_score
    val_score_out = val_score
    rp = _repetition_penalty(fml)
    if rp > 0:
        reward = reward - rp
        val_score_out = val_score_out - rp
    corr_slice = (folds[0]["train_start"], folds[0]["train_end"])
    reward = engine._apply_corr_penalty(reward, res, corr_slice)
    val_score_out = engine._apply_corr_penalty(val_score_out, res, corr_slice)
    return {
        "idx": idx,
        "status": "ok",
        "reward": reward.item(),
        "val_score": val_score_out.item(),
        "ic_full": ic_full.item(),
        "ic_stab": ic_stab_full.item(),
        "ic_i": ic_i,
        "res": res,
        "fml": fml,
    }


@pytest.mark.parametrize("n_symbols,reward_mode", [(1, "ftmo"), (1, "forex"), (3, "ftmo")])
def test_p1b_cached_wf_is_bitwise_equal_to_reference(monkeypatch, n_symbols, reward_mode):
    monkeypatch.setattr(ModelConfig, "REWARD_MODE", reward_mode)
    torch.manual_seed(991 + n_symbols)
    t = 610
    factor = torch.randn(n_symbols, t)
    target_ret = torch.randn(n_symbols, t) * 0.001
    feat = torch.zeros(n_symbols, 1, t)
    folds = _build_walk_forward_folds(t, n_folds=5, gap=20)

    engine = AlphaEngine(data_manager=None)
    engine.vm.execute = lambda _fml, _feat: factor
    # Exercise correlation-penalty equality too.
    engine.factor_pool = [(1.0, 1, factor.clone())]
    fml = [0]

    ref = _reference_eval(engine, 0, fml, feat, target_ret, folds, True)
    got = engine._eval_formula_task(0, fml, feat, target_ret, folds, True, list(engine.factor_pool))

    assert got["status"] == ref["status"]
    for key in ("reward", "val_score", "ic_full", "ic_stab", "ic_i"):
        assert got[key] == ref[key], (key, got[key], ref[key])
    assert torch.equal(got["res"], ref["res"])


def test_precomputed_sortino_is_bitwise_equal(monkeypatch):
    monkeypatch.setattr(ModelConfig, "REWARD_MODE", "ftmo")
    torch.manual_seed(77)
    factor = torch.randn(1, 400)
    target = torch.randn(1, 400) * 0.001
    position = torch.tanh(factor)
    prev = torch.roll(position, 1, dims=1)
    prev[:, 0] = 0
    pnl = position * target - (position - prev).abs() * 0.0003
    bt = MT5Backtest(cost_rate=0.0003)
    sortino = bt._sortino(pnl)
    a = bt._multi_objective(factor, target, pnl, position, eval_bars=400)
    b = bt._multi_objective(
        factor,
        target,
        pnl,
        position,
        eval_bars=400,
        _precomputed_sortino=sortino,
    )
    assert torch.equal(a, b)
