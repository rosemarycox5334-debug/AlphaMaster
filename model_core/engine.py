import copy
import heapq
import json
import pathlib
import random

import torch
from torch.distributions import Categorical
from tqdm import tqdm

from .config import ModelConfig
from .alphagpt import AlphaGPT, NewtonSchulzLowRankDecay, StableRankMonitor
from .vm import StackVM
from .backtest import MT5Backtest
from .vocab import FORMULA_VOCAB, VOCAB_VERSION, VocabVersionMismatchError  # task 12.2

# P3：冠军在场时间稳健性校验所需
try:
    from strategy_manager.signal import compute_target_positions_stateless
except ImportError:
    # 兼容无 strategy_manager 的测试环境
    def compute_target_positions_stateless(factors):  # type: ignore
        import torch as _torch
        return _torch.sign(_torch.tanh(factors))

try:
    from config import Config as _RootConfig
    _STRATEGY_FILE  = _RootConfig.STRATEGY_FILE
    _CHECKPOINT_DIR = pathlib.Path(getattr(_RootConfig, 'CHECKPOINT_DIR', 'checkpoints'))
except ImportError:
    _STRATEGY_FILE  = "best_mt5_strategy.json"
    _CHECKPOINT_DIR = pathlib.Path("checkpoints")


def _strategy_file_for_symbol(symbol: str | None) -> str:
    """返回该品种对应的策略文件路径。

    单品种训练时使用 strategies/best_{symbol}.json，
    多品种/未指定品种时回退到默认路径。
    """
    if symbol:
        return str(pathlib.Path("strategies") / f"best_{symbol}.json")
    return _STRATEGY_FILE


# ─────────────────────────────────────────────────────────────────────────────
# Walk-Forward 折叠构建
# ─────────────────────────────────────────────────────────────────────────────

def _build_walk_forward_folds(T: int, n_folds: int = 5, gap: int = 20) -> list[dict]:
    fold_size = T // n_folds
    if fold_size < 2:
        return [{"train_start": 0, "train_end": T, "val_start": 0, "val_end": T, "gap": 0}]
    total_required = fold_size * n_folds + gap * (n_folds - 1)
    if total_required > T:
        gap = max(0, (T - fold_size * n_folds) // n_folds)
    folds = []
    for k in range(1, n_folds):
        train_end = fold_size * k
        val_start = train_end + gap
        val_end   = val_start + fold_size if k < n_folds - 1 else T
        if val_start >= T or val_end > T:
            break
        folds.append({"train_start": 0, "train_end": train_end,
                      "val_start": val_start, "val_end": val_end, "gap": gap})
    if not folds:
        return [{"train_start": 0, "train_end": T, "val_start": 0, "val_end": T, "gap": 0}]
    return folds


def _repetition_penalty(formula: list[int]) -> float:
    if not formula:
        return 0.0
    penalty, count = 0.0, 1
    for i in range(1, len(formula)):
        if formula[i] == formula[i - 1]:
            count += 1
            if count >= 2:
                penalty += 0.3
        else:
            count = 1
    return penalty


# ─────────────────────────────────────────────────────────────────────────────
# ConstrainedSampler — 保证 100% 合法公式
# ─────────────────────────────────────────────────────────────────────────────

class ConstrainedSampler:
    def __init__(self, vocab_size: int, feat_offset: int, arity_map: dict[int, int]):
        self.vocab_size  = vocab_size
        self.feat_offset = feat_offset
        self.arity_map   = arity_map
        self.delta: dict[int, int] = {}
        for tid in range(vocab_size):
            if tid < feat_offset:
                self.delta[tid] = 1
            else:
                a = arity_map.get(tid, 1)
                self.delta[tid] = 1 - a

    def valid_mask(self, stack_depth: int, step_idx: int,
                   total_steps: int, device: torch.device) -> torch.Tensor:
        remaining = total_steps - step_idx
        mask = torch.ones(self.vocab_size, dtype=torch.bool, device=device)
        for tid in range(self.vocab_size):
            d         = self.delta[tid]
            new_depth = stack_depth + d
            if new_depth < 1:
                mask[tid] = False;  continue
            min_future = new_depth + (remaining - 1) * (-2)
            max_future = new_depth + (remaining - 1) * 1
            if 1 < min_future or 1 > max_future:
                mask[tid] = False
        if not mask.any():
            for tid in range(self.vocab_size):
                if stack_depth + self.delta[tid] >= 1:
                    mask[tid] = True
        return mask

    def apply_mask_to_logits(self, logits: torch.Tensor, stack_depths: list[int],
                              step_idx: int, total_steps: int) -> torch.Tensor:
        masked = logits.clone()
        device = logits.device
        for b, depth in enumerate(stack_depths):
            vmask = self.valid_mask(depth, step_idx, total_steps, device)
            masked[b][~vmask] = -1e9
        return masked


# ─────────────────────────────────────────────────────────────────────────────
# AlphaEngine — __init__ 与静态辅助方法
# ─────────────────────────────────────────────────────────────────────────────

class AlphaEngine:
    def __init__(self, data_manager=None, use_lord_regularization=True,
                 lord_decay_rate=1e-3, lord_num_iterations=5, n_folds: int = 5,
                 target_symbol: str | None = None):
        self.data_manager  = data_manager
        self.n_folds       = n_folds
        self.target_symbol = target_symbol   # None = 多品种模式，str = 单品种模式
        self.model   = AlphaGPT().to(ModelConfig.DEVICE)
        self.opt     = torch.optim.AdamW(self.model.parameters(), lr=1e-3)

        self.use_lord = use_lord_regularization
        if self.use_lord:
            self.lord_opt = NewtonSchulzLowRankDecay(
                self.model.named_parameters(),
                decay_rate=lord_decay_rate,
                num_iterations=lord_num_iterations,
                target_keywords=["attention", "qk_norm"],
            )
            self.rank_monitor = StableRankMonitor(
                self.model, target_keywords=["in_proj", "out_proj", "qk_norm"]
            )
        else:
            self.lord_opt = None
            self.rank_monitor = None

        self.vm = StackVM()
        self.bt = MT5Backtest()

        from .vocab import FORMULA_VOCAB as _v
        self.sampler = ConstrainedSampler(
            vocab_size=_v.size, feat_offset=_v.operator_offset, arity_map=self.vm.arity_map
        )

        self.best_score   = -float('inf')
        self.best_formula = None
        self._best_snapshot: dict | None = None

        self.training_history = {
            'step': [], 'avg_reward': [], 'best_score': [], 'val_score': [], 'stable_rank': []
        }
        self._restart_count      = 0
        self.factor_pool: list[tuple[float, int, torch.Tensor]] = []
        self._factor_pool_counter = 0

        # Elite Replay pool: (val_score, counter, formula_tokens, birth_step)
        self._elite_pool: list[tuple[float, int, list[int], int]] = []
        self._elite_counter = 0

        # 自适应噪声：记录 best 刷新步数
        self._best_update_step = 0
        self._stagnation_steps = 0

    # ── IC computation ────────────────────────────────────────────────────────

    @staticmethod
    def _compute_ic(factor: torch.Tensor, target_ret: torch.Tensor
                    ) -> tuple[torch.Tensor, torch.Tensor]:
        """时序 IC（每品种内部 factor[t] vs ret[t+1]）的均值与稳定性。

        对 5 品种宇宙，时序 IC 比横截面 IC 统计意义更强。
        """
        N, T = factor.shape
        if T < 2:
            z = torch.zeros(1, device=factor.device)
            return z, z

        ic_list = []
        for n in range(N):
            x  = factor[n, :-1]
            y  = target_ret[n, 1:]
            xm = x - x.mean()
            ym = y - y.mean()
            sx = (xm ** 2).mean().sqrt()
            sy = (ym ** 2).mean().sqrt()
            if sx < 1e-6 or sy < 1e-6:
                continue
            ic = (xm * ym).mean() / (sx * sy + 1e-8)
            ic_list.append(ic)

        if not ic_list:
            z = torch.zeros(1, device=factor.device)
            return z, z

        ic_tensor = torch.stack(ic_list)
        ic_mean   = ic_tensor.mean()
        ic_stab   = (ic_mean / (ic_tensor.std(unbiased=False) + 1e-6)
                     if ic_tensor.numel() >= 2
                     else torch.zeros(1, device=factor.device))
        return ic_mean, ic_stab

    # ── IC gate: direction-based, dimension-agnostic ──────────────────────────

    @staticmethod
    def _apply_ic_gate(reward: torch.Tensor, ic_mean) -> torch.Tensor:
        """IC 门控：用 IC 符号而非量值调整 reward，完全规避量纲问题。
        IC > thresh  → reward × IC_GATE_MULT  (正向预测，奖励)
        IC < -thresh → reward × IC_NEG_MULT   (反向预测，惩罚)
        |IC| ≤ thresh→ 不修改                  (噪声区)
        """
        ic_val = ic_mean.item() if isinstance(ic_mean, torch.Tensor) else float(ic_mean)
        t = ModelConfig.IC_GATE_THRESH
        if ic_val > t:
            return reward * ModelConfig.IC_GATE_MULT
        elif ic_val < -t:
            return reward * ModelConfig.IC_NEG_MULT
        return reward


    # ── Elite pool ────────────────────────────────────────────────────────────

    @staticmethod
    def _dedup_elite_pool(
        pool: list[tuple[float, int, list[int], int]]
    ) -> list[tuple[float, int, list[int], int]]:
        """对精英池去重：相同 tokens 只保留得分最高的一条，重建最小堆。"""
        best: dict[str, tuple[float, int, list[int], int]] = {}
        for sc, cnt, toks, birth in pool:
            key = str(toks)
            if key not in best or sc > best[key][0]:
                best[key] = (sc, cnt, toks, birth)
        deduped = list(best.values())
        heapq.heapify(deduped)
        return deduped

    def _update_elite_pool(self, val_score: float, formula: list[int], step: int = 0) -> None:
        """维护精英公式池（最小堆，Top-ELITE_POOL_SIZE 个历史最优公式，自动去重）。

        去重逻辑：若 formula 已在池中，只在新得分更高时原地更新，不插入重复副本。
        这防止了单一公式垄断 elite pool，保持多样性。
        新增：记录 birth_step 用于 elite decay。
        """
        k = ModelConfig.ELITE_POOL_SIZE

        # 检查是否已有相同公式
        for idx, (sc, cnt, toks, birth) in enumerate(self._elite_pool):
            if toks == formula:
                if val_score <= sc:
                    return  # 已有更高分的相同公式，不更新
                # 分数更高：从堆中移除旧条目，插入新条目
                self._elite_pool[idx] = self._elite_pool[-1]
                self._elite_pool.pop()
                heapq.heapify(self._elite_pool)  # O(k)，k≤20，可接受
                break

        entry = (val_score, self._elite_counter, list(formula), step)
        self._elite_counter += 1
        if len(self._elite_pool) < k:
            heapq.heappush(self._elite_pool, entry)
        elif val_score > self._elite_pool[0][0]:
            heapq.heapreplace(self._elite_pool, entry)

    # ── Factor pool ───────────────────────────────────────────────────────────

    def _update_factor_pool(self, val_score: float, factor: torch.Tensor) -> None:
        k     = ModelConfig.FACTOR_TOP_K
        f_gpu = factor.detach()
        entry = (val_score, self._factor_pool_counter, f_gpu)
        self._factor_pool_counter += 1
        if len(self.factor_pool) < k:
            heapq.heappush(self.factor_pool, entry)
        elif val_score > self.factor_pool[0][0]:
            heapq.heapreplace(self.factor_pool, entry)

    def _apply_corr_penalty(self, reward: torch.Tensor, factor: torch.Tensor) -> torch.Tensor:
        if not self.factor_pool:
            return reward
        f_flat = factor.detach().reshape(-1).float()
        if f_flat.std() < 1e-4:
            return reward
        pool_vecs = torch.stack(
            [f.reshape(-1).float() for _, _cnt, f in self.factor_pool], dim=0
        )
        f_c  = f_flat - f_flat.mean()
        p_c  = pool_vecs - pool_vecs.mean(dim=1, keepdim=True)
        cov  = (p_c * f_c).sum(dim=1)
        sx   = f_c.norm() + 1e-8
        sy   = p_c.norm(dim=1) + 1e-8
        corr = (cov / (sx * sy)).abs()
        if (corr > ModelConfig.CORR_THRESHOLD).any():
            reward = reward * ModelConfig.CORR_PENALTY
        return reward


    # ── Main training loop ────────────────────────────────────────────────────

    def train(self, start_step: int = 0, end_step: int | None = None,
              migration_hook=None, verbose_header: bool = True):
        if self.data_manager is None:
            raise RuntimeError("AlphaEngine requires a data_manager.")

        if end_step is None:
            end_step = ModelConfig.TRAIN_STEPS

        if verbose_header:
            print("🚀 Starting MT5 Alpha Mining" +
                  (" with LoRD Regularization..." if self.use_lord else "..."))
            print(f"   Entropy: thresh={ModelConfig.ENTROPY_COLLAPSE_THRESH}  "
                  f"coeff_max={ModelConfig.ENTROPY_COEFF_MAX}  "
                  f"collapse_steps={ModelConfig.ENTROPY_COLLAPSE_STEPS}")
            print(f"   Elite Replay: frac={ModelConfig.ELITE_REPLAY_FRAC}  "
                  f"pool={ModelConfig.ELITE_POOL_SIZE}")
            print(f"   IC gate: ±{ModelConfig.IC_GATE_THRESH}  "
                  f"pos×{ModelConfig.IC_GATE_MULT}  neg×{ModelConfig.IC_NEG_MULT}")
            print(f"   Max restarts: {ModelConfig.MAX_RESTARTS}  "
                  f"noise={ModelConfig.RESTART_NOISE}")

        T     = self.data_manager.target_ret.shape[1]
        folds = _build_walk_forward_folds(T, self.n_folds,
                                          gap=getattr(ModelConfig, 'WF_GAP', 20))
        use_wf = len(folds) > 1 and not (
            folds[0]["train_start"] == 0 and folds[0]["train_end"] == T
        )
        if verbose_header:
            if use_wf:
                print(f"   Walk-Forward: {len(folds)} 折  T={T}")
                for k, f in enumerate(folds):
                    print(f"  折{k+1}: train[0,{f['train_end']}) "
                          f"gap={f['gap']} val[{f['val_start']},{f['val_end']})")
            else:
                print(f"   退化为全量评估（T={T}）")

        # 因果安全：features.py 的 _robust_norm 已改为滚动因果实现
        # 每个 t 的归一化参数只用 [t-w+1..t]，walk-forward 折叠切片无泄露
        feat  = self.data_manager.feat_tensor.to(ModelConfig.DEVICE)
        t_ret = self.data_manager.target_ret.to(ModelConfig.DEVICE)
        bs      = ModelConfig.BATCH_SIZE
        n_elite = max(1, int(bs * ModelConfig.ELITE_REPLAY_FRAC))
        n_new   = bs - n_elite

        remaining = end_step - start_step
        if remaining <= 0:
            print(f"[Train] start_step={start_step} >= end_step={end_step}, nothing to do.")
            return

        pbar               = tqdm(range(start_step, end_step),
                                  total=end_step,
                                  initial=start_step)
        low_entropy_streak = 0

        for step in pbar:
            # ── Part A: Sample n_new new formulas ────────────────────
            inp_new = torch.zeros((n_new, 1), dtype=torch.long,
                                  device=ModelConfig.DEVICE)
            lp_new, tok_new, ent_new = [], [], []
            sd_new = [0] * n_new

            for si in range(ModelConfig.MAX_FORMULA_LEN):
                lg, _, _ = self.model(inp_new)
                lg = self.sampler.apply_mask_to_logits(lg, sd_new, si,
                                                       ModelConfig.MAX_FORMULA_LEN)
                d  = Categorical(logits=lg)
                a  = d.sample()
                lp_new.append(d.log_prob(a))
                tok_new.append(a)
                ent_new.append(d.entropy())
                inp_new = torch.cat([inp_new, a.unsqueeze(1)], dim=1)
                for b in range(n_new):
                    sd_new[b] += self.sampler.delta[a[b].item()]

            seqs_new = torch.stack(tok_new, dim=1)


            # ── Part B: Elite Replay ─────────────────────────────────
            elite_formulas: list[list[int]] = []
            if self._elite_pool and n_elite > 0:
                ps = []
                pt = []
                weights = []
                for sc, cnt, toks, birth in self._elite_pool:
                    age = max(0, step - birth)
                    decay = 1.0
                    if ModelConfig.ELITE_DECAY:
                        half = max(1, ModelConfig.ELITE_DECAY_HALF_LIFE)
                        decay = 0.5 ** (age / half)
                    ps.append(sc)
                    pt.append(toks)
                    weights.append(decay)
                ps_min  = min(ps)
                ps_max  = max(ps)
                # 软温度采样：避免最高分公式垄断
                # 先归一到 [0,1]，再除以温度 T=0.5 后做 softmax
                # T<1 → 高分公式仍被偏好，但不再独占
                if ps_max > ps_min:
                    normalized = [(s - ps_min) / (ps_max - ps_min + 1e-8) for s in ps]
                else:
                    normalized = [1.0] * len(ps)
                temp = 0.5
                exp_s = [weights[i] * (2.0 ** (normalized[i] / temp)) for i in range(len(ps))]
                exp_sum = sum(exp_s)
                probs = [e / exp_sum for e in exp_s]
                idx_e   = random.choices(range(len(self._elite_pool)),
                                         weights=probs, k=n_elite)
                elite_formulas = [pt[i] for i in idx_e]

                # 详细日志：Elite Replay 衰减状态（每 100 步打印一次）
                if step % 100 == 0:
                    avg_decay = sum(weights) / len(weights)
                    max_age = max(max(0, step - birth) for _, _, _, birth in self._elite_pool)
                    age_list = sorted([max(0, step - birth) for _, _, _, birth in self._elite_pool])
                    tqdm.write(
                        f"[EliteReplay @ step {step}] pool={len(self._elite_pool)} "
                        f"avg_decay={avg_decay:.3f} max_age={max_age} ages={age_list} "
                        f"sampled_scores=[{', '.join(f'{ps[i]:.3f}' for i in idx_e[:3])}...]"
                    )
            else:
                elite_formulas = seqs_new[:n_elite].tolist()

            lp_elite, ent_elite = [], []
            if elite_formulas:
                ne     = len(elite_formulas)
                inp_e  = torch.zeros((ne, 1), dtype=torch.long,
                                     device=ModelConfig.DEVICE)
                sd_e   = [0] * ne
                tok_e_t = torch.tensor(elite_formulas, dtype=torch.long,
                                       device=ModelConfig.DEVICE)
                for si in range(ModelConfig.MAX_FORMULA_LEN):
                    lg_e, _, _ = self.model(inp_e)
                    lg_e = self.sampler.apply_mask_to_logits(
                        lg_e, sd_e, si, ModelConfig.MAX_FORMULA_LEN
                    )
                    d_e  = Categorical(logits=lg_e)
                    tk   = tok_e_t[:, si]
                    lp_elite.append(d_e.log_prob(tk))
                    ent_elite.append(d_e.entropy())
                    inp_e = torch.cat([inp_e, tk.unsqueeze(1)], dim=1)
                    for b in range(ne):
                        sd_e[b] += self.sampler.delta[tk[b].item()]


            # ── Part C: Evaluate all formulas ────────────────────────
            all_fmls = seqs_new.tolist() + elite_formulas
            tot      = len(all_fmls)
            rewards    = torch.zeros(tot, device=ModelConfig.DEVICE)
            val_scores = torch.zeros(tot, device=ModelConfig.DEVICE)

            ok_cnt = none_cnt = const_cnt = 0
            step_max_val = -float('inf');  step_best_f = None
            bic, bis, bsor = [], [], []

            for i, fml in enumerate(all_fmls):
                with torch.no_grad():
                    res = self.vm.execute(fml, feat)
                if res is None:
                    rewards[i] = val_scores[i] = -5.0
                    none_cnt += 1;  continue
                if res.std() < 1e-4:
                    rewards[i] = val_scores[i] = -2.0
                    const_cnt += 1;  continue
                ok_cnt += 1

                with torch.no_grad():
                    if use_wf:
                        fold_tr, fold_vl, fold_ic = [], [], []
                        for fold in folds:
                            # res[:, train_start:train_end] 是在已无泄露的因子上切片，正确
                            tr_sc, vl_sc = self.bt.evaluate_fold(
                                res, t_ret,
                                fold["train_start"], fold["train_end"],
                                fold["val_start"],   fold["val_end"],
                            )
                            ic_m, _ = AlphaEngine._compute_ic(
                                res[:, fold["train_start"]:fold["train_end"]],
                                t_ret[:, fold["train_start"]:fold["train_end"]],
                            )
                            tr_adj = AlphaEngine._apply_ic_gate(tr_sc, ic_m)
                            fold_tr.append(ModelConfig.REWARD_ALPHA * tr_adj)
                            fold_vl.append(vl_sc)
                            fold_ic.append(ic_m.item())
                        train_score = torch.stack(fold_tr).mean()
                        val_score   = torch.stack(fold_vl).mean()
                        ic_i        = sum(fold_ic) / len(fold_ic)
                    else:
                        train_score, _ = self.bt.evaluate(res, {}, t_ret)
                        ic_m0, _  = AlphaEngine._compute_ic(res, t_ret)
                        train_score = AlphaEngine._apply_ic_gate(
                            ModelConfig.REWARD_ALPHA * train_score, ic_m0
                        )
                        val_score = train_score
                        ic_i      = ic_m0.item()
                    ic_full, ic_stab_full = AlphaEngine._compute_ic(res, t_ret)

                rewards[i]    = train_score
                val_scores[i] = val_score
                bic.append(ic_full.item());  bis.append(ic_stab_full.item())
                bsor.append(val_score.item())

                # 重复惩罚和相关性惩罚同时施加到 rewards 和 val_scores
                # 保证 best_score / elite_pool 选优时已含所有惩罚
                rp = _repetition_penalty(fml)
                if rp > 0:
                    rewards[i]    -= rp
                    val_scores[i] -= rp
                rewards[i]    = self._apply_corr_penalty(rewards[i], res)
                val_scores[i] = self._apply_corr_penalty(val_scores[i], res)

                # 用含惩罚的 val_scores[i] 选全局最优
                final_val = val_scores[i].item()
                if final_val > step_max_val:
                    step_max_val = final_val;  step_best_f = fml

                if final_val > self.best_score:
                    # P3：冠军选择稳健性校验——连续仓位均值 < 5% 的极稀疏公式拒绝登顶
                    pos_check = compute_target_positions_stateless(res)
                    exposure = pos_check.abs().mean().item()  # 连续仓位：均值
                    if exposure < 0.05:
                        # 极稀疏：参与梯度更新但不登顶，但记录日志方便排查
                        tqdm.write(
                            f"[SparseSkip @ step {step}] Val={final_val:.3f} "
                            f"IC={ic_i:.4f} Exp={exposure:.1%} | too sparse to be king"
                        )
                        pass
                    else:
                        old_best = self.best_score
                        self.best_score   = final_val
                        self.best_formula = fml
                        self._best_snapshot = copy.deepcopy(self.model.state_dict())
                        self._best_update_step = step
                        self._stagnation_steps = 0
                        self._update_factor_pool(final_val, res)
                        # 即时保存：任何时刻进程退出都有最新最优公式（防终端回收丢策略）
                        self._save_strategy_live()
                        tqdm.write(
                            f"[!] New King @ step {step}: Val={final_val:.3f} "
                            f"(was {old_best:.3f}, +{final_val-old_best:.3f}) "
                            f"IC={ic_i:.4f} Exp={exposure:.1%} | "
                            f"{fml}\n    {self._decode_formula(fml)}"
                        )
                self._update_elite_pool(final_val, fml, step)


            # ── Part D: REINFORCE gradient update ────────────────────
            rew_std = rewards.std().clamp(min=0.1)
            adv     = (rewards - rewards.mean()) / (rew_std + 1e-5)
            adv_new   = adv[:n_new]
            adv_elite = adv[n_new:]

            policy_loss = torch.zeros(1, device=ModelConfig.DEVICE)
            for ti in range(len(lp_new)):
                policy_loss += (-lp_new[ti] * adv_new).mean()
            if lp_elite and adv_elite.shape[0] > 0:
                for ti in range(len(lp_elite)):
                    lpe = lp_elite[ti]
                    if lpe.shape[0] == adv_elite.shape[0]:
                        policy_loss += (-lpe * adv_elite
                                        * ModelConfig.ELITE_REWARD_SCALE).mean()

            if ent_new:
                mean_ent_new = torch.stack(ent_new).mean()
            else:
                mean_ent_new = torch.zeros(1, device=ModelConfig.DEVICE)
            if ent_elite:
                mean_ent_elite = torch.stack(ent_elite).mean()
                mean_ent = (
                    mean_ent_new * n_new + mean_ent_elite * n_elite
                ) / (n_new + n_elite)
            else:
                mean_ent = mean_ent_new
            ent_val   = mean_ent.item()
            ent_coeff = ModelConfig.ENTROPY_COEFF_MAX / (
                (1.0 + ent_val) ** ModelConfig.ENTROPY_COEFF_POWER
            )
            loss = policy_loss - ent_coeff * mean_ent

            self.opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.opt.step()
            if self.use_lord:
                self.lord_opt.step()

            # ── Part E: Logging & history & checkpoint ───────────────
            avg_rew = rewards.mean().item()
            avg_val = val_scores.mean().item()
            bim  = sum(bic)  / len(bic)  if bic  else 0.0
            bis_ = sum(bis)  / len(bis)  if bis  else 0.0
            bsor_= sum(bsor) / len(bsor) if bsor else 0.0

            self._stagnation_steps = step - self._best_update_step
            tqdm.write(
                f"[{step+1}/{end_step}] "
                f"new={n_new} elite={n_elite} | "
                f"ok={ok_cnt} none={none_cnt} const={const_cnt} | "
                f"Rew={avg_rew:.3f} Val={avg_val:.3f} | "
                f"IC={bim:.4f} | H={ent_val:.3f}(c={ent_coeff:.3f}) | "
                f"Best={self.best_score:.3f} stagnation={self._stagnation_steps} "
                f"epool={len(self._elite_pool)} restarts={self._restart_count}"
            )
            pbar.set_postfix({
                'Val': f"{avg_val:.3f}", 'Best': f"{self.best_score:.3f}",
                'H':   f"{ent_val:.2f}", 'IC':   f"{bim:.4f}",
                'stag': f"{self._stagnation_steps}",
            })

            if self.use_lord and step % 10 == 0:
                sr = self.rank_monitor.compute()
                self.training_history['stable_rank'].append(sr)

            self.training_history['step'].append(step)
            self.training_history['avg_reward'].append(avg_rew)
            self.training_history['val_score'].append(avg_val)
            self.training_history['best_score'].append(self.best_score)
            self.training_history.setdefault('entropy', []).append(ent_val)
            self.training_history.setdefault('ic_mean', []).append(bim)
            self.training_history.setdefault('ic_stability', []).append(bis_)
            self.training_history.setdefault('sortino', []).append(bsor_)
            self.training_history.setdefault('elite_pool_size', []).append(
                len(self._elite_pool))

            if self.best_formula is not None:
                from .vocab import VOCAB_VERSION
                strategy_data = {
                    "vocab_version": VOCAB_VERSION,
                    "symbol": self.target_symbol,
                    "formula": self.best_formula,
                    "best_score": self.best_score,
                }
                save_path = _strategy_file_for_symbol(self.target_symbol)
                pathlib.Path(save_path).parent.mkdir(parents=True, exist_ok=True)
                with open(save_path, "w") as fp:
                    json.dump(strategy_data, fp, indent=2)

            if (step + 1) % 20 == 0 or (step + 1) == end_step:
                ckpt = self.save_checkpoint(step + 1)
                tqdm.write(f"[Checkpoint] → {ckpt} (best={self.best_score:.3f})")

            # ── Part F: Migration hook（多岛训练时交换精英）────────────
            if migration_hook is not None and (step + 1) % ModelConfig.MIGRATION_INTERVAL == 0:
                tqdm.write(f"[MigrationHook @ step {step+1}] calling registered hook")
                migration_hook(self, step + 1)

            # ── Part G: Entropy collapse detection & restart ─────────
            if ent_val < ModelConfig.ENTROPY_COLLAPSE_THRESH:
                low_entropy_streak += 1
            else:
                low_entropy_streak  = 0

            if low_entropy_streak >= ModelConfig.ENTROPY_COLLAPSE_STEPS:
                # ── 自适应噪声：根据 stagnation 调整 ─────────────────────
                self._stagnation_steps = step - self._best_update_step
                stagnation_ratio = self._stagnation_steps / max(1, ModelConfig.STAGNATION_WINDOW)
                base_noise = ModelConfig.RESTART_NOISE
                if ModelConfig.ADAPTIVE_NOISE:
                    raw_noise = base_noise + ModelConfig.NOISE_BOOST_FACTOR * 0.1 * min(stagnation_ratio, 3.0)
                    noise = max(ModelConfig.NOISE_MIN, min(ModelConfig.NOISE_MAX, raw_noise))
                else:
                    noise = base_noise

                max_r = ModelConfig.MAX_RESTARTS
                if self._restart_count < max_r:
                    self._restart_count  += 1
                    low_entropy_streak    = 0
                    if self._best_snapshot is not None:
                        self.model.load_state_dict(self._best_snapshot)
                        with torch.no_grad():
                            if ModelConfig.PARTIAL_RESET:
                                perturbed_layers = []
                                for nm, p in self.model.named_parameters():
                                    if any(k in nm for k in ModelConfig.PARTIAL_RESET_LAYERS):
                                        p.add_(torch.randn_like(p) * noise)
                                        perturbed_layers.append(nm)
                            else:
                                perturbed_layers = []
                                for nm, p in self.model.named_parameters():
                                    if 'ffn' in nm or 'attention' in nm or nm.startswith('blocks'):
                                        p.add_(torch.randn_like(p) * noise)
                                        perturbed_layers.append(nm)
                        tqdm.write(
                            f"[Restart {self._restart_count}/{max_r} @ step {step}] "
                            f"mode={'partial' if ModelConfig.PARTIAL_RESET else 'FFN/Attn'} "
                            f"noise={noise:.4f}(base={base_noise:.3f}, ratio={stagnation_ratio:.2f}) "
                            f"stagnation={self._stagnation_steps} "
                            f"H={ent_val:.3f} "
                            f"perturbed_layers={len(perturbed_layers)}"
                        )
                    else:
                        with torch.no_grad():
                            for p in self.model.parameters():
                                p.add_(torch.randn_like(p) * noise)
                        tqdm.write(
                            f"[Restart {self._restart_count}/{max_r} @ step {step}] "
                            f"mode=full_param "
                            f"noise={noise:.4f}(base={base_noise:.3f}, ratio={stagnation_ratio:.2f}) "
                            f"stagnation={self._stagnation_steps} "
                            f"H={ent_val:.3f} | no best snapshot"
                        )
                    self.opt = torch.optim.AdamW(self.model.parameters(), lr=1e-3)
                else:
                    # 训练时间不敏感：超过重启上限后不再 Early Stop 终止，
                    # 改为「全参数强扰动 + 重置流计数」继续探索，直到跑满 TRAIN_STEPS。
                    # 从 best_snapshot 恢复（若有）以保住已发现的最优结构，再加大扰动。
                    low_entropy_streak = 0
                    hard_noise = min(ModelConfig.NOISE_MAX, noise * 2.0)
                    if self._best_snapshot is not None:
                        self.model.load_state_dict(self._best_snapshot)
                    with torch.no_grad():
                        for p in self.model.parameters():
                            p.add_(torch.randn_like(p) * hard_noise)
                    self.opt = torch.optim.AdamW(self.model.parameters(), lr=1e-3)
                    tqdm.write(
                        f"[Hard Restart @ step {step}] max_reached={max_r} "
                        f"H={ent_val:.3f} hard_noise={hard_noise:.4f} "
                        f"continue training without early stop"
                    )

        # ── End of training ──────────────────────────────────────────
        # 仅当跑满最终步时才保存最终 strategy 和历史
        if end_step == ModelConfig.TRAIN_STEPS:
            if self.best_formula is not None:
                from .vocab import VOCAB_VERSION
                strategy_data = {
                    "vocab_version": VOCAB_VERSION,
                    "symbol": self.target_symbol,
                    "formula": self.best_formula,
                    "best_score": self.best_score,
                }
                save_path = _strategy_file_for_symbol(self.target_symbol)
                pathlib.Path(save_path).parent.mkdir(parents=True, exist_ok=True)
                with open(save_path, "w") as fp:
                    json.dump(strategy_data, fp, indent=2)

            sym_tag = f"[{self.target_symbol}] " if self.target_symbol else ""
            self.training_history.pop('_low_entropy_streak', None)
            hist_path = (
                f"training_history_{self.target_symbol}.json"
                if self.target_symbol else "training_history.json"
            )
            with open(hist_path, "w") as fp:
                json.dump(self.training_history, fp)

            print(f"\n✓ {sym_tag}Training completed!")
            print(f"  Best val score : {self.best_score:.4f}")
            print(f"  Best formula   : {self.best_formula}")
            print(f"  Readable       : {self._decode_formula(self.best_formula)}")
            print(f"  Elite pool     : {len(self._elite_pool)}")
            print(f"  Elite decay    : enabled={ModelConfig.ELITE_DECAY}, half_life={ModelConfig.ELITE_DECAY_HALF_LIFE}")
            print(f"  Adaptive noise : enabled={ModelConfig.ADAPTIVE_NOISE}, range=[{ModelConfig.NOISE_MIN}, {ModelConfig.NOISE_MAX}]")
            print(f"  Partial reset  : enabled={ModelConfig.PARTIAL_RESET}, layers={ModelConfig.PARTIAL_RESET_LAYERS}")
            print(f"  Restarts       : {self._restart_count}")
            print(f"  Strategy saved : {save_path}")


    # ── 实时保存最优公式（防进程意外退出丢失）────────────────────────────────
    def _save_strategy_live(self) -> None:
        """每次 best_formula 更新时立即保存 strategy json。
        即使训练中途进程被杀（OOM/终端回收/Ctrl+C），也能保留最新最优公式。
        """
        if self.best_formula is None:
            return
        try:
            from .vocab import VOCAB_VERSION
            strategy_data = {
                "vocab_version": VOCAB_VERSION,
                "symbol":        self.target_symbol,
                "formula":       self.best_formula,
                "best_score":    self.best_score,
            }
            save_path = _strategy_file_for_symbol(self.target_symbol)
            pathlib.Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, "w") as fp:
                json.dump(strategy_data, fp, indent=2)
        except Exception:
            pass

    # ── Checkpoint save / load ────────────────────────────────────────────────

    def save_checkpoint(self, step: int, path: str | None = None) -> str:
        _CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        if path is None:
            sym_tag = f"_{self.target_symbol}" if self.target_symbol else ""
            path = str(_CHECKPOINT_DIR / f"ckpt{sym_tag}_step_{step:04d}.pt")
        ckpt = {
            "step":                 step,
            "vocab_version":        VOCAB_VERSION,   # task 12.2: 版本校验所需
            "model_state_dict":     self.model.state_dict(),
            "optimizer_state_dict": self.opt.state_dict(),
            "best_score":           self.best_score,
            "best_formula":         self.best_formula,
            "best_snapshot":        self._best_snapshot,
            "factor_pool":          self.factor_pool,
            "factor_pool_counter":  self._factor_pool_counter,
            "elite_pool":           self._elite_pool,
            "elite_counter":        self._elite_counter,
            "restart_count":        self._restart_count,
            "training_history":     {
                k: v for k, v in self.training_history.items()
                if k != '_low_entropy_streak'
            },
        }
        torch.save(ckpt, path)
        return path

    def load_checkpoint(self, path: str) -> int:
        ckpt = torch.load(path, map_location=ModelConfig.DEVICE)

        # ── Task 12.2：版本校验（R3.7）──────────────────────────────────────
        # 从 checkpoint 读取 vocab_version；若字段缺失（旧版 checkpoint），视为
        # 版本不匹配并抛错——拒绝加载、不消费任何 token。
        artifact_version = ckpt.get("vocab_version")
        if artifact_version is None:
            raise VocabVersionMismatchError(
                f"checkpoint '{path}' 不含 vocab_version 字段（旧版产物），"
                f"当前词表版本 {FORMULA_VOCAB.version!r}；需重新训练后加载"
            )
        # verify() 版本不匹配时抛 VocabVersionMismatchError，拒绝加载
        FORMULA_VOCAB.verify(artifact_version)
        # ── 版本校验通过，继续加载 ────────────────────────────────────────

        self.model.load_state_dict(ckpt["model_state_dict"])
        self.opt.load_state_dict(ckpt["optimizer_state_dict"])
        self.best_score          = ckpt.get("best_score",  -float('inf'))
        self.best_formula        = ckpt.get("best_formula", None)
        self._best_snapshot      = ckpt.get("best_snapshot", None)
        self.factor_pool         = ckpt.get("factor_pool", [])
        self._factor_pool_counter = ckpt.get("factor_pool_counter", 0)
        self._elite_pool         = ckpt.get("elite_pool", [])
        self._elite_counter      = ckpt.get("elite_counter", 0)
        self._restart_count      = ckpt.get("restart_count", 0)
        for k, v in ckpt.get("training_history", {}).items():
            self.training_history[k] = v

        # 清理 elite pool 中的重复条目（保留各公式的最高分版本）
        self._elite_pool = self._dedup_elite_pool(self._elite_pool)

        completed = ckpt.get("step", 0)
        tqdm.write(f"[Checkpoint] 已从 {path} 恢复。"
                   f" step={completed}  best={self.best_score:.4f}"
                   f"  elite={len(self._elite_pool)}(去重后)")
        return completed

    # ── Decode formula tokens to readable string ──────────────────────────────

    def _decode_formula(self, tokens: list[int] | None) -> str:
        if tokens is None:
            return "N/A"
        from .vocab import FORMULA_VOCAB
        names = FORMULA_VOCAB.token_names
        return " -> ".join(names[t] if 0 <= t < len(names) else f"?{t}"
                           for t in tokens)
