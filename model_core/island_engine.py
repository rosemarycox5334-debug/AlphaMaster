"""
model_core/island_engine.py — 多起点并行训练（Island Model）

同时维护 N 个独立的 AlphaEngine（island），每个 island 独立探索不同的
公式空间区域。每隔 migration_interval 步，把所有 island 的 elite 公式
汇总，取 Top-K 注入到其他 island 的 elite pool 中，实现"精英迁移"。

注意：这里的"并行"是算法层面的多群体演化，不是 Python multiprocessing。
CPU 训练下串行轮流训练每个 island 一个小阶段效率更高，且输出不混乱。
"""
import copy
import heapq
import json
from pathlib import Path

import torch

from .config import ModelConfig
from .engine import AlphaEngine


class IslandAlphaEngine:
    """管理多个 AlphaEngine 组成 island population。"""

    def __init__(self, data_manager, n_islands: int | None = None,
                 migration_interval: int | None = None,
                 migration_top_k: int | None = None):
        self.data_manager = data_manager
        self.n_islands = n_islands or ModelConfig.N_ISLANDS
        self.migration_interval = migration_interval or ModelConfig.MIGRATION_INTERVAL
        self.migration_top_k = migration_top_k or ModelConfig.MIGRATION_TOP_K

        self.islands: list[AlphaEngine] = []
        for i in range(self.n_islands):
            isl = AlphaEngine(data_manager=data_manager)
            # 给每个 island 不同的随机初始化，增加多样性
            torch.manual_seed(2026 + i * 17)
            isl.model = isl.model.__class__().to(ModelConfig.DEVICE)
            isl.opt = torch.optim.AdamW(isl.model.parameters(), lr=1e-3)
            self.islands.append(isl)

        self.global_best_score = -float('inf')
        self.global_best_formula = None
        self.global_best_island = -1
        self._step = 0

    def _migrate_elites(self, step: int):
        """在所有 islands 之间交换 Top-K elite 公式。"""
        # 收集所有 island 的 elite
        all_elites = []
        for isl in self.islands:
            all_elites.extend(isl._elite_pool)

        if len(all_elites) < 2:
            return

        # 去重：相同公式保留最高分和最新 birth_step
        best_by_formula = {}
        for sc, cnt, toks, birth in all_elites:
            key = str(toks)
            if key not in best_by_formula or sc > best_by_formula[key][0]:
                best_by_formula[key] = (sc, cnt, toks, birth)

        # 按得分排序取 Top-K
        sorted_elites = sorted(
            best_by_formula.values(), key=lambda x: x[0], reverse=True
        )
        top_elites = sorted_elites[:self.migration_top_k]

        # 注入到每个 island（替换低分 elite）
        injected = 0
        for isl in self.islands:
            for sc, cnt, toks, birth in top_elites:
                # 避免注入 island 已存在的公式（_update_elite_pool 会处理去重）
                isl._update_elite_pool(sc, list(toks), step)
                injected += 1

        print(f"\n[Island Migration @ step {step}] "
              f"collected {len(all_elites)} elites, "
              f"deduped to {len(best_by_formula)}, "
              f"injected {injected} top elites across {self.n_islands} islands\n")

    def _update_global_best(self):
        """从所有 island 中更新全局最优。"""
        for i, isl in enumerate(self.islands):
            if isl.best_score > self.global_best_score:
                self.global_best_score = isl.best_score
                self.global_best_formula = isl.best_formula
                self.global_best_island = i

    def train(self):
        """主训练循环：每个 island 轮流训练一个阶段，然后迁移 elite。"""
        total_steps = ModelConfig.TRAIN_STEPS
        n_phases = total_steps // self.migration_interval
        if n_phases == 0:
            n_phases = 1

        print(f"\n{'='*60}")
        print(f"  Island Alpha Training")
        print(f"  islands={self.n_islands}  migration_every={self.migration_interval}")
        print(f"  total_steps={total_steps}  phases={n_phases}")
        print(f"{'='*60}\n")

        for phase in range(n_phases):
            start = phase * self.migration_interval
            end = min((phase + 1) * self.migration_interval, total_steps)

            for i, isl in enumerate(self.islands):
                print(f"\n>>> Phase {phase+1}/{n_phases} — Island {i+1}/{self.n_islands} "
                      f"steps [{start}:{end}]")
                # 每个 island 独立训练一个阶段
                isl.train(start_step=start, end_step=end,
                          migration_hook=None, verbose_header=False)
                self._update_global_best()

            # 阶段结束：迁移 elite
            self._migrate_elites(end)

            # 同步全局最优到每个 island 的 best_snapshot
            # 这样下次 restart 时可以从全局最优恢复，而非局部最优
            for isl in self.islands:
                if self.global_best_score > isl.best_score:
                    isl.best_score = self.global_best_score
                    isl.best_formula = copy.deepcopy(self.global_best_formula)

        # 最终保存全局最优
        self._update_global_best()
        if self.global_best_formula is not None:
            from .vocab import VOCAB_VERSION
            strategy_data = {
                "vocab_version": VOCAB_VERSION,
                "formula": self.global_best_formula,
                "best_score": self.global_best_score,
                "island_engine": True,
                "n_islands": self.n_islands,
            }
            save_path = Path("strategies") / "best_island_strategy.json"
            save_path.parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, "w") as fp:
                json.dump(strategy_data, fp, indent=2)
            print(f"\n✓ Island training completed!")
            print(f"  Global best score : {self.global_best_score:.4f}")
            print(f"  From island       : {self.global_best_island + 1}")
            print(f"  Formula           : {self.global_best_formula}")
            sample_island = self.islands[self.global_best_island] if self.global_best_island >= 0 else self.islands[0]
            readable = sample_island._decode_formula(self.global_best_formula)
            print(f"  Readable          : {readable}")
            print(f"  Saved to          : {save_path}")

    def get_global_best(self):
        return self.global_best_formula, self.global_best_score
