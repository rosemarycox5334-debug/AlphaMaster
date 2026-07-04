"""
model_core/config.py — 模型层配置

仅保留模型训练所需的参数。
品种、数据、风控等全局配置统一由根目录 config.py 的 Config 类管理。
"""
import math

import torch
from .vocab import FORMULA_VOCAB


class ModelConfig:
    # ── 训练设备 ─────────────────────────────────────────────────────────
    # 注意：本任务 CPU 训练速度反而比 GPU 快（实测约 2.3 倍），故强制用 CPU。
    # 原因：
    #   1. 张量太小——forex 组仅 (2 品种 × 3508 × 20 特征)，单个算子的
    #      计算量小于 CUDA kernel 启动开销（数十微秒），GPU 算得快但启动慢。
    #   2. 训练循环是 Python 串行调度：每 step 逐条跑 128 条公式 × 8 个
    #      VM 步 × 4 个 walk-forward 折，GPU 被切成上万个碎片段，吃不满。
    #   3. host↔device 拷贝 + kernel 启动延迟主导总耗时，而非张量计算本身。
    #   4. 实测 GPU 利用率 ~51%，正是 GPU 一半时间在干等 Python 喂下一个
    #      kernel 的证据（不是“还能压榨”，而是“调度瓶颈”）。
    # 基准测试（forex 组, 50 步, RTX 4060, 2026-07-03）：
    #   cuda: 4.48 s/步  Best=4.875
    #   cpu : 1.91 s/步  Best=5.103
    #   加速比 = 0.43x（GPU 反而慢 2.3 倍）
    # 若后续改为批量并行公式评估（一次喂大批张量进 GPU），再切回 cuda。
    DEVICE = torch.device("cpu")

    # ── 训练参数（大搜索空间适配版，2026-07-04 重构）─────────────────────
    # 背景：特征库扩展到 65、算子库扩展到 66（vocab=131），8-token 搜索空间
    #   从旧版 ~7亿 暴增到 ~8.67×10^16（1.2 亿倍）。旧的采样预算（128×3000）
    #   覆盖率趋近于零，导致熵坍塌 Early Stop、公式退化。
    # 对策（训练时间不敏感场景）：
    #   1. 特征剪枝（active_features.json）把 vocab 降到 ~90，空间缩小约 20 倍
    #   2. 放大采样预算：BATCH_SIZE 128→256，TRAIN_STEPS 3000→8000
    #   3. 更大精英池（60）保留更多历史最优
    BATCH_SIZE      = 192   # 每步采样公式数（原 128，1.5x 提升覆盖率）
    TRAIN_STEPS     = 5000  # 每组训练步数（原 3000）
    MAX_FORMULA_LEN = 8     # 公式长度上限（暂保持 8，防过拟合）

    # ── 特征维度（由 vocab.py 自动派生，无需手动修改）──────────────────
    INPUT_DIM: int = FORMULA_VOCAB.feature_count  # == 10

    # ── Reward：Sortino 为主，IC 做门控 ──────────────────────────────────
    # IC_NEG_MULT 0.30→0.50：0.30 对反向因子惩罚过重，可能误杀非线性高收益因子。
    # 收益优先模式下，只要年化收益是正的，适当负 IC 可以接受。
    REWARD_ALPHA:      float = 1.0
    IC_GATE_THRESH:    float = 0.01
    IC_GATE_MULT:      float = 1.15
    IC_NEG_MULT:       float = 0.50   # 0.30 → 0.50

    # ── 熵保护（大空间加强版）──────────────────────────────────────────
    # ENTROPY_COEFF_MAX 0.5→1.0：加倍探索压力，对抗大 vocab 的过早收敛。
    # ENTROPY_COLLAPSE_THRESH 改为相对阈值 0.15×ln(vocab)：大 vocab 最大熵更高
    #   （ln(131)≈4.87 vs ln(54)≈3.99），绝对阈值 0.5 不再合理。
    # ENTROPY_COLLAPSE_STEPS 15→40：给模型更长的自我恢复窗口，不急于重启。
    ENTROPY_COEFF_MAX:   float = 1.0
    ENTROPY_COEFF_POWER: float = 1.3
    ENTROPY_COLLAPSE_THRESH: float = 0.15 * math.log(FORMULA_VOCAB.size)
    ENTROPY_COLLAPSE_STEPS:  int   = 40

    # ── Elite Replay ──────────────────────────────────────────────────
    ELITE_REPLAY_FRAC:  float = 0.25
    ELITE_POOL_SIZE:    int   = 60    # 30→60：大空间需要更大的精英记忆
    ELITE_REWARD_SCALE: float = 1.2

    # ── 坍塌重启（大空间加强版）─────────────────────────────────────────
    # MAX_RESTARTS 8→25、RESTART_NOISE 0.05→0.1：时间不敏感，多给机会+更强扰动。
    # 配合 engine.py：超过 MAX_RESTARTS 后不再 Early Stop，改为强扰动继续训练。
    MAX_RESTARTS:   int   = 25
    RESTART_NOISE:  float = 0.1

    # ── 因子去相关参数 ────────────────────────────────────────────────
    FACTOR_TOP_K:     int   = 10
    CORR_THRESHOLD:   float = 0.7
    CORR_PENALTY:     float = 0.5

    # ── Walk-Forward Gap ───────────────────────────────────────────────
    WF_GAP: int = 20
