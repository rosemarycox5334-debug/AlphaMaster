import torch
from .ops import OPS_CONFIG
from .vocab import FORMULA_VOCAB

# ── 恒正算子集 ────────────────────────────────────────────────────────────
# 这些算子输出值域非负（或几乎恒正），连续使用会丢失符号信息，
# 导致因子退化成「永远做多」的 beta 因子。
# TS_RANK_*: 输出 [0, 1)，永远非负
# ABS: 取绝对值，丢失符号
# TS_SUM_*: 本身不恒正，但如果输入已经非负则输出更正
# TS_MAX_*: 取最大值，如果输入含正数则偏向正
# 我们用「感染性算子」概念：一旦前面出现了恒正算子，后续的 TS_SUM/TS_MEAN/TS_MAX
# 都不会让值域变回有正有负，反而会强化正值。只有 SUB/NEG/DIV/TS_ZSCORE/CS_NEUTRALIZE
# 等算子才能「恢复」符号信息。
POSITIVE_ONLY_OPS = {"TS_RANK_5", "TS_RANK_10", "TS_RANK_20", "ABS"}
# 感染传播算子：在恒正值域上使用时，输出仍为恒正
# TS_SUM_*: 非负值的和仍非负
# TS_MEAN_*: 非负值的均值仍非负
# TS_MAX_*: 非负值的最大值仍非负
# CLIP: clamp(-3,3) 不改变符号，但如果输入全非负则输出也全非负
# SQRT: sign(x)*sqrt(|x|)，如果输入全非负则输出全非负
# POWER: sign(x)*|x|^2，如果输入全非负则输出全非负
# SIGNED_LOG: sign(x)*log1p(|x|)，如果输入全非负则输出全非负
# SIGMOID: 2*sigmoid-1，对非负输入输出正值
# TANH_SQUASH: tanh，对非负输入输出正值
INFECTED_PROPAGATING_OPS = {
    "TS_RANK_5", "TS_RANK_10", "TS_RANK_20", "ABS",
    "TS_SUM_5", "TS_SUM_10", "TS_SUM_20",
    "TS_MEAN_5", "TS_MEAN_10", "TS_MEAN_20",
    "TS_MAX_10", "TS_MAX_20",
    "CLIP", "SQRT", "POWER", "SIGNED_LOG",
    "SIGMOID", "TANH_SQUASH", "WINSORIZE",
    "WMA", "EMA_5", "EMA_20", "DECAY", "DECAY_LINEAR_5",
    "TS_DECAY_EXP_5",
}
# 恢复算子：能够把恒正值域重新变成有正有负
SIGN_RESTORE_OPS = {
    "SUB", "DIV", "NEG", "GATE", "IF_GT",
    "TS_ZSCORE_10", "TS_ZSCORE_20",
    "CS_NEUTRALIZE", "CS_RANK", "CS_SCALE",
    "TS_STD_5", "TS_STD_10", "TS_STD_20",
    "TS_CORR_10", "TS_SKEW_10", "TS_QUANTILE_10",
    "DELTA", "DELTA_5", "MOMENTUM_5", "MOMENTUM_10",
    "PPO",  # 但 PPO 是特征不是算子
}


def is_positive_only_op(token_name: str) -> bool:
    """判断算子是否输出恒正值（可能丢失符号信息）。"""
    return token_name in POSITIVE_ONLY_OPS


def is_infected_propagating(token_name: str) -> bool:
    """判断算子是否会传播恒正感染（在恒正输入上输出仍恒正）。"""
    return token_name in INFECTED_PROPAGATING_OPS


def is_sign_restoring(token_name: str) -> bool:
    """判断算子是否能恢复符号信息（把恒正值域变回有正有负）。"""
    return token_name in SIGN_RESTORE_OPS


def validate_formula_structure(formula_tokens: list[int], vocab_names: tuple[str, ...]) -> list[str]:
    """校验公式结构，返回违规原因列表（空列表 = 合法）。
    
    使用「感染模型」：一旦公式中出现恒正算子（如 TS_RANK），
    后续如果连续使用传播算子（如 TS_SUM/TS_MEAN/CLIP/SQRT），
    值域会一直保持非负，导致因子退化成 beta。
    只有恢复算子（如 SUB/TS_ZSCORE/CS_NEUTRALIZE）才能打破感染。
    
    规则：
    1. 禁止恒正算子后连续 2 个以上传播算子（感染链太长）
    2. 公式末尾如果是感染状态（恒正且未恢复），标记违规
    """
    violations = []
    feat_offset = FORMULA_VOCAB.operator_offset
    
    infected = False  # 当前值域是否已被感染（恒正）
    infected_chain_len = 0  # 感染链长度
    last_positive_op = None  # 最后一个引发感染的算子名
    
    for i, token in enumerate(formula_tokens):
        token = int(token)
        if token < feat_offset:
            # 特征 token：不改变感染状态
            continue
        name = vocab_names[token] if token < len(vocab_names) else f"op_{token}"
        
        if is_positive_only_op(name):
            # 恒正算子：开始/延续感染
            if not infected:
                infected = True
                last_positive_op = name
            infected_chain_len += 1
            
        elif infected and is_sign_restoring(name):
            # 恢复算子：打破感染
            infected = False
            infected_chain_len = 0
            last_positive_op = None
            
        elif infected and is_infected_propagating(name):
            # 传播算子：感染继续
            infected_chain_len += 1
            # 规则1：感染链超过 3 个算子时报警
            if infected_chain_len >= 3:
                violations.append(
                    f"步骤{i}: 恒正感染链过长（从 {last_positive_op} 起 {infected_chain_len} 个传播算子），"
                    f"因子将退化为 beta"
                )
        # else: 非感染相关算子（如 ADD/MUL），不改变感染状态
    
    # 规则2：公式末尾处于感染状态
    if infected and infected_chain_len >= 2:
        violations.append(
            f"公式末尾处于恒正感染状态（链长 {infected_chain_len}），"
            f"因子输出将偏向单方向"
        )
    
    return violations

# ── 扩展后词表规模说明（task 12.1）──────────────────────────────────────────
#
# 本次扩展（factor-operator-library-expansion）后：
#   - 特征数 F  = len(FORMULA_VOCAB.feature_names)  （当前 65，覆盖 8 大类）
#   - 算子数 O  = len(OPS_CONFIG)                    （当前 66）
#   - 词表总 size = F + O = 131
#   - feat_offset = F = 65（feature token id ∈ [0, 64]）
#   - operator token id ∈ [F, F+O-1] = [65, 130]
#
# StackVM 的 op_map / arity_map **完全动态**从 FORMULA_VOCAB 与 OPS_CONFIG 派生，
# 不硬编码任何 token 数或偏移值，因此无需在此处做任何结构变更。后续再次扩展
# 特征或算子时只需更新注册表，VM 自动消费。
#
# Cross-sectional 算子（CS_RANK / CS_SCALE / CS_NEUTRALIZE）沿 N 维逐时间步
# 操作，输入/输出形状均为 [N, T]，满足统一的 [N,T]→[N,T] 契约（R2.9）；
# VM 主循环无需对它们做任何特殊处理。


class StackVM:
    def __init__(self):
        # feat_offset 动态从 FORMULA_VOCAB.operator_offset 读取（= feature_count = F）。
        self.feat_offset = FORMULA_VOCAB.operator_offset
        # op_map / arity_map 动态从 OPS_CONFIG 构建。
        self.op_map = {i + self.feat_offset: cfg[1] for i, cfg in enumerate(OPS_CONFIG)}
        self.arity_map = {i + self.feat_offset: cfg[2] for i, cfg in enumerate(OPS_CONFIG)}
        # 恒正算子 token id 集合（用于采样时约束）
        self.positive_only_ids = set()
        for i, cfg in enumerate(OPS_CONFIG):
            if cfg[0] in POSITIVE_ONLY_OPS:
                self.positive_only_ids.add(i + self.feat_offset)

    @staticmethod
    def _normalize_output(x: torch.Tensor) -> torch.Tensor:
        """
        对因子输出做标准化，确保幅度足够触发 neutral band 入场。

        策略（三级降级）：
        1. 截面 zscore（跨品种，每时间步）：适合因子跨品种有分散
        2. 时序 zscore（每品种，全局）：当截面 std 太小时使用
        3. 若两级都失败（因子是常数）：返回原值，由 const_cnt 拦截

        Returns:
            [N, T] clip 到 [-3, 3]，若是常数则返回原值（engine 会过滤）
        """
        N, T = x.shape

        # 检测是否是全局常数（标准化无意义）
        global_std = x.std()
        if global_std < 1e-6:
            return x   # 常数因子，由 engine 的 const_cnt 拦截

        # ── 截面标准化（跨品种，每时间步；N=1 时跳过）──────────────
        if N > 1:
            cs_mean = x.mean(dim=0, keepdim=True)
            cs_std  = x.std(dim=0, keepdim=True).clamp(min=1e-8)
            cs_z    = (x - cs_mean) / cs_std
            if cs_z.std() >= 0.3:
                return torch.clamp(cs_z, -3.0, 3.0)

        # ── 时序标准化（每品种独立）─────────────────────────────────
        ts_mean = x.mean(dim=1, keepdim=True)
        ts_std  = x.std(dim=1, keepdim=True).clamp(min=1e-8)
        ts_z    = (x - ts_mean) / ts_std

        if ts_z.std() >= 0.1:
            return torch.clamp(ts_z, -3.0, 3.0)

        # ── 两级均失败：因子无区分度，返回原值让 engine 过滤 ────────
        return x

    def execute(self, formula_tokens, feat_tensor):
        stack = []
        try:
            for token in formula_tokens:
                token = int(token)
                if token < self.feat_offset:
                    if token >= feat_tensor.shape[1]:
                        return None
                    stack.append(feat_tensor[:, token, :])
                elif token in self.op_map:
                    arity = self.arity_map[token]
                    if len(stack) < arity: return None
                    args = []
                    for _ in range(arity):
                        args.append(stack.pop())
                    args.reverse()
                    func = self.op_map[token]
                    res = func(*args)
                    if torch.isnan(res).any() or torch.isinf(res).any():
                        res = torch.nan_to_num(res, nan=0.0, posinf=1.0, neginf=-1.0)
                    stack.append(res)
                else:
                    return None
            if len(stack) == 1:
                result = stack[0]
                # 最终输出标准化：保证因子幅度足够，避免全程空仓
                return self._normalize_output(result)
            else:
                return None
        except Exception:
            return None
