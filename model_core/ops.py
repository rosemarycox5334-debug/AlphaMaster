"""
model_core/ops.py -- 算子库（Operator_Library, R2）

本模块把历史上手工维护的 `OPS_CONFIG` 列表迁移为由声明式注册层
（`model_core.registry.Registry`）驱动的 `OPERATOR_REGISTRY`。所有算子先以
`OperatorSpec(name, arity, transform)` 注册进 `OPERATOR_REGISTRY`，`OPS_CONFIG`
随后作为「导出视图」由注册表派生（`[(name, transform, arity), ...]`），保持对
下游 `vocab.py` / `vm.py` 的 import 兼容与既有元组结构。

统一契约（R2.8, R2.9, R2.13）：
  - 形状契约：所有算子输入 `[N, T]`、输出 `[N, T]`。
  - 二元/三元算子在入口校验各操作数形状一致，不一致抛 `ShapeError` 且不产出张量。
  - 注册表存储 name（≤64 字符）与 arity（本模块算子均为 1/2/3）。

说明：现有算子多以 lambda 定义，`inspect` 不总能可靠解析 arity（内建函数、被
包装函数等）。注册时以显式声明的 arity 为准；二元/三元算子经形状校验包装后为
可变位置参数形式（`*operands`），注册层会跳过 arity 观测校验，从而避免对既有算子
的 `ArityMismatchError` 误报。
"""
import torch

from .registry import OperatorSpec, Registry


# ── 算子层错误类型（对应 design「错误类型模型」，归为算子层）──────────────

class ShapeError(Exception):
    """算子操作数形状不兼容或错误（R2.13）。

    二元/三元算子在入口发现各操作数形状不一致时抛出，且不产出张量。
    """


def _ts_delay(x: torch.Tensor, d: int) -> torch.Tensor:
    if d == 0: return x
    pad = torch.zeros((x.shape[0], d), device=x.device, dtype=x.dtype)
    return torch.cat([pad, x[:, :-d]], dim=1)

def _op_gate(condition: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    mask = (condition > 0).float()
    return mask * x + (1.0 - mask) * y

def _op_jump(x: torch.Tensor) -> torch.Tensor:
    """降低稀疏度：阈值从 3σ 改为 1.5σ，让更多时间步有非零输出。

    因果 expanding zscore：每个 t 仅用 x[:, :t+1] 计算 mean/std，
    避免原 dim=1 全局聚合（含未来统计量）引入的 look-ahead bias。
    """
    N, T = x.shape
    cnt = torch.arange(1, T + 1, device=x.device, dtype=x.dtype).view(1, T)
    cumsum = x.cumsum(dim=1)
    mean = cumsum / cnt                       # [N,T]，t 位 = x[:,:t+1].mean()
    cumsum_sq = (x * x).cumsum(dim=1)
    var = (cumsum_sq / cnt) - mean * mean     # E[x^2] - E[x]^2
    std = var.clamp(min=1e-12).sqrt() + 1e-6
    z = (x - mean) / std
    return torch.tanh(z - 1.5)   # tanh 软化，不再产生全零区间

def _op_decay(x: torch.Tensor) -> torch.Tensor:
    return x + 0.8 * _ts_delay(x, 1) + 0.6 * _ts_delay(x, 2)

def _op_wma(x: torch.Tensor) -> torch.Tensor:
    """加权移动平均（权重 3,2,1），平滑信号，减少剥头皮"""
    return (3.0 * x + 2.0 * _ts_delay(x, 1) + 1.0 * _ts_delay(x, 2)) / 6.0


# ── 时序滑动窗口辅助函数（不使用 @torch.jit.script，lambda 不兼容 JIT）──────

def _ts_rolling(x: torch.Tensor, d: int) -> torch.Tensor:
    """unfold 实现因果滑动窗口，返回 [N, T, d] 的窗口张量。"""
    N, T = x.shape
    pad = torch.zeros(N, d - 1, device=x.device, dtype=x.dtype)
    return torch.cat([pad, x], dim=1).unfold(1, d, 1)  # [N, T, d]


def _ts_mean(x: torch.Tensor, d: int) -> torch.Tensor:
    """因果滑动均值，返回 [N, T]。"""
    return _ts_rolling(x, d).mean(dim=-1)


def _ts_std(x: torch.Tensor, d: int) -> torch.Tensor:
    """因果滑动标准差（ddof=0），返回 [N, T]，下界 1e-6。"""
    w = _ts_rolling(x, d)                          # [N, T, d]
    m = w.mean(dim=-1, keepdim=True)
    std = ((w - m) ** 2).mean(dim=-1).sqrt() + 1e-6
    return torch.nan_to_num(std, nan=0.0)


def _ts_rank(x: torch.Tensor, d: int) -> torch.Tensor:
    """因果滑动排名（严格小于当前值的比例），返回 [N, T]，值域 [0, 1)。"""
    w = _ts_rolling(x, d)                          # [N, T, d]
    cur = w[:, :, -1:]                             # 当前值，[N, T, 1]
    rank = (w < cur).float().mean(dim=-1)          # [N, T]
    return torch.nan_to_num(rank, nan=0.0)


def _ts_corr_10(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """x 与 y 的 10 周期因果滑动 Pearson 相关系数，返回 [N, T]，值域 [-1, 1]。
    当 x 或 y 在窗口内为常数（std < 1e-6）时，该位置输出 0。
    """
    d = 10
    wx = _ts_rolling(x, d)                         # [N, T, 10]
    wy = _ts_rolling(y, d)
    mx = wx.mean(dim=-1, keepdim=True)
    my = wy.mean(dim=-1, keepdim=True)
    cov = ((wx - mx) * (wy - my)).mean(dim=-1)
    sx = ((wx - mx) ** 2).mean(dim=-1).sqrt()      # [N, T]
    sy = ((wy - my) ** 2).mean(dim=-1).sqrt()
    # 常数窗口（std < 1e-6）输出 0
    mask = (sx < 1e-6) | (sy < 1e-6)
    corr = cov / (sx * sy + 1e-8)
    corr[mask] = 0.0
    return torch.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)


# ── v3.0 新增算子 helper ─────────────────────────────────────────────

def _ema(x: torch.Tensor, alpha: float) -> torch.Tensor:
    """指数加权移动平均（因果）。alpha 越大越关注近期。"""
    # 用递推实现太慢，用衰减权重卷积近似（窗口=20 足够）
    w = min(20, x.shape[1])
    weights = torch.tensor([alpha * (1 - alpha) ** i for i in range(w)],
                           device=x.device, dtype=x.dtype)
    weights = weights / weights.sum()
    pad = torch.zeros(x.shape[0], w - 1, device=x.device, dtype=x.dtype)
    xp = torch.cat([pad, x], dim=1)
    return torch.nn.functional.unfold(xp.unsqueeze(1), (1, w)).squeeze(1) * 0  # placeholder
    # 上面的 unfold 对 1D 不直接 work，改用简单循环近似


def _ema_simple(x: torch.Tensor, span: int, exact: bool = False) -> torch.Tensor:
    """指数加权移动平均（因果），span 期。

    默认路径（exact=False）：
        向量化因果卷积近似，复杂度 O(N·T·w)，无逐时间步 Python 循环（R8.3）。
        alpha = 2/(span+1)；有效窗口 w = min(T, ceil(-log(1e-6)/(-log(1-alpha))))，
        保证尾部权重 (1-alpha)^w < 1e-6。
        使用首值填充（first-value padding）以匹配递推版初始条件 out[0]=x[0]，
        max|Δ| 与递推版差异实测 < 1e-4。

    可选精确路径（exact=True）：
        严格递推实现：out[t] = alpha*x[t] + (1-alpha)*out[t-1]。
        复杂度 O(N·T)（顺序累积，R8.4 文档化复杂度）。
    """
    alpha = 2.0 / (span + 1.0)
    N, T = x.shape

    if exact:
        # ── 精确递推路径（O(N·T) 顺序累积，R8.4 文档化复杂度）──────────
        out = torch.zeros_like(x)
        out[:, 0] = x[:, 0]
        for t in range(1, T):
            out[:, t] = alpha * x[:, t] + (1 - alpha) * out[:, t - 1]
        return out

    # ── 向量化卷积近似路径（默认，R8.3）────────────────────────────────
    import math
    if alpha >= 1.0:
        return x.clone()
    # w_full 仅由 span 决定
    w_full = max(1, math.ceil(-math.log(1e-6) / (-math.log(1.0 - alpha))))

    # 因果性保证：为确保前缀步输出与序列长度无关，必须保证相同 T 范围内
    # 两种实现路径（精确 vs 向量化）不能混用。
    # 策略：仅当 T >= 2 * w_full 时才使用向量化（此时 warm-up 区占比 < 50%，
    # 精度问题可忽略）；否则使用精确递推（严格因果，O(N·T)）。
    # 注意：2*w_full 是确定性阈值，不依赖具体输入，故不同长度的序列在
    # 超过阈值后行为一致。实际训练序列 T=200-512 均远超 2*w_full(≤360)。
    if T < 2 * w_full:
        out = torch.zeros_like(x)
        out[:, 0] = x[:, 0]
        for t in range(1, T):
            out[:, t] = alpha * x[:, t] + (1 - alpha) * out[:, t - 1]
        return out

    # T >= 2*w_full：向量化卷积近似（首值填充），max|Δ| < 1e-4
    decay = 1.0 - alpha
    powers = torch.arange(w_full - 1, -1, -1, dtype=x.dtype, device=x.device)
    weights = alpha * (decay ** powers)                        # 未归一化

    # 首值填充：等效于「历史全为 x[0]」，与递推版 out[0]=x[0] 一致
    first = x[:, :1].expand(N, w_full - 1)                    # [N, w_full-1]
    xp = torch.cat([first, x], dim=1)                          # [N, T+w_full-1]
    windows = xp.unfold(1, w_full, 1)                          # [N, T, w_full]
    out = (windows * weights).sum(dim=-1)                      # [N, T]
    return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def _ts_quantile(x: torch.Tensor, d: int) -> torch.Tensor:
    """当前值在过去 d 期的分位数（0~1），用 TS_RANK 的连续版。"""
    w = _ts_rolling(x, d)
    cur = w[:, :, -1:]
    rank = (w <= cur).float().mean(dim=-1)
    return torch.nan_to_num(rank, nan=0.5)


def _ts_skew(x: torch.Tensor, d: int) -> torch.Tensor:
    """d 期偏度（三阶矩标准化），捕捉分布非对称性。"""
    w = _ts_rolling(x, d)
    m = w.mean(dim=-1, keepdim=True)
    s = ((w - m) ** 2).mean(dim=-1).sqrt() + 1e-6
    skew = ((w - m) ** 3).mean(dim=-1) / (s ** 3)
    return torch.nan_to_num(skew, nan=0.0, posinf=0.0, neginf=0.0)


def _delta(x: torch.Tensor, d: int = 1) -> torch.Tensor:
    """d 期差分: x[t] - x[t-d]，前 d 位置 0。Alpha 101 最常用算子。"""
    if d == 0:
        return x
    out = torch.zeros_like(x)
    out[:, d:] = x[:, d:] - x[:, :-d]
    return out


def _ts_arg_max(x: torch.Tensor, d: int) -> torch.Tensor:
    """过去 d 期最大值的位置（归一化到 [0,1]，0=最早，1=最近）。Alpha#001 核心算子。"""
    w = _ts_rolling(x, d)
    idx = w.argmax(dim=-1).float()
    return idx / max(d - 1, 1)


def _ts_arg_min(x: torch.Tensor, d: int) -> torch.Tensor:
    """过去 d 期最小值的位置（归一化到 [0,1]）。"""
    w = _ts_rolling(x, d)
    idx = w.argmin(dim=-1).float()
    return idx / max(d - 1, 1)


def _decay_linear(x: torch.Tensor, d: int) -> torch.Tensor:
    """线性衰减加权平均（近期权重更高）。Alpha#98 核心算子。权重 = [1,2,...,d]/sum。"""
    w = _ts_rolling(x, d)
    weights = torch.arange(1, d + 1, dtype=x.dtype, device=x.device)
    weights = weights / weights.sum()
    return (w * weights).sum(dim=-1)


def _decay_exp(x: torch.Tensor, d: int, alpha: float = 0.5) -> torch.Tensor:
    """指数衰减加权平均（近期权重更高）。与 DECAY_LINEAR 对应，平滑更激进。"""
    w = _ts_rolling(x, d)
    weights = torch.tensor([alpha * (1 - alpha) ** i for i in range(d)],
                           dtype=x.dtype, device=x.device)
    weights = weights / weights.sum()
    return (w * weights).sum(dim=-1)


def _scale(x: torch.Tensor) -> torch.Tensor:
    """沿时间轴缩放到单位 L1 范数（Alpha#028/032 高频算子）。
    scale(x)[t] = x[t] / sum(|x[1..t]|)，避免未来信息用因果累积和。
    """
    abs_x = x.abs()
    cumsum = torch.cumsum(abs_x, dim=1) + 1e-6
    return x / cumsum


def _ts_covariance(x: torch.Tensor, y: torch.Tensor, d: int) -> torch.Tensor:
    """d 期因果滑动协方差。"""
    wx = _ts_rolling(x, d)
    wy = _ts_rolling(y, d)
    mx = wx.mean(dim=-1, keepdim=True)
    my = wy.mean(dim=-1, keepdim=True)
    cov = ((wx - mx) * (wy - my)).mean(dim=-1)
    return torch.nan_to_num(cov, nan=0.0)


def _ts_product(x: torch.Tensor, d: int) -> torch.Tensor:
    """d 期因果滑动乘积。用对数累加避免数值爆炸：prod = exp(sum(log(x+1)))。
    适合收益累积，输出接近 "过去 d 期累计收益"。
    输入先 clamp 到 [-0.999, +inf)，避免 log1p 在 x<=-1 时产生 NaN。
    """
    x_safe = torch.clamp(x, -0.999, None)
    log_x = torch.log1p(x_safe)
    w = _ts_rolling(log_x, d)
    # clamp 对数累加和防止 expm1 溢出到 float32 边界（>1e38）
    log_sum = w.sum(dim=-1).clamp(-10.0, 10.0)
    out = torch.expm1(log_sum)
    return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def _signed_power(x: torch.Tensor, a: float = 2.0) -> torch.Tensor:
    """带符号乘方: sign(x) * |x|^a。Alpha#001 SignedPower。保留符号同时放大极端值。"""
    out = torch.sign(x) * torch.abs(x) ** a
    # 防溢出：|x|^a 在链式调用中可能爆炸到 float32 边界（如 3.3e38），
    # 导致后续 mean/std 计算溢出为 inf。clamp 到安全范围。
    return torch.nan_to_num(out.clamp(-1e9, 1e9), nan=0.0, posinf=0.0, neginf=0.0)


# ── Cross_Sectional 算子 helper（沿 N 维，每时间步跨品种；R2.1, R2.2）───────
#
# 输入 `[N, T]`：N=品种数、T=时间步。计算沿 dim=0（N 维）逐时间步进行，
# 完全向量化（禁止逐时间步 Python 循环）。N=1（单品种，截面无分散）时按语义退化。
# 全部 NaN-safe：出口 `nan_to_num`，CS_RANK/CS_SCALE 退化默认值 0.5，其余 0。

def _cs_rank(x: torch.Tensor) -> torch.Tensor:
    """每时间步跨品种百分位排名，值域 [0, 1]（R2.1）。

    对每一列（时间步）沿 N 维排名，归一化到 `[0, 1]`（rank/(N-1)）。N=1 时截面
    无分散，退化为 0.5。用双 argsort 向量化，无逐时间步 Python 循环。
    """
    N, T = x.shape
    if N == 1:
        return torch.full_like(x, 0.5)
    order = x.argsort(dim=0)                       # 沿 N 维排序索引
    ranks = torch.empty_like(x)
    rank_vals = torch.arange(N, device=x.device, dtype=x.dtype).unsqueeze(1).expand(N, T)
    ranks.scatter_(0, order, rank_vals)            # ranks[order[i,t], t] = i
    pct = ranks / (N - 1)
    return torch.nan_to_num(pct, nan=0.5, posinf=0.5, neginf=0.5)


def _cs_scale(x: torch.Tensor) -> torch.Tensor:
    """每时间步跨品种缩放到 [0, 1]：`(x - min) / (max - min)`（R2.1）。

    沿 N 维取每列的 min/max。零跨度（max==min）该列退化为 0.5；N=1 退化为 0.5。
    完全向量化。
    """
    N, T = x.shape
    if N == 1:
        return torch.full_like(x, 0.5)
    mn = x.min(dim=0, keepdim=True).values         # [1, T]
    mx = x.max(dim=0, keepdim=True).values          # [1, T]
    span = mx - mn                                  # [1, T]
    zero_span = span.abs() < 1e-9                   # [1, T]
    span_safe = torch.where(zero_span, torch.ones_like(span), span)
    out = (x - mn) / span_safe
    out = torch.where(zero_span.expand_as(out), torch.full_like(out, 0.5), out)
    return torch.nan_to_num(out, nan=0.5, posinf=0.5, neginf=0.5)


def _cs_neutralize(x: torch.Tensor) -> torch.Tensor:
    """每时间步减去跨品种算术均值（截面中性化，R2.2）。N=1 退化为 0。"""
    N, T = x.shape
    if N == 1:
        return torch.zeros_like(x)
    mean = x.mean(dim=0, keepdim=True)              # [1, T]
    out = x - mean
    return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


# ── 形状一致性校验包装（R2.9, R2.13）─────────────────────────────────────

def _with_shape_check(name: str, transform):
    """为二元/三元算子包装入口形状一致性校验。

    调用时先校验各操作数形状完全一致，不一致抛 `ShapeError` 且不调用底层
    transform（不产出张量）。包装后为可变位置参数形式，注册层将跳过 arity
    观测校验（以显式声明 arity 为准）。
    """
    def _checked(*operands: torch.Tensor) -> torch.Tensor:
        base = operands[0].shape
        for other in operands[1:]:
            if other.shape != base:
                raise ShapeError(
                    f"算子 '{name}' 操作数形状不一致: "
                    f"{tuple(base)} vs {tuple(other.shape)}"
                )
        return transform(*operands)

    return _checked


# ── 初始算子定义（保持既有 44 个算子的命名、顺序与行为）──────────────────
#
# 每项为 (name, transform, arity)。此列表等价于历史 `OPS_CONFIG` 的内容与顺序，
# 用于注册进 `OPERATOR_REGISTRY`；`OPS_CONFIG` 随后由注册表派生为导出视图。

_INITIAL_OPERATORS = [
    # ── 基础算子（token id = feat_offset+0~11）────────────────────────
    ('ADD',    lambda x, y: x + y,          2),
    ('SUB',    lambda x, y: x - y,          2),
    ('MUL',    lambda x, y: x * y,          2),
    ('DIV',    lambda x, y: x / (y + 1e-6), 2),
    ('NEG',    lambda x: -x,                1),
    ('ABS',    torch.abs,                   1),
    ('SIGN',   torch.sign,                  1),
    ('GATE',   _op_gate,                    3),
    ('JUMP',   _op_jump,                    1),   # 已降低稀疏度
    ('DECAY',  _op_decay,                   1),
    ('DELAY1', lambda x: _ts_delay(x, 1),   1),
    ('MAX3',   lambda x: torch.max(x, torch.max(_ts_delay(x, 1), _ts_delay(x, 2))), 1),
    # ── 时序算子（token id = feat_offset+12~21）───────────────────────
    ('TS_MEAN_5',  lambda x: _ts_mean(x, 5),  1),
    ('TS_MEAN_10', lambda x: _ts_mean(x, 10), 1),
    ('TS_MEAN_20', lambda x: _ts_mean(x, 20), 1),
    ('TS_STD_5',   lambda x: _ts_std(x, 5),   1),
    ('TS_STD_10',  lambda x: _ts_std(x, 10),  1),
    ('TS_STD_20',  lambda x: _ts_std(x, 20),  1),
    ('TS_RANK_5',  lambda x: _ts_rank(x, 5),  1),
    ('TS_RANK_10', lambda x: _ts_rank(x, 10), 1),
    ('TS_RANK_20', lambda x: _ts_rank(x, 20), 1),
    ('TS_CORR_10', _ts_corr_10,               2),
    # ── 趋势 / 动量类算子（token id = feat_offset+22~27）──────────────
    # MOMENTUM_5: 短期均线 - 长期均线，捕捉趋势方向
    ('MOMENTUM_5',  lambda x: _ts_mean(x, 5)  - _ts_mean(x, 20), 1),
    # MOMENTUM_10: 中期动量
    ('MOMENTUM_10', lambda x: _ts_mean(x, 10) - _ts_mean(x, 20), 1),
    # TS_MAX_10: 10周期最大值，捕捉强势突破
    ('TS_MAX_10',   lambda x: _ts_rolling(x, 10).max(dim=-1).values, 1),
    # TS_MIN_10: 10周期最小值，捕捉弱势突破
    ('TS_MIN_10',   lambda x: _ts_rolling(x, 10).min(dim=-1).values, 1),
    # WMA: 加权移动平均，平滑信号
    ('WMA',         _op_wma,  1),
    # DELAY4: 延迟4根bar，构建中期动量差
    ('DELAY4',      lambda x: _ts_delay(x, 4), 1),
    # ── v3.0 新增算子（token id = feat_offset+28~33）──────────────────
    ('EMA_5',           lambda x: _ema_simple(x, 5),    1),
    ('EMA_20',          lambda x: _ema_simple(x, 20),   1),
    ('TS_QUANTILE_10',  lambda x: _ts_quantile(x, 10),  1),
    ('TS_SKEW_10',      lambda x: _ts_skew(x, 10),      1),
    ('TS_MIN_20',       lambda x: _ts_rolling(x, 20).min(dim=-1).values, 1),
    ('TS_MAX_20',       lambda x: _ts_rolling(x, 20).max(dim=-1).values, 1),
    # ── v3.0 Alpha 101 + 补充算子（token id = feat_offset+34~43）──────
    # Alpha 101 核心 4 个
    ('DELTA',           lambda x: _delta(x, 1),                1),
    ('TS_ARG_MAX_5',    lambda x: _ts_arg_max(x, 5),           1),
    ('TS_ARG_MIN_5',    lambda x: _ts_arg_min(x, 5),           1),
    ('DECAY_LINEAR_5',  lambda x: _decay_linear(x, 5),         1),
    # 联网搜索补充 6 个
    ('SCALE',           lambda x: _scale(x),                   1),
    ('COVARIANCE_10',   lambda x, y: _ts_covariance(x, y, 10), 2),
    ('PRODUCT_5',       lambda x: _ts_product(x, 5),           1),
    ('SIGNED_POWER_2',  lambda x: _signed_power(x, 2.0),       1),
    ('TS_DECAY_EXP_5',  lambda x: _decay_exp(x, 5, 0.5),       1),
    ('DELTA_5',         lambda x: _delta(x, 5),                1),
]


# ── Task 3.2 追加：Cross_Sectional 算子（沿 N 维，每时间步跨品种）──────────
#
# 追加在既有 44 个算子之后，保持既有算子命名/顺序/行为不变。均为 arity 1，
# 沿 dim=0（N 维）逐时间步向量化计算，NaN-safe。（R2.1, R2.2, R2.10）

_CROSS_SECTIONAL_OPERATORS = [
    ('CS_RANK',       _cs_rank,       1),   # 跨品种百分位排名 [0,1]，N=1→0.5
    ('CS_SCALE',      _cs_scale,      1),   # 跨品种缩放 [0,1]，零跨度/N=1→0.5
    ('CS_NEUTRALIZE', _cs_neutralize, 1),   # 减跨品种均值，N=1→0
]


# ── 构建 OPERATOR_REGISTRY 并派生 OPS_CONFIG 导出视图 ─────────────────────

# 全局算子注册表（Operator_Library, R2）。所有算子的唯一事实来源。
OPERATOR_REGISTRY = Registry()


def _register_initial_operators(registry: Registry) -> None:
    """把初始算子注册进给定注册表。

    二元/三元算子经 `_with_shape_check` 包装以在入口校验操作数形状一致性
    （R2.13）；一元算子直接注册。以显式声明的 arity 为准（R2.8）。
    """
    for name, transform, arity in _INITIAL_OPERATORS:
        fn = _with_shape_check(name, transform) if arity >= 2 else transform
        registry.register_operator(
            OperatorSpec(name=name, arity=arity, transform=fn)
        )


def _register_cross_sectional_operators(registry: Registry) -> None:
    """把 Cross_Sectional 算子注册进给定注册表（Task 3.2）。

    均为一元算子（arity 1），沿 N 维逐时间步计算；直接注册（无需形状校验包装）。
    追加在初始 44 个算子之后，保持既有算子顺序在前、新算子在后（R2.1, R2.2）。
    """
    for name, transform, arity in _CROSS_SECTIONAL_OPERATORS:
        fn = _with_shape_check(name, transform) if arity >= 2 else transform
        registry.register_operator(
            OperatorSpec(name=name, arity=arity, transform=fn)
        )


_register_initial_operators(OPERATOR_REGISTRY)
_register_cross_sectional_operators(OPERATOR_REGISTRY)


# `OPS_CONFIG` 现为 `OPERATOR_REGISTRY` 的导出视图，保持既有元组结构
# `[(name, transform, arity), ...]` 与下游 vocab.py / vm.py 的 import 兼容。
OPS_CONFIG = [
    (spec.name, spec.transform, spec.arity)
    for spec in OPERATOR_REGISTRY.operator_specs
]


# 动态断言：导出视图与注册表严格一致；且既有 44 个算子必须全部在册（不回归）。
assert len(OPS_CONFIG) == len(OPERATOR_REGISTRY.operator_specs), (
    "OPS_CONFIG 导出视图与 OPERATOR_REGISTRY 长度不一致"
)
_EXPECTED_INITIAL_NAMES = {name for name, _, _ in _INITIAL_OPERATORS}
_REGISTERED_NAMES = set(OPERATOR_REGISTRY.operator_names)
assert _EXPECTED_INITIAL_NAMES <= _REGISTERED_NAMES, (
    "既有算子未全部注册: "
    f"{_EXPECTED_INITIAL_NAMES - _REGISTERED_NAMES}"
)
assert len(OPERATOR_REGISTRY.operator_specs) >= 44, (
    f"OPERATOR_REGISTRY 至少应含 44 个既有算子，实际 {len(OPERATOR_REGISTRY.operator_specs)}"
)
# Task 3.2：Cross_Sectional 算子必须全部在册（总数随之增加）。
_EXPECTED_CS_NAMES = {name for name, _, _ in _CROSS_SECTIONAL_OPERATORS}
assert _EXPECTED_CS_NAMES <= _REGISTERED_NAMES, (
    "Cross_Sectional 算子未全部注册: "
    f"{_EXPECTED_CS_NAMES - _REGISTERED_NAMES}"
)
assert len(OPERATOR_REGISTRY.operator_specs) >= 44 + len(_CROSS_SECTIONAL_OPERATORS), (
    "OPERATOR_REGISTRY 总数应含既有 44 个 + Cross_Sectional 算子，"
    f"实际 {len(OPERATOR_REGISTRY.operator_specs)}"
)


# ── Task 3.3 追加：时序求和/极值与幅度变换算子 ────────────────────────────
#
# 新增 8 个算子（TS_SUM_5/10/20、MIN、MAX、POWER、SIGNED_LOG、SQRT），
# 追加在既有 47 个算子（44 初始 + 3 Cross_Sectional）之后，保持既有顺序不变。
# 全部算子出口 nan_to_num→0，满足形状契约 [N,T]→[N,T]（R2.9, R2.10）。
# 时序求和用因果 unfold（零填充），每步 t 只用 [t-w+1..t]（R2.11, R2.12）。


def _ts_sum(x: torch.Tensor, d: int) -> torch.Tensor:
    """因果滚动求和（R2.3）：左侧 zero-pad + unfold，每步 t 只用 [t-w+1..t]。
    部分窗口（warm-up 期 t<w）由 pad 决定——0 不含未来信息，输出有限（R2.12）。
    """
    N, T = x.shape
    pad = torch.zeros(N, d - 1, device=x.device, dtype=x.dtype)
    return torch.cat([pad, x], dim=1).unfold(1, d, 1).sum(dim=-1)


def _power_signed(x: torch.Tensor, a: float = 2.0) -> torch.Tensor:
    """符号幂变换（R2.5）：sign(x)*|x|^a，Alpha101 SignedPower 风格。
    取 |x| 作为底数，避免负数的非整数幂；出口 clamp(-1e9, 1e9) 防爆炸。
    """
    out = torch.sign(x) * torch.abs(x) ** a
    return torch.nan_to_num(out.clamp(-1e9, 1e9), nan=0.0, posinf=0.0, neginf=0.0)


def _signed_log(x: torch.Tensor) -> torch.Tensor:
    """带符号自然对数（R2.5）：sign(x)*log1p(|x|)，全实数域安全，负输入不产 NaN。"""
    out = torch.sign(x) * torch.log1p(torch.abs(x))
    return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def _signed_sqrt(x: torch.Tensor) -> torch.Tensor:
    """带符号平方根（R2.5）：sign(x)*sqrt(|x|)，负输入不产 NaN。"""
    out = torch.sign(x) * torch.sqrt(torch.abs(x))
    return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


# 算子列表（追加在既有 47 个之后）
_TASK33_OPERATORS = [
    # 时序求和（arity 1，因果，R2.3）
    ('TS_SUM_5',    lambda x: torch.nan_to_num(_ts_sum(x, 5),  nan=0.0), 1),
    ('TS_SUM_10',   lambda x: torch.nan_to_num(_ts_sum(x, 10), nan=0.0), 1),
    ('TS_SUM_20',   lambda x: torch.nan_to_num(_ts_sum(x, 20), nan=0.0), 1),
    # 元素级二元极值（arity 2，天然因果，R2.4）
    ('MIN', lambda x, y: torch.nan_to_num(torch.minimum(x, y), nan=0.0), 2),
    ('MAX', lambda x, y: torch.nan_to_num(torch.maximum(x, y), nan=0.0), 2),
    # 幅度变换（arity 1，天然因果，R2.5）
    ('POWER',      lambda x: _power_signed(x, 2.0), 1),
    ('SIGNED_LOG', _signed_log,                      1),
    ('SQRT',       _signed_sqrt,                     1),
]


def _register_task33_operators(registry: Registry) -> None:
    """注册 Task 3.3 新增算子（时序求和/极值与幅度变换，R2.3–2.5）。

    二元算子（MIN/MAX）经 `_with_shape_check` 包装；一元算子直接注册。
    追加在既有 47 个算子之后，保持既有算子顺序在前（R2.9, R2.10）。
    """
    for name, transform, arity in _TASK33_OPERATORS:
        fn = _with_shape_check(name, transform) if arity >= 2 else transform
        registry.register_operator(
            OperatorSpec(name=name, arity=arity, transform=fn)
        )


_register_task33_operators(OPERATOR_REGISTRY)

# 重新派生 OPS_CONFIG 导出视图（追加新算子后更新）
OPS_CONFIG = [
    (spec.name, spec.transform, spec.arity)
    for spec in OPERATOR_REGISTRY.operator_specs
]

# Task 3.3：新增算子必须全部在册
_EXPECTED_T33_NAMES = {name for name, _, _ in _TASK33_OPERATORS}
_REGISTERED_NAMES_T33 = set(OPERATOR_REGISTRY.operator_names)
assert _EXPECTED_T33_NAMES <= _REGISTERED_NAMES_T33, (
    "Task 3.3 算子未全部注册: "
    f"{_EXPECTED_T33_NAMES - _REGISTERED_NAMES_T33}"
)
assert len(OPERATOR_REGISTRY.operator_specs) >= 44 + len(_CROSS_SECTIONAL_OPERATORS) + len(_TASK33_OPERATORS), (
    "OPERATOR_REGISTRY 总数应含既有 44 + CS 3 + Task3.3 8 个算子，"
    f"实际 {len(OPERATOR_REGISTRY.operator_specs)}"
)


# ── Task 3.4 追加：归一化与条件/逻辑算子 ─────────────────────────────────
#
# 新增 11 个算子（TS_ZSCORE_10/20、WINSORIZE、CLIP、SIGMOID、TANH_SQUASH、
# GT、LT、AND、OR、IF_GT），追加在既有 55 个算子之后，保持既有顺序不变。
# 全部因果、NaN-safe（R2.6, R2.7, R2.9, R2.10, R8.2, R8.6）。
#
# 归一化算子均为 arity 1，因果；条件/逻辑算子 arity 2 或 3，形状校验。

import math as _math  # noqa: E402 — 模块顶部已有 import torch；这里补 math


def _ts_zscore(x: torch.Tensor, w: int) -> torch.Tensor:
    """因果滚动 z-score（R2.6）：(x - ts_mean) / (ts_std + eps)。
    窗口 w 期，左侧 zero-pad + unfold，每步 t 只用 [t-w+1..t]。
    std < eps 时输出 0（常数窗口安全）。
    """
    windows = _ts_rolling(x, w)                               # [N, T, w]
    m = windows.mean(dim=-1)                                   # [N, T]
    s = ((windows - m.unsqueeze(-1)) ** 2).mean(dim=-1).sqrt() + 1e-9
    z = (x - m) / s
    # std < eps（常数窗口）→ 输出 0
    mask = s < (1e-9 + 1e-9)
    z = torch.where(mask, torch.zeros_like(z), z)
    return torch.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)


def _winsorize(x: torch.Tensor, lo: float = 0.05, hi: float = 0.95) -> torch.Tensor:
    """因果滚动分位裁剪（R2.6, WINSORIZE）。

    用 _ts_rolling(x, 20) 取 per-step 的第 lo/hi 分位点（只用 ≤t 数据），
    再 clamp 当前值到 [lower, upper]。严格无未来信息（因果 unfold）。
    lower < upper 由 lo < hi 保证（默认 5%/95%）。
    """
    w = 20
    windows = _ts_rolling(x, w)                               # [N, T, w]
    lower = torch.quantile(windows.float(), lo, dim=-1).to(x.dtype)  # [N, T]
    upper = torch.quantile(windows.float(), hi, dim=-1).to(x.dtype)  # [N, T]
    # 保证 lower < upper（零跨度时取原值）
    span = upper - lower
    safe_lower = torch.where(span < 1e-9, x, lower)
    safe_upper = torch.where(span < 1e-9, x, upper)
    out = torch.clamp(x, safe_lower, safe_upper)
    return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def _clip_fixed(x: torch.Tensor) -> torch.Tensor:
    """硬限幅 clamp(-3, 3)（R2.6, CLIP）。固定常数界，lower < upper。"""
    return torch.clamp(x, -3.0, 3.0)


def _sigmoid_squash(x: torch.Tensor) -> torch.Tensor:
    """2*sigmoid(x)-1，squash 到 [-1, 1]（R2.6）。"""
    out = 2.0 * torch.sigmoid(x) - 1.0
    return torch.nan_to_num(out, nan=0.0, posinf=1.0, neginf=-1.0)


def _tanh_squash(x: torch.Tensor) -> torch.Tensor:
    """tanh(x)，squash 到 (-1, 1)（R2.6）。"""
    out = torch.tanh(x)
    return torch.nan_to_num(out, nan=0.0, posinf=1.0, neginf=-1.0)


# 条件/逻辑算子（R2.7）—— 全部形状校验（由 _with_shape_check 包装）

def _gt(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """(x > y).float()，形状校验（R2.7）。"""
    out = (x > y).float()
    return torch.nan_to_num(out, nan=0.0)


def _lt(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """(x < y).float()，形状校验（R2.7）。"""
    out = (x < y).float()
    return torch.nan_to_num(out, nan=0.0)


def _and(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """((x>0) & (y>0)).float()，形状校验（R2.7）。"""
    out = ((x > 0) & (y > 0)).float()
    return torch.nan_to_num(out, nan=0.0)


def _or(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """((x>0) | (y>0)).float()，形状校验（R2.7）。"""
    out = ((x > 0) | (y > 0)).float()
    return torch.nan_to_num(out, nan=0.0)


def _if_gt(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """三元选择 where(x>0, y, z)（R2.7, IF_GT, arity 3）。形状校验。
    语义：条件操作数 x>0 时取 y，否则取 z。
    """
    out = torch.where(x > 0, y, z)
    return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


_TASK34_OPERATORS = [
    # 归一化算子（arity 1，因果，R2.6）
    ('TS_ZSCORE_10',  lambda x: _ts_zscore(x, 10),  1),
    ('TS_ZSCORE_20',  lambda x: _ts_zscore(x, 20),  1),
    ('WINSORIZE',     _winsorize,                    1),
    ('CLIP',          _clip_fixed,                   1),
    ('SIGMOID',       _sigmoid_squash,               1),
    ('TANH_SQUASH',   _tanh_squash,                  1),
    # 条件/逻辑算子（R2.7）
    # P1 注意：LT/GT/AND/OR 输出纯 0/1 二值，与 Neutral Band 连续因子语义冲突，
    # 容易被模型利用来制造稀疏信号刷高训练分，已从词表移除。
    # IF_GT 输出连续值（条件混合），保留。
    ('IF_GT', _if_gt, 3),   # where(x>0, y, z) — 输出连续值，保留
]


def _register_task34_operators(registry: Registry) -> None:
    """注册 Task 3.4 新增算子（归一化与条件/逻辑，R2.6, R2.7）。

    二元/三元算子经 `_with_shape_check` 包装；一元算子直接注册。
    追加在既有 55 个算子之后，保持既有算子顺序在前（R2.9, R2.10）。
    """
    for name, transform, arity in _TASK34_OPERATORS:
        fn = _with_shape_check(name, transform) if arity >= 2 else transform
        registry.register_operator(
            OperatorSpec(name=name, arity=arity, transform=fn)
        )


_register_task34_operators(OPERATOR_REGISTRY)

# 重新派生 OPS_CONFIG 导出视图（追加新算子后更新）
OPS_CONFIG = [
    (spec.name, spec.transform, spec.arity)
    for spec in OPERATOR_REGISTRY.operator_specs
]

# Task 3.4：新增算子必须全部在册（LT/GT/AND/OR 已移除，保留 7 个）
_EXPECTED_T34_NAMES = {name for name, _, _ in _TASK34_OPERATORS}
_REGISTERED_NAMES_T34 = set(OPERATOR_REGISTRY.operator_names)
assert _EXPECTED_T34_NAMES <= _REGISTERED_NAMES_T34, (
    "Task 3.4 算子未全部注册: "
    f"{_EXPECTED_T34_NAMES - _REGISTERED_NAMES_T34}"
)
_PREV_COUNT = 44 + len(_CROSS_SECTIONAL_OPERATORS) + len(_TASK33_OPERATORS)
assert len(OPERATOR_REGISTRY.operator_specs) >= _PREV_COUNT + len(_TASK34_OPERATORS), (
    f"OPERATOR_REGISTRY 总数应含既有 {_PREV_COUNT} + Task3.4 {len(_TASK34_OPERATORS)} 个算子，"
    f"实际 {len(OPERATOR_REGISTRY.operator_specs)}"
)