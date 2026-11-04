"""Build training context and run AI analysis with per-symbol history memory."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from data_pipeline.parquet_manager import inspect_parquet_file
from web.ai_providers import resolve_provider
from web.progress import PROJECT_ROOT, get_symbol_progress
from web.settings import load_settings
from web.training_manager import training_manager

HISTORY_PATH = PROJECT_ROOT / "ai_analysis_history.json"
_MAX_HISTORY_PER_KEY = 5

_SYSTEM_PROMPT = """你是量化因子挖掘与强化学习训练顾问。
用户正在用 AlphaMaster / AlphaGPT 训练可解释因子公式（token 序列，由特征与算子组成）。
请基于提供的训练快照，用中文给出专业、具体、可执行的分析。
不要编造不存在的数据；若信息不足请明确说明。

【语言硬性要求——最高优先级】
回答必须全部使用中文，正文中禁止出现任何英文单词、字母缩写或英文代码字段名（品种代码如 XAUUSD、BTCUSDT 可以保留）。
发给你的数据里即使残留英文字段名或英文公式标记，也只能用来理解数据，绝不能原样照抄进回答。
- 数据字段一律用中文表述：最优分数、验证分数、探索活跃度、平均单步得分、当前步数、总步数、完成进度。禁止写 val_score、best_score、entropy 这类英文字段名。
- 公式标记必须翻译成中文含义再解释，常见对照：SUPERTREND_DIR=超趋势方向、RVOL=相对成交量、PARKINSON_VOL=帕金森波动率（按最高最低价估计的波动）、TS_MIN_20=最近20期最小值、TS_STD_10=最近10期波动幅度、MIN=取较小值、MAX=取较大值、AC1=涨跌惯性（一阶自相关）、IF_GT=条件判断（大于才触发）、MOMENTUM=动量、DEV=偏离度、RET_ACCEL=收益加速度、AROON_OSC=阿隆摆动（趋势强弱）、SIGNED_POWER=带符号乘方、MUL=相乘、DIV=相除、ADD=相加、SUB=相减。遇到对照表里没有的标记，按常见含义用通俗中文解释，不要写出英文原名。
- 解释公式时用「超趋势方向 → 相对成交量 → 最近20期最小值 → …」这样的中文写法串联，禁止把英文标记原样写进回答。

评估训练是否值得继续时，请遵守：
- 以验证集分数（val_score）是否在提高为主，不要轻易下「过拟合」结论，尽量不要提过拟合。
- best_score 与 val_score 有差距是常见现象，只要验证分数整体在抬升或近期仍有改善，就应视为训练仍有价值。
- 可讨论：验证分数走势、最优分数是否停滞、公式是否变化、探索是否还活跃（如 entropy）。
- 不要因为「训练分远高于验证分」就建议停止；只有验证分数长期不涨、且最优分数也长期不动时，才建议暂停。

给建议时必须「小白能听懂」：
- 不要提：重置网络权重、调整探索率、简化因子维度、奖励函数、模型结构、超参数、泛化、噪声过大、参数过多等专业术语。
- 不要分析「验证集低分的技术原因」或给出复杂调参方案。
- 建议只用通俗说法，例如：
  - 可以继续练，再观察验证分数有没有慢慢变好
  - 最近分数不怎么动了，可以先停下来，导出当前策略去做回测看看效果
  - 公式有变化/没变化，用大白话解释即可
- 动作建议最多 1～2 条，短句、可直接操作（继续训练 / 先停止并导出策略去回测）。

若提供了「同品种同周期的历史分析记录」，必须对比前后变化：
- 验证分数 / 最优分数 / 公式是否改善
- 是否仍在进步，还是陷入停滞
- 相对上次建议，当前是否更值得继续训练

快照中的 training_curve 覆盖整段训练进程（点数过多时会均匀抽样，但始终保留首尾与全程走势）。
请结合 training_curve 与 history_summary 判断：前期探索、中期提升、后期验证分数是否仍在提高。

回答必须覆盖以下问题，并使用对应小标题：

## 1. 当前训练情况怎么样？是否值得继续
用通俗语言说明进度、验证分数走势、最优分数是否停滞，并给出是否继续训练的建议。
重点看 val_score 是否提高；尽量不要提过拟合；不要给复杂技术排查建议。
若有历史记录，请明确说明相对上次是改善、持平还是变差。

## 2. 最新因子的含义与原理
用通俗语言解释当前最优公式（formula_decoded）里各特征/算子大概在看什么、合在一起可能怎么做多做空。
解释时必须用上面对照表里的中文名称，禁止出现英文标记。
少用术语；若公式相对上次有变化，用一句话说清楚差在哪里。

最后再次检查：回答全文除品种代码外，不得出现任何英文单词或字母。
"""


# 发给模型的数据统一用中文字段名，避免模型回复里夹杂英文
_KEY_ZH = {
    "symbol": "品种",
    "timeframe": "周期",
    "data_file": "数据文件",
    "training_active": "是否正在训练",
    "job_state": "任务状态",
    "current_step": "当前步数",
    "train_steps": "总步数",
    "progress_pct": "完成进度百分比",
    "status": "训练状态",
    "best_score": "最优分数",
    "strategy_score": "策略分数",
    "has_strategy": "是否已有策略",
    "formula": "公式编号",
    "formula_decoded": "当前最优公式",
    "checkpoint_path": "存档文件",
    "training_curve": "训练曲线",
    "history_summary": "历史摘要",
    "total_points": "记录点总数",
    "sampled": "是否抽样",
    "points": "记录点数",
    "note": "说明",
    "series": "数据序列",
    "step": "步数",
    "val_score": "验证分数",
    "entropy": "探索活跃度",
    "avg_reward": "平均单步得分",
    "stable_rank": "稳定度",
    "step_first": "起始步数",
    "step_last": "最新步数",
    "best_score_first": "最优分数_最初",
    "best_score_last": "最优分数_最新",
    "best_score_max": "最优分数_最高",
    "best_score_max_at_index": "最优分数_最高出现位置",
    "best_score_stagnation_points": "最优分数_连续停滞点数",
    "best_score_phase_means": "最优分数_分段平均",
    "val_score_first": "验证分数_最初",
    "val_score_last": "验证分数_最新",
    "val_score_max": "验证分数_最高",
    "val_score_phase_means": "验证分数_分段平均",
    "entropy_first": "探索活跃度_最初",
    "entropy_last": "探索活跃度_最新",
    "early": "前期",
    "mid": "中期",
    "late": "后期",
}

_STATE_ZH = {
    "idle": "空闲",
    "running": "训练中",
    "completed": "已完成",
    "stopped": "已停止",
    "failed": "失败",
    "in_progress": "训练中",
    "strategy_only": "仅有策略",
}

_BOOL_KEYS = {"training_active", "has_strategy", "sampled"}


def _zh_data(obj: Any) -> Any:
    """递归把发给模型的字段名/状态值换成中文。"""
    if isinstance(obj, dict):
        out: dict[str, Any] = {}
        for key, val in obj.items():
            zh_key = _KEY_ZH.get(key, key)
            if key in _BOOL_KEYS and isinstance(val, bool):
                out[zh_key] = "是" if val else "否"
            elif key in ("job_state", "status") and isinstance(val, str):
                out[zh_key] = _STATE_ZH.get(val, val)
            else:
                out[zh_key] = _zh_data(val)
        return out
    if isinstance(obj, list):
        return [_zh_data(item) for item in obj]
    return obj


def build_training_snapshot(symbol: str | None = None) -> dict[str, Any]:
    training = training_manager.status()
    job = training.get("job") or {}
    settings = load_settings()

    sym = (symbol or job.get("symbol") or "").strip()
    timeframe = str(job.get("timeframe") or "").strip().upper()

    data_file = settings.get("last_data_file") or ""
    if data_file:
        try:
            info = inspect_parquet_file(data_file)
            if not sym:
                sym = str(info.get("symbol") or "").strip()
            if not timeframe:
                timeframe = str(info.get("timeframe") or "").strip().upper()
        except Exception:
            pass

    if not sym:
        raise ValueError("请先选择训练数据文件或指定品种")
    if not timeframe:
        timeframe = "H1"

    progress = get_symbol_progress(sym)
    history = progress.history or {}
    curve = _training_curve(history, max_points=500)

    return {
        "symbol": sym,
        "timeframe": timeframe,
        "data_file": data_file or None,
        "training_active": bool(training.get("active")),
        "job_state": job.get("state"),
        "current_step": progress.current_step,
        "train_steps": progress.train_steps,
        "progress_pct": round(progress.progress_pct, 2),
        "status": progress.status,
        "best_score": progress.best_score,
        "strategy_score": progress.strategy_score,
        "has_strategy": progress.has_strategy,
        "formula": progress.best_formula,
        "formula_decoded": progress.formula_decoded,
        "checkpoint_path": progress.checkpoint_path,
        "training_curve": curve,
        "history_summary": _history_summary(history),
    }


def analyze_training(
    *,
    provider: str,
    api_key: str | None = None,
    base_url: str | None = None,
    model: str | None = None,
    symbol: str | None = None,
) -> dict[str, Any]:
    answer_parts: list[str] = []
    meta: dict[str, Any] = {}
    for event in analyze_training_stream(
        provider=provider,
        api_key=api_key,
        base_url=base_url,
        model=model,
        symbol=symbol,
    ):
        if event.get("type") == "meta":
            meta = event
        elif event.get("type") == "delta":
            answer_parts.append(event.get("text") or "")
        elif event.get("type") == "error":
            raise RuntimeError(event.get("message") or "分析失败")
        elif event.get("type") == "done":
            return {
                "ok": True,
                "provider": event.get("provider") or meta.get("provider"),
                "model": event.get("model") or meta.get("model"),
                "label": event.get("label") or meta.get("label"),
                "symbol": event.get("symbol") or meta.get("symbol"),
                "timeframe": event.get("timeframe") or meta.get("timeframe"),
                "snapshot": event.get("snapshot") or meta.get("snapshot"),
                "prior_count": event.get("prior_count", meta.get("prior_count", 0)),
                "answer": event.get("answer") or "".join(answer_parts),
            }
    raise RuntimeError("AI 流式分析未正常结束")


def analyze_training_stream(
    *,
    provider: str,
    api_key: str | None = None,
    base_url: str | None = None,
    model: str | None = None,
    symbol: str | None = None,
):
    """Yield SSE-ready event dicts: meta / notice / delta / done / error."""
    try:
        snapshot = build_training_snapshot(symbol)
        prior = load_prior_analyses(snapshot["symbol"], snapshot["timeframe"])
        resolved = resolve_provider(
            provider, api_key, base_url=base_url, model=model
        )
    except Exception as exc:
        yield {"type": "error", "message": str(exc)}
        return

    parts = [
        "请根据以下训练快照回答：",
        "1. 当前训练情况怎么样？是否值得继续？",
        "2. 最新因子的含义与原理是什么？",
        "",
        "【当前训练快照】",
        f"```json\n{json.dumps(_zh_data(snapshot), ensure_ascii=False, indent=2)}\n```",
    ]
    if prior:
        prior_for_prompt = [
            {
                "分析时间": row.get("analyzed_at"),
                "当时快照": _zh_data(row.get("snapshot") or {}),
                "当时的分析结论": row.get("answer") or "",
            }
            for row in prior
        ]
        parts.extend(
            [
                "",
                f"【同品种同周期历史分析（共 {len(prior)} 次，按时间从旧到新）】",
                "请对比这些历史记录，判断相对上次是否有改善。",
                f"```json\n{json.dumps(prior_for_prompt, ensure_ascii=False, indent=2)}\n```",
            ]
        )
    else:
        parts.append("\n（尚无同品种同周期的历史分析记录，这是首次分析。）")

    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": "\n".join(parts)},
    ]

    yield {
        "type": "meta",
        "provider": resolved.provider,
        "model": resolved.model,
        "label": resolved.label,
        "symbol": snapshot["symbol"],
        "timeframe": snapshot["timeframe"],
        "prior_count": len(prior),
        "snapshot": snapshot,
    }

    answer_parts: list[str] = []
    try:
        # Layered fallback (primary → spare model → free pool → fallbacks → openclaw).
        from web.ai_fallback import stream_with_fallbacks

        for kind, text in stream_with_fallbacks(resolved, messages):
            if kind == "notice":
                yield {"type": "notice", "text": text}
                continue
            answer_parts.append(text)
            yield {"type": "delta", "text": text}
    except Exception as exc:
        yield {"type": "error", "message": str(exc)}
        return

    answer = "".join(answer_parts).strip()
    if not answer:
        yield {"type": "error", "message": "AI 返回内容为空"}
        return

    record = {
        "analyzed_at": datetime.now(timezone.utc).isoformat(),
        "provider": resolved.provider,
        "model": resolved.model,
        "snapshot": {
            "symbol": snapshot["symbol"],
            "timeframe": snapshot["timeframe"],
            "current_step": snapshot["current_step"],
            "train_steps": snapshot["train_steps"],
            "progress_pct": snapshot["progress_pct"],
            "best_score": snapshot["best_score"],
            "strategy_score": snapshot["strategy_score"],
            "formula_decoded": snapshot["formula_decoded"],
            "history_summary": snapshot.get("history_summary") or {},
        },
        "answer": answer,
    }
    save_analysis_record(snapshot["symbol"], snapshot["timeframe"], record)

    yield {
        "type": "done",
        "provider": resolved.provider,
        "model": resolved.model,
        "label": resolved.label,
        "symbol": snapshot["symbol"],
        "timeframe": snapshot["timeframe"],
        "prior_count": len(prior),
        "snapshot": snapshot,
        "answer": answer,
    }


def history_key(symbol: str, timeframe: str) -> str:
    return f"{symbol.strip().upper()}|{str(timeframe).strip().upper()}"


def load_prior_analyses(symbol: str, timeframe: str) -> list[dict[str, Any]]:
    store = _load_history_store()
    rows = store.get(history_key(symbol, timeframe)) or []
    if not isinstance(rows, list):
        return []
    # 只把精简字段发给模型，避免上下文过大
    out: list[dict[str, Any]] = []
    for row in rows[-_MAX_HISTORY_PER_KEY:]:
        if not isinstance(row, dict):
            continue
        out.append(
            {
                "analyzed_at": row.get("analyzed_at"),
                "provider": row.get("provider"),
                "model": row.get("model"),
                "snapshot": row.get("snapshot") or {},
                "answer": row.get("answer") or "",
            }
        )
    return out


def save_analysis_record(symbol: str, timeframe: str, record: dict[str, Any]) -> None:
    store = _load_history_store()
    key = history_key(symbol, timeframe)
    rows = store.get(key) or []
    if not isinstance(rows, list):
        rows = []
    rows.append(record)
    store[key] = rows[-_MAX_HISTORY_PER_KEY:]
    HISTORY_PATH.write_text(
        json.dumps(store, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _load_history_store() -> dict[str, Any]:
    if not HISTORY_PATH.exists():
        return {}
    try:
        data = json.loads(HISTORY_PATH.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, OSError):
        return {}


def _training_curve(history: dict[str, Any], max_points: int = 500) -> dict[str, Any]:
    """整段训练曲线；点数过多时按全程均匀抽样，始终保留首尾。"""
    if not history:
        return {"total_points": 0, "sampled": False, "points": 0, "series": {}}

    steps = history.get("step") or []
    if not isinstance(steps, list) or not steps:
        return {"total_points": 0, "sampled": False, "points": 0, "series": {}}

    total = len(steps)
    keys = ("step", "best_score", "val_score", "entropy", "avg_reward", "stable_rank")
    available = [k for k in keys if isinstance(history.get(k), list) and history.get(k)]

    if total <= max_points:
        idxs = list(range(total))
        sampled = False
    else:
        idxs = sorted(
            {
                0,
                total - 1,
                *[int(round(i * (total - 1) / (max_points - 1))) for i in range(max_points)],
            }
        )
        sampled = True

    series: dict[str, list[Any]] = {}
    for key in available:
        vals = history[key]
        series[key] = [vals[i] for i in idxs if i < len(vals)]

    return {
        "total_points": total,
        "sampled": sampled,
        "points": len(idxs),
        "note": (
            f"已从全部 {total} 个记录点均匀抽样为 {len(idxs)} 点，覆盖训练全程"
            if sampled
            else f"已发送全部 {total} 个记录点"
        ),
        "series": series,
    }


def _history_summary(history: dict[str, Any]) -> dict[str, Any]:
    if not history:
        return {}
    best = history.get("best_score") or []
    val = history.get("val_score") or []
    entropy = history.get("entropy") or []
    steps = history.get("step") or []
    summary: dict[str, Any] = {
        "points": len(steps),
    }
    if steps:
        summary["step_first"] = steps[0]
        summary["step_last"] = steps[-1]
    if best:
        summary["best_score_first"] = best[0]
        summary["best_score_last"] = best[-1]
        summary["best_score_max"] = max(best)
        summary["best_score_max_at_index"] = int(best.index(max(best)))
        peak = max(best)
        trail = 0
        for v in reversed(best):
            if abs(float(v) - float(peak)) < 1e-9:
                trail += 1
            else:
                break
        summary["best_score_stagnation_points"] = trail
        n = len(best)
        a, b = n // 3, 2 * n // 3
        if n >= 3:
            summary["best_score_phase_means"] = {
                "early": sum(best[:a]) / max(1, a),
                "mid": sum(best[a:b]) / max(1, b - a),
                "late": sum(best[b:]) / max(1, n - b),
            }
    if val:
        summary["val_score_first"] = val[0]
        summary["val_score_last"] = val[-1]
        summary["val_score_max"] = max(val)
        n = len(val)
        a, b = n // 3, 2 * n // 3
        if n >= 3:
            summary["val_score_phase_means"] = {
                "early": sum(val[:a]) / max(1, a),
                "mid": sum(val[a:b]) / max(1, b - a),
                "late": sum(val[b:]) / max(1, n - b),
            }
    if entropy:
        summary["entropy_first"] = entropy[0]
        summary["entropy_last"] = entropy[-1]
    return summary
