"""Strategy JSON inspection for the web UI."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from model_core.vocab import VOCAB_VERSION
from web.progress import (
    STRATEGIES_DIR,
    _decode_formula,
    _load_checkpoint_meta,
    _load_strategy,
    checkpoint_glob,
)

_BEST_NAME_RE = re.compile(r"^best_(.+)\.json$", re.IGNORECASE)
_STRATEGY_EXPORT_RE = re.compile(
    r"^strategy_(.+)_step(\d+)(?:_score([\d.]+))?\.json$",
    re.IGNORECASE,
)


def strategy_path_for_symbol(symbol: str) -> Path:
    return STRATEGIES_DIR / f"best_{symbol}.json"


def symbol_from_strategy_path(path: Path) -> str | None:
    name = path.name
    m = _BEST_NAME_RE.match(name)
    if m:
        return m.group(1)
    m = _STRATEGY_EXPORT_RE.match(name)
    if m:
        return m.group(1)
    return None


def inspect_strategy_file(path: str) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"文件不存在: {p}")

    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise ValueError(f"无法解析策略 JSON: {exc}") from exc

    if isinstance(data, list):
        formula = data
        symbol = symbol_from_strategy_path(p)
        best_score = None
        vocab_version = "legacy"
    elif isinstance(data, dict):
        formula = data.get("formula")
        symbol = data.get("symbol") or symbol_from_strategy_path(p)
        best_score = data.get("best_score")
        vocab_version = data.get("vocab_version")
    else:
        raise ValueError("策略文件格式无效")

    if not formula:
        raise ValueError("策略文件缺少 formula 字段")

    formula_decoded = None
    if isinstance(data, dict):
        formula_decoded = data.get("formula_decoded") or _decode_formula(formula)

    return {
        "strategy_file": str(p.resolve()),
        "filename": p.name,
        "symbol": symbol or "",
        "best_score": best_score,
        "vocab_version": vocab_version,
        "formula_decoded": formula_decoded,
        "valid": True,
        "message": "",
    }


def resolve_strategy_file(
    saved_path: str,
    train_symbol: str | None = None,
) -> str:
    """优先使用已保存路径；否则回退到训练品种对应的 best_{symbol}.json。"""
    if saved_path:
        p = Path(saved_path)
        if p.exists():
            return str(p.resolve())

    if train_symbol:
        default = strategy_path_for_symbol(train_symbol)
        if default.exists():
            return str(default.resolve())

    return saved_path or ""


def _step_from_export_name(path: Path) -> int:
    m = _STRATEGY_EXPORT_RE.match(path.name)
    if not m:
        return 0
    try:
        return int(m.group(2))
    except (TypeError, ValueError):
        return 0


def sync_best_strategy_for_symbol(symbol: str) -> dict[str, Any] | None:
    """在策略文件与检查点中选出最高分策略，写入 strategies/best_{symbol}.json。"""
    candidates: list[tuple[float, list[int], int]] = []

    strat = _load_strategy(symbol)
    if strat and strat.get("formula") and strat.get("best_score") is not None:
        step = int(strat.get("train_step") or strat.get("current_step") or 0)
        candidates.append((float(strat["best_score"]), strat["formula"], step))

    for ckpt_path in checkpoint_glob(symbol):
        meta = _load_checkpoint_meta(ckpt_path)
        score = meta.get("best_score")
        formula = meta.get("best_formula")
        if score is None or not formula:
            continue
        candidates.append((float(score), formula, int(meta.get("step") or 0)))

    safe = symbol.replace(".", "_")
    for path in STRATEGIES_DIR.glob(f"strategy_{safe}_*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        formula = data.get("formula")
        score = data.get("best_score")
        if not formula or score is None:
            continue
        candidates.append((float(score), formula, _step_from_export_name(path)))

    if not candidates:
        existing = strategy_path_for_symbol(symbol)
        if existing.exists():
            return inspect_strategy_file(str(existing.resolve()))
        return None

    best_score, best_formula, best_step = max(candidates, key=lambda row: row[0])
    out_path = strategy_path_for_symbol(symbol)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "vocab_version": VOCAB_VERSION,
        "symbol": symbol,
        "formula": best_formula,
        "best_score": best_score,
        "formula_decoded": _decode_formula(best_formula),
        "train_step": best_step,
    }
    if strat:
        for key in ("timeframe", "data_file", "mode", "train_steps"):
            if strat.get(key) is not None:
                payload[key] = strat[key]
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return inspect_strategy_file(str(out_path.resolve()))
