"""Persisted UI settings for the training web console."""
from __future__ import annotations

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SETTINGS_PATH = PROJECT_ROOT / "web_settings.json"

_DEFAULT = {
    "last_data_file": "",
    "last_strategy_file": "",
    "debug_mode": False,
    "ai_provider": "deepseek",
    "ai_api_key": "",
    # 回测单边成本（单位 %）：手续费 0.02% + 滑点 0.01% ≈ 常见加密货币轻度成本
    "bt_commission_pct": 0.02,
    "bt_slippage_pct": 0.01,
    # 实时分析监控清单：[{source, symbol, timeframe, strategy_file}, ...]
    "realtime_watches": [],
    # 飞书机器人（信号转折提醒，仅文本）
    "feishu_enabled": False,
    "feishu_webhook_url": "",
    "feishu_secret": "",
}


def _as_pct(value, default: float) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return default
    if v < 0:
        return default
    return v


def load_settings() -> dict:
    if not SETTINGS_PATH.exists():
        return dict(_DEFAULT)
    try:
        data = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return dict(_DEFAULT)
    out = dict(_DEFAULT)
    out.update({k: v for k, v in data.items() if k in _DEFAULT})
    out["debug_mode"] = bool(out.get("debug_mode", False))
    out["last_strategy_file"] = str(out.get("last_strategy_file") or "").strip()
    out["ai_provider"] = str(out.get("ai_provider") or "deepseek").strip().lower()
    if out["ai_provider"] not in ("deepseek", "openclaw", "openclaw_wb"):
        out["ai_provider"] = "deepseek"
    out["ai_api_key"] = str(out.get("ai_api_key") or "").strip()
    out["bt_commission_pct"] = _as_pct(
        out.get("bt_commission_pct"), _DEFAULT["bt_commission_pct"]
    )
    out["bt_slippage_pct"] = _as_pct(
        out.get("bt_slippage_pct"), _DEFAULT["bt_slippage_pct"]
    )
    watches = out.get("realtime_watches")
    if not isinstance(watches, list):
        watches = []
    cleaned = []
    for w in watches:
        if not isinstance(w, dict):
            continue
        src = str(w.get("source") or "").strip()
        sym = str(w.get("symbol") or "").strip()
        tf = str(w.get("timeframe") or "").strip()
        sf = str(w.get("strategy_file") or "").strip()
        if src and sym and tf and sf:
            cleaned.append(
                {"source": src, "symbol": sym, "timeframe": tf, "strategy_file": sf}
            )
    out["realtime_watches"] = cleaned
    out["feishu_enabled"] = bool(out.get("feishu_enabled", False))
    out["feishu_webhook_url"] = str(out.get("feishu_webhook_url") or "").strip()
    out["feishu_secret"] = str(out.get("feishu_secret") or "").strip()
    return out


def save_settings(data: dict) -> dict:
    current = load_settings()
    if "last_data_file" in data:
        current["last_data_file"] = str(data["last_data_file"] or "").strip()
    if "last_strategy_file" in data:
        current["last_strategy_file"] = str(data["last_strategy_file"] or "").strip()
    if "debug_mode" in data:
        current["debug_mode"] = bool(data["debug_mode"])
    if "ai_provider" in data:
        provider = str(data["ai_provider"] or "deepseek").strip().lower()
        current["ai_provider"] = (
            provider if provider in ("deepseek", "openclaw", "openclaw_wb") else "deepseek"
        )
    if "ai_api_key" in data:
        current["ai_api_key"] = str(data["ai_api_key"] or "").strip()
    if "bt_commission_pct" in data:
        current["bt_commission_pct"] = _as_pct(
            data["bt_commission_pct"], _DEFAULT["bt_commission_pct"]
        )
    if "bt_slippage_pct" in data:
        current["bt_slippage_pct"] = _as_pct(
            data["bt_slippage_pct"], _DEFAULT["bt_slippage_pct"]
        )
    if "realtime_watches" in data:
        watches = data["realtime_watches"]
        if not isinstance(watches, list):
            watches = []
        cleaned = []
        for w in watches:
            if not isinstance(w, dict):
                continue
            src = str(w.get("source") or "").strip()
            sym = str(w.get("symbol") or "").strip()
            tf = str(w.get("timeframe") or "").strip()
            sf = str(w.get("strategy_file") or "").strip()
            if src and sym and tf and sf:
                cleaned.append(
                    {
                        "source": src,
                        "symbol": sym,
                        "timeframe": tf,
                        "strategy_file": sf,
                    }
                )
        current["realtime_watches"] = cleaned
    if "feishu_enabled" in data:
        current["feishu_enabled"] = bool(data["feishu_enabled"])
    if "feishu_webhook_url" in data:
        current["feishu_webhook_url"] = str(data["feishu_webhook_url"] or "").strip()
    if "feishu_secret" in data:
        current["feishu_secret"] = str(data["feishu_secret"] or "").strip()
    SETTINGS_PATH.write_text(
        json.dumps(current, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return current
