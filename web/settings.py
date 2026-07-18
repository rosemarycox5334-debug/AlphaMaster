"""Persisted UI settings for the training web console."""
from __future__ import annotations

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SETTINGS_PATH = PROJECT_ROOT / "web_settings.json"
STRATEGIES_DIR = PROJECT_ROOT / "strategies"

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
    # tqsdk 天勤量化账号（国内期货实时数据源）
    "tqsdk_user": "七斗居士",
    "tqsdk_password": "ghhkphs8",
}


def _as_pct(value, default: float) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return default
    if v < 0:
        return default
    return v


def _is_ephemeral_data_path(path: str) -> bool:
    """Stale pytest temp parquet paths (not valid training data)."""
    norm = str(path or "").replace("\\", "/").lower()
    if "pytest-of-" not in norm:
        return False
    return (
        "/appdata/local/temp/" in norm
        or norm.startswith("/tmp/")
        or "/temp/" in norm
    )


def _is_production_settings_path() -> bool:
    try:
        return SETTINGS_PATH.resolve() == (PROJECT_ROOT / "web_settings.json").resolve()
    except OSError:
        return False


def _is_usable_data_file(path: str) -> bool:
    p = Path(str(path or "").strip())
    return p.is_file() and p.suffix.lower() == ".parquet"


def _should_replace_last_data_file(path: str) -> bool:
    cur = str(path or "").strip()
    if not cur:
        return True
    if not Path(cur).is_file():
        return True
    return _is_ephemeral_data_path(cur)


def _data_file_from_strategy_json(path: str) -> str | None:
    p = Path(str(path or "").strip())
    if not p.is_file():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    candidate = str(data.get("data_file") or "").strip()
    if _is_usable_data_file(candidate):
        return str(Path(candidate).resolve())
    return None


def _recover_last_data_file(current: dict) -> str:
    cur = str(current.get("last_data_file") or "").strip()
    if cur and not _should_replace_last_data_file(cur):
        return str(Path(cur).resolve())

    for strategy_path in (
        str(current.get("last_strategy_file") or "").strip(),
        *(str(p) for p in sorted(STRATEGIES_DIR.glob("best_*.json")) if p.is_file()),
    ):
        if not strategy_path:
            continue
        candidate = _data_file_from_strategy_json(strategy_path)
        if candidate:
            return candidate
    return cur


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
    out["tqsdk_user"] = str(out.get("tqsdk_user") or "").strip()
    out["tqsdk_password"] = str(out.get("tqsdk_password") or "").strip()
    recovered = _recover_last_data_file(out)
    if recovered != out.get("last_data_file") and _is_production_settings_path():
        out["last_data_file"] = recovered
        if recovered:
            SETTINGS_PATH.write_text(
                json.dumps(out, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
    elif recovered != out.get("last_data_file"):
        out["last_data_file"] = recovered
    return out


def save_settings(data: dict) -> dict:
    current = load_settings()
    if "last_data_file" in data:
        path = str(data["last_data_file"] or "").strip()
        if (
            path
            and _is_ephemeral_data_path(path)
            and _is_production_settings_path()
        ):
            data = {k: v for k, v in data.items() if k != "last_data_file"}
        else:
            current["last_data_file"] = path
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
    if "tqsdk_user" in data:
        current["tqsdk_user"] = str(data["tqsdk_user"] or "").strip()
    if "tqsdk_password" in data:
        current["tqsdk_password"] = str(data["tqsdk_password"] or "").strip()
    SETTINGS_PATH.write_text(
        json.dumps(current, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return current
