"""火山方舟 (Ark) 网关适配：模型别名、轮换、模型池与账号级错误识别。

与 PA_Agent 的 ``pa_agent/ai/ark_gateway.py`` 保持同一套实测结论：

* 别名：``deepseek-flash`` -> ``deepseek-v4-1-flash-economy-261017``；
* 强制 low 思考强度（方舟只接受 low/medium/high，low 最快）；
* 每次分析按 ``ARK_ROTATION_POOL`` 轮换一个模型（不论设置里填了什么）；
* 某个模型没额度/未开通时在同网关内换下一个；
* 方舟池全挂后再走免费池，最后才用配置的备用通道；
* 账号欠费（账号级 403）时换模型没有意义，直接跳过整池。
"""
from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

ARK_BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"

#: 用户输入的习惯名 -> 方舟真实模型 ID。
ARK_MODEL_ALIASES: dict[str, str] = {
    "deepseek-flash": "deepseek-v4-1-flash-economy-261017",
    "deepseek": "deepseek-v4-1-flash-261017",
    "deepseek-chat": "deepseek-v4-1-flash-261017",
    "glm-flash": "glm-5-3-flash-260828",
    "doubao-flash": "doubao-seed-2-1-lite-260915",
}

#: 出错时依次尝试的方舟模型（额度充足、速度可接受优先）。
ARK_MODEL_POOL: tuple[str, ...] = (
    "doubao-seed-evolving",
    "glm-5-3-flash-260828",
    "deepseek-v4-pro-ga-260813",
    "deepseek-v4-flash-ga-260731",
    "glm-5-2-260617",
    "doubao-seed-2-1-turbo-260628",
    "doubao-seed-2-1-pro-260915",
    "doubao-seed-2-0-lite-260428",
    "doubao-seed-2-0-mini-260428",
    "kimi-k3-1-pro-261024",
    "doubao-seed-2-1-lite-260915",
    "deepseek-v4-1-flash-261017",
    "deepseek-v4-1-flash-economy-261017",
)

#: 每次分析轮换一次的顺序（与 PA_Agent 一致：Kimi → 经济版 → 2.1-lite → …）。
ARK_ROTATION_POOL: tuple[str, ...] = (
    "kimi-k3-1-pro-261024",
    "deepseek-v4-1-flash-economy-261017",
    "doubao-seed-2-1-lite-260915",
    "glm-5-3-flash-260828",
    "doubao-seed-evolving",
    "deepseek-v4-flash-ga-260731",
    "deepseek-v4-pro-ga-260813",
    "glm-5-2-260617",
    "doubao-seed-2-1-turbo-260628",
    "doubao-seed-2-1-pro-260915",
    "doubao-seed-2-0-lite-260428",
    "doubao-seed-2-0-mini-260428",
    "deepseek-v4-1-flash-261017",
)

ARK_THINKING_EFFORT = "low"
ARK_MAX_OUTPUT_TOKENS = 32_768

_ROTATION_LOCK = threading.Lock()


def is_ark_base(base_url: str | None) -> bool:
    """True when *base_url* points at the Ark OpenAI-compatible gateway."""
    url = (base_url or "").strip().lower()
    return "volces.com" in url or "volcengine" in url or "ark.cn-beijing" in url


def resolve_ark_model(model: str | None) -> str:
    """Translate a friendly alias to the real Ark model ID (identity otherwise)."""
    raw = (model or "").strip()
    return ARK_MODEL_ALIASES.get(raw.lower(), raw)


def rotation_state_path() -> Path:
    return Path(__file__).resolve().parents[1] / "ark_rotation_state.json"


def next_rotation_model(state_path: Path | None = None) -> str:
    """Return this analysis's model and advance the persisted cursor."""
    path = state_path or rotation_state_path()
    with _ROTATION_LOCK:
        try:
            cursor = int(json.loads(path.read_text(encoding="utf-8")).get("cursor", 0))
        except (OSError, ValueError, TypeError, AttributeError):
            cursor = 0
        model = ARK_ROTATION_POOL[cursor % len(ARK_ROTATION_POOL)]
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(
                    {"cursor": (cursor + 1) % len(ARK_ROTATION_POOL), "last_model": model},
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
        except OSError:
            pass
        return model


def fallback_models(exclude: str | None = None) -> list[str]:
    """Ark pool minus the model already in use, order preserved."""
    current = (exclude or "").strip()
    return [m for m in ARK_MODEL_POOL if m != current]


def is_account_level_error(exc: BaseException | str | None) -> bool:
    """True for账号级封禁（欠费/冻结），换同一网关的模型也没用。"""
    msg = (str(exc) if exc is not None else "").lower()
    return any(
        marker in msg
        for marker in ("accountoverdue", "overdue balance", "account has been suspended", "arrear")
    )


def chat_payload_extra() -> dict[str, Any]:
    """Request-body fields Ark needs: thinking enabled + low effort."""
    return {"thinking": {"type": "enabled"}, "reasoning_effort": ARK_THINKING_EFFORT}
