"""Layered AI fallback for the training-analysis chat (SenseNova gateways).

Order — mirrors PA_Agent's five-layer ladder:

  1. primary        whatever the user configured (base_url / model / key)
  2. spare model    sensenova-6.8-flash-lite, same gateway and key,
                    used only for rate limits (tpm/quota)
  3. free pool      Kilo keyless ``:free`` models, any failure after that
  4. fallbacks      entries from ``web_settings.json -> ai_fallbacks``
  5. openclaw       the local QClaw gateway, when one is detected

Every switch reports a single neutral line: 正在正常分析，请稍等。

Notes drawn from PA_Agent's measurements (2026-10-30):

* the spare model answers 200 with an EMPTY body when the message list ends
  with an ``assistant`` turn — so the tail is folded into the user turn;
* its thinking shares ``max_tokens`` with the answer, hence the large budget;
* the string ``thinking`` field is rejected by this gateway, so only
  ``reasoning_effort`` is sent.
"""
from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from dataclasses import replace
from typing import Any, Callable, Iterator

from web.ai_providers import ResolvedProvider, stream_chat_completions

logger = logging.getLogger(__name__)

#: The one line shown whenever a fallback layer takes over.
NEUTRAL_PROGRESS_TEXT = "正在正常分析，请稍等。"

#: The spare model served by the same SenseNova gateway.
SPARE_MODEL = "sensenova-6.8-flash-lite"
#: Thinking and answer share this budget (1024/2048 end up empty).
SPARE_MAX_TOKENS = 16384
SPARE_REASONING_EFFORT = "low"

KILO_BASE = "https://api.kilo.ai/api/gateway"
#: Keyless free models, best first (same ordering knowledge as PA_Agent).
KILO_FREE_MODELS = (
    "reflection/beam-free",
    "cohere/north-mini-code:free",
    "recursive/flint-v1.0:free",
    "nvidia/nemotron-3.5-lightning:free",
    "nvidia/nemotron-3.5-super-vl:free",
)

_RATE_LIMIT_MARKERS = (
    "429",
    "rate limit",
    "rate_limit",
    "tpm",
    "rpm",
    "quota",
    "insufficient_quota",
    "too many requests",
)

ProgressFn = Callable[[str], None] | None


def is_sensenova_base(base_url: str | None) -> bool:
    return "sensenova.cn" in (base_url or "").lower()


def is_rate_limit_error(exc: BaseException) -> bool:
    text = f"{type(exc).__name__}: {exc}".lower()
    return any(marker in text for marker in _RATE_LIMIT_MARKERS)


def shape_user_tail(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return a copy that ends with a ``user`` turn (the gateway needs this)."""
    msgs = [
        {"role": str(m.get("role") or "user"), "content": m.get("content") or ""}
        for m in messages
    ]
    if not msgs or msgs[-1]["role"] != "assistant":
        return msgs
    last = msgs.pop()
    tail = "\n\n【上一轮回复，供继续参考】\n" + str(last.get("content") or "")
    if msgs and msgs[-1]["role"] == "user":
        msgs[-1]["content"] = str(msgs[-1].get("content") or "") + tail
    else:
        msgs.append({"role": "user", "content": "请基于以上内容继续完成本次任务。" + tail})
    return msgs


def spare_provider(resolved: ResolvedProvider) -> ResolvedProvider:
    """Same gateway and key, different (cheaper/faster) model."""
    return replace(resolved, model=SPARE_MODEL, label=f"{resolved.label} · 备用模型")


def _stream_kilo(model: str, messages: list[dict[str, Any]], *, max_tokens: int, timeout: float) -> Iterator[str]:
    """Stream one keyless Kilo free model."""
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "stream": True,
    }
    req = urllib.request.Request(
        f"{KILO_BASE}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
            "User-Agent": "AlphaMaster-AI-Analyze",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for raw in resp:
            line = raw.decode("utf-8", errors="replace").strip()
            if not line or not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if not data or data == "[DONE]":
                if data == "[DONE]":
                    break
                continue
            try:
                chunk = json.loads(data)
            except json.JSONDecodeError:
                continue
            choices = chunk.get("choices") or []
            if not choices:
                continue
            delta = choices[0].get("delta") or {}
            text = delta.get("content") or delta.get("reasoning_content") or ""
            if text:
                yield text


def load_fallbacks() -> list[ResolvedProvider]:
    """Read ``ai_fallbacks`` from web_settings.json (missing/empty -> none)."""
    try:
        from web.settings import load_settings

        raw = load_settings().get("ai_fallbacks") or []
    except Exception:  # noqa: BLE001
        logger.debug("load_fallbacks: settings unavailable", exc_info=True)
        return []
    out: list[ResolvedProvider] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        model = str(item.get("model") or "").strip()
        base = str(item.get("base_url") or "").strip()
        key = str(item.get("api_key") or "").strip()
        if not (model and base and key):
            continue
        out.append(
            ResolvedProvider(
                provider="deepseek",
                model=model,
                base_url=base.rstrip("/"),
                api_key=key,
                label=str(item.get("label") or "备用通道"),
                needs_user_key=False,
            )
        )
    return out


def _openclaw_provider() -> ResolvedProvider | None:
    try:
        from web.ai_providers import detect_qclaw, resolve_provider

        if not detect_qclaw(require_alive=True):
            return None
        return resolve_provider("openclaw")
    except Exception:  # noqa: BLE001
        logger.debug("openclaw layer unavailable", exc_info=True)
        return None


def _progress(on_progress: ProgressFn) -> None:
    if on_progress is None:
        return
    try:
        on_progress(NEUTRAL_PROGRESS_TEXT)
    except Exception:  # noqa: BLE001
        logger.debug("progress callback failed", exc_info=True)


def stream_with_fallbacks(
    resolved: ResolvedProvider,
    messages: list[dict[str, Any]],
    *,
    max_tokens: int = 4096,
    timeout: float = 180.0,
    on_progress: ProgressFn = None,
) -> Iterator[tuple[str, str]]:
    """Yield ``("text", chunk)`` / ``("notice", line)`` across the five layers.

    Once a layer has produced output it is never abandoned for a later layer:
    partial text must not be duplicated. An empty answer counts as a failure.
    """
    sense = is_sensenova_base(resolved.base_url)
    errors: list[str] = []

    def _run(fn, layer: str) -> Iterator[tuple[str, str]]:
        produced = False
        for chunk in fn():
            produced = True
            yield ("text", chunk)
        if not produced:
            raise RuntimeError(f"{layer} returned no content")

    # ── Layer 1: primary ────────────────────────────────────────────────────
    try:
        yield from _run(
            lambda: stream_chat_completions(
                resolved, messages, max_tokens=max_tokens, timeout=timeout
            ),
            "primary",
        )
        return
    except Exception as exc:  # noqa: BLE001
        errors.append(f"primary: {exc}")
        primary_exc = exc

    rate_limited = is_rate_limit_error(primary_exc)
    if sense:
        _progress(on_progress)
        yield ("notice", NEUTRAL_PROGRESS_TEXT)

    # ── Layer 2: spare model (rate limits only) ─────────────────────────────
    if sense and rate_limited:
        try:
            spare = spare_provider(resolved)
            yield from _run(
                lambda: stream_chat_completions(
                    spare,
                    shape_user_tail(messages),
                    max_tokens=max(SPARE_MAX_TOKENS, max_tokens),
                    timeout=max(timeout, 300.0),
                    reasoning_effort=SPARE_REASONING_EFFORT,
                ),
                "spare",
            )
            return
        except Exception as exc:  # noqa: BLE001
            errors.append(f"spare: {exc}")

    # ── Layer 3: free pool (SenseNova gateways) ─────────────────────────────
    if sense:
        for model in KILO_FREE_MODELS:
            try:
                yield from _run(
                    lambda m=model: _stream_kilo(
                        m, shape_user_tail(messages), max_tokens=max_tokens, timeout=timeout
                    ),
                    f"free:{model}",
                )
                return
            except Exception as exc:  # noqa: BLE001
                errors.append(f"free:{model}: {exc}")
                continue

    # ── Layer 4: configured fallbacks ──────────────────────────────────────
    for fb in load_fallbacks():
        try:
            yield from _run(
                lambda f=fb: stream_chat_completions(
                    f, messages, max_tokens=max_tokens, timeout=timeout
                ),
                f"fallback:{fb.model}",
            )
            return
        except Exception as exc:  # noqa: BLE001
            errors.append(f"fallback:{fb.model}: {exc}")
            continue

    # ── Layer 5: local openclaw gateway ────────────────────────────────────
    oc = _openclaw_provider()
    if oc is not None:
        try:
            yield from _run(
                lambda: stream_chat_completions(
                    oc, messages, max_tokens=max_tokens, timeout=timeout
                ),
                "openclaw",
            )
            return
        except Exception as exc:  # noqa: BLE001
            errors.append(f"openclaw: {exc}")

    raise RuntimeError(errors[-1] if errors else "AI 请求失败")
