"""Layered fallback ladder for the training-analysis chat."""
from __future__ import annotations

import pytest

from web.ai_fallback import (
    NEUTRAL_PROGRESS_TEXT,
    SPARE_MAX_TOKENS,
    SPARE_MODEL,
    is_rate_limit_error,
    is_sensenova_base,
    shape_user_tail,
    spare_provider,
    stream_with_fallbacks,
)
from web.ai_providers import ResolvedProvider

SENSENOVA = "https://token.sensenova.cn/v1"
OTHER = "https://api.deepseek.com"


def _prov(base_url: str = SENSENOVA, model: str = "deepseek-flash") -> ResolvedProvider:
    return ResolvedProvider(
        provider="deepseek",
        model=model,
        base_url=base_url,
        api_key="sk-test",
        label="SenseNova",
        needs_user_key=True,
    )


def _patch(monkeypatch, fake):
    monkeypatch.setattr("web.ai_fallback.stream_chat_completions", fake)


# ── helpers ─────────────────────────────────────────────────────────────────

def test_is_sensenova_base():
    assert is_sensenova_base(SENSENOVA) is True
    assert is_sensenova_base(OTHER) is False
    assert is_sensenova_base(None) is False


@pytest.mark.parametrize("msg", [
    "HTTP 429: tpm exhausted",
    "quota_exceeded_error",
    "rate limit reached",
])
def test_rate_limit_detection_positive(msg):
    assert is_rate_limit_error(RuntimeError(msg)) is True


@pytest.mark.parametrize("msg", ["connection reset", "HTTP 500", "timeout"])
def test_rate_limit_detection_negative(msg):
    assert is_rate_limit_error(RuntimeError(msg)) is False


def test_shape_user_tail_folds_assistant_turn():
    msgs = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "a"},
    ]
    out = shape_user_tail(msgs)
    assert out[-1]["role"] == "user"
    assert "q" in out[-1]["content"] and "a" in out[-1]["content"]
    assert msgs[-1]["role"] == "assistant"  # input untouched


def test_spare_provider_keeps_gateway_and_key():
    spare = spare_provider(_prov())
    assert spare.model == SPARE_MODEL
    assert spare.base_url == SENSENOVA
    assert spare.api_key == "sk-test"


# ── layer 1 ─────────────────────────────────────────────────────────────────

def test_primary_success_emits_only_text(monkeypatch):
    def fake(res, msgs, **kw):
        yield "hello"

    _patch(monkeypatch, fake)
    out = list(stream_with_fallbacks(_prov(), [{"role": "user", "content": "hi"}]))
    assert out == [("text", "hello")]


# ── layer 2: spare model ────────────────────────────────────────────────────

def test_rate_limit_switches_to_spare_model(monkeypatch):
    calls = []

    def fake(res, msgs, **kw):
        calls.append({"model": res.model, "kw": kw, "roles": [m["role"] for m in msgs]})
        if res.model == "deepseek-flash":
            raise RuntimeError("HTTP 429: tpm exhausted")
        yield "spare answer"

    _patch(monkeypatch, fake)
    msgs = [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]
    out = list(stream_with_fallbacks(_prov(), msgs))

    assert ("notice", NEUTRAL_PROGRESS_TEXT) in out
    assert ("text", "spare answer") in out

    spare = calls[1]
    assert spare["model"] == SPARE_MODEL
    assert spare["kw"]["reasoning_effort"] == "low"
    assert spare["kw"]["max_tokens"] >= SPARE_MAX_TOKENS
    assert spare["roles"][-1] == "user", "assistant-tailed input must be folded"


def test_non_rate_limit_error_skips_the_spare_model(monkeypatch):
    models = []

    def fake(res, msgs, **kw):
        models.append(res.model)
        if res.model == "deepseek-flash":
            raise RuntimeError("connection reset")
        yield "fallback answer"

    def free(model, msgs, **kw):
        raise RuntimeError("free down")
        yield  # pragma: no cover

    _patch(monkeypatch, fake)
    monkeypatch.setattr("web.ai_fallback._stream_kilo", free)
    monkeypatch.setattr(
        "web.ai_fallback.load_fallbacks", lambda: [_prov(model="glm-4-flash")]
    )
    out = list(stream_with_fallbacks(_prov(), [{"role": "user", "content": "hi"}]))

    assert SPARE_MODEL not in models
    assert ("text", "fallback answer") in out


# ── layer 3: free pool ──────────────────────────────────────────────────────

def test_spare_failure_falls_through_to_free_pool(monkeypatch):
    def fake(res, msgs, **kw):
        raise RuntimeError("429 rate limit")
        yield  # pragma: no cover

    def free(model, msgs, **kw):
        if model.endswith("beam-free"):
            raise RuntimeError("502 upstream")
        yield "free answer"

    _patch(monkeypatch, fake)
    monkeypatch.setattr("web.ai_fallback._stream_kilo", free)
    out = list(stream_with_fallbacks(_prov(), [{"role": "user", "content": "hi"}]))

    assert ("text", "free answer") in out


# ── layers 4/5 and total failure ────────────────────────────────────────────

def test_non_sensenova_goes_straight_to_configured_fallbacks(monkeypatch):
    models = []

    def fake(res, msgs, **kw):
        models.append(res.model)
        if res.model == "deepseek-flash":
            raise RuntimeError("HTTP 429")
        yield "fb answer"

    _patch(monkeypatch, fake)
    monkeypatch.setattr(
        "web.ai_fallback.load_fallbacks", lambda: [_prov(base_url=OTHER, model="glm-4-flash")]
    )
    out = list(stream_with_fallbacks(_prov(base_url=OTHER), [{"role": "user", "content": "hi"}]))

    assert models == ["deepseek-flash", "glm-4-flash"]
    assert ("text", "fb answer") in out


def test_all_layers_failing_raises(monkeypatch):
    def fake(res, msgs, **kw):
        raise RuntimeError("primary down")
        yield  # pragma: no cover

    def free(model, msgs, **kw):
        raise RuntimeError("free down")
        yield  # pragma: no cover

    _patch(monkeypatch, fake)
    monkeypatch.setattr("web.ai_fallback._stream_kilo", free)
    monkeypatch.setattr("web.ai_fallback.load_fallbacks", lambda: [])
    monkeypatch.setattr("web.ai_fallback._openclaw_provider", lambda: None)

    with pytest.raises(RuntimeError):
        list(stream_with_fallbacks(_prov(), [{"role": "user", "content": "hi"}]))


def test_empty_answer_counts_as_failure(monkeypatch):
    def fake(res, msgs, **kw):
        return iter(())

    def free(model, msgs, **kw):
        yield "rescued"

    _patch(monkeypatch, fake)
    monkeypatch.setattr("web.ai_fallback._stream_kilo", free)
    out = list(stream_with_fallbacks(_prov(), [{"role": "user", "content": "hi"}]))
    assert ("text", "rescued") in out
