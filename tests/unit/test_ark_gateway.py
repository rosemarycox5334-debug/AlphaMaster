"""火山方舟接入 AlphaMaster 的单元测试。

覆盖与 PA_Agent 一致的五条行为：
  1. 别名 deepseek-flash -> 经济版；
  2. max_tokens 按方舟上限裁剪；
  3. 请求体固定带 thinking=enabled + effort=low；
  4. 每次分析轮换模型（游标持久化、重启继续）；
  5. 降级顺序：同网关模型池 -> 免费池 -> 配置的备用通道。
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from web.ark_gateway import (
    ARK_BASE_URL,
    ARK_ROTATION_POOL,
    ARK_THINKING_EFFORT,
    fallback_models,
    is_account_level_error,
    is_ark_base,
    next_rotation_model,
    resolve_ark_model,
)
from web.ai_providers import ResolvedProvider, _clamp_max_tokens, resolve_provider


def _resolved(model: str = "kimi-k3-1-pro-261024") -> ResolvedProvider:
    return ResolvedProvider(
        provider="deepseek", model=model, base_url=ARK_BASE_URL,
        api_key="ark-test", label="火山方舟",
    )


@pytest.fixture()
def state_file(tmp_path, monkeypatch):
    path = tmp_path / "ark_rotation_state.json"
    monkeypatch.setattr("web.ark_gateway.rotation_state_path", lambda: path)
    return path


def test_is_ark_base():
    assert is_ark_base(ARK_BASE_URL)
    assert is_ark_base("https://ark.volcengine.com/api/v3")
    assert not is_ark_base("https://token.sensenova.cn/v1")


def test_alias_mapping():
    assert resolve_ark_model("deepseek-flash") == "deepseek-v4-1-flash-economy-261017"
    assert resolve_ark_model("glm-flash") == "glm-5-3-flash-260828"
    assert resolve_ark_model("doubao-seed-evolving") == "doubao-seed-evolving"


def test_max_tokens_clamped_for_ark():
    assert _clamp_max_tokens(ARK_BASE_URL, "glm-5-3-flash-260828", 400_000) == 32_768
    assert _clamp_max_tokens(ARK_BASE_URL, "doubao-seed-evolving", 4096) == 4096


def test_resolution_rotates_models(state_file):
    seen = []
    for _ in range(3):
        r = resolve_provider("deepseek", "ark-test", base_url=ARK_BASE_URL, model="deepseek-flash")
        seen.append(r.model)
        assert r.label.startswith("火山方舟")
    assert seen == ["kimi-k3-1-pro-261024",
                    "deepseek-v4-1-flash-economy-261017",
                    "doubao-seed-2-1-lite-260915"]


def test_rotation_ignores_configured_model(state_file):
    """不管 model 栏填什么，方舟下都按轮换池走。"""
    a = resolve_provider("deepseek", "k", base_url=ARK_BASE_URL, model="whatever-user-typed")
    b = resolve_provider("deepseek", "k", base_url=ARK_BASE_URL, model="doubao-seed-evolving")
    assert a.model == ARK_ROTATION_POOL[0]
    assert b.model == ARK_ROTATION_POOL[1]


def test_rotation_persists_across_calls(state_file):
    first = next_rotation_model(state_file)
    second = next_rotation_model(state_file)
    assert first != second
    assert json.loads(state_file.read_text(encoding="utf-8"))["cursor"] == 2


def test_non_ark_gateway_does_not_rotate(state_file):
    r = resolve_provider("deepseek", "k", base_url="https://token.sensenova.cn/v1",
                         model="deepseek-v4-flash")
    assert r.model == "deepseek-v4-flash"
    assert not state_file.exists()


def test_fallback_pool_excludes_current():
    models = fallback_models("glm-5-3-flash-260828")
    assert "glm-5-3-flash-260828" not in models
    assert len(models) == 12


def test_account_level_error_detection():
    assert is_account_level_error("403 AccountOverdueError: overdue balance")
    assert not is_account_level_error("429 rate limit")


def test_chat_payload_extra_forces_low():
    from web.ark_gateway import chat_payload_extra

    extra = chat_payload_extra()
    assert extra["thinking"]["type"] == "enabled"
    assert extra["reasoning_effort"] == ARK_THINKING_EFFORT == "low"


def test_request_body_carries_ark_thinking(tmp_path):
    """真正发出去的请求体必须带 thinking=enabled 且不超 max_tokens 上限。"""
    import web.ai_providers as ap

    captured = {}

    class _Resp:
        def __iter__(self):
            yield b'data: {"choices":[{"delta":{"content":"OK"}}]}\n'
            yield b'data: [DONE]\n'

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def fake_urlopen(req, timeout=0):
        captured["body"] = json.loads(req.data.decode("utf-8"))
        return _Resp()

    with patch("urllib.request.urlopen", fake_urlopen):
        text = "".join(ap.stream_chat_completions(
            _resolved(), [{"role": "user", "content": "hi"}], max_tokens=400_000))
    assert text == "OK"
    body = captured["body"]
    assert body["thinking"] == {"type": "enabled"}
    assert body["reasoning_effort"] == "low"
    assert body["max_tokens"] == 32_768
    assert "temperature" not in body


def test_fallback_order_ark_then_free_then_configured(monkeypatch):
    """方舟失败后：先同网关模型池，再免费池，最后才是配置的备用通道。"""
    from web import ai_fallback as fb

    calls: list[tuple[str, str]] = []

    def fake_stream(resolved, messages, **kw):
        if "volces" in resolved.base_url:
            calls.append(("ark", resolved.model))
        else:
            calls.append(("fallback", resolved.model))
        raise RuntimeError("HTTP 500 server error")
        yield ""  # pragma: no cover

    def fake_kilo(model, messages, **kw):
        calls.append(("free", model))
        raise RuntimeError("free down")
        yield ""  # pragma: no cover

    monkeypatch.setattr(fb, "stream_chat_completions", fake_stream)
    monkeypatch.setattr(fb, "_stream_kilo", fake_kilo)
    monkeypatch.setattr(fb, "load_fallbacks", lambda: [
        ResolvedProvider(provider="deepseek", model="glm-4-flash",
                         base_url="https://open.bigmodel.cn/api/paas/v4",
                         api_key="glm-key", label="GLM 备用")
    ])

    events = []
    try:
        for event in fb.stream_with_fallbacks(_resolved(), [{"role": "user", "content": "hi"}]):
            events.append(event)
    except RuntimeError:
        pass  # 所有层都失败时会抛错，本测试只关心调用顺序
    kinds = [k for k, _ in calls]
    assert "ark" in kinds
    assert kinds.index("free") > max(i for i, k in enumerate(kinds) if k == "ark")
    assert kinds[-1] == "fallback"
    assert any(kind == "notice" for kind, _ in events)


def test_account_overdue_skips_ark_pool(monkeypatch):
    """账号欠费时不应逐个试方舟模型，直接去免费池。"""
    from web import ai_fallback as fb

    calls: list[str] = []

    def fake_stream(resolved, messages, **kw):
        calls.append("primary")
        raise RuntimeError("403 AccountOverdueError: overdue balance")
        yield ""  # pragma: no cover

    def fake_kilo(model, messages, **kw):
        calls.append("free")
        yield "免费OK"

    monkeypatch.setattr(fb, "stream_chat_completions", fake_stream)
    monkeypatch.setattr(fb, "_stream_kilo", fake_kilo)
    monkeypatch.setattr(fb, "load_fallbacks", lambda: [])

    out = [v for k, v in fb.stream_with_fallbacks(_resolved(), [{"role": "user", "content": "hi"}]) if k == "text"]
    assert out == ["免费OK"]
    assert calls == ["primary", "free"]
