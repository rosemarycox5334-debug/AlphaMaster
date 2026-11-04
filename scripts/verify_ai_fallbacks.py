"""Verify AlphaMaster's layered AI fallback against the real SenseNova gateway.

The primary model is forced to fail with 429 (what the gateway really does on a
large prompt), then the ladder takes over exactly as it does in production.

Usage (any directory):
    python scripts/verify_ai_fallbacks.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import web.ai_fallback as fb  # noqa: E402
from web.ai_providers import ResolvedProvider  # noqa: E402

SENSENOVA = "https://token.sensenova.cn/v1"
#: SenseNova key lives in PA_Agent's local config (never committed here).
PA_SETTINGS = Path(r"D:\cl\PA_Agent\config\settings.json")


def load_key() -> str:
    try:
        data = json.loads(PA_SETTINGS.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        print(f"[!] 读不到 SenseNova key：{exc}")
        return ""
    return str((data.get("provider") or {}).get("api_key") or "").strip()


def main() -> int:
    key = load_key()
    if not key:
        return 2
    resolved = ResolvedProvider(
        provider="deepseek",
        model="deepseek-flash",
        base_url=SENSENOVA,
        api_key=key,
        label="SenseNova",
        needs_user_key=True,
    )
    print(f"primary model={resolved.model} base_url={resolved.base_url}")
    print(f"spare model={fb.SPARE_MODEL}\n")

    real = fb.stream_chat_completions

    def fake(res, msgs, **kw):
        if res.model == resolved.model:
            raise RuntimeError("HTTP 429: tpm exhausted (simulated primary)")
        return real(res, msgs, **kw)

    fb.stream_chat_completions = fake  # type: ignore[assignment]

    messages = [{"role": "user", "content": "只回答两个字：收到"}]
    events: list[tuple[str, str]] = []
    text = ""
    t0 = time.monotonic()
    try:
        for kind, chunk in fb.stream_with_fallbacks(resolved, messages, max_tokens=2048):
            events.append((kind, chunk))
            if kind == "text":
                text += chunk
    except Exception as exc:  # noqa: BLE001
        print(f"[x] 全层失败（{time.monotonic() - t0:.1f}s）：{type(exc).__name__}: {exc}")
        return 1
    finally:
        fb.stream_chat_completions = real  # type: ignore[assignment]

    elapsed = time.monotonic() - t0
    notices = [c for k, c in events if k == "notice"]
    print(f"[ok] 用时 {elapsed:.1f}s，事件 {len(events)} 个")
    print(f"[ok] 中性提示：{sorted(set(notices))}")
    print(f"[ok] 正文 {len(text)} 字：{text.strip()[:120]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
