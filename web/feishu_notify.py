"""飞书自定义机器人通知（信号方向转折时推送文本）。

参考 PA_Agent 的 webhook + 可选签名校验写法。
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
import urllib.error
import urllib.request
from typing import Any

from web.settings import load_settings

_DIR_CN = {
    "LONG": "看涨",
    "SHORT": "看跌",
    "FLAT": "不确定",
}


def direction_cn(direction: str | None) -> str:
    if not direction:
        return "未知"
    return _DIR_CN.get(str(direction).upper(), str(direction))


def strength_cn(strength: float | None, direction: str | None) -> str:
    if direction == "FLAT" or direction is None:
        return "没把握"
    s = max(0.0, min(1.0, float(strength or 0.0)))
    if s < 0.2:
        return "一点把握"
    if s < 0.4:
        return "把握不大"
    if s < 0.6:
        return "一半把握"
    if s < 0.8:
        return "比较有把握"
    return "很有把握"


def _gen_sign(secret: str, timestamp: int) -> str:
    string_to_sign = f"{timestamp}\n{secret}"
    digest = hmac.new(string_to_sign.encode("utf-8"), digestmod=hashlib.sha256).digest()
    return base64.b64encode(digest).decode("utf-8")


def send_text(
    text: str,
    *,
    webhook_url: str | None = None,
    secret: str | None = None,
    timeout_s: float = 10.0,
) -> tuple[bool, str]:
    """向飞书群发送纯文本。返回 (ok, message)。"""
    settings = load_settings()
    url = (webhook_url if webhook_url is not None else settings.get("feishu_webhook_url") or "").strip()
    if not url:
        return False, "未配置 Webhook URL"
    sec = (secret if secret is not None else settings.get("feishu_secret") or "").strip()

    payload: dict[str, Any] = {
        "msg_type": "text",
        "content": {"text": text},
    }
    if sec:
        ts = int(time.time())
        payload["timestamp"] = str(ts)
        payload["sign"] = _gen_sign(sec, ts)

    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
        data = json.loads(raw) if raw else {}
    except urllib.error.HTTPError as exc:
        try:
            detail = exc.read().decode("utf-8", errors="replace")
        except Exception:
            detail = str(exc)
        return False, f"HTTP {exc.code}: {detail}"
    except Exception as exc:  # noqa: BLE001
        return False, str(exc)

    if data.get("code") == 0 or data.get("StatusCode") == 0:
        return True, "ok"

    code = data.get("code", data.get("StatusCode", "?"))
    msg = data.get("msg", data.get("StatusMessage", ""))
    hint = ""
    if code == 19021:
        hint = "（签名校验失败，请检查密钥或留空禁用签名）"
    elif code == 19024:
        hint = "（关键词校验失败，请检查机器人自定义关键词）"
    elif code == 19022:
        hint = "（IP 不在白名单）"
    return False, f"飞书返回 code={code} msg={msg}{hint}"


def notify_direction_flip(
    *,
    symbol: str,
    timeframe: str,
    strategy_name: str,
    prev_direction: str,
    new_direction: str,
    strength: float | None = None,
    factor_value: float | None = None,
) -> tuple[bool, str]:
    """信号方向发生转折时推送提醒。"""
    settings = load_settings()
    if not settings.get("feishu_enabled"):
        return False, "飞书通知未启用"
    if not (settings.get("feishu_webhook_url") or "").strip():
        return False, "未配置 Webhook URL"

    prev_cn = direction_cn(prev_direction)
    new_cn = direction_cn(new_direction)
    grasp = strength_cn(strength, new_direction)
    factor_s = f"{factor_value:+.4f}" if factor_value is not None else "—"

    text = (
        f"【AlphaMaster 信号转折】\n"
        f"{symbol} · {timeframe}\n"
        f"上次判断：{prev_cn}\n"
        f"本次判断：{new_cn}（{grasp}）\n"
        f"策略：{strategy_name}\n"
        f"因子：{factor_s}"
    )
    return send_text(text)
