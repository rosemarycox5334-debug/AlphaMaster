"""AI provider resolution for training analysis.

Supports:
  - deepseek: fixed model deepseek-v4-flash @ https://api.deepseek.com (user API key)
  - openclaw: local QClaw gateway Agent (token from ~/.qclaw/openclaw.json)
  - openclaw_wb: WorkBuddy / copilot.tencent.com (local session token)

Does NOT support openclaw_cs (Cursor SDK).
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEEPSEEK_MODEL = "deepseek-v4-flash"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"

_QCLAW_CONFIG_CANDIDATES = (
    Path.home() / ".qclaw" / "openclaw.json",
    Path("~/.qclaw/openclaw.json").expanduser(),
)
_OPENCLAW_MODEL = "openclaw"

_WORKBUDDY_MODEL = "openclaw_wb"
_WORKBUDDY_API_MODEL = "auto"
_WORKBUDDY_DEFAULT_ENDPOINT = "https://copilot.tencent.com"
_WORKBUDDY_API_PATH = "/v2"
_WORKBUDDY_CONFIG_DIR = Path(
    os.environ.get("WORKBUDDY_CONFIG_DIR", "") or (Path.home() / ".workbuddy")
)
_WORKBUDDY_TOKEN_FILE = _WORKBUDDY_CONFIG_DIR / ".wb_token"
_WORKBUDDY_SESSION_PATH = _WORKBUDDY_CONFIG_DIR / "app" / "session"
_WORKBUDDY_LOCAL_STATE = _WORKBUDDY_SESSION_PATH / "Local State"
_WORKBUDDY_AUTH_EXPECTED = (
    Path(os.environ.get("LOCALAPPDATA", "") or Path.home() / "AppData" / "Local")
    / "CodeBuddyExtension"
    / "Data"
    / "Public"
    / "auth"
    / "workbuddy-desktop.info"
)

PROVIDERS = ("deepseek", "openclaw", "openclaw_wb")


@dataclass
class ResolvedProvider:
    provider: str
    model: str
    base_url: str
    api_key: str
    label: str
    needs_user_key: bool = False


def detect_qclaw(*, require_alive: bool = False) -> bool:
    """True when local QClaw config looks usable.

    When *require_alive* is True, also require the gateway /models probe to succeed.
    """
    info = _qclaw_gateway_info()
    if info is None:
        return False
    config_path = _find_qclaw_config()
    if config_path is None:
        return False
    data = _read_json(config_path)
    if not data:
        return False
    chat = (
        data.get("gateway", {})
        .get("http", {})
        .get("endpoints", {})
        .get("chatCompletions", {})
    )
    if not (bool(chat.get("enabled", False)) and bool(info[2])):
        return False
    if not require_alive:
        return True
    host, port, token = info
    base = f"http://{host}:{port}/v1"
    return _probe_qclaw_gateway(base, token)


def detect_workbuddy() -> bool:
    if os.environ.get("CLIENT_INFO_PRODUCT_NAME", "") == "WorkBuddy":
        return True
    if os.environ.get("WORKBUDDY_CONFIG_DIR"):
        return True
    if _workbuddy_auth_session_candidates():
        return True
    return _WORKBUDDY_CONFIG_DIR.exists()


def _probe_qclaw_gateway(base_url: str, token: str, *, timeout: float = 5.0) -> bool:
    """Return True when the local QClaw gateway responds."""
    import urllib.error
    import urllib.request

    headers = {"Authorization": f"Bearer {token}", "User-Agent": "AlphaMaster"}
    for path in ("/models", "/health"):
        try:
            req = urllib.request.Request(
                f"{base_url.rstrip('/')}{path}",
                headers=headers,
                method="GET",
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                if resp.status == 200:
                    return True
        except urllib.error.HTTPError as exc:
            if exc.code in (401, 403):
                return True
        except Exception as exc:
            logger.debug("QClaw probe %s failed: %s", path, exc)
    return False


def provider_status() -> dict[str, Any]:
    qclaw_cfg = detect_qclaw(require_alive=False)
    qclaw_reachable = detect_qclaw(require_alive=True) if qclaw_cfg else False
    wb_ok = bool(_workbuddy_token())
    wb_env = detect_workbuddy()
    if qclaw_cfg and qclaw_reachable:
        qclaw_hint = "已检测到本地 QClaw Gateway（可连通）"
    elif qclaw_cfg:
        qclaw_hint = (
            "已读取 QClaw 配置，将自动连接本地 Gateway。"
            "若 QClaw 已打开仍显示未连通，可直接尝试分析（分析时会重新检测）。"
        )
    else:
        qclaw_hint = "未检测到 QClaw（需 ~/.qclaw/openclaw.json 且 chatCompletions 已启用）"
    if wb_ok:
        wb_hint = "已自动读取 WorkBuddy 登录 token"
    elif wb_env:
        wb_hint = (
            "已检测到 WorkBuddy，但未找到登录 token。"
            f"请打开 WorkBuddy 并登录（会话文件：{_WORKBUDDY_AUTH_EXPECTED}），"
            "或设置 WORKBUDDY_API_TOKEN / 写入 ~/.workbuddy/.wb_token"
        )
    else:
        wb_hint = "未检测到 WorkBuddy（请先安装并登录 WorkBuddy）"
    return {
        "providers": [
            {
                "id": "deepseek",
                "label": "DeepSeek (deepseek-v4-flash)",
                "available": True,
                "needs_user_key": True,
                "hint": "固定模型 deepseek-v4-flash · https://api.deepseek.com",
            },
            {
                "id": "openclaw",
                "label": "openclaw (QClaw)",
                "available": qclaw_cfg,
                "gateway_reachable": qclaw_reachable,
                "needs_user_key": False,
                "hint": qclaw_hint,
            },
            {
                "id": "openclaw_wb",
                "label": "openclaw_wb (WorkBuddy)",
                "available": wb_ok,
                "needs_user_key": False,
                "hint": wb_hint,
            },
        ]
    }


def _alias_provider_from_key(api_key: str | None) -> str | None:
    """Map Key 输入里的 openclaw / openclaw_wb 别名到通道 id。"""
    key_lower = (api_key or "").strip().lower()
    if not key_lower:
        return None
    # openclaw_wb 必须先于 openclaw，避免被 openclaw 前缀误伤
    if key_lower in ("openclaw_wb",) or key_lower.startswith("openclaw_wb/"):
        return "openclaw_wb"
    if key_lower in ("openclaw",) or key_lower.startswith("openclaw/"):
        return "openclaw"
    return None


def resolve_provider(provider: str, api_key: str | None = None) -> ResolvedProvider:
    pid = (provider or "deepseek").strip().lower()
    key = (api_key or "").strip()

    # API Key 里直接填 openclaw / openclaw_wb 时，自动切换通道并读取本地 token
    aliased = _alias_provider_from_key(key)
    if aliased:
        pid = aliased
        key = ""

    if pid not in PROVIDERS:
        raise ValueError(f"不支持的 AI 通道: {provider}（可选: {', '.join(PROVIDERS)}）")

    if pid == "deepseek":
        if not key:
            raise ValueError("请填写 DeepSeek API Key")
        return ResolvedProvider(
            provider="deepseek",
            model=DEEPSEEK_MODEL,
            base_url=DEEPSEEK_BASE_URL,
            api_key=key,
            label="DeepSeek",
            needs_user_key=True,
        )

    if pid == "openclaw":
        info = _qclaw_gateway_info()
        if info is None or not detect_qclaw(require_alive=False):
            raise ValueError(
                "未检测到本地 QClaw。请确认已安装 QClaw，"
                "且 ~/.qclaw/openclaw.json 中 chatCompletions 已启用、token 已配置。"
            )
        host, port, token = info
        base = f"http://{host}:{port}/v1"
        if not _probe_qclaw_gateway(base, token):
            raise ValueError(
                f"已读取 QClaw 配置，但暂时无法连接 Gateway（{base}）。"
                "请确认 QClaw 已打开；若刚启动请稍等几秒后重试。"
            )
        model = str(_pick_openclaw_model(base, token) or _OPENCLAW_MODEL)
        return ResolvedProvider(
            provider="openclaw",
            model=model,
            base_url=base,
            api_key=token,
            label="openclaw (QClaw)",
            needs_user_key=False,
        )

    # openclaw_wb：自动从 WorkBuddy 会话 / .wb_token / 环境变量 / Electron DPAPI 读取
    token = _workbuddy_token()
    if not token:
        raise ValueError(_workbuddy_token_missing_message())
    endpoint = _workbuddy_endpoint()
    base = f"{endpoint.rstrip('/')}{_WORKBUDDY_API_PATH}"
    return ResolvedProvider(
        provider="openclaw_wb",
        model=_WORKBUDDY_API_MODEL,
        base_url=base,
        api_key=token,
        label="openclaw_wb (WorkBuddy)",
        needs_user_key=False,
    )


def chat_completions(
    resolved: ResolvedProvider,
    messages: list[dict[str, str]],
    *,
    max_tokens: int = 4096,
    timeout: float = 120.0,
) -> str:
    """Call OpenAI-compatible /chat/completions and return assistant text."""
    parts: list[str] = []
    for chunk in stream_chat_completions(
        resolved, messages, max_tokens=max_tokens, timeout=timeout
    ):
        parts.append(chunk)
    content = "".join(parts).strip()
    if not content:
        raise RuntimeError("AI 返回内容为空")
    return content


def stream_chat_completions(
    resolved: ResolvedProvider,
    messages: list[dict[str, str]],
    *,
    max_tokens: int = 4096,
    timeout: float = 180.0,
):
    """Yield text deltas from OpenAI-compatible streaming chat completions."""
    import urllib.error
    import urllib.request

    url = resolved.base_url.rstrip("/") + "/chat/completions"
    payload: dict[str, Any] = {
        "model": resolved.model,
        "messages": messages,
        "max_tokens": max_tokens,
        "stream": True,
    }
    if resolved.provider == "openclaw":
        payload["tool_choice"] = "none"

    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={
            "Authorization": f"Bearer {resolved.api_key}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
            "User-Agent": "AlphaMaster-AI-Analyze",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            for raw in resp:
                line = raw.decode("utf-8", errors="replace").strip()
                if not line:
                    continue
                if line.startswith(":"):
                    continue
                if not line.startswith("data:"):
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
                text = delta.get("content") or ""
                if not text:
                    text = delta.get("reasoning_content") or ""
                if text:
                    yield text
    except urllib.error.HTTPError as exc:
        err_body = exc.read().decode("utf-8", errors="replace")[:800]
        raise RuntimeError(f"AI 请求失败 HTTP {exc.code}: {err_body}") from exc
    except Exception as exc:
        raise RuntimeError(f"AI 请求失败: {exc}") from exc


def _find_qclaw_config() -> Path | None:
    for path in _QCLAW_CONFIG_CANDIDATES:
        if path.exists():
            return path
    return None


def _read_json(path: Path) -> dict | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except (json.JSONDecodeError, OSError):
        return None


def _qclaw_gateway_info() -> tuple[str, int, str] | None:
    config_path = _find_qclaw_config()
    if config_path is None:
        return None
    data = _read_json(config_path)
    if not data:
        return None
    gw = data.get("gateway") or {}
    token = str((gw.get("auth") or {}).get("token") or "").strip()
    if not token:
        return None
    port = int(gw.get("port") or 51187)
    host = "127.0.0.1"
    bind = str(gw.get("bind") or "127.0.0.1")
    if bind and bind not in ("0.0.0.0", "loopback"):
        host = bind
    return host, port, token


def _pick_openclaw_model(
    base_url: str, token: str, *, probe_only: bool = False
) -> str | bool:
    """Probe QClaw /models.

    When *probe_only* is True, return whether the gateway responded.
    Otherwise return a preferred model id (falls back to ``openclaw``).
    """
    if probe_only:
        return _probe_qclaw_gateway(base_url, token)
    try:
        import urllib.request

        req = urllib.request.Request(
            f"{base_url.rstrip('/')}/models",
            headers={"Authorization": f"Bearer {token}", "User-Agent": "AlphaMaster"},
            method="GET",
        )
        with urllib.request.urlopen(req, timeout=5.0) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        ids = [str(m.get("id", "")) for m in data.get("data", []) if m.get("id")]
        if _OPENCLAW_MODEL in ids:
            return _OPENCLAW_MODEL
        for mid in ids:
            if mid.startswith("openclaw"):
                return mid
    except Exception as exc:
        logger.debug("QClaw /models probe failed: %s", exc)
    return _OPENCLAW_MODEL


def _workbuddy_auth_dirs() -> list[Path]:
    """Candidate directories where WorkBuddy stores FileAuthenticationStorage sessions."""
    local_app = os.environ.get("LOCALAPPDATA", "").strip()
    home = Path.home()
    roots: list[Path] = []
    if local_app:
        roots.append(Path(local_app))
    roots.append(home / "AppData" / "Local")
    names = ("CodeBuddyExtension", "WorkBuddyExtension")
    out: list[Path] = []
    for root in roots:
        for name in names:
            auth_dir = root / name / "Data" / "Public" / "auth"
            if auth_dir.is_dir() and auth_dir not in out:
                out.append(auth_dir)
    return out


def _workbuddy_auth_session_candidates() -> list[Path]:
    preferred = (
        os.environ.get("WORKBUDDY_AUTH_FILE", "").strip(),
        "workbuddy-desktop.info",
        "auth.info",
    )
    out: list[Path] = []
    for auth_dir in _workbuddy_auth_dirs():
        for name in preferred:
            if not name:
                continue
            path = auth_dir / name
            if path.exists() and path not in out:
                out.append(path)
        # timestamped backups from logout: workbuddy-desktop.2026-....info
        try:
            for path in sorted(auth_dir.glob("*.info"), key=lambda p: p.stat().st_mtime, reverse=True):
                if path.name.endswith(".logged-out"):
                    continue
                if path not in out:
                    out.append(path)
        except OSError:
            pass
    return out


def _read_workbuddy_auth_token(path: Path) -> str | None:
    data = _read_json(path)
    if not data:
        return None
    auth = data.get("auth")
    if not isinstance(auth, dict):
        return None
    token = str(auth.get("accessToken") or auth.get("access_token") or "").strip()
    if not token:
        return None
    expires_at = auth.get("expiresAt")
    if isinstance(expires_at, (int, float)) and expires_at > 0:
        import time

        if expires_at <= time.time() * 1000:
            logger.debug("WorkBuddy auth session expired: %s", path)
            return None
    return token


def _workbuddy_token_missing_message() -> str:
    expected = _WORKBUDDY_AUTH_EXPECTED
    lines = [
        "未检测到 WorkBuddy token，请确认：",
        "1. 已打开 WorkBuddy 并完成登录",
        f"2. 存在会话文件：{expected}",
        f"3. 或写入 token 文件：{_WORKBUDDY_TOKEN_FILE}",
        "4. 或设置环境变量 WORKBUDDY_API_TOKEN",
    ]
    if detect_qclaw(require_alive=False):
        lines.append("若要用本地 QClaw，请先启动 QClaw。")
    return "\n".join(lines)


def _workbuddy_token() -> str | None:
    """Extract WorkBuddy token (same layered strategy as PA_Agent).

    1. Desktop auth session (CodeBuddyExtension/.../auth/*.info)
    2. ~/.workbuddy/.wb_token
    3. WORKBUDDY_API_TOKEN / CODEBUDDY_AUTH_TOKEN / ACC_AUTH_TOKEN
    4. DPAPI-decrypted Electron session storage (Windows)
    """
    for path in _workbuddy_auth_session_candidates():
        token = _read_workbuddy_auth_token(path)
        if token:
            logger.debug("Using WorkBuddy token from auth session %s", path)
            return token
    if _WORKBUDDY_TOKEN_FILE.exists():
        try:
            token = _WORKBUDDY_TOKEN_FILE.read_text(encoding="utf-8").strip()
            if token:
                logger.debug("Using WorkBuddy token from %s", _WORKBUDDY_TOKEN_FILE)
                return token
        except OSError:
            pass
    for env_name in ("WORKBUDDY_API_TOKEN", "CODEBUDDY_AUTH_TOKEN", "ACC_AUTH_TOKEN"):
        token = os.environ.get(env_name, "").strip()
        if token:
            logger.debug("Using WorkBuddy token from env %s", env_name)
            return token
    token = _decrypt_electron_token()
    if token:
        logger.debug("Using WorkBuddy token from DPAPI Electron storage")
        return token
    return None


def _read_workbuddy_local_state() -> dict | None:
    if not _WORKBUDDY_LOCAL_STATE.exists():
        return None
    try:
        return json.loads(_WORKBUDDY_LOCAL_STATE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.debug("Failed to read WorkBuddy Local State: %s", exc)
        return None


def _dpapi_decrypt(blob: bytes) -> bytes | None:
    import ctypes
    from ctypes import wintypes

    crypt32 = ctypes.windll.crypt32
    kernel32 = ctypes.windll.kernel32

    class DATA_BLOB(ctypes.Structure):
        _fields_ = [
            ("cbData", wintypes.DWORD),
            ("pbData", ctypes.POINTER(ctypes.c_ubyte)),
        ]

    buf_in = (ctypes.c_ubyte * len(blob))(*blob)
    blob_in = DATA_BLOB(len(blob), buf_in)
    blob_out = DATA_BLOB()
    ok = crypt32.CryptUnprotectData(
        ctypes.byref(blob_in),
        None,
        None,
        None,
        None,
        0x1,
        ctypes.byref(blob_out),
    )
    if not ok:
        return None
    try:
        size = blob_out.cbData
        buf = ctypes.cast(blob_out.pbData, ctypes.POINTER(ctypes.c_ubyte * size))
        return bytes(buf.contents)
    finally:
        kernel32.LocalFree(blob_out.pbData)


def _decrypt_electron_token() -> str | None:
    """Try extracting auth token from Electron DPAPI-encrypted storage (Windows)."""
    import sys

    if sys.platform != "win32":
        return None

    local_state = _read_workbuddy_local_state()
    if local_state is None:
        return None
    encrypted_key_b64 = (local_state.get("os_crypt") or {}).get("encrypted_key") or ""
    if not encrypted_key_b64:
        return None
    try:
        import base64

        encrypted_key = base64.b64decode(encrypted_key_b64)
    except Exception:
        return None
    if not encrypted_key.startswith(b"DPAPI"):
        return None
    aes_key = _dpapi_decrypt(encrypted_key[5:])
    if aes_key is None:
        return None
    try:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM

        aesgcm = AESGCM(aes_key)
    except ImportError:
        logger.debug("cryptography not installed; skipping WorkBuddy DPAPI decryption")
        return None

    search_dirs = [
        _WORKBUDDY_SESSION_PATH / "Local Storage" / "leveldb",
        _WORKBUDDY_SESSION_PATH / "Session Storage",
        _WORKBUDDY_SESSION_PATH / "Network",
        _WORKBUDDY_SESSION_PATH / "Partitions",
    ]
    for search_dir in search_dirs:
        if not search_dir.exists():
            continue
        files = [search_dir] if search_dir.is_file() else list(search_dir.rglob("*"))
        for entry in files:
            if not entry.is_file():
                continue
            try:
                if entry.stat().st_size > 20_000_000:
                    continue
                data = entry.read_bytes()
            except OSError:
                continue
            for prefix in (b"v10", b"v11"):
                idx = 0
                while True:
                    idx = data.find(prefix, idx)
                    if idx == -1:
                        break
                    encrypted_val = data[idx + 3 : idx + 3 + 2048]
                    if len(encrypted_val) >= 27:
                        try:
                            plain = aesgcm.decrypt(
                                encrypted_val[:12], encrypted_val[12:], None
                            )
                            plain_str = plain.decode("utf-8", errors="replace")
                            looks_like_token = (
                                plain_str.startswith("eyJ")
                                or (
                                    len(plain_str) >= 40
                                    and plain_str.strip().isascii()
                                    and not plain_str.startswith("{")
                                    and not plain_str.startswith("[")
                                    and "\x00" not in plain_str
                                )
                                or (
                                    "accessToken" in plain_str
                                    or "access_token" in plain_str
                                    or "bearerToken" in plain_str
                                )
                            )
                            if looks_like_token:
                                if "accessToken" in plain_str or "access_token" in plain_str:
                                    try:
                                        obj = json.loads(plain_str)
                                        for k in (
                                            "accessToken",
                                            "access_token",
                                            "token",
                                            "bearerToken",
                                        ):
                                            if k in obj:
                                                return str(obj[k])
                                    except json.JSONDecodeError:
                                        pass
                                return plain_str.strip("\x00").strip()
                        except Exception:
                            pass
                    idx += 1
    return None


def _workbuddy_endpoint() -> str:
    for env_name in ("WORKBUDDY_API_ENDPOINT", "WORKBUDDY_API_URL"):
        endpoint = os.environ.get(env_name, "").strip()
        if endpoint:
            return endpoint
    acc_config = os.environ.get("ACC_PRODUCT_CONFIG_V3", "")
    if acc_config:
        try:
            config = json.loads(acc_config)
            endpoint = str(config.get("endpoint") or "").strip()
            if endpoint:
                return endpoint
        except json.JSONDecodeError:
            pass
    return _WORKBUDDY_DEFAULT_ENDPOINT
