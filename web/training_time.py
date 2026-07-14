"""Per-symbol cumulative training session durations for the Web UI."""
from __future__ import annotations

import json
import re
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = PROJECT_ROOT / "logs"

_lock = threading.Lock()
_backfilled: set[str] = set()


def _stats_path(symbol: str) -> Path:
    safe = symbol.replace(".", "_")
    return PROJECT_ROOT / f"training_time_{safe}.json"


def _parse_iso(ts: str | None) -> datetime | None:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except ValueError:
        return None


def _duration_seconds(started_at: str, finished_at: str | None) -> int:
    start = _parse_iso(started_at)
    if start is None:
        return 0
    end = _parse_iso(finished_at) if finished_at else datetime.now(timezone.utc)
    if end.tzinfo is None:
        end = end.replace(tzinfo=timezone.utc)
    if start.tzinfo is None:
        start = start.replace(tzinfo=timezone.utc)
    return max(0, int((end - start).total_seconds()))


def _load(symbol: str) -> dict[str, Any]:
    path = _stats_path(symbol)
    if not path.exists():
        return {"symbol": symbol, "sessions": []}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {"symbol": symbol, "sessions": []}
    if not isinstance(data.get("sessions"), list):
        data["sessions"] = []
    data["symbol"] = symbol
    return data


def _save(symbol: str, data: dict[str, Any]) -> None:
    path = _stats_path(symbol)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _known_log_paths(data: dict[str, Any]) -> set[str]:
    out: set[str] = set()
    for row in data.get("sessions") or []:
        if isinstance(row, dict):
            lp = str(row.get("log_path") or "").strip()
            if lp:
                out.add(lp.replace("\\", "/"))
    return out


def _backfill_from_logs(symbol: str, data: dict[str, Any]) -> dict[str, Any]:
    """One-time import of past train logs (mtime - ctime) for this symbol."""
    if symbol in _backfilled:
        return data
    _backfilled.add(symbol)

    safe = symbol.replace(".", "_")
    pattern = re.compile(rf"^train_{re.escape(safe)}_(\d{{8}})_(\d{{6}})\.log$")
    known = _known_log_paths(data)
    sessions: list[dict[str, Any]] = list(data.get("sessions") or [])

    if not LOG_DIR.exists():
        return data

    for path in sorted(LOG_DIR.glob(f"train_{safe}_*.log")):
        rel = str(path.relative_to(PROJECT_ROOT)).replace("\\", "/")
        if rel in known:
            continue
        m = pattern.match(path.name)
        if not m:
            continue
        try:
            st = path.stat()
        except OSError:
            continue
        date_part, time_part = m.group(1), m.group(2)
        try:
            started = datetime.strptime(
                f"{date_part}{time_part}", "%Y%m%d%H%M%S"
            ).replace(tzinfo=timezone.utc)
        except ValueError:
            started = datetime.fromtimestamp(st.st_ctime, tz=timezone.utc)
        finished = datetime.fromtimestamp(st.st_mtime, tz=timezone.utc)
        seconds = max(0, int((finished - started).total_seconds()))
        if seconds <= 0:
            seconds = max(1, int(finished.timestamp() - st.st_ctime))
        sessions.append(
            {
                "log_path": rel,
                "started_at": started.isoformat(),
                "finished_at": finished.isoformat(),
                "seconds": seconds,
                "source": "log_backfill",
            }
        )
        known.add(rel)

    data["sessions"] = sessions
    if sessions:
        _save(symbol, data)
    return data


def record_training_session(
    *,
    symbol: str,
    started_at: str,
    finished_at: str | None,
    log_path: str,
) -> None:
    """Persist one completed training subprocess duration."""
    if not symbol or not started_at or not log_path:
        return
    rel_log = log_path.replace("\\", "/")
    seconds = _duration_seconds(started_at, finished_at)
    if seconds <= 0:
        return

    with _lock:
        data = _backfill_from_logs(symbol, _load(symbol))
        if rel_log in _known_log_paths(data):
            return
        sessions = list(data.get("sessions") or [])
        sessions.append(
            {
                "log_path": rel_log,
                "started_at": started_at,
                "finished_at": finished_at,
                "seconds": seconds,
                "source": "web_job",
            }
        )
        data["sessions"] = sessions
        _save(symbol, data)


@dataclass
class TrainingTimeSummary:
    session_seconds: int | None
    history_total_seconds: int


def get_training_time_summary(
    symbol: str,
    *,
    job: dict[str, Any] | None = None,
    active: bool = False,
) -> TrainingTimeSummary:
    """Return current-session and cumulative seconds for *symbol*."""
    if not symbol:
        return TrainingTimeSummary(session_seconds=None, history_total_seconds=0)

    with _lock:
        data = _backfill_from_logs(symbol, _load(symbol))
        past_seconds = sum(
            int(row.get("seconds") or 0)
            for row in (data.get("sessions") or [])
            if isinstance(row, dict)
        )

    session_seconds: int | None = None
    if job and str(job.get("symbol") or "") == symbol:
        session_seconds = _duration_seconds(
            str(job.get("started_at") or ""),
            None if active else job.get("finished_at"),
        )

    if session_seconds and active:
        history_total = past_seconds + session_seconds
    else:
        history_total = past_seconds

    return TrainingTimeSummary(
        session_seconds=session_seconds,
        history_total_seconds=history_total,
    )
