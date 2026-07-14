"""Tests for per-symbol training time accounting."""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from web import training_time as tt


@pytest.fixture
def isolated_training_time(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(tt, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(tt, "LOG_DIR", tmp_path / "logs")
    (tmp_path / "logs").mkdir(parents=True, exist_ok=True)
    tt._backfilled.clear()
    yield tmp_path


def test_record_and_summarize_session(isolated_training_time: Path) -> None:
    start = datetime(2026, 7, 14, 10, 0, 0, tzinfo=timezone.utc)
    end = start + timedelta(hours=2, minutes=30)
    tt.record_training_session(
        symbol="XAUUSD",
        started_at=start.isoformat(),
        finished_at=end.isoformat(),
        log_path="logs/train_XAUUSD_20260714_100000.log",
    )

    summary = tt.get_training_time_summary("XAUUSD")
    assert summary.history_total_seconds == 9000
    assert summary.session_seconds is None

    job = {
        "symbol": "XAUUSD",
        "started_at": (end + timedelta(minutes=5)).isoformat(),
        "finished_at": None,
    }
    live = tt.get_training_time_summary("XAUUSD", job=job, active=True)
    assert live.session_seconds == pytest.approx(0, abs=5)
    assert live.history_total_seconds >= 9000


def test_backfill_from_logs(isolated_training_time: Path) -> None:
    log = isolated_training_time / "logs" / "train_XAUUSD_20260714_120000.log"
    log.write_text("train\n", encoding="utf-8")
    start_ts = datetime(2026, 7, 14, 12, 0, 0, tzinfo=timezone.utc).timestamp()
    end_ts = start_ts + 3600
    import os

    os.utime(log, (start_ts, end_ts))

    summary = tt.get_training_time_summary("XAUUSD")
    assert summary.history_total_seconds >= 3500
    data = json.loads((isolated_training_time / "training_time_XAUUSD.json").read_text(encoding="utf-8"))
    assert len(data["sessions"]) >= 1
