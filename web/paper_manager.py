"""
web/paper_manager.py — A股回放任务管理器

仿 backtest_manager：subprocess 跑 run_paper_replay.py，日志写 logs/paper_*.log，
读 paper_trading/output/ 下的 equity/trades/metrics JSON 供前端展示。
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)
OUT_DIR = PROJECT_ROOT / "paper_trading" / "output"


class JobState(str, Enum):
    IDLE = "idle"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    STOPPED = "stopped"


@dataclass
class PaperJob:
    strategy_file: str
    start: str
    end: str
    sim_start: str
    state: JobState = JobState.RUNNING
    pid: int | None = None
    log_path: str = ""
    started_at: str = ""
    finished_at: str | None = None
    exit_code: int | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["state"] = self.state.value
        return d


class PaperManager:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._proc: subprocess.Popen | None = None
        self._job: PaperJob | None = None
        self._log_fp = None

    def status(self) -> dict[str, Any]:
        with self._lock:
            self._refresh()
            return {
                "active": self._proc is not None and self._proc.poll() is None,
                "job": self._job.to_dict() if self._job else None,
            }

    def start(self, strategy_file: str, start: str, end: str,
              sim_start: str) -> PaperJob:
        with self._lock:
            self._refresh()
            if self._proc is not None and self._proc.poll() is None:
                raise RuntimeError("已有回放任务在运行")
            ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            log_path = LOG_DIR / f"paper_{ts}.log"
            cmd = [sys.executable, "-u", "run_paper_replay.py",
                   "--strategy", strategy_file, "--start", start,
                   "--end", end, "--sim-start", sim_start]
            self._log_fp = open(log_path, "w", encoding="utf-8", buffering=1)
            env = os.environ.copy()
            env.update(PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
            flags = subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0
            self._proc = subprocess.Popen(
                cmd, cwd=PROJECT_ROOT, stdout=self._log_fp,
                stderr=subprocess.STDOUT, env=env, creationflags=flags)
            self._job = PaperJob(
                strategy_file=strategy_file, start=start, end=end,
                sim_start=sim_start, pid=self._proc.pid,
                log_path=str(log_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                started_at=datetime.now(timezone.utc).isoformat())
            return self._job

    def stop(self) -> bool:
        with self._lock:
            if self._proc is None or self._proc.poll() is not None:
                return False
            try:
                self._proc.terminate()
            except Exception:
                pass
            return True

    def _read_json(self, name: str, default):
        path = OUT_DIR / name
        if not path.exists():
            return default
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return default

    def equity(self) -> list:
        return self._read_json("equity.json", [])

    def trades(self) -> list:
        return self._read_json("trades.json", [])

    def metrics(self) -> dict:
        return self._read_json("metrics.json", {})

    def _refresh(self) -> None:
        if self._proc is None or self._job is None:
            return
        code = self._proc.poll()
        if code is None:
            return
        self._job.exit_code = code
        self._job.finished_at = datetime.now(timezone.utc).isoformat()
        if self._job.state == JobState.RUNNING:
            self._job.state = JobState.COMPLETED if code == 0 else JobState.FAILED
        if self._log_fp:
            try:
                self._log_fp.close()
            except Exception:
                pass
            self._log_fp = None
        self._proc = None


paper_manager = PaperManager()
