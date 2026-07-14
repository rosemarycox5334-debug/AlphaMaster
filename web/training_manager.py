"""Subprocess manager for train_file.py jobs."""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.train_logging import strip_ansi

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)


class JobState(str, Enum):
    IDLE = "idle"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    STOPPED = "stopped"


@dataclass
class TrainingJob:
    data_file: str
    symbol: str
    timeframe: str
    mode: str
    state: JobState = JobState.RUNNING
    pid: int | None = None
    log_path: str = ""
    started_at: str = ""
    finished_at: str | None = None
    exit_code: int | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "data_file": self.data_file,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "mode": self.mode,
            "state": self.state.value,
            "pid": self.pid,
            "log_path": self.log_path,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "exit_code": self.exit_code,
            "error": self.error,
        }


class TrainingManager:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._proc: subprocess.Popen | None = None
        self._job: TrainingJob | None = None
        self._log_fp = None
        self._stopped_by_user = False
        self._recorded_log_paths: set[str] = set()

    def status(self) -> dict[str, Any]:
        with self._lock:
            self._refresh_state()
            return {
                "active": self._job is not None and self._job.state == JobState.RUNNING,
                "job": self._job.to_dict() if self._job else None,
            }

    def start(
        self,
        data_file: str,
        symbol: str,
        timeframe: str,
        mode: str = "ftmo",
        *,
        from_scratch: bool = False,
    ) -> TrainingJob:
        with self._lock:
            self._refresh_state()
            if self._proc is not None and self._proc.poll() is None:
                sym = self._job.symbol if self._job else "unknown"
                raise RuntimeError(f"已有训练任务在运行: {sym}")

            ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            safe_sym = symbol.replace(".", "_")
            log_path = LOG_DIR / f"train_{safe_sym}_{ts}.log"

            hist_path = PROJECT_ROOT / f"training_history_{symbol}.json"
            try:
                hist_path.unlink(missing_ok=True)
            except OSError:
                pass

            cmd = [
                sys.executable,
                "-u",
                "train_file.py",
                "--data-file",
                data_file,
            ]
            if from_scratch:
                cmd.append("--from-scratch")

            self._log_fp = open(log_path, "w", encoding="utf-8", buffering=1)
            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"
            env["PYTHONIOENCODING"] = "utf-8"
            env["PYTHONUTF8"] = "1"
            env["LOGURU_COLORIZE"] = "0"

            creationflags = 0
            if sys.platform == "win32":
                creationflags = subprocess.CREATE_NEW_PROCESS_GROUP

            self._stopped_by_user = False
            self._proc = subprocess.Popen(
                cmd,
                cwd=PROJECT_ROOT,
                stdout=self._log_fp,
                stderr=subprocess.STDOUT,
                env=env,
                creationflags=creationflags,
            )
            self._job = TrainingJob(
                data_file=data_file,
                symbol=symbol,
                timeframe=timeframe,
                mode=mode,
                pid=self._proc.pid,
                log_path=str(log_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                started_at=datetime.now(timezone.utc).isoformat(),
            )
            return self._job

    def stop(self) -> bool:
        with self._lock:
            if self._proc is None or self._proc.poll() is not None:
                return False
            self._stopped_by_user = True
            try:
                self._proc.terminate()
            except Exception:
                self._proc.kill()
            return True

    def parse_step_from_log(self) -> int | None:
        """从日志尾部解析当前步数，用于 checkpoint 写入前的进度展示。"""
        import re

        for line in reversed(self.tail_log(80)):
            m = re.search(r"\[(\d+)/\d+\]", line)
            if m:
                return int(m.group(1))
        return None

    def tail_log(self, lines: int = 200) -> list[str]:
        with self._lock:
            if not self._job or not self._job.log_path:
                return []
            path = PROJECT_ROOT / self._job.log_path
            if not path.exists():
                return []
            try:
                content = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                return []
            return [strip_ansi(line) for line in content.splitlines()[-lines:]]

    def _refresh_state(self) -> None:
        if self._proc is None or self._job is None:
            return
        code = self._proc.poll()
        if code is None:
            return
        self._job.exit_code = code
        self._job.finished_at = datetime.now(timezone.utc).isoformat()
        if self._job.state == JobState.RUNNING:
            if self._stopped_by_user:
                self._job.state = JobState.STOPPED
            elif code == 0:
                self._job.state = JobState.COMPLETED
            elif code in (-signal.SIGTERM, 1) and sys.platform != "win32":
                self._job.state = JobState.STOPPED
            elif code < 0:
                self._job.state = JobState.STOPPED
            else:
                self._job.state = JobState.FAILED
        if self._job.state == JobState.FAILED and self._job.error is None:
            self._job.error = f"训练进程异常退出 (exit_code={code})"
            try:
                if self._job.log_path:
                    path = PROJECT_ROOT / self._job.log_path
                    with path.open("a", encoding="utf-8") as fp:
                        fp.write(f"\n[Web] 训练进程已结束，退出码: {code}\n")
            except OSError:
                pass
        if self._log_fp:
            try:
                self._log_fp.flush()
                self._log_fp.close()
            except Exception:
                pass
            self._log_fp = None
        self._record_session_time()
        self._proc = None

    def _record_session_time(self) -> None:
        job = self._job
        if job is None or not job.log_path or not job.started_at:
            return
        rel = job.log_path.replace("\\", "/")
        if rel in self._recorded_log_paths:
            return
        if job.state == JobState.RUNNING:
            return
        from web.training_time import record_training_session

        record_training_session(
            symbol=job.symbol,
            started_at=job.started_at,
            finished_at=job.finished_at,
            log_path=rel,
        )
        self._recorded_log_paths.add(rel)


training_manager = TrainingManager()
