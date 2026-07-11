"""实时信号引擎：多品种 / 多周期并发调度。

- 每个「监控项」= (数据源, 品种, 周期, 策略因子)。
- 后台线程按周期自适应节奏轮询，出现新 bar 才重算，信号取最后已收盘 bar。
- 共享 (源,品种,周期) 的 K 线抓取结果做短 TTL 缓存，避免重复请求。
- 监控清单持久化到 web_settings，重启恢复。
"""
from __future__ import annotations

import json
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from model_core.vocab import VOCAB_VERSION
from strategy_manager.live_signal import evaluate_signal, min_exposure
from web.data_sources.base import bars_to_raw_dict
from web.data_sources.factory import SOURCE_KINDS, get_source
from web.settings import load_settings, save_settings

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 每个周期的轮询节奏（秒）
_CADENCE = {
    "1m": 15, "5m": 30, "15m": 45, "30m": 60,
    "1h": 60, "4h": 120, "1d": 300, "1w": 600, "1M": 600,
}
# K 线周期长度（秒）；用于推算「下一根已收盘 bar」时间
_TF_SECONDS = {
    "1m": 60,
    "5m": 300,
    "15m": 900,
    "30m": 1800,
    "1h": 3600,
    "4h": 14400,
    "1d": 86400,
    "1w": 604800,
    "1M": 2592000,  # 近似 30 天
}
_DEFAULT_CADENCE = 60
_N_BARS = 500                 # 每次拉取的历史 bar 数（喂给特征引擎）
_HISTORY_LEN = 60             # 保留的信号强度历史点数（供 sparkline）
_VALID_KINDS = {k for k, _ in SOURCE_KINDS}


def _cadence_for(tf: str) -> int:
    return _CADENCE.get(tf, _DEFAULT_CADENCE)


def _next_bar_close_at(last_bar_open: int | None, timeframe: str, now: float | None = None) -> int | None:
    """根据最后已收盘 bar 的开盘时间，推算下次收盘（即下次信号更新）的 Unix 秒。

    若最后一根已收盘 bar 已过时太久（超过约 2 个周期仍无新 bar），视为休市/断档，
    返回 None，避免在周末等时段虚构「几分钟后更新」的倒计时。
    """
    if last_bar_open is None:
        return None
    period = _TF_SECONDS.get(timeframe)
    if not period:
        return None
    now_i = int(now if now is not None else time.time())
    last_open = int(last_bar_open)
    last_close = last_open + period
    # 仍未到收盘（常见于 MT5 终端时钟快于本机、或未剔除形成中 bar）
    if last_close > now_i:
        return last_close
    # 正常交易中：上一根收盘距今至多约 1 个周期；再放宽到 2 个周期容错拉取延迟
    if now_i - last_close > period * 2:
        return None
    # last_open 开盘 → last_close 收盘；当前形成中的 bar 在 +2*period 收盘
    nxt = last_open + 2 * period
    while nxt <= now_i:
        nxt += period
        if nxt - last_close > period * 2:
            return None
    return nxt


def _ensure_closed_bars(bars: list, timeframe: str, now: float | None = None) -> list:
    """按本机时钟再剔掉尚未收盘的 K 线（防止 MT5 时钟偏快时把形成中 bar 当已收盘）。"""
    period = _TF_SECONDS.get(timeframe)
    if not period or not bars:
        return bars
    now_i = int(now if now is not None else time.time())
    out = list(bars)
    while out and int(out[-1].ts) + period > now_i:
        out.pop()
    return out


def _load_strategy_meta(path: str) -> dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, list):
        return {"formula": data, "vocab_version": "legacy", "symbol": None, "timeframe": None, "best_score": None}
    return {
        "formula": data.get("formula"),
        "vocab_version": data.get("vocab_version"),
        "symbol": data.get("symbol"),
        "timeframe": data.get("timeframe"),
        "best_score": data.get("best_score"),
    }


@dataclass
class WatchTask:
    id: str
    source: str
    symbol: str
    timeframe: str
    strategy_file: str
    strategy_name: str
    formula: list[int]
    vocab_version: str | None
    strategy_symbol: str | None
    strategy_timeframe: str | None
    best_score: float | None
    cadence_s: int
    # 运行时状态
    state: str = "pending"          # pending|ok|insufficient|error
    direction: str | None = None
    strength: float | None = None
    position: float | None = None
    factor_value: float | None = None
    bars_used: int | None = None
    last_bar_ts: int | None = None
    updated_at: float | None = None
    message: str = ""
    warn: str = ""
    next_due: float = 0.0
    history: deque = field(default_factory=lambda: deque(maxlen=_HISTORY_LEN))

    def to_public(self) -> dict[str, Any]:
        now = time.time()
        next_close = _next_bar_close_at(self.last_bar_ts, self.timeframe, now)
        live = next_close is not None
        return {
            "id": self.id,
            "source": self.source,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "strategy_name": self.strategy_name,
            "strategy_symbol": self.strategy_symbol,
            "strategy_timeframe": self.strategy_timeframe,
            "best_score": self.best_score,
            "state": self.state,
            "direction": self.direction,
            "strength": self.strength,
            "position": self.position,
            "factor_value": self.factor_value,
            "bars_used": self.bars_used,
            "last_bar_ts": self.last_bar_ts,
            "session_live": live,
            "next_bar_close_at": next_close,
            "seconds_to_next": (
                max(0, int(next_close - now)) if next_close is not None else None
            ),
            "updated_at": self.updated_at,
            "message": self.message,
            "warn": self.warn,
            "threshold": min_exposure(),
            "history": list(self.history),
        }

    def persist_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "source": self.source,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "strategy_file": self.strategy_file,
        }


class RealtimeManager:
    def __init__(self) -> None:
        self._tasks: dict[str, WatchTask] = {}
        self._lock = threading.RLock()
        self._running = False
        self._thread: threading.Thread | None = None
        self._executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="rt")
        self._inflight: set[str] = set()
        self._inflight_lock = threading.Lock()
        # K线缓存：(kind,symbol,tf) -> (monotonic_ts, bars)
        self._bar_cache: dict[tuple[str, str, str], tuple[float, list]] = {}
        self._loaded = False

    # ── 持久化 ──────────────────────────────────────────────────────────
    def load_persisted(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        watches = load_settings().get("realtime_watches") or []
        for w in watches:
            try:
                self._add_task_internal(
                    w["source"], w["symbol"], w["timeframe"], w["strategy_file"],
                    persist=False,
                )
            except Exception:
                continue
        if self._tasks:
            self._ensure_thread()

    def _persist(self) -> None:
        save_settings({"realtime_watches": [t.persist_dict() for t in self._tasks.values()]})

    # ── 增删 ────────────────────────────────────────────────────────────
    def add_watch(self, source: str, symbol: str, timeframe: str, strategy_file: str) -> dict[str, Any]:
        task = self._add_task_internal(source, symbol, timeframe, strategy_file, persist=True)
        self._ensure_thread()
        return task.to_public()

    def _add_task_internal(
        self, source: str, symbol: str, timeframe: str, strategy_file: str, persist: bool
    ) -> WatchTask:
        source = (source or "").strip()
        symbol = (symbol or "").strip()
        timeframe = (timeframe or "").strip()
        if source not in _VALID_KINDS:
            raise ValueError(f"未知数据源: {source}")
        if not symbol:
            raise ValueError("请填写品种")
        src = get_source(source)
        if timeframe not in src.supported_timeframes():
            raise ValueError(f"{src.label} 不支持周期 {timeframe}")

        path = strategy_file
        if not Path(path).is_absolute():
            path = str((PROJECT_ROOT / path).resolve())
        if not Path(path).exists():
            raise ValueError(f"策略文件不存在: {strategy_file}")
        meta = _load_strategy_meta(path)
        if not meta.get("formula"):
            raise ValueError("策略文件缺少 formula")

        name = Path(path).stem
        task_id = f"{source}:{symbol}:{timeframe}:{name}"

        warn = ""
        if meta.get("vocab_version") and meta["vocab_version"] not in (VOCAB_VERSION, "legacy"):
            warn = f"词表版本不符（{meta['vocab_version']} vs {VOCAB_VERSION}），信号可能失真"
        elif meta.get("symbol") and meta["symbol"] != symbol:
            warn = f"该因子为 {meta['symbol']} 训练，跨品种运行仅供参考"

        task = WatchTask(
            id=task_id,
            source=source,
            symbol=symbol,
            timeframe=timeframe,
            strategy_file=path,
            strategy_name=name,
            formula=[int(t) for t in meta["formula"]],
            vocab_version=meta.get("vocab_version"),
            strategy_symbol=meta.get("symbol"),
            strategy_timeframe=meta.get("timeframe"),
            best_score=meta.get("best_score"),
            cadence_s=_cadence_for(timeframe),
            warn=warn,
            next_due=0.0,
        )
        with self._lock:
            self._tasks[task_id] = task
            if persist:
                self._persist()
        return task

    def remove_watch(self, task_id: str) -> bool:
        with self._lock:
            existed = self._tasks.pop(task_id, None) is not None
            if existed:
                self._persist()
        return existed

    def clear(self) -> None:
        with self._lock:
            self._tasks.clear()
            self._persist()

    # ── 状态 ────────────────────────────────────────────────────────────
    def status(self) -> dict[str, Any]:
        with self._lock:
            watches = [t.to_public() for t in self._tasks.values()]
        nearest = None
        for w in watches:
            s = w.get("seconds_to_next")
            if s is None:
                continue
            if nearest is None or s < nearest:
                nearest = s
        return {
            "running": self._running,
            "count": len(watches),
            "watches": watches,
            "server_time": time.time(),
            "nearest_seconds_to_next": nearest,
        }

    # ── 调度线程 ────────────────────────────────────────────────────────
    def start(self) -> None:
        self._ensure_thread()

    def _ensure_thread(self) -> None:
        with self._lock:
            if self._running and self._thread and self._thread.is_alive():
                return
            self._running = True
            self._thread = threading.Thread(target=self._loop, name="realtime", daemon=True)
            self._thread.start()

    def stop(self) -> None:
        self._running = False

    def _loop(self) -> None:
        while self._running:
            try:
                self._tick()
            except Exception:
                pass
            time.sleep(1.5)

    def _tick(self) -> None:
        now = time.monotonic()
        with self._lock:
            due = [t for t in self._tasks.values() if now >= t.next_due]
        for task in due:
            with self._inflight_lock:
                if task.id in self._inflight:
                    continue
                self._inflight.add(task.id)
            # 预置下次到期，避免重复提交
            task.next_due = now + task.cadence_s
            self._executor.submit(self._evaluate_task, task)

    def _get_bars(self, source: str, symbol: str, timeframe: str):
        """带短 TTL 缓存的 K 线抓取（同一 源/品种/周期 的多因子复用）。"""
        key = (source, symbol, timeframe)
        ttl = max(10.0, _cadence_for(timeframe) * 0.8)
        now = time.monotonic()
        cached = self._bar_cache.get(key)
        if cached and (now - cached[0]) < ttl:
            return cached[1]
        src = get_source(source)
        bars = src.fetch_bars(symbol, timeframe, _N_BARS, drop_forming=True)
        bars = _ensure_closed_bars(bars, timeframe)
        self._bar_cache[key] = (now, bars)
        return bars

    def _evaluate_task(self, task: WatchTask) -> None:
        try:
            bars = self._get_bars(task.source, task.symbol, task.timeframe)
            if not bars:
                self._set_error(task, "未获取到 K 线")
                return
            last_ts = bars[-1].ts
            raw = bars_to_raw_dict(bars)
            result = evaluate_signal(task.formula, raw)

            task.state = result.get("state", "error")
            task.message = result.get("message", "")
            task.bars_used = result.get("bars_used", len(bars))
            task.last_bar_ts = last_ts
            task.updated_at = time.time()
            if task.state == "ok":
                new_dir = result["direction"]
                prev_dir = task.direction
                task.direction = new_dir
                task.strength = result["strength"]
                task.position = result["position"]
                task.factor_value = result["factor_value"]
                task.history.append(round(result["strength"], 4))
                # 已有上次方向且发生转折时推飞书（首次算出方向不打扰）
                if prev_dir and new_dir and prev_dir != new_dir:
                    try:
                        from web.feishu_notify import notify_direction_flip

                        notify_direction_flip(
                            symbol=task.symbol,
                            timeframe=task.timeframe,
                            strategy_name=task.strategy_name,
                            prev_direction=prev_dir,
                            new_direction=new_dir,
                            strength=task.strength,
                            factor_value=task.factor_value,
                        )
                    except Exception:
                        pass
        except Exception as exc:  # noqa: BLE001
            self._set_error(task, str(exc))
        finally:
            with self._inflight_lock:
                self._inflight.discard(task.id)

    def _set_error(self, task: WatchTask, message: str) -> None:
        task.state = "error"
        task.message = message
        task.updated_at = time.time()


realtime_manager = RealtimeManager()
