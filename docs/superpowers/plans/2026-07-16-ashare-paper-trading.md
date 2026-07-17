# AlphaMaster A股虚拟炒股平台 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 AlphaMaster 之上搭建面向 A 股的虚拟炒股平台：100 万现金账户，自主截面选股/训练/买卖，历史回放与实盘两种模式共用一套决策引擎。

**Architecture:** 唤醒被单品种模式雪藏的多品种截面因子挖掘（`AlphaEngine(target_symbol=None)`）。新建 A 股数据管理器产出与 `MT5DataManager` 相同的 `[N,T]` 张量接口，下游 model_core 零改动。因子训练加 `REWARD_MODE="ashare"`（截面 Rank-IC 主权重）。全新 `paper_trading/` 模块实现 T+1 组合账户引擎与 Top10 等权选股。Web 层复用 subprocess + 日志轮询的 manager 模式。

**Tech Stack:** Python 3.12（uv 管理）、akshare（A股数据）、PyTorch（复用因子内核）、pandas/pyarrow（parquet 缓存）、FastAPI（Web）、pytest + hypothesis（测试）。

---

## 文件结构

**新建：**
- `data_pipeline/ashare_fetcher.py` — akshare 拉取 + parquet 缓存（成份股名单、交易日历、日线）
- `data_pipeline/ashare_manager.py` — A股数据管理器（`[N,T]` 张量 + valid_mask + trade_dates）
- `paper_trading/__init__.py`
- `paper_trading/config.py` — A股交易规则常量（成本、涨跌停、T+1、Top-K）
- `paper_trading/factor_ranker.py` — 因子公式 → 当日 Top10 选股
- `paper_trading/account.py` — 账户数据结构（Holding / Account）与成交/成本原语
- `paper_trading/portfolio_engine.py` — 逐日组合引擎 `step(date, next_date, target_codes, bar)`
- `paper_trading/data_feed.py` — ReplayFeed / LiveFeed 数据喂法
- `paper_trading/metrics.py` — 绩效指标计算
- `train_ashare.py` — A股截面因子训练入口脚本
- `run_paper_replay.py` — 历史回放 CLI 入口（Web subprocess 调用）
- `web/paper_manager.py` — Web 任务管理器
- `web/static/paper.html` — 前端页面
- 测试：`tests/unit/test_ashare_manager.py`、`tests/unit/test_account.py`、`tests/property/test_portfolio.py`、`tests/smoke/test_paper_smoke.py`

**小改：**
- `model_core/backtest.py` — 加 `REWARD_MODE="ashare"` 分支 + 截面 Rank-IC 方法
- `config.py` — 加 A股配置段
- `web/app.py` — import paper_manager + 注册 `/api/paper/*` 路由
- `requirements.txt` — 加 akshare

---

## 任务概览

- Task 1: 环境与依赖（akshare 装机、A股配置段）
- Task 2: AShareFetcher — akshare 拉取 + parquet 缓存
- Task 3: AShareDataManager — [N,T] 张量 + valid_mask + 交易日历对齐
- Task 4: backtest.py 截面 Rank-IC reward 分支
- Task 5: train_ashare.py — 截面因子训练入口
- Task 6: paper_trading/config.py + account.py — 账户结构与成交成本原语
- Task 7: PortfolioEngine — T+1 逐日组合引擎
- Task 8: FactorRanker — 因子 → Top10 选股
- Task 9: metrics.py — 绩效指标
- Task 10: ReplayFeed + run_paper_replay.py — 历史回放端到端
- Task 11: Web 层 — paper_manager + 路由 + 页面
- Task 12: LiveFeed — 实盘模式（状态持久化续跑）

---

## Task 1: 环境与依赖

**Files:**
- Modify: `requirements.txt`
- Modify: `config.py`（在 `Config` 类内 `COST_RATE` 段之后加 A股配置段）
- Test: `tests/unit/test_ashare_config.py`

- [ ] **Step 1: 用 uv 安装 akshare**

Run: `uv pip install akshare`
Expected: 安装成功。若无 uv 环境先 `uv venv`。验证：`uv run python -c "import akshare; print(akshare.__version__)"` 打印版本号。

- [ ] **Step 2: requirements.txt 追加 akshare**

在 `# ===== Trading Platforms / Market Data =====` 段下、`tushare` 行附近追加：

```
akshare>=1.12.0
```

- [ ] **Step 3: 写 A股配置测试**

创建 `tests/unit/test_ashare_config.py`：

```python
from config import Config


def test_ashare_config_present():
    assert Config.ASHARE_INITIAL_CAPITAL == 1_000_000.0
    assert Config.ASHARE_TOP_K == 10
    assert Config.ASHARE_COMMISSION_RATE == 0.00025
    assert Config.ASHARE_MIN_COMMISSION == 5.0
    assert Config.ASHARE_STAMP_TAX == 0.001
    assert Config.ASHARE_LIMIT_PCT == 0.10
    assert Config.ASHARE_LOT_SIZE == 100
    assert isinstance(Config.ASHARE_CACHE_DIR, str)
```

- [ ] **Step 4: 运行测试确认失败**

Run: `uv run pytest tests/unit/test_ashare_config.py -v`
Expected: FAIL — `AttributeError: type object 'Config' has no attribute 'ASHARE_INITIAL_CAPITAL'`

- [ ] **Step 5: config.py 加 A股配置段**

在 `config.py` 的 `Config` 类内，`COST_RATE = 0.0001` 行之后插入：

```python
    # ── A股虚拟炒股平台配置（2026-07-16）───────────────────────
    ASHARE_INITIAL_CAPITAL = 1_000_000.0   # 起始资金 100 万
    ASHARE_TOP_K           = 10            # 每日持仓只数（等权）
    ASHARE_COMMISSION_RATE = 0.00025       # 双边佣金费率 万2.5
    ASHARE_MIN_COMMISSION  = 5.0           # 单笔最低佣金 5 元
    ASHARE_STAMP_TAX       = 0.001         # 卖出印花税 千1
    ASHARE_LIMIT_PCT       = 0.10          # 涨跌停幅度 ±10%（封板判定容差）
    ASHARE_LOT_SIZE        = 100           # 最小交易单位 100 股
    ASHARE_CACHE_DIR       = os.getenv("ASHARE_CACHE_DIR", "ashare_cache")
    ASHARE_MIN_BARS        = 250           # 个股最少交易日（约1年），不足剔除
```

- [ ] **Step 6: 运行测试确认通过**

Run: `uv run pytest tests/unit/test_ashare_config.py -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add requirements.txt config.py tests/unit/test_ashare_config.py
git commit -m "feat(ashare): 环境依赖与A股平台配置段"
```

---

## Task 2: AShareFetcher — akshare 拉取 + parquet 缓存

**Files:**
- Create: `data_pipeline/ashare_fetcher.py`
- Test: `tests/unit/test_ashare_fetcher.py`

**接口契约：**
```python
class AShareFetcher:
    def __init__(self, cache_dir: str = None): ...
    def universe_codes(self) -> list[str]:
        """沪深300+中证500 当前成份股代码（去重），如 ['600000','000001',...]。"""
    def trade_calendar(self, start: str, end: str) -> list[str]:
        """交易日列表 ['2023-01-03', ...]（YYYY-MM-DD）。"""
    def daily(self, code: str, start: str, end: str) -> pd.DataFrame:
        """单只日线，列 [date, open, high, low, close, volume]，parquet 缓存增量更新。"""
```

- [ ] **Step 1: 写 fetcher 测试（用 mock 隔离网络）**

创建 `tests/unit/test_ashare_fetcher.py`：

```python
import pandas as pd
import pytest
from data_pipeline.ashare_fetcher import AShareFetcher, _normalize_daily_df


def test_normalize_daily_df_columns():
    # akshare 返回中文列名，需归一化为标准英文列
    raw = pd.DataFrame({
        "日期": ["2023-01-03", "2023-01-04"],
        "开盘": [10.0, 10.5], "最高": [10.8, 10.9],
        "最低": [9.9, 10.4], "收盘": [10.5, 10.6],
        "成交量": [1000, 1200],
    })
    out = _normalize_daily_df(raw)
    assert list(out.columns) == ["date", "open", "high", "low", "close", "volume"]
    assert len(out) == 2
    assert out.iloc[0]["close"] == 10.5


def test_daily_uses_cache(tmp_path, monkeypatch):
    # 第二次调用应命中缓存，不再访问网络
    f = AShareFetcher(cache_dir=str(tmp_path))
    calls = {"n": 0}

    def fake_hist(*args, **kwargs):
        calls["n"] += 1
        return pd.DataFrame({
            "日期": pd.to_datetime(["2023-01-03", "2023-01-04"]),
            "开盘": [10.0, 10.5], "最高": [10.8, 10.9],
            "最低": [9.9, 10.4], "收盘": [10.5, 10.6],
            "成交量": [1000, 1200],
        })

    monkeypatch.setattr("data_pipeline.ashare_fetcher.ak.stock_zh_a_hist", fake_hist)
    df1 = f.daily("600000", "2023-01-01", "2023-01-04")
    df2 = f.daily("600000", "2023-01-01", "2023-01-04")
    assert calls["n"] == 1               # 第二次命中缓存
    assert len(df1) == len(df2) == 2
    assert list(df1.columns) == ["date", "open", "high", "low", "close", "volume"]
```

- [ ] **Step 2: 运行测试确认失败**

Run: `uv run pytest tests/unit/test_ashare_fetcher.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'data_pipeline.ashare_fetcher'`

- [ ] **Step 3: 实现 ashare_fetcher.py（先写归一化与缓存骨架）**

创建 `data_pipeline/ashare_fetcher.py`：

```python
"""
data_pipeline/ashare_fetcher.py — A股数据拉取与本地缓存

用 akshare 拉取沪深300+中证500成份股、交易日历、日线行情，
单只股票以 parquet 缓存到 ASHARE_CACHE_DIR，增量更新。
所有网络调用带重试；缓存优先。
"""
from __future__ import annotations

import time
from pathlib import Path

import akshare as ak
import pandas as pd
from loguru import logger

from config import Config

_COL_MAP = {
    "日期": "date", "开盘": "open", "最高": "high",
    "最低": "low", "收盘": "close", "成交量": "volume",
}


def _normalize_daily_df(raw: pd.DataFrame) -> pd.DataFrame:
    """akshare 中文列名 → 标准英文列，只保留 OHLCV+date。"""
    df = raw.rename(columns=_COL_MAP)
    keep = ["date", "open", "high", "low", "close", "volume"]
    df = df[[c for c in keep if c in df.columns]].copy()
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.reset_index(drop=True)


def _retry(fn, *args, tries: int = 3, delay: float = 1.0, **kwargs):
    """带指数退避的重试包装，全部失败则抛最后一次异常。"""
    last = None
    for i in range(tries):
        try:
            return fn(*args, **kwargs)
        except Exception as e:  # noqa: BLE001 — akshare 抛各种网络异常
            last = e
            logger.warning(f"akshare 调用失败({i+1}/{tries}): {e}")
            time.sleep(delay * (2 ** i))
    raise last
```

- [ ] **Step 4: 追加 AShareFetcher 类**

在 `data_pipeline/ashare_fetcher.py` 末尾追加：

```python
class AShareFetcher:
    def __init__(self, cache_dir: str | None = None) -> None:
        self.cache_dir = Path(cache_dir or Config.ASHARE_CACHE_DIR)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def universe_codes(self) -> list[str]:
        """沪深300 + 中证500 当前成份股代码（6位数字，去重排序）。"""
        hs300 = _retry(ak.index_stock_cons, symbol="000300")
        zz500 = _retry(ak.index_stock_cons, symbol="000905")
        codes = set()
        for df in (hs300, zz500):
            col = "品种代码" if "品种代码" in df.columns else df.columns[0]
            codes.update(str(c).zfill(6) for c in df[col].tolist())
        return sorted(codes)

    def trade_calendar(self, start: str, end: str) -> list[str]:
        """交易日历（YYYY-MM-DD），区间 [start, end]。"""
        cal = _retry(ak.tool_trade_date_hist_sina)
        dates = pd.to_datetime(cal["trade_date"]).dt.strftime("%Y-%m-%d")
        mask = (dates >= start) & (dates <= end)
        return dates[mask].tolist()

    def daily(self, code: str, start: str, end: str) -> pd.DataFrame:
        """单只日线，parquet 缓存增量更新。列 [date,open,high,low,close,volume]。"""
        cache_path = self.cache_dir / f"{code}.parquet"
        cached = None
        if cache_path.exists():
            cached = pd.read_parquet(cache_path)
            if not cached.empty and cached["date"].max() >= end:
                sub = cached[(cached["date"] >= start) & (cached["date"] <= end)]
                if not sub.empty:
                    return sub.reset_index(drop=True)
        s = start.replace("-", "")
        e = end.replace("-", "")
        raw = _retry(ak.stock_zh_a_hist, symbol=code, period="daily",
                     start_date=s, end_date=e, adjust="qfq")
        if raw is None or len(raw) == 0:
            return cached if cached is not None else pd.DataFrame(
                columns=["date", "open", "high", "low", "close", "volume"])
        fresh = _normalize_daily_df(raw)
        merged = fresh if cached is None else pd.concat([cached, fresh])
        merged = merged.drop_duplicates("date", keep="last").sort_values("date")
        merged = merged.reset_index(drop=True)
        merged.to_parquet(cache_path, index=False)
        sub = merged[(merged["date"] >= start) & (merged["date"] <= end)]
        return sub.reset_index(drop=True)
```

- [ ] **Step 5: 运行测试确认通过**

Run: `uv run pytest tests/unit/test_ashare_fetcher.py -v`
Expected: PASS（两条用例）

- [ ] **Step 6: 冒烟验证真实网络（可选，需联网）**

Run: `uv run python -c "from data_pipeline.ashare_fetcher import AShareFetcher; f=AShareFetcher(); c=f.universe_codes(); print('universe', len(c)); print(f.daily(c[0],'2024-01-01','2024-02-01').head())"`
Expected: 打印约 800 只代码数与一段日线。若限流则重试。

- [ ] **Step 7: Commit**

```bash
git add data_pipeline/ashare_fetcher.py tests/unit/test_ashare_fetcher.py
git commit -m "feat(ashare): akshare 拉取器与 parquet 缓存"
```

---

## Task 3: AShareDataManager — [N,T] 张量 + valid_mask + 交易日历对齐

**Files:**
- Create: `data_pipeline/ashare_manager.py`
- Test: `tests/unit/test_ashare_manager.py`

**接口契约（与 `MT5DataManager` 对齐，下游零改动）：** `raw_dict{field:[N,T]}`、`feat_tensor[N,F,T]`、`target_ret[N,T]`、`valid_mask[N,T]`、`symbols`、`trade_dates`。

- [ ] **Step 1: 写 manager 测试（用假 fetcher 注入）**

创建 `tests/unit/test_ashare_manager.py`：

```python
import pandas as pd
import torch
from data_pipeline.ashare_manager import AShareDataManager


class FakeFetcher:
    """两只股票：A 全区间有数据，B 中间停牌一天、且上市晚一天。"""
    def universe_codes(self):
        return ["AAA", "BBB"]

    def trade_calendar(self, start, end):
        return ["2023-01-03", "2023-01-04", "2023-01-05", "2023-01-06"]

    def daily(self, code, start, end):
        if code == "AAA":
            dates = ["2023-01-03", "2023-01-04", "2023-01-05", "2023-01-06"]
            close = [10.0, 10.5, 11.0, 10.8]
        else:  # BBB：缺 2023-01-03（未上市）与 2023-01-05（停牌）
            dates = ["2023-01-04", "2023-01-06"]
            close = [20.0, 21.0]
        return pd.DataFrame({
            "date": dates, "open": close, "high": close,
            "low": close, "close": close, "volume": [1000] * len(dates),
        })


def test_shapes_and_calendar():
    mgr = AShareDataManager(fetcher=FakeFetcher())
    mgr.load(start="2023-01-03", end="2023-01-06")
    assert mgr.symbols == ["AAA", "BBB"]
    assert mgr.trade_dates == ["2023-01-03", "2023-01-04", "2023-01-05", "2023-01-06"]
    assert mgr.raw_dict["close"].shape == (2, 4)   # [N=2, T=4]
    assert mgr.target_ret.shape == (2, 4)
    assert mgr.feat_tensor.shape[0] == 2 and mgr.feat_tensor.shape[2] == 4


def test_valid_mask_marks_suspension_and_prelisting():
    mgr = AShareDataManager(fetcher=FakeFetcher())
    mgr.load(start="2023-01-03", end="2023-01-06")
    vm = mgr.valid_mask                             # [2, 4] bool
    # AAA 全有效
    assert vm[0].all()
    # BBB: 01-03 未上市无效, 01-05 停牌无效, 01-04/01-06 有效
    assert not vm[1, 0]     # 未上市
    assert vm[1, 1]         # 有效
    assert not vm[1, 2]     # 停牌
    assert vm[1, 3]         # 有效


def test_forward_fill_price_continuity():
    # 停牌日价格前向填充，保证特征计算连续（无 NaN）
    mgr = AShareDataManager(fetcher=FakeFetcher())
    mgr.load(start="2023-01-03", end="2023-01-06")
    close = mgr.raw_dict["close"]
    assert not torch.isnan(close).any()
    # BBB 01-05 停牌 → 用 01-04 的 20.0 填充
    assert close[1, 2].item() == 20.0
    # BBB 01-03 未上市 → 用首个有效值 20.0 回填
    assert close[1, 0].item() == 20.0
```

- [ ] **Step 2: 运行测试确认失败**

Run: `uv run pytest tests/unit/test_ashare_manager.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'data_pipeline.ashare_manager'`

- [ ] **Step 3: 实现 ashare_manager.py**

创建 `data_pipeline/ashare_manager.py`：

```python
"""
data_pipeline/ashare_manager.py — A股多品种数据管理器

产出与 MT5DataManager 相同的接口契约（raw_dict/feat_tensor/target_ret/symbols），
下游 model_core 零改动。额外提供 valid_mask（停牌/未上市剔除）与 trade_dates。

关键：用固定交易日历对齐（不用时间戳交集），停牌日前向填充价格但标记无效。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from loguru import logger

from config import Config
from data_pipeline.ashare_fetcher import AShareFetcher


class AShareDataManager:
    def __init__(self, fetcher=None) -> None:
        self._fetcher = fetcher or AShareFetcher()
        self._symbols: list[str] = []
        self._trade_dates: list[str] = []
        self._raw_dict: dict[str, torch.Tensor] | None = None
        self._target_ret: torch.Tensor | None = None
        self._valid_mask: torch.Tensor | None = None

    def load(self, codes: list[str] | None = None,
             start: str = "2023-01-01", end: str = "2026-06-30") -> None:
        codes = codes or self._fetcher.universe_codes()
        calendar = self._fetcher.trade_calendar(start, end)
        cal_idx = pd.Index(calendar)
        T = len(calendar)

        fields = ["open", "high", "low", "close", "volume"]
        price_rows = {f: [] for f in fields}
        mask_rows = []
        valid_codes = []

        for code in codes:
            df = self._fetcher.daily(code, start, end)
            if df is None or len(df) < 1:
                continue
            df = df.drop_duplicates("date", keep="last").set_index("date")
            df = df.reindex(cal_idx)                       # 对齐到固定日历
            present = df["close"].notna().values           # 该日有真实成交=有效
            if present.sum() < 1:
                continue
            # 停牌/未上市：价格前向+后向填充保证连续，mask 记录真实有效位
            df_filled = df[fields].ffill().bfill()
            for f in fields:
                price_rows[f].append(df_filled[f].values.astype("float32"))
            mask_rows.append(present)
            valid_codes.append(code)

        if not valid_codes:
            raise ValueError("无有效股票数据")

        self._symbols = valid_codes
        self._trade_dates = list(calendar)
        self._raw_dict = {
            f: torch.tensor(np.array(price_rows[f]), dtype=torch.float32)
            for f in fields
        }
        # time 字段：交易日历转 Unix 秒，[N,T]（各股相同）
        secs = pd.to_datetime(calendar).astype("int64") // 10**9
        self._raw_dict["time"] = torch.tensor(
            np.tile(secs.values, (len(valid_codes), 1)), dtype=torch.int64)
        self._valid_mask = torch.tensor(np.array(mask_rows), dtype=torch.bool)
        self._target_ret = self._compute_target_ret(self._raw_dict["open"])
        logger.info(f"[A股] 加载 {len(valid_codes)} 只 × {T} 交易日")

    @staticmethod
    def _compute_target_ret(open_tensor: torch.Tensor) -> torch.Tensor:
        """target_ret[n,t] = log(open[t+2]/open[t+1])，末两位补0（与 MT5 口径一致）。"""
        n, t = open_tensor.shape
        target = torch.zeros(n, t, dtype=torch.float32)
        if t >= 3:
            num = open_tensor[:, 2:]
            den = open_tensor[:, 1:-1].clone()
            den[den == 0] = 1.0
            target[:, :t - 2] = torch.log(num / den)
        return target

    @property
    def symbols(self) -> list[str]:
        return list(self._symbols)

    @property
    def trade_dates(self) -> list[str]:
        return list(self._trade_dates)

    @property
    def raw_dict(self) -> dict:
        assert self._raw_dict is not None, "call load() first"
        return self._raw_dict

    @property
    def target_ret(self) -> torch.Tensor:
        assert self._target_ret is not None, "call load() first"
        return self._target_ret

    @property
    def valid_mask(self) -> torch.Tensor:
        assert self._valid_mask is not None, "call load() first"
        return self._valid_mask

    @property
    def feat_tensor(self) -> torch.Tensor:
        from model_core.features import MT5FeatureEngineer
        return MT5FeatureEngineer.compute_features(self.raw_dict)

    @property
    def bar_time(self) -> torch.Tensor:
        return self.raw_dict["time"][:, -1].long()
```

- [ ] **Step 4: 运行测试确认通过**

Run: `uv run pytest tests/unit/test_ashare_manager.py -v`
Expected: PASS（三条用例）

- [ ] **Step 5: Commit**

```bash
git add data_pipeline/ashare_manager.py tests/unit/test_ashare_manager.py
git commit -m "feat(ashare): 数据管理器（交易日历对齐+停牌mask）"
```

---

## Task 4: backtest.py 截面 Rank-IC reward 分支

**Files:**
- Modify: `model_core/backtest.py`（`_multi_objective` 内加 `ashare` 分支 + 新增 `_cross_sectional_ic` 方法）
- Test: `tests/unit/test_ashare_reward.py`

**背景：** A股选股是截面排序问题。当前 `_multi_objective` 的 N>1 分支用时序 Sortino/年化收益，不契合。新增 `REWARD_MODE="ashare"`：主权重给**截面 Rank-IC**（每交易日对 N 只股票的因子值与次日收益求 Spearman 相关，再对时间取均值），弱化外汇特有的 beta/感染惩罚。

- [ ] **Step 1: 写截面 IC 测试**

创建 `tests/unit/test_ashare_reward.py`：

```python
import torch
from model_core.backtest import MT5Backtest


def test_cross_sectional_ic_perfect_positive():
    # 因子值与次日收益截面完全同序 → Rank-IC ≈ +1
    bt = MT5Backtest()
    # N=4 只股, T=3 期。factor[:,t] 与 target_ret[:,t+1] 同序
    factor = torch.tensor([[1., 1., 1.],
                           [2., 2., 2.],
                           [3., 3., 3.],
                           [4., 4., 4.]])
    target = torch.tensor([[0.0, 0.1, 0.1],
                           [0.0, 0.2, 0.2],
                           [0.0, 0.3, 0.3],
                           [0.0, 0.4, 0.4]])
    ic = bt._cross_sectional_ic(factor, target)
    assert ic > 0.9


def test_cross_sectional_ic_perfect_negative():
    bt = MT5Backtest()
    factor = torch.tensor([[1., 1.], [2., 2.], [3., 3.], [4., 4.]])
    target = torch.tensor([[0.0, 0.4], [0.0, 0.3], [0.0, 0.2], [0.0, 0.1]])
    ic = bt._cross_sectional_ic(factor, target)
    assert ic < -0.9


def test_ashare_reward_prefers_higher_ic():
    import model_core.config as mc
    mc.ModelConfig.REWARD_MODE = "ashare"
    bt = MT5Backtest()
    good_f = torch.tensor([[1., 1.], [2., 2.], [3., 3.], [4., 4.]])
    good_t = torch.tensor([[0.0, 0.1], [0.0, 0.2], [0.0, 0.3], [0.0, 0.4]])
    pos = torch.tanh(good_f)
    r_good = bt._multi_objective(good_f, good_t, pos * good_t, pos)
    # 打乱因子与收益的截面对应 → IC 崩 → reward 更低
    bad_f = torch.tensor([[4., 4.], [1., 1.], [3., 3.], [2., 2.]])
    r_bad = bt._multi_objective(bad_f, good_t, torch.tanh(bad_f) * good_t, torch.tanh(bad_f))
    assert r_good.item() > r_bad.item()
```

- [ ] **Step 2: 运行测试确认失败**

Run: `uv run pytest tests/unit/test_ashare_reward.py -v`
Expected: FAIL — `AttributeError: 'MT5Backtest' object has no attribute '_cross_sectional_ic'`

- [ ] **Step 3: 加 `_cross_sectional_ic` 方法**

在 `model_core/backtest.py` 的 `_ts_ic_stability` 方法**之后**插入：

```python
    def _cross_sectional_ic(self, factors: Tensor, target_ret: Tensor) -> float:
        """截面 Rank-IC：每时间步对 N 只股票的 factor[t] 与 ret[t+1] 求
        Spearman 秩相关，再对时间取均值。适合 A股选股（N 大，横截面统计强）。

        Returns: float ∈ [-1,1]，正值代表因子选股方向正确。
        """
        N, T = factors.shape
        if N < 3 or T < 2:
            return 0.0

        def _rank(x: Tensor) -> Tensor:
            # 沿 N 维求秩（每列独立），返回 [N] 或 [N,T]
            order = x.argsort(dim=0)
            ranks = torch.zeros_like(x)
            idx = torch.arange(x.shape[0], dtype=x.dtype, device=x.device)
            if x.dim() == 1:
                ranks.scatter_(0, order, idx)
            else:
                ranks.scatter_(0, order, idx.unsqueeze(1).expand_as(x))
            return ranks

        ic_list = []
        for t in range(T - 1):
            fx = factors[:, t]
            fy = target_ret[:, t + 1]
            rx = _rank(fx)
            ry = _rank(fy)
            rxm = rx - rx.mean()
            rym = ry - ry.mean()
            sx = (rxm ** 2).mean().sqrt()
            sy = (rym ** 2).mean().sqrt()
            if sx < 1e-6 or sy < 1e-6:
                continue
            ic = (rxm * rym).mean() / (sx * sy + 1e-8)
            ic_list.append(ic.item())

        if not ic_list:
            return 0.0
        return float(sum(ic_list) / len(ic_list))
```

- [ ] **Step 4: 在 `_multi_objective` 加 ashare 分支**

在 `model_core/backtest.py` 的 `_multi_objective` 方法内，`N = pnl.shape[0]` 之后、现有 `ann_ret = ...` 计算之前插入（放在方法最前面优先返回）：

```python
        if ModelConfig.REWARD_MODE == "ashare":
            # A股截面选股模式：主权重给截面 Rank-IC + IC 稳定性，
            # 弱化外汇特有的年化收益/beta 惩罚（截面排序不关心绝对多空）。
            cs_ic = self._cross_sectional_ic(factors, target_ret)
            ts_ic = self._ts_ic_stability(factors, target_ret)   # IC_IR 作稳定性
            tq    = self._turnover_quality(position)
            exp_pen = self._exposure_penalty(position)
            return (
                torch.tensor(
                    0.70 * cs_ic          # 主目标：截面 Rank-IC（选股方向）
                    + 0.20 * ts_ic        # IC 稳定性（IR）
                    + 0.05 * tq,          # 换手质量
                    dtype=torch.float32,
                )
                + exp_pen                  # 稀疏惩罚（张量）
            )
```

注：`exp_pen` 是张量，前面的加权和先包成张量再相加，返回类型与其它分支一致（Tensor）。

- [ ] **Step 5: 运行测试确认通过**

Run: `uv run pytest tests/unit/test_ashare_reward.py -v`
Expected: PASS（三条用例）

- [ ] **Step 6: 回归——确认原有 reward 分支未破坏**

Run: `uv run pytest tests/unit/test_backtest.py -v`
Expected: PASS（若该文件存在；否则 `uv run pytest tests/ -k backtest -v`）

- [ ] **Step 7: Commit**

```bash
git add model_core/backtest.py tests/unit/test_ashare_reward.py
git commit -m "feat(ashare): 截面 Rank-IC reward 分支"
```

---

## Task 5: train_ashare.py — 截面因子训练入口

**Files:**
- Create: `train_ashare.py`
- Test: `tests/smoke/test_train_ashare_smoke.py`

**目标：** 用 `AShareDataManager` 喂 `AlphaEngine(target_symbol=None)`，`REWARD_MODE="ashare"` 训练，产出 `strategies/best_ashare_universe.json`。

- [ ] **Step 1: 写训练入口冒烟测试（小样本、少步数）**

创建 `tests/smoke/test_train_ashare_smoke.py`：

```python
import pandas as pd
from train_ashare import train_ashare


class TinyFetcher:
    def universe_codes(self):
        return [f"C{i:03d}" for i in range(8)]      # 8 只股

    def trade_calendar(self, start, end):
        return pd.bdate_range("2024-01-01", periods=120).strftime("%Y-%m-%d").tolist()

    def daily(self, code, start, end):
        import numpy as np
        dates = self.trade_calendar(start, end)
        seed = int(code[1:])
        rng = np.random.default_rng(seed)
        price = 10 + np.cumsum(rng.normal(0, 0.2, len(dates)))
        price = abs(price) + 1
        return pd.DataFrame({
            "date": dates, "open": price, "high": price * 1.01,
            "low": price * 0.99, "close": price, "volume": rng.integers(1e5, 1e6, len(dates)),
        })


def test_train_ashare_runs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    import model_core.config as mc
    monkeypatch.setattr(mc.ModelConfig, "TRAIN_STEPS", 5, raising=False)
    engine = train_ashare(fetcher=TinyFetcher(), start="2024-01-01",
                          end="2024-06-30", steps=5)
    assert engine is not None
    out = tmp_path / "strategies" / "best_ashare_universe.json"
    assert out.exists()
```

- [ ] **Step 2: 运行测试确认失败**

Run: `uv run pytest tests/smoke/test_train_ashare_smoke.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'train_ashare'`

- [ ] **Step 3: 实现 train_ashare.py**

创建 `train_ashare.py`：

```python
"""
train_ashare.py — A股截面因子训练入口

用 AShareDataManager 喂 AlphaEngine 多品种截面模式，REWARD_MODE="ashare"，
产出 strategies/best_ashare_universe.json（可解释 token 公式 + vocab 校验）。

用法:
    python train_ashare.py --start 2023-01-01 --end 2026-05-31
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from config import Config
from data_pipeline.ashare_manager import AShareDataManager
from model_core.config import ModelConfig
from model_core.engine import AlphaEngine
from model_core.vocab import VOCAB_VERSION

_UNIVERSE_TAG = "ashare_universe"


def train_ashare(fetcher=None, start="2023-01-01", end="2026-05-31",
                 steps: int | None = None) -> AlphaEngine | None:
    ModelConfig.REWARD_MODE = "ashare"
    if steps is not None:
        ModelConfig.TRAIN_STEPS = steps

    mgr = AShareDataManager(fetcher=fetcher)
    mgr.load(start=start, end=end)
    print(f"[A股训练] {len(mgr.symbols)} 只 × {len(mgr.trade_dates)} 交易日")

    engine = AlphaEngine(data_manager=mgr, target_symbol=None)
    engine.mode = "ashare_universe"
    engine.train(start_step=0)
    _save_strategy(engine)
    return engine


def _save_strategy(engine: AlphaEngine) -> None:
    path = pathlib.Path("strategies") / f"best_{_UNIVERSE_TAG}.json"
    path.parent.mkdir(exist_ok=True)
    data = {
        "vocab_version": VOCAB_VERSION,
        "symbol": _UNIVERSE_TAG,
        "mode": "ashare_universe",
        "formula": engine.best_formula,
        "formula_decoded": engine._decode_formula(engine.best_formula)
        if engine.best_formula else None,
        "best_score": engine.best_score,
        "train_steps": ModelConfig.TRAIN_STEPS,
    }
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[A股训练] 策略已保存: {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2023-01-01")
    ap.add_argument("--end", default="2026-05-31")
    ap.add_argument("--steps", type=int, default=None)
    args = ap.parse_args()
    eng = train_ashare(start=args.start, end=args.end, steps=args.steps)
    if eng and eng.best_formula:
        print(f"最优公式: {eng._decode_formula(eng.best_formula)}  分数={eng.best_score:.4f}")
```

- [ ] **Step 4: 运行测试确认通过**

Run: `uv run pytest tests/smoke/test_train_ashare_smoke.py -v`
Expected: PASS。注意：若 `AlphaEngine` 因 walk-forward 折叠对 120 期报错，测试里已用 120 期规避；如仍失败读报错调整最小期数。

- [ ] **Step 5: Commit**

```bash
git add train_ashare.py tests/smoke/test_train_ashare_smoke.py
git commit -m "feat(ashare): 截面因子训练入口 train_ashare.py"
```

---

## Task 6: paper_trading/config.py + account.py — 账户结构与成交成本原语

**Files:**
- Create: `paper_trading/__init__.py`（空文件）
- Create: `paper_trading/config.py`（从根 Config 读 A股常量的薄封装）
- Create: `paper_trading/account.py`
- Test: `tests/unit/test_account.py`

- [ ] **Step 1: 写账户原语测试**

创建 `tests/unit/test_account.py`：

```python
from paper_trading.account import (
    Account, Holding, buy_commission, sell_cost, max_buyable_shares,
)


def test_buy_commission_min_floor():
    # 成交额小 → 命中最低 5 元
    assert buy_commission(1000.0) == 5.0
    # 成交额大 → 按万2.5
    assert abs(buy_commission(100_000.0) - 25.0) < 1e-9


def test_sell_cost_includes_stamp_tax():
    # 卖出 = 佣金(万2.5,最低5) + 印花税(千1)
    turnover = 100_000.0
    expected = max(turnover * 0.00025, 5.0) + turnover * 0.001
    assert abs(sell_cost(turnover) - expected) < 1e-9


def test_max_buyable_shares_lot_rounding():
    # 现金 10000, 价格 9.9 → 理论 1010 股, 向下取整到 100 → 1000 股
    # 但需预留佣金, 精确到 100 股整数倍且总花费<=现金
    shares = max_buyable_shares(cash=10_000.0, price=9.9)
    assert shares % 100 == 0
    cost = shares * 9.9 + buy_commission(shares * 9.9)
    assert cost <= 10_000.0
    # 再多买 100 股就会超预算
    over = (shares + 100) * 9.9 + buy_commission((shares + 100) * 9.9)
    assert over > 10_000.0


def test_account_nav():
    acc = Account(cash=100_000.0)
    acc.holdings["600000"] = Holding(
        code="600000", shares=1000, cost_price=10.0,
        buy_date="2024-01-02", sellable_date="2024-01-03")
    nav = acc.nav({"600000": 11.0})     # 收盘价 11
    assert abs(nav - (100_000.0 + 1000 * 11.0)) < 1e-9
```

- [ ] **Step 2: 运行测试确认失败**

Run: `uv run pytest tests/unit/test_account.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'paper_trading'`

- [ ] **Step 3: 创建包与配置**

创建 `paper_trading/__init__.py`（空）。

创建 `paper_trading/config.py`：

```python
"""paper_trading/config.py — A股交易规则常量（从根 Config 读取）。"""
from config import Config

INITIAL_CAPITAL = Config.ASHARE_INITIAL_CAPITAL
TOP_K           = Config.ASHARE_TOP_K
COMMISSION_RATE = Config.ASHARE_COMMISSION_RATE
MIN_COMMISSION  = Config.ASHARE_MIN_COMMISSION
STAMP_TAX       = Config.ASHARE_STAMP_TAX
LIMIT_PCT       = Config.ASHARE_LIMIT_PCT
LOT_SIZE        = Config.ASHARE_LOT_SIZE
```

- [ ] **Step 4: 实现 account.py**

创建 `paper_trading/account.py`：

```python
"""
paper_trading/account.py — 账户数据结构与成交/成本原语

Holding: 单只持仓（含 T+1 sellable_date）。
Account: 现金 + 持仓 + 净值/流水记录。
成本原语: 买入佣金、卖出成本（佣金+印花税）、可买股数（100股整数倍且不透支）。
"""
from __future__ import annotations

from dataclasses import dataclass, field

from paper_trading.config import (
    COMMISSION_RATE, LOT_SIZE, MIN_COMMISSION, STAMP_TAX,
)


def buy_commission(turnover: float) -> float:
    """买入佣金：max(成交额 × 费率, 最低佣金)。"""
    return max(turnover * COMMISSION_RATE, MIN_COMMISSION)


def sell_cost(turnover: float) -> float:
    """卖出总成本：佣金 + 印花税。"""
    return max(turnover * COMMISSION_RATE, MIN_COMMISSION) + turnover * STAMP_TAX


def max_buyable_shares(cash: float, price: float) -> int:
    """在不透支（含买入佣金）前提下，最多可买多少股（100 股整数倍）。"""
    if price <= 0 or cash <= 0:
        return 0
    lots = int(cash // (price * LOT_SIZE))     # 先粗估手数上限
    while lots > 0:
        shares = lots * LOT_SIZE
        turnover = shares * price
        if turnover + buy_commission(turnover) <= cash:
            return shares
        lots -= 1
    return 0


@dataclass
class Holding:
    code: str
    shares: int
    cost_price: float
    buy_date: str
    sellable_date: str


@dataclass
class Account:
    cash: float
    holdings: dict[str, Holding] = field(default_factory=dict)
    nav_history: list[tuple[str, float]] = field(default_factory=list)
    trades: list[dict] = field(default_factory=list)

    def nav(self, close_prices: dict[str, float]) -> float:
        """总净值 = 现金 + Σ(持股数 × 当日收盘价)。停牌股用传入的最后有效价。"""
        mkt = sum(h.shares * close_prices.get(c, h.cost_price)
                  for c, h in self.holdings.items())
        return self.cash + mkt
```

- [ ] **Step 5: 运行测试确认通过**

Run: `uv run pytest tests/unit/test_account.py -v`
Expected: PASS（四条用例）

- [ ] **Step 6: Commit**

```bash
git add paper_trading/__init__.py paper_trading/config.py paper_trading/account.py tests/unit/test_account.py
git commit -m "feat(paper): 账户结构与成交成本原语"
```

---

## Task 7: PortfolioEngine — T+1 逐日组合引擎

**Files:**
- Create: `paper_trading/portfolio_engine.py`
- Test: `tests/property/test_portfolio.py`

**核心 `step(date, next_date, target_codes, bar)`：** T 日收盘后决策 → 用 `bar`（含各股 open/close/limit 状态）在 T+1 开盘成交。
- `bar` 结构：`{code: {"open": float, "close": float, "limit_up": bool, "limit_down": bool, "tradable": bool}}`
- 卖出：持仓中不在 target 且 `sellable_date <= date` 的，按 open 卖出（封跌停或不可交易跳过）。
- 买入：target 中未持有的，等权用现金买入（封涨停或不可交易跳过）。
- 估值：按 close mark-to-market，记 nav。

- [ ] **Step 1: 写 property 测试（账户恒等式 + T+1 + 涨跌停）**

创建 `tests/property/test_portfolio.py`：

```python
from paper_trading.portfolio_engine import PortfolioEngine
from paper_trading.account import buy_commission, sell_cost


def _bar(prices, limit_up=(), limit_down=(), untradable=()):
    return {c: {"open": p, "close": p,
                "limit_up": c in limit_up, "limit_down": c in limit_down,
                "tradable": c not in untradable}
            for c, p in prices.items()}


def test_nav_identity_holds():
    # 任意步后：记录的 nav == cash + Σ shares*close
    eng = PortfolioEngine(initial_capital=1_000_000.0)
    eng.step("2024-01-02", "2024-01-03", ["A", "B"],
             _bar({"A": 10.0, "B": 20.0}))
    close = {"A": 10.0, "B": 20.0}
    nav = eng.account.cash + sum(h.shares * close[c] for c, h in eng.account.holdings.items())
    assert abs(eng.account.nav_history[-1][1] - nav) < 1e-6


def test_t_plus_1_not_violated():
    # 当日买入不可当日卖出：即使次日 target 不含该股, buy_date==决策日的持仓
    # sellable_date 必须严格大于买入日
    eng = PortfolioEngine(initial_capital=1_000_000.0)
    eng.step("2024-01-02", "2024-01-03", ["A"], _bar({"A": 10.0}))
    h = eng.account.holdings["A"]
    assert h.buy_date == "2024-01-03"          # T+1 开盘成交, 买入日=成交日
    assert h.sellable_date > h.buy_date        # 次一交易日才可卖


def test_limit_up_blocks_buy():
    # 封涨停 → 买不进, 现金不变, 无持仓
    eng = PortfolioEngine(initial_capital=1_000_000.0)
    before = eng.account.cash
    eng.step("2024-01-02", "2024-01-03", ["A"],
             _bar({"A": 10.0}, limit_up=["A"]))
    assert "A" not in eng.account.holdings
    assert eng.account.cash == before


def test_limit_down_blocks_sell():
    # 持有 A, 次日跌停封板 → 卖不出, 仍持有
    eng = PortfolioEngine(initial_capital=1_000_000.0)
    eng.step("2024-01-02", "2024-01-03", ["A"], _bar({"A": 10.0}))
    # 次日 target 清空且 A 跌停
    eng.step("2024-01-03", "2024-01-04", [],
             _bar({"A": 9.0}, limit_down=["A"]))
    assert "A" in eng.account.holdings         # 跌停卖不出

def test_cash_never_negative():
    eng = PortfolioEngine(initial_capital=100_000.0)
    eng.step("2024-01-02", "2024-01-03", ["A", "B", "C"],
             _bar({"A": 10.0, "B": 20.0, "C": 30.0}))
    assert eng.account.cash >= 0
```

- [ ] **Step 2: 运行测试确认失败**

Run: `uv run pytest tests/property/test_portfolio.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'paper_trading.portfolio_engine'`

- [ ] **Step 3: 实现 portfolio_engine.py**

创建 `paper_trading/portfolio_engine.py`：

```python
"""
paper_trading/portfolio_engine.py — T+1 逐日组合账户引擎

step(date, next_date, target_codes, bar): T日收盘决策 → T+1(next_date)开盘成交。
处理 T+1 交收、涨跌停封板不成交、真实成本、等权 Top-K。
"""
from __future__ import annotations

from paper_trading.account import (
    Account, Holding, buy_commission, max_buyable_shares, sell_cost,
)
from paper_trading.config import INITIAL_CAPITAL, LOT_SIZE, TOP_K


class PortfolioEngine:
    def __init__(self, initial_capital: float = INITIAL_CAPITAL) -> None:
        self.account = Account(cash=float(initial_capital))

    def step(self, date: str, next_date: str,
             target_codes: list[str], bar: dict) -> None:
        """在 next_date 开盘按 bar 成交，收盘估值记 nav。

        Args:
            date:         决策日（T日，已收盘）。
            next_date:    成交日（T+1，用其开盘价成交、收盘价估值）。
            target_codes: 目标持仓代码列表（因子选出的 Top-K）。
            bar:          {code: {open, close, limit_up, limit_down, tradable}}。
        """
        acc = self.account
        target = set(target_codes)

        # ── 1. 卖出：不在 target 且已过 T+1 ────────────────────────
        # T+1 口径：holding.buy_date 存买入日；只有买入日 < 成交日（next_date）
        # 才可卖出（严格小于 = 至少隔一个交易日）。
        for code in list(acc.holdings.keys()):
            if code in target:
                continue
            h = acc.holdings[code]
            if h.buy_date >= next_date:
                continue                        # 当日买入，T+1 未到，不可卖
            b = bar.get(code)
            if b is None or not b["tradable"] or b["limit_down"]:
                continue                        # 停牌/跌停封板，卖不出
            price = b["open"]
            turnover = h.shares * price
            acc.cash += turnover - sell_cost(turnover)
            acc.trades.append({
                "date": next_date, "code": code, "side": "SELL",
                "price": price, "shares": h.shares, "cost": sell_cost(turnover),
            })
            del acc.holdings[code]

        # ── 2. 买入：target 中未持有的，等权分配 ─────────────────────
        to_buy = [c for c in target_codes if c not in acc.holdings]
        buyable = [c for c in to_buy
                   if bar.get(c) and bar[c]["tradable"] and not bar[c]["limit_up"]]
        if buyable:
            # 等权目标：每只用 (总资产/TOP_K)，但不超过当前可用现金
            nav_now = acc.nav({c: b["open"] for c, b in bar.items()})
            per_budget = nav_now / TOP_K
            for code in buyable:
                b = bar[code]
                price = b["open"]
                budget = min(per_budget, acc.cash)
                shares = max_buyable_shares(cash=budget, price=price)
                if shares < LOT_SIZE:
                    continue
                turnover = shares * price
                fee = buy_commission(turnover)
                if turnover + fee > acc.cash:
                    continue
                acc.cash -= turnover + fee
                acc.holdings[code] = Holding(
                    code=code, shares=shares, cost_price=price,
                    buy_date=next_date, sellable_date=next_date,
                )
                acc.trades.append({
                    "date": next_date, "code": code, "side": "BUY",
                    "price": price, "shares": shares, "cost": fee,
                })

        # ── 3. 收盘估值 ────────────────────────────────────────────
        close_prices = {c: b["close"] for c, b in bar.items()}
        acc.nav_history.append((next_date, acc.nav(close_prices)))
```

注：`Holding.sellable_date` 存买入日本身；T+1 语义完全由卖出分支的 `h.buy_date >= next_date` 严格比较实现（买入日 == 成交日时不可卖，隔日才可）。`sellable_date` 字段保留供展示与未来扩展。

- [ ] **Step 4: 运行测试确认通过**

Run: `uv run pytest tests/property/test_portfolio.py -v`
Expected: PASS（五条用例）

- [ ] **Step 5: Commit**

```bash
git add paper_trading/portfolio_engine.py tests/property/test_portfolio.py
git commit -m "feat(paper): T+1 逐日组合引擎（涨跌停/成本/等权）"
```

---

## Task 8: FactorRanker — 因子 → Top10 选股

**Files:**
- Create: `paper_trading/factor_ranker.py`
- Test: `tests/unit/test_factor_ranker.py`

**职责：** 用 `StackVM` 执行公式得到当日截面因子值 `[N]`，剔除无效股（停牌/未上市/封板），降序取 Top-K 代码。

- [ ] **Step 1: 写 ranker 测试**

创建 `tests/unit/test_factor_ranker.py`：

```python
import torch
from paper_trading.factor_ranker import FactorRanker


def test_rank_picks_top_k_by_factor():
    # 构造一个恒等公式：直接取特征0作为因子值
    # feat_slice [N, F, T]，公式 [0] 表示压入特征0
    ranker = FactorRanker(formula=[0])
    N, F, T = 5, 3, 4
    feat = torch.zeros(N, F, T)
    # 特征0在最后时间步的值：股票越靠后越大
    feat[:, 0, -1] = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    valid = torch.ones(N, dtype=torch.bool)
    codes = ["A", "B", "C", "D", "E"]
    picks = ranker.rank(feat, valid, codes, top_k=2)
    assert picks == ["E", "D"]                 # 因子值最大的两只


def test_rank_excludes_invalid():
    ranker = FactorRanker(formula=[0])
    N, F, T = 5, 3, 4
    feat = torch.zeros(N, F, T)
    feat[:, 0, -1] = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    valid = torch.tensor([True, True, True, True, False])  # E 停牌
    codes = ["A", "B", "C", "D", "E"]
    picks = ranker.rank(feat, valid, codes, top_k=2)
    assert "E" not in picks                    # 无效股被剔除
    assert picks == ["D", "C"]


def test_rank_handles_fewer_valid_than_k():
    ranker = FactorRanker(formula=[0])
    N, F, T = 5, 3, 2
    feat = torch.zeros(N, F, T)
    feat[:, 0, -1] = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    valid = torch.tensor([True, False, False, False, False])
    picks = ranker.rank(feat, valid, ["A", "B", "C", "D", "E"], top_k=3)
    assert picks == ["A"]                      # 只有 1 只有效
```

- [ ] **Step 2: 运行测试确认失败**

Run: `uv run pytest tests/unit/test_factor_ranker.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'paper_trading.factor_ranker'`

- [ ] **Step 3: 实现 factor_ranker.py**

创建 `paper_trading/factor_ranker.py`：

```python
"""
paper_trading/factor_ranker.py — 因子公式 → 当日 Top-K 选股

用 StackVM 执行 token 公式得到截面因子值，剔除无效股，降序取 Top-K。
不用 tanh：纯多头排序信号（A股只能做多）。
"""
from __future__ import annotations

import torch

from model_core.vm import StackVM


class FactorRanker:
    def __init__(self, formula: list[int]) -> None:
        self.formula = [int(t) for t in formula]
        self.vm = StackVM()

    def rank(self, feat_slice: torch.Tensor, valid_mask: torch.Tensor,
             codes: list[str], top_k: int = 10) -> list[str]:
        """对当日有效股票按因子值降序取 Top-K 代码。

        Args:
            feat_slice: [N, F, T] 截至当日的特征张量（用最后时间步作当日截面）。
            valid_mask: [N] bool，当日是否可交易（停牌/未上市/封板剔除）。
            codes:      [N] 股票代码，与张量 N 维一一对应。
            top_k:      取前 K 只。
        """
        factor = self.vm.execute(self.formula, feat_slice)   # [N, T] 或 None
        if factor is None:
            return []
        vals = factor[:, -1]                                 # 当日截面因子值 [N]
        vals = torch.nan_to_num(vals, nan=float("-inf"))
        # 无效股设为 -inf，排序自然沉底
        vals = torch.where(valid_mask, vals, torch.full_like(vals, float("-inf")))
        n_valid = int(valid_mask.sum().item())
        k = min(top_k, n_valid)
        if k <= 0:
            return []
        order = torch.argsort(vals, descending=True)[:k]
        return [codes[i] for i in order.tolist()]
```

- [ ] **Step 4: 运行测试确认通过**

Run: `uv run pytest tests/unit/test_factor_ranker.py -v`
Expected: PASS（三条用例）

- [ ] **Step 5: Commit**

```bash
git add paper_trading/factor_ranker.py tests/unit/test_factor_ranker.py
git commit -m "feat(paper): 因子→Top-K 选股 FactorRanker"
```

---

## Task 9: metrics.py — 绩效指标

**Files:**
- Create: `paper_trading/metrics.py`
- Test: `tests/unit/test_metrics.py`

- [ ] **Step 1: 写指标测试**

创建 `tests/unit/test_metrics.py`：

```python
import math
from paper_trading.metrics import compute_metrics


def test_metrics_basic():
    # 净值 100万→110万，单调上升无回撤
    nav = [("2024-01-02", 1_000_000.0),
           ("2024-01-03", 1_050_000.0),
           ("2024-01-04", 1_100_000.0)]
    m = compute_metrics(nav, initial_capital=1_000_000.0)
    assert abs(m["total_return"] - 0.10) < 1e-9
    assert m["max_drawdown"] == 0.0
    assert m["final_nav"] == 1_100_000.0


def test_metrics_drawdown():
    # 100万→120万→90万：峰值120万, 谷底90万, 回撤 = (120-90)/120 = 0.25
    nav = [("d1", 1_000_000.0), ("d2", 1_200_000.0), ("d3", 900_000.0)]
    m = compute_metrics(nav, initial_capital=1_000_000.0)
    assert abs(m["max_drawdown"] - 0.25) < 1e-9


def test_metrics_empty():
    m = compute_metrics([], initial_capital=1_000_000.0)
    assert m["total_return"] == 0.0
    assert m["final_nav"] == 1_000_000.0
```

- [ ] **Step 2: 运行测试确认失败**

Run: `uv run pytest tests/unit/test_metrics.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'paper_trading.metrics'`

- [ ] **Step 3: 实现 metrics.py**

创建 `paper_trading/metrics.py`：

```python
"""paper_trading/metrics.py — 组合绩效指标。"""
from __future__ import annotations

import math

_TRADING_DAYS = 244   # A股年化交易日数


def compute_metrics(nav_history: list[tuple[str, float]],
                    initial_capital: float) -> dict:
    """从净值序列计算总收益/最大回撤/年化/夏普。

    Args:
        nav_history: [(date, nav), ...] 逐日总净值。
        initial_capital: 起始资金。
    """
    if not nav_history:
        return {"total_return": 0.0, "max_drawdown": 0.0, "sharpe": 0.0,
                "annual_return": 0.0, "final_nav": initial_capital, "days": 0}

    navs = [v for _, v in nav_history]
    final = navs[-1]
    total_return = final / initial_capital - 1.0

    # 最大回撤
    peak = navs[0]
    max_dd = 0.0
    for v in navs:
        peak = max(peak, v)
        dd = (peak - v) / peak if peak > 0 else 0.0
        max_dd = max(max_dd, dd)

    # 日收益序列
    rets = [navs[i] / navs[i - 1] - 1.0 for i in range(1, len(navs))]
    if rets:
        mean_r = sum(rets) / len(rets)
        var = sum((r - mean_r) ** 2 for r in rets) / len(rets)
        std = math.sqrt(var)
        sharpe = (mean_r / std * math.sqrt(_TRADING_DAYS)) if std > 1e-12 else 0.0
        annual = mean_r * _TRADING_DAYS
    else:
        sharpe = annual = 0.0

    return {
        "total_return": total_return,
        "max_drawdown": max_dd,
        "sharpe": sharpe,
        "annual_return": annual,
        "final_nav": final,
        "days": len(navs),
    }
```

- [ ] **Step 4: 运行测试确认通过**

Run: `uv run pytest tests/unit/test_metrics.py -v`
Expected: PASS（三条用例）

- [ ] **Step 5: Commit**

```bash
git add paper_trading/metrics.py tests/unit/test_metrics.py
git commit -m "feat(paper): 绩效指标计算"
```

---

## Task 10: ReplayFeed + run_paper_replay.py — 历史回放端到端

**Files:**
- Create: `paper_trading/data_feed.py`（先只实现 ReplayFeed）
- Create: `run_paper_replay.py`
- Test: `tests/smoke/test_paper_smoke.py`

**ReplayFeed 职责：** 持有已加载的 `AShareDataManager`，把它切成逐日的「特征切片 + bar」喂给引擎。

**bar 构造：** 用 `raw_dict` 的 open/close 与前一日收盘价判定涨跌停（`open >= prev_close*(1+LIMIT_PCT)*0.999` 视为封涨停），`valid_mask` 给 tradable。

- [ ] **Step 1: 写端到端冒烟测试**

创建 `tests/smoke/test_paper_smoke.py`：

```python
import pandas as pd
from paper_trading.data_feed import ReplayFeed
from paper_trading.portfolio_engine import PortfolioEngine
from paper_trading.factor_ranker import FactorRanker
from paper_trading.metrics import compute_metrics
from data_pipeline.ashare_manager import AShareDataManager


class TinyFetcher:
    def universe_codes(self):
        return [f"C{i:03d}" for i in range(20)]

    def trade_calendar(self, start, end):
        return pd.bdate_range("2024-01-01", periods=80).strftime("%Y-%m-%d").tolist()

    def daily(self, code, start, end):
        import numpy as np
        dates = self.trade_calendar(start, end)
        rng = np.random.default_rng(int(code[1:]))
        price = abs(10 + np.cumsum(rng.normal(0, 0.15, len(dates)))) + 1
        return pd.DataFrame({
            "date": dates, "open": price, "high": price * 1.02,
            "low": price * 0.98, "close": price,
            "volume": rng.integers(1e5, 1e6, len(dates)),
        })


def test_replay_end_to_end():
    mgr = AShareDataManager(fetcher=TinyFetcher())
    mgr.load(start="2024-01-01", end="2024-05-01")
    feed = ReplayFeed(mgr)
    ranker = FactorRanker(formula=[0])          # 用特征0当因子
    engine = PortfolioEngine(initial_capital=1_000_000.0)

    dates = feed.trade_dates()
    # 从第40天开始模拟（前面留给特征warm-up），逐日推进
    for i in range(40, len(dates) - 1):
        d, nd = dates[i], dates[i + 1]
        feat_slice, valid = feed.slice_until(d)
        picks = ranker.rank(feat_slice, valid, mgr.symbols, top_k=10)
        bar = feed.bar_at(nd)
        engine.step(d, nd, picks, bar)

    assert len(engine.account.nav_history) > 0
    m = compute_metrics(engine.account.nav_history, 1_000_000.0)
    assert m["final_nav"] > 0
    assert "total_return" in m
    # 账户恒等式最终校验
    last_close = feed.bar_at(dates[-1])
    nav = engine.account.cash + sum(
        h.shares * last_close[c]["close"]
        for c, h in engine.account.holdings.items() if c in last_close)
    assert nav > 0
```

- [ ] **Step 2: 运行测试确认失败**

Run: `uv run pytest tests/smoke/test_paper_smoke.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'paper_trading.data_feed'`

- [ ] **Step 3: 实现 data_feed.py（ReplayFeed）**

创建 `paper_trading/data_feed.py`：

```python
"""
paper_trading/data_feed.py — 数据喂法

ReplayFeed: 历史回放，持有已加载 AShareDataManager，按日切片。
LiveFeed:   实盘（Task 12 实现）。
两者共用 PortfolioEngine.step。
"""
from __future__ import annotations

import torch

from paper_trading.config import LIMIT_PCT


class ReplayFeed:
    def __init__(self, manager) -> None:
        self.mgr = manager
        self._feat = manager.feat_tensor          # [N, F, T] 一次算好
        self._dates = manager.trade_dates
        self._date_idx = {d: i for i, d in enumerate(self._dates)}
        self._close = manager.raw_dict["close"]   # [N, T]
        self._open = manager.raw_dict["open"]     # [N, T]
        self._valid = manager.valid_mask          # [N, T]

    def trade_dates(self) -> list[str]:
        return list(self._dates)

    def slice_until(self, date: str) -> tuple[torch.Tensor, torch.Tensor]:
        """返回截至 date（含）的特征张量 [N,F,t+1] 与当日 valid_mask [N]。"""
        t = self._date_idx[date]
        feat_slice = self._feat[:, :, : t + 1]
        valid = self._valid[:, t]
        return feat_slice, valid

    def bar_at(self, date: str) -> dict:
        """返回该日各股票的成交 bar：{code:{open,close,limit_up,limit_down,tradable}}。

        涨跌停判定：当日 open 相对前一交易日 close 涨/跌达 ±LIMIT_PCT（留 0.1% 容差）
        且当日振幅极小（open≈close≈high≈low）视为封板。这里用简化口径：
        open >= prev_close*(1+LIMIT_PCT)*0.999 → 封涨停；对称判跌停。
        """
        t = self._date_idx[date]
        codes = self.mgr.symbols
        bar = {}
        for n, code in enumerate(codes):
            o = float(self._open[n, t])
            c = float(self._close[n, t])
            tradable = bool(self._valid[n, t])
            limit_up = limit_down = False
            if t > 0:
                pc = float(self._close[n, t - 1])
                if pc > 0:
                    if o >= pc * (1 + LIMIT_PCT) * 0.999:
                        limit_up = True
                    elif o <= pc * (1 - LIMIT_PCT) * 1.001:
                        limit_down = True
            bar[code] = {"open": o, "close": c, "limit_up": limit_up,
                         "limit_down": limit_down, "tradable": tradable}
        return bar
```

- [ ] **Step 4: 运行冒烟测试确认通过**

Run: `uv run pytest tests/smoke/test_paper_smoke.py -v`
Expected: PASS

- [ ] **Step 5: 实现 run_paper_replay.py（编排 + 落盘）**

创建 `run_paper_replay.py`：

```python
"""
run_paper_replay.py — A股历史回放 CLI（Web subprocess 调用）

流程：加载策略公式 → 加载A股数据 → 逐日回放 → 落盘净值/流水/指标。
产物：paper_trading/output/{equity.json, trades.json, metrics.json}

用法:
    python run_paper_replay.py --strategy strategies/best_ashare_universe.json \
        --start 2023-01-01 --end 2026-06-30 --sim-start 2026-06-01
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from data_pipeline.ashare_manager import AShareDataManager
from model_core.vocab import FORMULA_VOCAB
from paper_trading.config import INITIAL_CAPITAL, TOP_K
from paper_trading.data_feed import ReplayFeed
from paper_trading.factor_ranker import FactorRanker
from paper_trading.metrics import compute_metrics
from paper_trading.portfolio_engine import PortfolioEngine

OUT_DIR = Path("paper_trading") / "output"


def run_replay(strategy_file: str, start: str, end: str,
               sim_start: str, warmup: int = 60) -> dict:
    data = json.loads(Path(strategy_file).read_text(encoding="utf-8"))
    # vocab 校验：不匹配抛 VocabVersionMismatchError（避免旧公式对不上特征维）
    saved_ver = data.get("vocab_version")
    if saved_ver:
        FORMULA_VOCAB.verify(saved_ver)     # 不匹配则抛异常
    formula = data["formula"]

    mgr = AShareDataManager()
    mgr.load(start=start, end=end)
    feed = ReplayFeed(mgr)
    ranker = FactorRanker(formula=formula)
    engine = PortfolioEngine(initial_capital=INITIAL_CAPITAL)

    dates = feed.trade_dates()
    if sim_start in dates:
        start_i = dates.index(sim_start)
    else:
        start_i = max(warmup, 0)
    start_i = max(start_i, warmup)

    for i in range(start_i, len(dates) - 1):
        d, nd = dates[i], dates[i + 1]
        feat_slice, valid = feed.slice_until(d)
        picks = ranker.rank(feat_slice, valid, mgr.symbols, top_k=TOP_K)
        engine.step(d, nd, picks, feed.bar_at(nd))
        print(f"[{nd}] nav={engine.account.nav_history[-1][1]:,.0f} "
              f"持仓={len(engine.account.holdings)}")

    metrics = compute_metrics(engine.account.nav_history, INITIAL_CAPITAL)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "equity.json").write_text(
        json.dumps(engine.account.nav_history, ensure_ascii=False), encoding="utf-8")
    (OUT_DIR / "trades.json").write_text(
        json.dumps(engine.account.trades, ensure_ascii=False), encoding="utf-8")
    (OUT_DIR / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[回放完成] 总收益={metrics['total_return']:.2%} "
          f"最大回撤={metrics['max_drawdown']:.2%} 夏普={metrics['sharpe']:.2f}")
    return metrics


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", default="strategies/best_ashare_universe.json")
    ap.add_argument("--start", default="2023-01-01")
    ap.add_argument("--end", default="2026-06-30")
    ap.add_argument("--sim-start", default="2026-06-01")
    args = ap.parse_args()
    run_replay(args.strategy, args.start, args.end, args.sim_start)
```

注：`FORMULA_VOCAB.verify(artifact_version)` 已存在（`model_core/vocab.py:88`），版本不匹配时抛 `VocabVersionMismatchError`。import 用 `from model_core.vocab import FORMULA_VOCAB`。

- [ ] **Step 6: 加沪深300基准对比（spec §6）**

在 `data_pipeline/ashare_fetcher.py` 的 `AShareFetcher` 内追加基准指数方法：

```python
    def benchmark(self, start: str, end: str, symbol: str = "sh000300") -> dict[str, float]:
        """沪深300指数日线收盘，返回 {date: close}，用于超额收益对比。"""
        raw = _retry(ak.stock_zh_index_daily, symbol=symbol)
        df = raw.copy()
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
        df = df[(df["date"] >= start) & (df["date"] <= end)]
        return dict(zip(df["date"], df["close"].astype(float)))
```

在 `run_paper_replay.py` 的 `run_replay` 里，落盘 metrics 前计算基准收益并并入：

```python
    # 基准对比：同期沪深300收益 + 超额 alpha
    from data_pipeline.ashare_fetcher import AShareFetcher
    bench = AShareFetcher().benchmark(dates[start_i], dates[-1])
    bench_vals = [bench[d] for d in dates[start_i:] if d in bench]
    if len(bench_vals) >= 2:
        bench_ret = bench_vals[-1] / bench_vals[0] - 1.0
        metrics["benchmark_return"] = bench_ret
        metrics["excess_return"] = metrics["total_return"] - bench_ret
    else:
        metrics["benchmark_return"] = None
        metrics["excess_return"] = None
```

前端 `paper.html` 的 `renderCards` 追加两张卡（基准收益、超额 alpha），仿现有卡片写法。

- [ ] **Step 7: 重跑冒烟测试确认无回归**

Run: `uv run pytest tests/smoke/test_paper_smoke.py -v`
Expected: PASS（基准仅在 run_paper_replay 中调用，冒烟测试不受影响）。

- [ ] **Step 8: Commit**

```bash
git add paper_trading/data_feed.py run_paper_replay.py data_pipeline/ashare_fetcher.py tests/smoke/test_paper_smoke.py
git commit -m "feat(paper): ReplayFeed、历史回放端到端与沪深300基准对比"
```

---

## Task 11: Web 层 — paper_manager + 路由 + 页面

**Files:**
- Create: `web/paper_manager.py`（仿 `web/backtest_manager.py` 的 subprocess + 日志轮询）
- Create: `web/static/paper.html`
- Modify: `web/app.py`（import + 注册 `/api/paper/*` 路由）
- Test: `tests/unit/test_paper_manager.py`

- [ ] **Step 1: 写 manager 状态机测试**

创建 `tests/unit/test_paper_manager.py`：

```python
from web.paper_manager import PaperManager, JobState


def test_status_idle_initially():
    m = PaperManager()
    st = m.status()
    assert st["active"] is False
    assert st["job"] is None


def test_read_outputs_missing_returns_empty():
    m = PaperManager()
    # 无输出文件时返回空结构，不抛异常
    assert m.equity() == []
    assert m.trades() == []
    assert m.metrics() == {}
```

- [ ] **Step 2: 运行测试确认失败**

Run: `uv run pytest tests/unit/test_paper_manager.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'web.paper_manager'`

- [ ] **Step 3: 实现 paper_manager.py**

创建 `web/paper_manager.py`：

```python
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
```

- [ ] **Step 4: 运行测试确认通过**

Run: `uv run pytest tests/unit/test_paper_manager.py -v`
Expected: PASS

- [ ] **Step 5: app.py 注册路由**

在 `web/app.py` 的 import 段（`from web.backtest_manager import backtest_manager` 附近）加：

```python
from web.paper_manager import paper_manager
```

在 `web/app.py` 末尾路由段（`@app.get("/")` 之前）加：

```python
class PaperReplayReq(BaseModel):
    strategy_file: str = "strategies/best_ashare_universe.json"
    start: str = "2023-01-01"
    end: str = "2026-06-30"
    sim_start: str = "2026-06-01"


@app.post("/api/paper/replay/start")
def paper_replay_start(req: PaperReplayReq):
    try:
        job = paper_manager.start(req.strategy_file, req.start, req.end, req.sim_start)
        return {"ok": True, "job": job.to_dict()}
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))


@app.post("/api/paper/replay/stop")
def paper_replay_stop():
    return {"ok": paper_manager.stop()}


@app.get("/api/paper/status")
def paper_status():
    return paper_manager.status()


@app.get("/api/paper/equity")
def paper_equity():
    return {"equity": paper_manager.equity()}


@app.get("/api/paper/trades")
def paper_trades():
    return {"trades": paper_manager.trades()}


@app.get("/api/paper/metrics")
def paper_metrics():
    return {"metrics": paper_manager.metrics()}
```

- [ ] **Step 6: 验证路由注册**

Run: `uv run python -c "from web.app import app; print([r.path for r in app.routes if 'paper' in r.path])"`
Expected: 打印 6 条 `/api/paper/*` 路由。

- [ ] **Step 7: 创建前端页面**

创建 `web/static/paper.html`（最小可用：调 API 画净值曲线 + 指标卡 + 持仓/流水表）：

```html
<!DOCTYPE html>
<html lang="zh">
<head>
  <meta charset="utf-8">
  <title>A股虚拟炒股</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
  <style>
    body { font-family: system-ui, sans-serif; margin: 24px; }
    .cards { display: flex; gap: 16px; margin: 16px 0; }
    .card { border: 1px solid #ddd; border-radius: 8px; padding: 12px 20px; }
    .card b { font-size: 22px; }
    table { border-collapse: collapse; width: 100%; margin-top: 12px; }
    th, td { border: 1px solid #eee; padding: 6px 10px; text-align: right; }
    button { padding: 8px 16px; }
  </style>
</head>
<body>
  <h2>A股虚拟炒股平台（100万起始）</h2>
  <button onclick="startReplay()">启动历史回放</button>
  <span id="state"></span>
  <div class="cards" id="cards"></div>
  <canvas id="equity" height="90"></canvas>
  <h3>交易流水</h3>
  <table id="trades"><thead><tr>
    <th>日期</th><th>代码</th><th>方向</th><th>价格</th><th>股数</th><th>费用</th>
  </tr></thead><tbody></tbody></table>
  <script>
    let chart;
    async function startReplay() {
      await fetch('/api/paper/replay/start', {method:'POST',
        headers:{'Content-Type':'application/json'}, body:'{}'});
      poll();
    }
    async function poll() {
      const st = await (await fetch('/api/paper/status')).json();
      document.getElementById('state').textContent =
        st.active ? ' 运行中…' : (st.job ? ' 状态: '+st.job.state : '');
      await refresh();
      if (st.active) setTimeout(poll, 2000);
    }
    async function refresh() {
      const m = (await (await fetch('/api/paper/metrics')).json()).metrics || {};
      const eq = (await (await fetch('/api/paper/equity')).json()).equity || [];
      const tr = (await (await fetch('/api/paper/trades')).json()).trades || [];
      renderCards(m); renderEquity(eq); renderTrades(tr);
    }
    function renderCards(m) {
      const pct = x => (x==null?'-':(x*100).toFixed(2)+'%');
      document.getElementById('cards').innerHTML = `
        <div class="card">总收益<br><b>${pct(m.total_return)}</b></div>
        <div class="card">最大回撤<br><b>${pct(m.max_drawdown)}</b></div>
        <div class="card">年化<br><b>${pct(m.annual_return)}</b></div>
        <div class="card">夏普<br><b>${m.sharpe?m.sharpe.toFixed(2):'-'}</b></div>
        <div class="card">期末净值<br><b>${m.final_nav?Math.round(m.final_nav).toLocaleString():'-'}</b></div>`;
    }
    function renderEquity(eq) {
      const labels = eq.map(x=>x[0]), data = eq.map(x=>x[1]);
      if (chart) chart.destroy();
      chart = new Chart(document.getElementById('equity'), {
        type:'line', data:{labels, datasets:[{label:'净值', data, borderColor:'#2b7', pointRadius:0}]},
        options:{scales:{y:{beginAtZero:false}}}});
    }
    function renderTrades(tr) {
      document.querySelector('#trades tbody').innerHTML = tr.slice(-100).map(t=>
        `<tr><td>${t.date}</td><td>${t.code}</td><td>${t.side}</td>
         <td>${t.price.toFixed(2)}</td><td>${t.shares}</td><td>${t.cost.toFixed(2)}</td></tr>`).join('');
    }
    refresh();
  </script>
</body>
</html>
```

注：`web/app.py:1108` 已挂载 `app.mount("/static", StaticFiles(directory=STATIC_DIR))`，故页面访问 URL 为 `http://127.0.0.1:8765/static/paper.html`。

- [ ] **Step 8: Commit**

```bash
git add web/paper_manager.py web/static/paper.html web/app.py tests/unit/test_paper_manager.py
git commit -m "feat(paper): Web 层 paper_manager + 路由 + 页面"
```

---

## Task 12: LiveFeed — 实盘模式（状态持久化续跑）

**Files:**
- Modify: `paper_trading/data_feed.py`（追加 `LiveFeed`）
- Create: `paper_trading/state_store.py`（账户状态 JSON 持久化）
- Create: `run_paper_live.py`（每交易日推进一步）
- Test: `tests/unit/test_state_store.py`

**LiveFeed 与 ReplayFeed 的唯一差异：** 每次推进前拉「当日」增量数据追加到缓存，再重建管理器切片。共用 `PortfolioEngine.step`。账户状态落盘，进程重启可续。

- [ ] **Step 1: 写状态持久化测试**

创建 `tests/unit/test_state_store.py`：

```python
from paper_trading.account import Account, Holding
from paper_trading.state_store import save_account, load_account


def test_roundtrip(tmp_path):
    acc = Account(cash=500_000.0)
    acc.holdings["600000"] = Holding("600000", 1000, 10.0, "2026-07-15", "2026-07-15")
    acc.nav_history.append(("2026-07-15", 510_000.0))
    acc.trades.append({"date": "2026-07-15", "code": "600000", "side": "BUY",
                       "price": 10.0, "shares": 1000, "cost": 5.0})
    path = tmp_path / "account.json"
    save_account(acc, str(path))
    loaded = load_account(str(path))
    assert loaded.cash == 500_000.0
    assert loaded.holdings["600000"].shares == 1000
    assert loaded.nav_history[-1] == ("2026-07-15", 510_000.0)
    assert loaded.trades[0]["code"] == "600000"


def test_load_missing_returns_fresh(tmp_path):
    loaded = load_account(str(tmp_path / "none.json"), initial_capital=1_000_000.0)
    assert loaded.cash == 1_000_000.0
    assert loaded.holdings == {}
```

- [ ] **Step 2: 运行测试确认失败**

Run: `uv run pytest tests/unit/test_state_store.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'paper_trading.state_store'`

- [ ] **Step 3: 实现 state_store.py**

创建 `paper_trading/state_store.py`：

```python
"""paper_trading/state_store.py — 账户状态 JSON 持久化（实盘续跑用）。"""
from __future__ import annotations

import json
from pathlib import Path

from paper_trading.account import Account, Holding


def save_account(acc: Account, path: str) -> None:
    data = {
        "cash": acc.cash,
        "holdings": {c: vars(h) for c, h in acc.holdings.items()},
        "nav_history": [list(x) for x in acc.nav_history],
        "trades": acc.trades,
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2),
                          encoding="utf-8")


def load_account(path: str, initial_capital: float = 1_000_000.0) -> Account:
    p = Path(path)
    if not p.exists():
        return Account(cash=float(initial_capital))
    data = json.loads(p.read_text(encoding="utf-8"))
    acc = Account(cash=float(data["cash"]))
    for c, hd in data.get("holdings", {}).items():
        acc.holdings[c] = Holding(**hd)
    acc.nav_history = [tuple(x) for x in data.get("nav_history", [])]
    acc.trades = data.get("trades", [])
    return acc
```

- [ ] **Step 4: 运行测试确认通过**

Run: `uv run pytest tests/unit/test_state_store.py -v`
Expected: PASS（两条用例）

- [ ] **Step 5: 追加 LiveFeed 到 data_feed.py**

在 `paper_trading/data_feed.py` 末尾追加。`LiveFeed` 拉当日增量后复用与 `ReplayFeed` 相同的切片/bar 逻辑（继承之）：

```python
class LiveFeed(ReplayFeed):
    """实盘：每次 advance() 拉当日增量数据，重建底层管理器后复用切片逻辑。

    与 ReplayFeed 唯一差异是数据来源随时间增长；切片/bar 计算完全复用父类。
    """
    def __init__(self, manager, start: str) -> None:
        super().__init__(manager)
        self._start = start

    def advance(self, today: str, end: str) -> None:
        """拉取截至 today 的最新数据并重建内部张量（含 today 当日已收盘 K线）。"""
        self.mgr.load(start=self._start, end=end)
        # 重建父类缓存的张量视图
        self._feat = self.mgr.feat_tensor
        self._dates = self.mgr.trade_dates
        self._date_idx = {d: i for i, d in enumerate(self._dates)}
        self._close = self.mgr.raw_dict["close"]
        self._open = self.mgr.raw_dict["open"]
        self._valid = self.mgr.valid_mask
```

- [ ] **Step 6: 实现 run_paper_live.py**

创建 `run_paper_live.py`：

```python
"""
run_paper_live.py — A股实盘模拟：推进一个交易日

每个交易日收盘后运行一次（手动或定时）：拉当日数据 → 决策 → 成交 → 落盘账户。
状态存 paper_trading/state/account.json，进程重启可续跑。

用法（每交易日收盘后）:
    python run_paper_live.py --today 2026-07-16
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from data_pipeline.ashare_manager import AShareDataManager
from model_core.vocab import FORMULA_VOCAB
from paper_trading.config import INITIAL_CAPITAL, TOP_K
from paper_trading.data_feed import LiveFeed
from paper_trading.factor_ranker import FactorRanker
from paper_trading.portfolio_engine import PortfolioEngine
from paper_trading.state_store import load_account, save_account

STATE_PATH = "paper_trading/state/account.json"


def run_live_step(today: str, strategy_file: str, start: str = "2023-01-01") -> None:
    data = json.loads(Path(strategy_file).read_text(encoding="utf-8"))
    if data.get("vocab_version"):
        FORMULA_VOCAB.verify(data["vocab_version"])
    formula = data["formula"]

    mgr = AShareDataManager()
    feed = LiveFeed(mgr, start=start)
    feed.advance(today, end=today)
    dates = feed.trade_dates()
    if today not in dates:
        print(f"[{today}] 非交易日，跳过")
        return

    engine = PortfolioEngine(initial_capital=INITIAL_CAPITAL)
    engine.account = load_account(STATE_PATH, INITIAL_CAPITAL)

    # 用「上一交易日」的因子选股，在 today 开盘成交（T日决策→T+1成交口径）。
    t = dates.index(today)
    if t == 0:
        print("数据不足，无法决策")
        return
    prev = dates[t - 1]
    feat_slice, valid = feed.slice_until(prev)
    picks = ranker_picks(formula, feat_slice, valid, mgr.symbols)
    engine.step(prev, today, picks, feed.bar_at(today))

    save_account(engine.account, STATE_PATH)
    nav = engine.account.nav_history[-1][1] if engine.account.nav_history else INITIAL_CAPITAL
    print(f"[{today}] nav={nav:,.0f} 持仓={len(engine.account.holdings)} 已存档")


def ranker_picks(formula, feat_slice, valid, codes):
    return FactorRanker(formula=formula).rank(feat_slice, valid, codes, top_k=TOP_K)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--today", required=True)
    ap.add_argument("--strategy", default="strategies/best_ashare_universe.json")
    args = ap.parse_args()
    run_live_step(args.today, args.strategy)
```

- [ ] **Step 7: 运行全部测试回归**

Run: `uv run pytest tests/ -k "ashare or paper or account or portfolio or metrics or factor or state" -v`
Expected: 全部 PASS。

- [ ] **Step 8: Commit**

```bash
git add paper_trading/data_feed.py paper_trading/state_store.py run_paper_live.py tests/unit/test_state_store.py
git commit -m "feat(paper): LiveFeed 实盘模式与状态持久化续跑"
```

---

## 最终验证（全部任务完成后）

- [ ] **端到端真实数据验证（需联网）**

```bash
# 1. 训练截面因子（真实全800，约几分钟~数十分钟视步数）
uv run python train_ashare.py --start 2023-01-01 --end 2026-05-31 --steps 300
# 2. 历史回放 2026-06 整月
uv run python run_paper_replay.py --start 2023-01-01 --end 2026-06-30 --sim-start 2026-06-01
```
人工核对：`paper_trading/output/metrics.json` 的总收益/回撤合理；净值曲线与同期沪深300对比；流水无 T+1 违规、无封板成交。

- [ ] **全量测试**

Run: `uv run pytest tests/ -v`
Expected: 全绿（含原有测试无回归）。

