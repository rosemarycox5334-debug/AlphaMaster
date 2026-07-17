# AlphaMaster A股虚拟炒股平台 — 设计文档

**日期**：2026-07-16
**状态**：待实现
**作者**：Claude Code + yinfei

## 1. 目标

在现有 AlphaMaster（RL 因子挖掘 + MT5 外汇/加密实盘）之上，搭建一个面向 A 股市场的**虚拟炒股平台**：

- 起始资金 100 万元人民币现金账户。
- 系统自主完成：选股（截面因子打分）、训练（RL 挖掘因子公式）、买入、卖出。
- 持续运行一个月，输出最终收益、净值曲线、交易流水与绩效指标。
- 一套决策引擎，支持两种数据喂法：**历史回放**（快速，几分钟跑完一个月）与**实盘模拟**（每交易日推进一步，真实等 30 个自然日）。

## 2. 核心决策（已与用户确认）

| 维度 | 决策 | 理由 |
|------|------|------|
| 技术路线 | 方案 A：唤醒截面因子挖掘 | 最大化复用 RL 内核，截面算子本为此场景而生 |
| 股票池 | 沪深300 + 中证500 = 全 800（当前成份） | 流动性好、数据干净；当前成份口径实现简单（接受幸存者偏差） |
| 数据源 | akshare | 免费免 token，覆盖日线+成份股+基本面 |
| 训练节奏 | 训练一次冻结，再逐日交易 | 简单、无前瞻偏差 |
| 持仓 | Top10 等权 | 清晰可控，分散适中 |
| 调仓频率 | 每日调仓 | 信号最及时（真实成本已计入） |
| 交易成本 | 真实成本（佣金+印花税） | 贴近散户真实收益 |
| 风控 | 纯因子驱动，无止损/回撤控制 | 最能反映因子真实能力 |
| 首跑 | 历史回放 + 实盘两种都实现，先做回放验证 | 回放先验证系统正确性 |

## 3. 关键约束与本质问题

### 3.1 A 股规则与外汇/加密的本质差异

现有系统为 24h 连续交易、可双向、无涨跌停的外汇/加密设计。A 股必须处理：

1. **T+1 交收**：当日买入的股票次一交易日才能卖出。买入时记录 `sellable_date`，卖出前校验。
2. **只能做多**：无融券的普通账户不能做空。因此**不复用** `signal.py` 的 `tanh(factor)→(-1,+1)` 连续仓位，改为「截面排序取 Top10 等权做多」。
3. **涨跌停**：主板 ±10%、ST ±5%、科创/创业板 ±20%。**开盘即封板**（涨停无卖单/跌停无买单）的股票当日不能成交，该笔跳过。
4. **停牌/未上市**：800 只股票上市日期不一、时有停牌。现有多品种对齐用**时间戳交集**，一只次新股就会把整个时间轴砍没——**必须换成固定交易日历 + 停牌 mask**。

### 3.2 训练目标从「时序」转向「截面」

现有单品种模式的 reward 是时序 Sortino/Calmar，且 N=1 时截面算子（CS_RANK/CS_SCALE/CS_NEUTRALIZE）与截面特征（REL_*/CS_*）全部退化失效。A 股选股本质是**截面排序问题**：同一天对 800 只股票打分，选最强的。因此训练目标以**截面 Rank-IC** 为主。

## 4. 架构

### 4.1 数据流总览

```
akshare
  │  拉全800日线 + 交易日历 + 成份股名单
  ▼
data_pipeline/ashare_manager.py  (AShareDataManager)
  │  产出与 MT5DataManager 相同接口:
  │    raw_dict{field:[N,T]} / feat_tensor[N,F,T] / target_ret[N,T] / symbols
  │  新增: valid_mask[N,T] (停牌/未上市剔除)
  ▼
model_core (复用, 多品种截面模式 target_symbol=None)
  │  AlphaEngine.train() + MT5Backtest(REWARD_MODE="ashare")
  │  输出: strategies/best_ashare_universe.json (可解释 token 公式)
  ▼
paper_trading/ (全新核心模块)
  │  factor_ranker.py  每日截面因子值 → Top10 等权目标
  │  portfolio_engine.py  100万账户账本, T+1/涨跌停/成本, 逐日 step(date)
  │  data_feed.py  历史回放 / 实盘 两种数据喂法, 共用 step
  ▼
web/paper_manager.py + /api/paper/* + 前端页面
  │  净值曲线 / 持仓 / 流水 / 绩效指标
```

### 4.2 复用 / 新建 / 小改 清单

**原样复用**：`model_core/*`（AlphaGPT、engine、vm、vocab、registry、ops、features、evaluator）、`web/app.py` FastAPI 框架与 manager 模式。

**全新模块**：
- `data_pipeline/ashare_manager.py` — A 股数据管理器（含交易日历对齐、停牌 mask）
- `data_pipeline/ashare_fetcher.py` — akshare 拉取 + parquet 缓存
- `paper_trading/factor_ranker.py` — 因子 → Top10 选股
- `paper_trading/portfolio_engine.py` — T+1 组合账户引擎
- `paper_trading/data_feed.py` — 历史回放 / 实盘数据喂法
- `web/paper_manager.py` + `web/static/paper.html` — Web 展示

**小改**：`model_core/backtest.py` 增加 `REWARD_MODE="ashare"` 分支（截面 Rank-IC 主权重）；`config.py` 增加 A 股相关配置段。

## 5. 组件详细设计

### 5.1 AShareDataManager（数据层）

**职责**：把 800 只 A 股日线整理成模型可吃的 `[N, T]` 张量，处理停牌与未上市。

**接口契约**（与 `MT5DataManager` 对齐，下游零改动）：
```python
class AShareDataManager:
    def load(self, codes: list[str] | None = None,
             start: str = ..., end: str = ...) -> None: ...
    @property
    def raw_dict(self) -> dict[str, Tensor]:   # {open,high,low,close,volume,time}, 各 [N,T]
    @property
    def feat_tensor(self) -> Tensor:            # [N, F, T], 复用 MT5FeatureEngineer
    @property
    def target_ret(self) -> Tensor:             # [N, T]
    @property
    def valid_mask(self) -> Tensor:             # [N, T] bool, 新增
    @property
    def symbols(self) -> list[str]:             # 800 只股票代码
    @property
    def trade_dates(self) -> list[str]:         # T 个交易日 (YYYY-MM-DD)
```

**关键实现**：
- **固定交易日历**：用 akshare `tool_trade_date_hist_sina` 取交易日，构建统一时间轴 `[T]`。所有股票 reindex 到该日历（**不用交集**）。
- **停牌/未上市**：某股某日无成交（volume=0 或缺失）→ `valid_mask[n,t]=False`；价格用前值 forward-fill（供特征计算连续），但排序时剔除。未上市日同样 mask。
- **缓存**：`ashare_cache/{code}.parquet` 单只存储，增量更新（记录 last_date，只拉新增）。首次全量约几分钟。
- **target_ret**：复用现有定义 `log(open[t+2]/open[t+1])`（T 日决策、T+1 开盘成交口径下，预测的是 T+1→T+2 收益，无前瞻）。

### 5.2 因子挖掘层（model_core，复用）

- 实例化：`AlphaEngine(data_manager=ashare_mgr, target_symbol=None)` → 触发多品种截面模式。
- **REWARD_MODE="ashare"**（backtest.py 新增分支）：
  - 主权重：截面 Rank-IC 均值 + IC 稳定性（IC_IR）。复用 `_ts_ic_stability` / `_compute_ic`，但改为按 `valid_mask` 逐日截面计算 IC。
  - 次权重：换手惩罚（`_turnover_penalty`，抑制每日全换）、截面覆盖度（有效股票数不足时降权）。
  - 移除/弱化：外汇特有的 `beta_neutral_penalty`、感染链惩罚（截面选股中恒正算子不再是问题，因为最终是排序取 Top，不是绝对多空）。
- 训练产物：`strategies/best_ashare_universe.json`，含 `vocab_version` 校验。
- **vocab 影响**：A 股启用全部特征还是剪枝，由是否存在 `active_features.json` 决定；本设计默认全特征（不剪枝），后续可用 `prune_features.py` 优化。

### 5.3 FactorRanker（选股信号，新建）

**职责**：把训练好的因子公式在某个交易日的截面因子值 → Top10 等权目标持仓。

```python
class FactorRanker:
    def __init__(self, formula: list[int], vocab_version: str): ...
    def rank(self, feat_slice: Tensor, valid_mask_slice: Tensor,
             codes: list[str], top_k: int = 10) -> list[str]:
        """对当日有效股票用 StackVM 算因子值, 截面排序取 Top-K 代码。"""
```

- 用 `StackVM` 执行公式得到 `[N]` 因子值（当日截面）。
- `valid_mask=False`（停牌/未上市/当日涨跌停封板）的股票排除出候选。
- 按因子值降序取前 10，等权（各 10%）。
- **不用 tanh**：这是纯多头排序信号，与外汇连续仓位是两套口径。

### 5.4 PortfolioEngine（组合账户引擎，全新核心）

**职责**：100 万现金账户账本，逐日推进，处理 T+1、涨跌停、成本，产出净值与流水。

**账户状态**：
```python
@dataclass
class Holding:
    code: str
    shares: int            # 持股数（100 股整数倍）
    cost_price: float      # 买入均价
    buy_date: str          # 买入日
    sellable_date: str     # T+1: buy_date 的次一交易日

@dataclass
class Account:
    cash: float                        # 可用现金
    holdings: dict[str, Holding]       # 当前持仓
    nav_history: list[tuple[str,float]]  # (date, 总净值)
    trades: list[dict]                 # 交易流水
```

**核心方法 `step(date, target_codes)`**（T 日收盘后决策，T+1 开盘成交）：
1. **卖出**：不在 `target_codes` 中、且 `sellable_date <= date` 的持仓 → 按当日开盘价卖出。封跌停（开盘=跌停价且无买盘）跳过，留到下个交易日重试。扣佣金+印花税。
2. **买入**：`target_codes` 中尚未持有的 → 用可用现金等权买入（每只目标市值 = 总资产/10）。封涨停跳过。买入数量向下取整到 100 股整数倍。扣佣金。
3. **估值**：所有持仓按当日收盘价 mark-to-market，记录当日总净值 = cash + Σ(shares × close)。
4. 记录流水（日期、代码、方向、价格、数量、费用）。

**交易成本**（`config.py` 配置）：
- 买入佣金：`max(成交额 × 0.00025, 5)` 元
- 卖出佣金：`max(成交额 × 0.00025, 5)` 元
- 卖出印花税：`成交额 × 0.001`
- 过户费忽略（可选，沪市万0.1，量级极小）

**成交假设**（无前瞻）：T 日收盘后用 T 日及之前数据算因子/选股 → T+1 开盘价成交。回放与实盘一致。

### 5.5 DataFeed（两种数据喂法，新建）

一个抽象接口，两种实现，共用 `PortfolioEngine.step`：

```python
class DataFeed(Protocol):
    def trade_dates(self) -> list[str]: ...
    def slice_until(self, date: str) -> tuple[Tensor, Tensor, list[str]]:
        """返回截至 date（含）的 feat_tensor、valid_mask、codes，供 FactorRanker。"""
    def bar_at(self, date: str, field: str) -> dict[str, float]:
        """返回该日各股票的 open/close 等，供 PortfolioEngine 成交/估值。"""

class ReplayFeed(DataFeed):   # 历史回放: 全部数据已在缓存, 按日切片
class LiveFeed(DataFeed):     # 实盘: 每交易日收盘后拉当日, 追加缓存
```

- **回放**：一次性加载缓存 → 循环 `for date in trade_dates[sim_start:sim_end]: engine.step(...)`，几分钟跑完。
- **实盘**：每个交易日收盘后（定时或手动触发）拉当日数据 → 追加 → 推进一步。状态持久化到 `paper_trading/state/account.json`，进程重启可续跑。

### 5.6 Web 展示（复用框架）

- `web/paper_manager.py`：仿 `backtest_manager.py`，管理回放任务（subprocess/线程）与实盘状态读取。
- 新增路由 `/api/paper/*`：
  - `POST /api/paper/replay/start` — 启动历史回放（参数：训练区间、模拟区间）
  - `GET  /api/paper/status` — 当前进度/状态
  - `GET  /api/paper/equity` — 净值曲线数据
  - `GET  /api/paper/holdings` — 当前持仓
  - `GET  /api/paper/trades` — 交易流水
  - `GET  /api/paper/metrics` — 绩效指标
- 前端 `web/static/paper.html`：净值曲线图、持仓表、流水表、绩效卡片（总收益/年化/最大回撤/夏普/胜率/换手率）。

## 6. 绩效指标

- **总收益率** = (期末净值 - 100万) / 100万
- **最大回撤**：净值曲线峰谷最大跌幅
- **夏普比率**：日收益均值/标准差 × √244
- **胜率**：盈利交易笔数 / 总平仓笔数
- **换手率**：日均买卖成交额 / 总资产
- **基准对比**：同期沪深300指数收益（超额 alpha）

## 7. 错误处理

- **akshare 限流/超时**：重试 3 次 + 指数退避；缓存优先，网络失败时用已有缓存并告警。
- **停牌股估值**：用最后有效收盘价 mark-to-market（不影响可交易性，仅估值）。
- **现金不足**：买入按可用现金比例缩减，不允许透支（无杠杆）。
- **vocab 不匹配**：加载策略时 `FORMULA_VOCAB.verify()`，A 股 vocab 版本变化则拒绝加载并提示重训。
- **数据缺口**：某交易日全市场无数据（异常）→ 跳过该日并告警。

## 8. 测试策略

- **property 测试**（`tests/property/test_portfolio.py`）：
  - 账户恒等式：任意时刻 `总净值 == cash + Σ(shares × close)`（含费用扣减一致性）。
  - T+1 不可违反：`buy_date == date` 的持仓当日不可卖出。
  - 涨跌停封板不成交：封板日无对应方向成交记录。
  - 现金非负：任意时刻 `cash >= 0`。
- **冒烟测试**（`tests/smoke/test_paper_smoke.py`）：20 股 × 60 交易日小样本，端到端跑通回放，产出净值曲线。
- **数据层单测**：交易日历对齐正确、停牌 mask 正确、缓存增量更新正确。
- **验证运行**：全 800 股历史回放一个月，人工核对净值曲线与流水合理性、与沪深300基准对比。

## 9. 实施顺序（供 writing-plans 参考）

1. 环境：`uv` 装 akshare 等依赖（用 uv 管理 Python 环境）。
2. 数据层：ashare_fetcher + ashare_manager（含缓存、日历、mask）+ 单测。
3. 因子训练：backtest.py 加 ashare reward 分支，跑通全 800 截面训练。
4. 组合引擎：portfolio_engine + factor_ranker + property/smoke 测试。
5. 数据喂法：ReplayFeed 先行（验证系统），LiveFeed 后续。
6. Web：paper_manager + 路由 + 前端页面。
7. 端到端验证：历史回放一个月，核对结果。

## 10. 非目标（YAGNI）

- 不做盘中分钟级交易（仅日线、收盘决策）。
- 不做融资融券/做空/期权。
- 不做时点成份股（接受当前成份的幸存者偏差）。
- 不做组合级止损/回撤风控（纯因子驱动）。
- 不接入真实券商下单（纯虚拟账户）。
- 不做滞动重训（训练一次冻结）。


