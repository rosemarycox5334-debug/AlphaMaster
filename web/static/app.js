const API = "";
let selectedDataFile = null;
let selectedSymbol = null;
let selectedStrategyFile = null;
let selectedStrategySymbol = null;
let chart = null;
let chartSymbol = null;
let pollTimer = null;
let clientErrors = [];
let debugMode = false;
let lastDebugViewContent = "";

// 分页与回测状态
let currentPage = "train";
let btActive = false;
let btBuster = "";      // 图表缓存刷新键（用 job 时间戳）
let btPortfolioSig = ""; // 绩效卡签名：变化时才重建 + 播放数字动画，避免每次轮询重播
let lastEquityData = null; // 最近一次资金曲线数据，供绩效卡 sparkline 复用
let lastTrainingActive = false;
let btLastAlertKey = "";
let lastErrorPopupText = "";
let lastErrorPopupAt = 0;

const $ = (id) => document.getElementById(id);

const CPU_TRAINING_NOTE = `暂无报错

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【为什么用 CPU 训练，不用 GPU？】

你可以把 GPU 想象成一辆超大的货车，CPU 想象成一辆灵活的小电瓶车。

我们这个项目的训练，就像要做很多很多道「小题」：
每道题只算一点点数字，算完马上换下一道。
货车虽然一次能装很多，但每装卸一次都要准备很久才能再出发；
电瓶车一次装的少，但说走就走，一道接一道做，反而更快。

再打个比方：
GPU 像很多厨师一起做大锅饭，适合一次炒一大锅；
我们这个训练更像一道道菜分开炒，而且每道菜份量很小。
大锅饭团队每次开火、洗锅、集合都要时间，小菜一碟反而耽误在「准备」上。

所以具体原因是：
1. 每次要算的数据不多，GPU「启动一次计算」的等待，有时比真正算数还久。
2. 训练是一步接一步、一条公式接一条公式地指挥，GPU 经常闲着等下一道题，没法一直满负荷。
3. 数据还要在 CPU 和 GPU 之间来回搬运，也要花时间。

我们实测过（同样训练 50 步）：GPU 大约 4.5 秒一步，CPU 大约 1.9 秒一步。
这不是显卡坏了，也不是没装驱动，而是这个项目的做题方式，更适合 CPU。

说白了就是这个项目用CPU训练的速度比用GPU训练的速度更快`;

function emptyDebugMessage() {
  return debugMode ? "暂无日志" : CPU_TRAINING_NOTE;
}

function formatApiError(data, status, path) {
  const d = data?.detail;
  let detail = "";
  if (Array.isArray(d)) {
    detail = d.map((x) => x.msg || JSON.stringify(x)).join("; ");
  } else if (typeof d === "string") {
    detail = d;
  } else if (d) {
    detail = JSON.stringify(d);
  }
  if (data?.traceback) {
    detail += `\n\n${data.traceback}`;
  }
  return detail || `HTTP ${status} ${path}`;
}

async function logClientError(message, context = {}) {
  const entry = `[${new Date().toLocaleString()}] ${message}`;
  clientErrors.push(entry);
  if (clientErrors.length > 80) clientErrors = clientErrors.slice(-80);
  renderDebugView();
  const silent = !!context.silent;
  if (!silent) {
    const detail = context.detail ? `${message}\n\n${context.detail}` : message;
    showErrorPopup("出错了", detail);
  }
  try {
    await fetch(API + "/api/debug/client-log", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ level: "error", message, context }),
    });
  } catch (_) {
    /* server may be down */
  }
}

function showErrorPopup(title, detail) {
  const modal = $("errorModal");
  const titleEl = $("errorModalTitle");
  const detailEl = $("errorModalDetail");
  if (!modal || !detailEl) {
    window.alert(`${title}\n\n${detail}`);
    return;
  }
  const text = String(detail || "").trim() || "未知错误";
  const now = Date.now();
  if (text === lastErrorPopupText && now - lastErrorPopupAt < 2500) return;
  lastErrorPopupText = text;
  lastErrorPopupAt = now;
  if (titleEl) titleEl.textContent = title || "出错了";
  detailEl.textContent = text;
  modal.hidden = false;
}

function closeErrorPopup() {
  const modal = $("errorModal");
  if (modal) modal.hidden = true;
}

async function copyErrorPopupDetail() {
  const text = $("errorModalDetail")?.textContent || "";
  if (!text) return;
  try {
    await navigator.clipboard.writeText(text);
  } catch (_) {
    const ta = document.createElement("textarea");
    ta.value = text;
    document.body.appendChild(ta);
    ta.select();
    document.execCommand("copy");
    ta.remove();
  }
}

function isViewAtBottom(el, threshold = 40) {
  return el.scrollHeight - el.scrollTop - el.clientHeight < threshold;
}

function renderDebugView(serverLines = [], errorLines = []) {
  const parts = [];
  if (clientErrors.length) {
    parts.push("=== 前端报错 ===", ...clientErrors);
  }
  if (errorLines.length) {
    parts.push("\n=== 服务端错误日志 (logs/web_errors.log) ===", ...errorLines);
  }
  if (debugMode && serverLines.length) {
    parts.push("\n=== 服务端运行日志 (logs/web_server.log) ===", ...serverLines);
  }
  const el = $("debugView");
  const atBottom = isViewAtBottom(el);
  const next = parts.length ? parts.join("\n") : emptyDebugMessage();
  const changed = next !== lastDebugViewContent;
  el.textContent = next;
  if (changed && atBottom && lastDebugViewContent) {
    el.scrollTop = el.scrollHeight;
  }
  lastDebugViewContent = next;
}

async function setDebugMode(enabled) {
  debugMode = !!enabled;
  $("debugModeCheck").checked = debugMode;
  try {
    await fetchJSON("/api/settings", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ debug_mode: debugMode }),
    });
  } catch (e) {
    await logClientError("切换调试模式失败: " + e.message);
  }
  if (!debugMode) {
    renderDebugView([], []);
  } else {
    await refreshDebugLogs();
  }
}

async function refreshDebugLogs() {
  try {
    const data = await fetch(API + "/api/debug/logs?lines=120").then(async (res) => {
      const json = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(formatApiError(json, res.status, "/api/debug/logs"));
      return json;
    });
    $("debugLogPaths").textContent = `本地: ${data.error_log}`;
    renderDebugView(data.server_tail || [], data.error_tail || []);
  } catch (e) {
    renderDebugView();
  }
}

async function fetchJSON(path, opts = {}) {
  const silent = !!opts.silent;
  const fetchOpts = { ...opts };
  delete fetchOpts.silent;
  let res;
  try {
    res = await fetch(API + path, fetchOpts);
  } catch (e) {
    const msg = `网络错误 ${path}: ${e.message}`;
    await logClientError(msg, { path, silent });
    throw new Error(msg);
  }
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    const msg = formatApiError(data, res.status, path);
    await logClientError(`${path} -> ${msg}`, { path, status: res.status, silent });
    if (!silent) await refreshDebugLogs();
    throw new Error(msg);
  }
  return data;
}

function formatScore(v) {
  if (v == null || Number.isNaN(v)) return "—";
  return Number(v).toFixed(4);
}

function renderDataFileCard(info) {
  const card = $("dataFileCard");
  const startBtn = $("startBtn");

  if (!info || !info.data_file) {
    card.className = "data-file-card";
    card.innerHTML = '<div class="data-file-empty">尚未选择数据文件</div>';
    selectedDataFile = null;
    selectedSymbol = null;
    startBtn.disabled = true;
    if ($("retrainBtn")) $("retrainBtn").disabled = true;
    if ($("exportBtn")) $("exportBtn").disabled = true;
    if ($("exportTrainingBtn")) $("exportTrainingBtn").disabled = true;
    if ($("importTrainingBtn")) $("importTrainingBtn").disabled = true;
    return;
  }

  selectedDataFile = info.data_file;
  selectedSymbol = info.symbol || null;

  if (info.valid === false) {
    card.className = "data-file-card invalid";
    card.innerHTML = `
      <div class="data-file-error">${info.message || "文件无效"}</div>
      <div class="data-file-path">${info.data_file}</div>
    `;
    startBtn.disabled = true;
    if ($("retrainBtn")) $("retrainBtn").disabled = true;
    if ($("exportTrainingBtn")) $("exportTrainingBtn").disabled = true;
    if ($("importTrainingBtn")) $("importTrainingBtn").disabled = true;
    return;
  }

  card.className = "data-file-card valid";
  const yearsText = info.years_h1 != null ? `${info.years_h1} 年` : "—";
  card.innerHTML = `
    <div class="data-file-row">
      <div class="item"><span class="label">品种</span><span class="value sym">${info.symbol}</span></div>
      <div class="item"><span class="label">周期</span><span class="value">${info.timeframe}</span></div>
      <div class="item"><span class="label">K线</span><span class="value">${info.bars?.toLocaleString()}</span></div>
      <div class="item"><span class="label">数据年限</span><span class="value">${yearsText}</span></div>
      <div class="item"><span class="label">进度</span><span class="value" id="fileProgressPct">—</span></div>
      <div class="item"><span class="label">本次训练时长</span><span class="value" id="fileElapsedTime">—</span></div>
      <div class="item"><span class="label">历史训练总时长</span><span class="value" id="fileHistoryElapsedTime">—</span></div>
      <div class="item"><span class="label">最优分数</span><span class="value score-best" id="fileBestScore">—</span></div>
      <div class="item"><span class="label">验证分数</span><span class="value score-val" id="fileValScore">—</span></div>
    </div>
    <div class="path" title="${info.data_file}">${info.filename || info.data_file}</div>
  `;
  startBtn.disabled = false;
  if ($("retrainBtn")) $("retrainBtn").disabled = false;
}

function updateBtStartBtn() {
  const startBtn = $("btStartBtn");
  if (!startBtn) return;
  startBtn.disabled = btActive || !selectedStrategyFile;
  ["btCommissionInput", "btSlippageInput"].forEach((id) => {
    const el = $(id);
    if (el) el.disabled = btActive;
  });
}

function renderStrategyFileCard(info) {
  const card = $("btStrategyCard");
  if (!card) return;

  if (!info || !info.strategy_file) {
    card.className = "data-file-card";
    card.innerHTML = '<div class="data-file-empty">尚未选择策略文件</div>';
    selectedStrategyFile = null;
    selectedStrategySymbol = null;
    updateBtStartBtn();
    return;
  }

  if (info.valid === false) {
    card.className = "data-file-card invalid";
    card.innerHTML = `
      <div class="data-file-error">${info.message || "文件无效"}</div>
      <div class="data-file-path">${info.strategy_file}</div>
    `;
    selectedStrategyFile = null;
    selectedStrategySymbol = null;
    updateBtStartBtn();
    return;
  }

  selectedStrategyFile = info.strategy_file;
  selectedStrategySymbol = info.symbol || null;
  card.className = "data-file-card valid";
  const timeframeItem = info.timeframe
    ? `<div class="item"><span class="label">周期</span><span class="value">${info.timeframe}</span></div>`
    : "";
  const dataPath = info.data_file || "";
  const dataOk = info.data_file_exists;
  const dataHint = dataPath
    ? (dataOk ? dataPath : `（文件不存在）${dataPath}`)
    : "未记录数据路径 — 回测前请先在训练页选择同品种 Parquet";
  card.innerHTML = `
    <div class="data-file-row">
      <div class="item"><span class="label">品种</span><span class="value sym">${info.symbol || "—"}</span></div>
      ${timeframeItem}
      <div class="item"><span class="label">最优分数</span><span class="value score-best">${formatScore(info.best_score)}</span></div>
      <div class="item"><span class="label">词表版本</span><span class="value">${info.vocab_version || "—"}</span></div>
      <div class="item"><span class="label">公式长度</span><span class="value">${info.formula_decoded ? info.formula_decoded.split("→").length : "—"}</span></div>
    </div>
    <div class="path" title="${info.strategy_file}">策略: ${info.filename || info.strategy_file}</div>
    <div class="path ${dataPath && dataOk ? "" : "data-file-missing"}" title="${dataPath || ""}">数据: ${dataHint}</div>
  `;
  updateBtStartBtn();
}

function formatElapsed(startedAtIso, endAtIso) {
  if (!startedAtIso) return "—";
  const started = new Date(startedAtIso).getTime();
  if (Number.isNaN(started)) return "—";
  const end = endAtIso ? new Date(endAtIso).getTime() : Date.now();
  if (Number.isNaN(end)) return "—";
  return formatDurationSeconds(Math.max(0, Math.floor((end - started) / 1000)));
}

function formatDurationSeconds(secs) {
  if (secs == null || secs < 0) return "—";
  const total = Math.floor(secs);
  const h = Math.floor(total / 3600);
  const m = Math.floor((total % 3600) / 60);
  if (h > 0) return `${h}小时${m}分`;
  if (m > 0) return `${m}分钟`;
  return `${total}秒`;
}

function updateTrainingTimeFields(progress, training) {
  const sessionEl = $("fileElapsedTime");
  const historyEl = $("fileHistoryElapsedTime");
  if (!sessionEl && !historyEl) return;

  if (historyEl) {
    const hist = progress?.history_total_seconds;
    historyEl.textContent = hist != null ? formatDurationSeconds(hist) : "—";
  }

  const job = training?.job;
  const active = !!training?.active;
  if (!sessionEl) return;

  if (!job || job.state === "idle") {
    sessionEl.textContent = "—";
    return;
  }

  const elapsed = formatElapsed(job.started_at, active ? null : job.finished_at);
  sessionEl.textContent = active || elapsed === "—" ? elapsed : `${elapsed}（已停）`;
}

function updateFileProgress(progress) {
  const el = document.getElementById("fileProgressPct");
  if (el && progress) {
    el.textContent = `${progress.current_step} / ${progress.train_steps} (${progress.progress_pct}%)`;
  }
  const bestEl = document.getElementById("fileBestScore");
  if (bestEl) {
    bestEl.textContent = progress ? formatScore(progress.best_score) : "—";
  }
  const valEl = document.getElementById("fileValScore");
  if (valEl) {
    let val = progress?.val_score;
    if (val == null && progress?.history?.val_score?.length) {
      val = progress.history.val_score[progress.history.val_score.length - 1];
    }
    valEl.textContent = progress ? formatScore(val) : "—";
  }
}

const CHART_SERIES = [
  { key: "best_score", label: "最优分数", borderColor: "#34f5c8", fillRGB: "52, 245, 200", yAxisID: "y" },
  { key: "val_score", label: "验证分数", borderColor: "#38bdf8", fillRGB: "56, 189, 248", yAxisID: "y" },
];

// 让曲线在填充区形成竖向渐变
function makeGradient(ctx, area, rgb) {
  if (!area) return `rgba(${rgb}, 0.08)`;
  const g = ctx.createLinearGradient(0, area.top, 0, area.bottom);
  g.addColorStop(0, `rgba(${rgb}, 0.28)`);
  g.addColorStop(0.6, `rgba(${rgb}, 0.06)`);
  g.addColorStop(1, `rgba(${rgb}, 0)`);
  return g;
}

// 发光效果：在每条数据线绘制前设置对应颜色的柔和阴影
const glowPlugin = {
  id: "neonGlow",
  beforeDatasetDraw(chart, args) {
    const color = args?.meta?.dataset?.options?.borderColor;
    const ctx = chart.ctx;
    ctx.save();
    if (typeof color === "string") {
      ctx.shadowColor = color;
      ctx.shadowBlur = 10;
    }
  },
  afterDatasetDraw(chart) {
    chart.ctx.restore();
  },
};
if (window.Chart) Chart.register(glowPlugin);

const CHART_OPTIONS = {
  responsive: true,
  maintainAspectRatio: false,
  interaction: { mode: "index", intersect: false },
  animation: { duration: 450, easing: "easeOutQuart" },
  transitions: {
    active: { animation: { duration: 450, easing: "easeOutQuart" } },
  },
  plugins: {
    legend: {
      labels: {
        color: "#a9bccf",
        usePointStyle: true,
        pointStyle: "circle",
        boxWidth: 8,
        boxHeight: 8,
        padding: 16,
        font: { family: "'DM Sans'", size: 12, weight: "600" },
      },
    },
    tooltip: {
      backgroundColor: "rgba(8, 12, 20, 0.92)",
      borderColor: "rgba(94, 234, 212, 0.35)",
      borderWidth: 1,
      titleColor: "#e8edf4",
      bodyColor: "#a9bccf",
      titleFont: { family: "'JetBrains Mono'", size: 11 },
      bodyFont: { family: "'JetBrains Mono'", size: 11 },
      padding: 10,
      cornerRadius: 8,
      displayColors: true,
      usePointStyle: true,
    },
  },
  scales: {
    x: {
      ticks: { color: "#6b7d92", maxTicksLimit: 8, font: { family: "'JetBrains Mono'", size: 10 } },
      grid: { color: "rgba(120,190,235,0.05)" },
      border: { color: "rgba(120,190,235,0.12)" },
    },
    y: {
      title: { display: true, text: "分数（最优 / 验证）", color: "#7dd3fc", font: { family: "'DM Sans'", size: 10, weight: "600" } },
      ticks: { color: "#6b7d92", font: { family: "'JetBrains Mono'", size: 10 } },
      grid: { color: "rgba(120,190,235,0.05)" },
      border: { color: "rgba(120,190,235,0.12)" },
    },
  },
};

function buildChartDatasets(history) {
  return CHART_SERIES.filter((s) => history?.[s.key]?.length).map((s) => ({
    label: s.label,
    data: history[s.key],
    borderColor: s.borderColor,
    borderWidth: 2,
    tension: 0.35,
    pointRadius: 0,
    pointHoverRadius: 4,
    pointHoverBackgroundColor: s.borderColor,
    pointHoverBorderColor: "#05070d",
    fill: true,
    backgroundColor: (context) => {
      const { ctx, chartArea } = context.chart;
      return makeGradient(ctx, chartArea, s.fillRGB);
    },
    yAxisID: s.yAxisID,
  }));
}

function destroyChart() {
  if (chart) {
    chart.destroy();
    chart = null;
  }
  chartSymbol = null;
}

function createChart(ctx, steps, history) {
  return new Chart(ctx, {
    type: "line",
    data: { labels: steps, datasets: buildChartDatasets(history) },
    options: CHART_OPTIONS,
  });
}

function updateChartInPlace(steps, history) {
  const prevLen = chart.data.labels.length;
  chart.data.labels = steps;

  const next = buildChartDatasets(history);
  for (const ds of next) {
    const existing = chart.data.datasets.find((d) => d.label === ds.label);
    if (existing) {
      existing.data = ds.data;
    } else {
      chart.data.datasets.push(ds);
    }
  }

  const nextLabels = new Set(next.map((d) => d.label));
  chart.data.datasets = chart.data.datasets.filter((d) => nextLabels.has(d.label));

  const grew = steps.length > prevLen;
  chart.update(grew ? "active" : "none");
}

function renderChart(history, label, progress) {
  const ctx = $("mainChart").getContext("2d");
  const steps = history?.step || [];
  if (!steps.length) {
    destroyChart();
    if (progress?.current_step > 0) {
      $("chartHint").textContent = `训练中 第 ${progress.current_step}/${progress.train_steps} 步，曲线每步更新`;
    } else {
      $("chartHint").textContent = "暂无历史数据（首步约需 15–30 秒）";
    }
    return;
  }

  const sameSymbol = chart && chartSymbol === label;
  if (sameSymbol) {
    updateChartInPlace(steps, history);
  } else {
    destroyChart();
    chart = createChart(ctx, steps, history);
    chartSymbol = label;
  }

  $("chartTitle").textContent = `${label} 训练曲线`;
  $("chartHint").textContent = `${steps.length} 个记录点`;
}

async function loadSymbolChart(symbol, progress) {
  if (!symbol) return;
  try {
    const data = await fetchJSON(`/api/symbols/${encodeURIComponent(symbol)}`);
    renderChart(data.history, symbol, progress || data);
    $("formulaText").textContent = data.formula_decoded || "—";
  } catch (e) {
    $("formulaText").textContent = "—";
  }
}

function renderStrategies(rows) {
  const tbody = $("strategiesBody");
  if (!rows.length) {
    tbody.innerHTML = '<tr class="empty-row"><td colspan="4">暂无已保存策略</td></tr>';
    return;
  }
  tbody.innerHTML = rows
    .map(
      (r) => `
    <tr>
      <td>${r.symbol}</td>
      <td>${r.timeframe || "—"}</td>
      <td>${formatScore(r.best_score)}</td>
      <td><code>${r.formula_decoded || "—"}</code></td>
    </tr>`
    )
    .join("");
}

function updateTrainingUI(training, progress) {
  const job = training?.job;
  const active = training?.active;
  const pill = $("jobPill");
  const startBtn = $("startBtn");
  const retrainBtn = $("retrainBtn");
  const stopBtn = $("stopBtn");

  if (!job || job.state === "idle") {
    pill.innerHTML = '<i class="pill-dot"></i>空闲';
    pill.className = "pill";
    startBtn.disabled = !selectedDataFile;
    if (retrainBtn) retrainBtn.disabled = !selectedDataFile;
    stopBtn.disabled = true;
    $("logHint").textContent = "—";
    updateTrainingTimeFields(progress, training);
    return;
  }

  const stateLabel = {
    running: "训练中",
    completed: "已完成",
    failed: "失败",
    stopped: "已停止",
  };
  const label = job.symbol ? `${job.symbol} ${job.timeframe || ""}`.trim() : "训练";
  const stateText = stateLabel[job.state] || job.state;
  pill.innerHTML = `<i class="pill-dot"></i>${stateText} · ${label}`;
  pill.className = "pill " + (job.state === "running" ? "running" : job.state);

  startBtn.disabled = active;
  if (retrainBtn) retrainBtn.disabled = active;
  stopBtn.disabled = !active;
  $("logHint").textContent = job.log_path || "—";
  updateTrainingTimeFields(progress, training);

  const logView = $("logView");
  const atBottom = isViewAtBottom(logView);
  logView.textContent = (training.log_tail || []).join("\n") || "等待输出…";
  if (atBottom) logView.scrollTop = logView.scrollHeight;
}

async function refreshOverview() {
  let overview = { data_file: null, progress: null };
  let strategies = { strategies: [] };
  let training = { active: false, job: null, log_tail: [] };

  try {
    overview = await fetchJSON("/api/overview", { silent: true });
  } catch (_) {}

  try {
    strategies = await fetchJSON("/api/strategies", { silent: true });
  } catch (_) {}

  try {
    training = await fetchJSON("/api/training/status", { silent: true });
  } catch (_) {}

  if (overview.data_file) renderDataFileCard(overview.data_file);
  updateFileProgress(overview.progress);
  updateExportBtn(overview.progress, strategies.strategies);
  updateTrainingBtns(overview.progress, training);
  updateTrainingUI(training, overview.progress);
  renderStrategies(strategies.strategies);

  const sym = overview.progress?.symbol || selectedSymbol || training?.job?.symbol;
  const trainingActive = !!training?.active;
  if (lastTrainingActive && !trainingActive && sym) {
    await applyBestStrategyForBacktest(sym, null);
  }
  lastTrainingActive = trainingActive;

  if (sym && (training?.active || overview.progress)) {
    await loadSymbolChart(sym, overview.progress);
  }

  await refreshDebugLogs();
}

async function loadConfig() {
  const health = await fetch(API + "/api/health").then((r) => r.json()).catch(() => ({}));
  if (!health.version) {
    await logClientError(
      "后端版本过旧或未启动新版服务。请关闭旧进程后重新运行: python run_web.py",
      { health }
    );
  }

  const cfg = await fetchJSON("/api/config");
  debugMode = !!cfg.debug_mode;
  $("debugModeCheck").checked = debugMode;
  $("deviceMeta").textContent = `${cfg.train_steps} steps · batch ${cfg.batch_size} · ${cfg.device}`;
  if (cfg.error_log) {
    $("debugLogPaths").textContent = `本地: ${cfg.error_log}`;
  }
  if (cfg.data_file) renderDataFileCard(cfg.data_file);
  if (cfg.strategy_file) renderStrategyFileCard(cfg.strategy_file);
  applyBacktestCostDefaults(cfg);
  await initAiPanel(cfg);
}

function applyBacktestCostDefaults(cfg) {
  const cIn = $("btCommissionInput");
  const sIn = $("btSlippageInput");
  if (cIn && cfg.bt_commission_pct != null) cIn.value = Number(cfg.bt_commission_pct);
  if (sIn && cfg.bt_slippage_pct != null) sIn.value = Number(cfg.bt_slippage_pct);
  updateBtCostHint();
}

function readBacktestCosts() {
  const cRaw = Number($("btCommissionInput")?.value);
  const sRaw = Number($("btSlippageInput")?.value);
  const commission = Number.isFinite(cRaw) && cRaw >= 0 ? cRaw : 0.02;
  const slippage = Number.isFinite(sRaw) && sRaw >= 0 ? sRaw : 0.01;
  return { commission_pct: commission, slippage_pct: slippage };
}

function updateBtCostHint() {
  const hint = $("btCostSumHint");
  if (!hint) return;
  const { commission_pct, slippage_pct } = readBacktestCosts();
  const fee = Number((commission_pct + slippage_pct).toFixed(4));
  hint.textContent = `单边成本 ${fee}%`;
}

async function initAiPanel(cfg) {
  const keyInput = $("aiApiKeyInput");
  if (!keyInput) return;

  if (cfg?.ai_api_key) keyInput.value = cfg.ai_api_key;
  else if (cfg?.ai_provider === "openclaw" || cfg?.ai_provider === "openclaw_wb") {
    keyInput.value = cfg.ai_provider;
  }

  try {
    const status = await fetchJSON("/api/ai/providers");
    window.__aiProviderStatus = status;
  } catch (_) {
    window.__aiProviderStatus = null;
  }
  updateAiChannelHint();
}

function resolveAiFromKey(raw) {
  const v = (raw || "").trim().toLowerCase();
  // openclaw_wb 必须先于 openclaw，避免前缀误匹配
  if (v === "openclaw_wb" || v.startsWith("openclaw_wb/")) {
    return { provider: "openclaw_wb", apiKey: raw.trim(), isAlias: true };
  }
  if (v === "openclaw" || v.startsWith("openclaw/")) {
    return { provider: "openclaw", apiKey: raw.trim(), isAlias: true };
  }
  return { provider: "deepseek", apiKey: (raw || "").trim(), isAlias: false };
}

function updateAiChannelHint() {
  const hint = $("aiChannelHint");
  const headHint = $("aiProviderHint");
  const keyInput = $("aiApiKeyInput");
  if (!hint || !keyInput) return;

  const resolved = resolveAiFromKey(keyInput.value);
  const status = window.__aiProviderStatus;
  const row = (status?.providers || []).find((p) => p.id === resolved.provider);

  if (resolved.provider === "deepseek") {
    if (headHint) headHint.textContent = "DeepSeek · deepseek-v4-flash";
    hint.textContent = "当前：DeepSeek（deepseek-v4-flash · https://api.deepseek.com）。";
  } else if (resolved.provider === "openclaw") {
    if (headHint) headHint.textContent = row?.available ? "openclaw (QClaw) · 已匹配" : "openclaw (QClaw) · 未就绪";
    hint.textContent = row?.hint || "已匹配 openclaw：将自动使用本地 QClaw token。";
  } else {
    if (headHint) headHint.textContent = row?.available ? "openclaw_wb · 已匹配" : "openclaw_wb · 未就绪";
    hint.textContent = row?.hint || "已匹配 openclaw_wb：将自动使用 WorkBuddy token。";
  }
}

function openUnlimitedModal() {
  const modal = $("aiUnlimitedModal");
  if (modal) modal.hidden = false;
}

function closeUnlimitedModal() {
  const modal = $("aiUnlimitedModal");
  if (modal) modal.hidden = true;
}

async function runAiAnalyze() {
  const btn = $("aiAnalyzeBtn");
  const view = $("aiAnswerView");
  const rawKey = $("aiApiKeyInput")?.value || "";
  const resolved = resolveAiFromKey(rawKey);
  if (!view) return;

  if (resolved.provider === "deepseek" && !resolved.apiKey) {
    view.className = "ai-answer error";
    view.textContent = "请填写 DeepSeek API Key，或在 Key 中输入 openclaw / openclaw_wb";
    return;
  }

  if (btn) btn.disabled = true;
  view.className = "ai-answer loading";
  view.textContent = `正在通过 ${resolved.provider} 连接并流式分析…`;

  let header = "";
  let answer = "";

  try {
    const res = await fetch(API + "/api/ai/analyze-training", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        provider: resolved.provider,
        api_key: resolved.apiKey,
        symbol: selectedSymbol || null,
      }),
    });
    if (!res.ok) {
      const data = await res.json().catch(() => ({}));
      throw new Error(formatApiError(data, res.status, "/api/ai/analyze-training"));
    }
    if (!res.body) throw new Error("浏览器不支持流式响应");

    const reader = res.body.getReader();
    const decoder = new TextDecoder("utf-8");
    let buffer = "";
    view.className = "ai-answer streaming";
    view.textContent = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const chunks = buffer.split("\n\n");
      buffer = chunks.pop() || "";
      for (const block of chunks) {
        const line = block
          .split("\n")
          .map((l) => l.trim())
          .find((l) => l.startsWith("data:"));
        if (!line) continue;
        let event;
        try {
          event = JSON.parse(line.slice(5).trim());
        } catch (_) {
          continue;
        }
        if (event.type === "meta") {
          header =
            `[${event.label || event.provider || resolved.provider} · ${event.model || ""} · ${event.symbol || ""}${event.timeframe ? " " + event.timeframe : ""}]` +
            (event.prior_count
              ? ` · 已带入前 ${event.prior_count} 次同品种同周期分析`
              : " · 首次分析") +
            `\n\n`;
          view.textContent = header;
          view.scrollTop = view.scrollHeight;
        } else if (event.type === "delta") {
          answer += event.text || "";
          view.textContent = header + answer;
          view.scrollTop = view.scrollHeight;
        } else if (event.type === "error") {
          throw new Error(event.message || "分析失败");
        } else if (event.type === "done") {
          answer = event.answer || answer;
          view.className = "ai-answer";
          view.textContent = header + (answer || "（无内容）");
        }
      }
    }
    if (!answer && view.className.includes("streaming")) {
      throw new Error("流式分析中断，未收到完整回复");
    }
    view.className = "ai-answer";
  } catch (e) {
    view.className = "ai-answer error";
    view.textContent = `分析失败: ${e.message}`;
  } finally {
    if (btn) btn.disabled = false;
  }
}

async function browseStrategyFile() {
  try {
    const res = await fetchJSON("/api/strategy-file/browse", { method: "POST" });
    if (res.cancelled) return;
    renderStrategyFileCard(res);
  } catch (e) {
    $("debugView")?.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }
}

async function applyBestStrategyForBacktest(symbol, strategyFile) {
  if (strategyFile) {
    renderStrategyFileCard(strategyFile);
    return;
  }
  if (!symbol) return;
  try {
    const res = await fetchJSON(
      `/api/strategy-file/sync-best?symbol=${encodeURIComponent(symbol)}`,
      { method: "POST" }
    );
    renderStrategyFileCard(res);
  } catch (_) {
    await loadBacktestStrategyContext();
  }
}

async function loadBacktestStrategyContext() {
  const sym = selectedStrategySymbol || selectedSymbol;
  if (sym) {
    await applyBestStrategyForBacktest(sym, null);
    return;
  }
  try {
    const cfg = await fetchJSON("/api/config");
    if (cfg.strategy_file) renderStrategyFileCard(cfg.strategy_file);
  } catch (_) {
    /* ignore */
  }
}

async function browseDataFile() {
  try {
    const res = await fetchJSON("/api/data-file/browse", { method: "POST" });
    if (res.cancelled) return;
    renderDataFileCard(res);
    selectedSymbol = res.symbol;
    await loadSymbolChart(res.symbol);
  } catch (e) {
    $("debugView").scrollIntoView({ behavior: "smooth", block: "nearest" });
  }
}

async function startTraining() {
  if (!selectedDataFile) {
    await logClientError("请先选择数据文件");
    return;
  }
  try {
    const res = await fetchJSON("/api/training/start", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ data_file: selectedDataFile, from_scratch: false }),
    });
    selectedSymbol = res.data_file?.symbol || res.job?.symbol;
    renderDataFileCard(res.data_file);
    await refreshOverview();
  } catch (e) {
    $("debugView").scrollIntoView({ behavior: "smooth", block: "nearest" });
  }
}

async function retrainFromScratch() {
  if (!selectedDataFile) {
    await logClientError("请先选择数据文件");
    return;
  }
  const ok = window.confirm(
    "重新训练会清除该品种的检查点，从第 0 步重新搜索。\n" +
      "已有的更优策略会保留，只有挖到更高分才会覆盖。\n\n" +
      "确定要重新训练吗？"
  );
  if (!ok) return;
  try {
    const res = await fetchJSON("/api/training/start", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ data_file: selectedDataFile, from_scratch: true }),
    });
    selectedSymbol = res.data_file?.symbol || res.job?.symbol;
    renderDataFileCard(res.data_file);
    await refreshOverview();
  } catch (e) {
    $("debugView").scrollIntoView({ behavior: "smooth", block: "nearest" });
  }
}

function updateExportBtn(progress, strategies) {
  const sym = progress?.symbol || selectedSymbol;
  const hasStrategy = progress?.has_strategy || (strategies || []).some((s) => s.symbol === sym);
  const btn = $("exportBtn");
  if (btn) btn.disabled = !sym || !hasStrategy;
}

function updateTrainingBtns(progress, training) {
  const sym = progress?.symbol || selectedSymbol;
  const active = training?.active;
  const hasCheckpoint = Boolean(progress?.has_checkpoint);
  const exportBtn = $("exportTrainingBtn");
  const importBtn = $("importTrainingBtn");

  let exportTitle = "打包 checkpoint、训练曲线与策略为 zip";
  if (!sym) {
    exportTitle = "请先选择数据文件";
  } else if (active) {
    exportTitle = "训练进行中，请停止后再导出";
  } else if (!hasCheckpoint) {
    exportTitle = "该品种尚无检查点：至少训练满 20 步后才会生成（每 20 步保存一次）";
  }

  if (exportBtn) {
    exportBtn.disabled = !sym || !hasCheckpoint || !!active;
    exportBtn.title = exportTitle;
  }
  if (importBtn) {
    importBtn.disabled = !sym || !!active;
    importBtn.title = active ? "训练进行中，请停止后再导入" : "上传 .zip 或 .pt，下次训练断点续训";
  }
}

async function exportTraining() {
  const sym = selectedSymbol;
  if (!sym) {
    await logClientError("请先选择数据文件");
    return;
  }
  const path = `/api/training/${encodeURIComponent(sym)}/export`;
  try {
    const res = await fetch(API + path);
    if (!res.ok) {
      const data = await res.json().catch(() => ({}));
      throw new Error(formatApiError(data, res.status, path));
    }
    const blob = await res.blob();
    const disp = res.headers.get("Content-Disposition") || "";
    const m = /filename="([^"]+)"/.exec(disp);
    const filename = m ? m[1] : `training_${sym.replace(/\./g, "_")}.zip`;
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
  } catch (e) {
    await logClientError(`导出训练失败: ${e.message}`);
    $("debugView").scrollIntoView({ behavior: "smooth", block: "nearest" });
  }
}

function triggerImportTraining() {
  const input = $("importTrainingFile");
  if (input) {
    input.value = "";
    input.click();
  }
}

async function handleImportTrainingFile(event) {
  const input = event.target;
  const file = input.files?.[0];
  if (!file) return;

  const sym = selectedSymbol;
  if (!sym) {
    await logClientError("请先选择数据文件");
    return;
  }

  const form = new FormData();
  form.append("file", file);

  try {
    const res = await fetch(`${API}/api/training/import?symbol=${encodeURIComponent(sym)}`, {
      method: "POST",
      body: form,
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) {
      throw new Error(formatApiError(data, res.status, "/api/training/import"));
    }
    if (data.symbol && data.symbol !== sym) {
      selectedSymbol = data.symbol;
    }
    clientErrors.push(`[${new Date().toLocaleString()}] ${data.message || "训练文件导入成功"}`);
    if (clientErrors.length > 80) clientErrors = clientErrors.slice(-80);
    renderDebugView();
    await refreshOverview();
  } catch (e) {
    await logClientError(`导入训练失败: ${e.message}`);
    $("debugView").scrollIntoView({ behavior: "smooth", block: "nearest" });
  } finally {
    input.value = "";
  }
}

function parseContentDispositionFilename(header) {
  if (!header) return null;
  const utf8 = /filename\*=UTF-8''([^;]+)/i.exec(header);
  if (utf8) return decodeURIComponent(utf8[1]);
  const plain = /filename="([^"]+)"/i.exec(header) || /filename=([^;]+)/i.exec(header);
  return plain ? plain[1].trim() : null;
}

async function exportStrategy() {
  const sym = selectedSymbol;
  if (!sym) {
    await logClientError("请先选择数据文件");
    return;
  }
  const path = `/api/strategies/${encodeURIComponent(sym)}/export`;
  try {
    const res = await fetch(API + path);
    if (!res.ok) {
      const data = await res.json().catch(() => ({}));
      throw new Error(formatApiError(data, res.status, path));
    }
    const blob = await res.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download =
      parseContentDispositionFilename(res.headers.get("Content-Disposition")) ||
      `strategy_${sym.replace(/\./g, "_")}.json`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
  } catch (e) {
    await logClientError(`导出策略失败: ${e.message}`);
    $("debugView").scrollIntoView({ behavior: "smooth", block: "nearest" });
  }
}

async function stopTraining() {
  try {
    const res = await fetchJSON("/api/training/stop", { method: "POST" });
    await refreshOverview();
    const sym = res.training?.job?.symbol || selectedSymbol;
    await applyBestStrategyForBacktest(sym, res.strategy_file);
  } catch (e) {
    $("debugView").scrollIntoView({ behavior: "smooth", block: "nearest" });
  }
}

// ═══════════════════════════════════════════════════════════════════
// 分页切换
// ═══════════════════════════════════════════════════════════════════
function switchPage(page) {
  if (page !== "train" && page !== "backtest" && page !== "realtime") return;
  currentPage = page;
  document.querySelectorAll(".stepper .step").forEach((s) => {
    s.classList.toggle("active", s.dataset.page === page);
  });
  document.querySelectorAll(".page").forEach((p) => {
    p.classList.toggle("active", p.id === `page-${page}`);
  });
  if (page === "backtest") {
    loadBacktestStrategyContext();
    refreshBacktest();
  } else if (page === "realtime") {
    initRealtimeOnce();
    refreshRealtime();
  }
}

// ═══════════════════════════════════════════════════════════════════
// 回测：格式化辅助
// ═══════════════════════════════════════════════════════════════════
function fmtPct(v, digits = 2) {
  if (v == null || Number.isNaN(v)) return "—";
  return (v >= 0 ? "+" : "") + (v * 100).toFixed(digits) + "%";
}
function fmtSigned(v, digits = 3) {
  if (v == null || Number.isNaN(v)) return "—";
  return (v >= 0 ? "+" : "") + Number(v).toFixed(digits);
}

// ═══════════════════════════════════════════════════════════════════
// 回测：状态轮询 + UI 更新
// ═══════════════════════════════════════════════════════════════════
async function refreshBacktest() {
  let st;
  try {
    st = await fetchJSON("/api/backtest/status", { silent: true });
  } catch (_) {
    return;
  }
  btActive = !!st.active;
  const job = st.job;
  const state = job?.state || "idle";

  // 按钮
  const stopBtn = $("btStopBtn");
  updateBtStartBtn();
  if (stopBtn) stopBtn.disabled = !btActive;

  // 缓存刷新键：用最近一次任务的结束/开始时间
  btBuster = job?.finished_at || job?.started_at || btBuster;

  // 日志
  const logView = $("btLogView");
  const logText = (st.log_tail || []).join("\n") || "等待任务…";
  if (logView) {
    const atBottom = isViewAtBottom(logView);
    logView.textContent = logText;
    if (atBottom) logView.scrollTop = logView.scrollHeight;
  }
  if ($("btLogHint")) $("btLogHint").textContent = job?.log_path || "—";

  // 阶段进度条
  updateBacktestPhase(st, state);

  if (state === "failed") {
    const alertKey = `${job?.log_path || ""}|${job?.finished_at || ""}|${job?.exit_code ?? ""}`;
    if (alertKey && alertKey !== btLastAlertKey) {
      btLastAlertKey = alertKey;
      const errLine = job?.error ? `\n错误: ${job.error}` : "";
      showErrorPopup(
        "回测失败",
        `退出码: ${job?.exit_code ?? "?"}${errLine}\n日志: ${job?.log_path || "—"}\n\n${logText}`
      );
    }
  }

  // 结果报告（非运行态时刷新，运行态保留上次结果）
  if (!btActive) {
    await refreshBacktestReport();
  }
}

const BT_STATE_LABEL = {
  running: "回测中",
  completed: "已完成",
  failed: "失败",
  stopped: "已停止",
  idle: "待机",
};

function updateBacktestPhase(st, state) {
  const fill = $("btPhaseFill");
  const label = $("btPhaseLabel");
  if (!fill || !label) return;

  const total = st.phase_total || 7;
  const idx = st.phase_index || 0;

  let pct;
  if (btActive) {
    pct = Math.min(96, Math.round(((idx + 1) / total) * 100));
    label.textContent = `${st.phase_label || "回测中"}…`;
    fill.classList.add("animate");
  } else if (state === "completed") {
    pct = 100;
    label.textContent = "完成";
    fill.classList.remove("animate");
  } else if (state === "failed" || state === "stopped") {
    pct = Math.min(96, Math.round(((idx + 1) / total) * 100));
    label.textContent = BT_STATE_LABEL[state];
    fill.classList.remove("animate");
  } else {
    pct = 0;
    label.textContent = "待机";
    fill.classList.remove("animate");
  }
  fill.style.width = pct + "%";
}

async function refreshBacktestReport() {
  let data;
  const sym = selectedStrategySymbol || selectedSymbol;
  const url = sym
    ? `/api/backtest/report?symbol=${encodeURIComponent(sym)}`
    : "/api/backtest/report";
  try {
    data = await fetchJSON(url, { silent: true });
  } catch (_) {
    return;
  }
  if (!data.available || !data.report) {
    if ($("btPortfolioHint")) $("btPortfolioHint").textContent = "尚未运行回测";
    lastEquityData = null;
    btPortfolioSig = "";
    renderEquity(null);
    return;
  }
  // 先取资金曲线（写入 lastEquityData），再渲染绩效卡，让 sparkline 用上真实数据
  await refreshEquityCurve();
  renderPortfolio(data.report);
  renderBacktestTable(data.report.symbols || {});
}

async function refreshEquityCurve() {
  const sym = selectedStrategySymbol || selectedSymbol;
  const url = sym
    ? `/api/backtest/equity?symbol=${encodeURIComponent(sym)}`
    : "/api/backtest/equity";
  try {
    const data = await fetchJSON(url, { silent: true });
    lastEquityData = data?.available ? data.data : null;
    renderEquity(data);
  } catch (_) {
    lastEquityData = null;
    renderEquity(null);
  }
}

// ═══════════════════════════════════════════════════════════════════
// 迷你 sparkline + 数字滚动动画（终端仪表盘质感）
// ═══════════════════════════════════════════════════════════════════
const METRIC_FMT = {
  pct: (v) => (v >= 0 ? "+" : "") + (v * 100).toFixed(2) + "%",
  signed: (v) => (v >= 0 ? "+" : "") + v.toFixed(3),
  ratio: (v) => v.toFixed(3),
  int: (v) => Math.round(v).toLocaleString(),
  winrate: (v) => (v * 100).toFixed(1) + "%",
  strength: (v) => Math.round(v * 100) + "%",
};

function prefersReducedMotion() {
  return !!(window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches);
}

// 短促 count-up（≈420ms, easeOutCubic），克制不浮夸
function animateCount(el, to, fmt) {
  const fn = METRIC_FMT[fmt] || ((v) => String(v));
  if (!Number.isFinite(to)) {
    el.textContent = "—";
    return;
  }
  if (prefersReducedMotion()) {
    el.textContent = fn(to);
    return;
  }
  const dur = 420;
  const t0 = performance.now();
  function frame(now) {
    const p = Math.min(1, (now - t0) / dur);
    const e = 1 - Math.pow(1 - p, 3); // easeOutCubic
    el.textContent = fn(to * e);
    if (p < 1) requestAnimationFrame(frame);
    else el.textContent = fn(to);
  }
  requestAnimationFrame(frame);
}

function runCountUp(root) {
  if (!root) return;
  root.querySelectorAll("[data-count]").forEach((el) => {
    animateCount(el, parseFloat(el.dataset.count), el.dataset.fmt || "");
  });
}

// 均匀降采样为 <= target 个有限点
function downsampleSeries(arr, target) {
  const clean = (arr || [])
    .map(Number)
    .filter((v) => Number.isFinite(v));
  if (clean.length <= target) return clean;
  const out = [];
  const step = (clean.length - 1) / (target - 1);
  for (let i = 0; i < target; i++) out.push(clean[Math.round(i * step)]);
  return out;
}

// 生成极小趋势微线（内联 SVG，轻量、清晰）
function sparklineSVG(values, { color = "#5eead4", fillRGB = null, w = 74, h = 22 } = {}) {
  const v = downsampleSeries(values, 56);
  if (v.length < 2) return "";
  const min = Math.min(...v);
  const max = Math.max(...v);
  const range = max - min || 1;
  const n = v.length;
  const x = (i) => (i / (n - 1)) * w;
  const y = (val) => h - 2 - ((val - min) / range) * (h - 4);
  const line = "M" + v.map((val, i) => `${x(i).toFixed(1)} ${y(val).toFixed(1)}`).join(" L ");
  const area = fillRGB
    ? `<path d="${line} L ${w} ${h} L 0 ${h} Z" fill="rgba(${fillRGB},0.14)" stroke="none"/>`
    : "";
  const dot = `<circle cx="${x(n - 1).toFixed(1)}" cy="${y(v[n - 1]).toFixed(1)}" r="1.6" fill="${color}"/>`;
  return `<svg class="spark-svg" viewBox="0 0 ${w} ${h}" preserveAspectRatio="none" aria-hidden="true">${area}<path d="${line}" fill="none" stroke="${color}" stroke-width="1.4" stroke-linejoin="round" stroke-linecap="round"/>${dot}</svg>`;
}

// 取当前主资金曲线序列（组合优先，否则第一个品种）
function mainEquitySeries() {
  const d = lastEquityData;
  if (!d) return null;
  if (d.portfolio) return d.portfolio;
  const syms = d.symbols || {};
  const names = Object.keys(syms);
  return names.length ? syms[names[0]] : null;
}

function renderPortfolio(report) {
  const grid = $("btPortfolioGrid");
  if (!grid) return;
  const p = report.portfolio || {};
  const focus = report.focus_symbol || Object.keys(report.symbols || {})[0] || "";
  const symData = focus ? (report.symbols || {})[focus] : null;

  if (!Object.keys(p).length) {
    grid.innerHTML = '<div class="metric-empty">回测结果无绩效数据</div>';
    btPortfolioSig = "";
    return;
  }

  const plNum = Number(symData?.profit_loss_ratio ?? p.profit_loss_ratio);
  const nTrades = symData?.n_trades ?? p.n_trades;
  const winRate = symData?.win_rate;

  // sparkline 数据源：主资金曲线 + 滚动夏普
  const eq = mainEquitySeries();
  const posColor = p.total_return >= 0 ? "#4ade80" : "#f87171";
  const posRGB = p.total_return >= 0 ? "74, 222, 128" : "248, 113, 113";
  const equitySpark = eq ? sparklineSVG(eq.equity, { color: posColor, fillRGB: posRGB }) : "";
  const rollSpark = eq ? sparklineSVG(eq.rolling_sharpe, { color: "#5eead4", fillRGB: "94, 234, 212" }) : "";

  const cards = [
    { label: "总收益", raw: p.total_return, fmt: "pct", cls: p.total_return >= 0 ? "pos" : "neg", spark: equitySpark },
    { label: "Sharpe", raw: p.sharpe, fmt: "signed", cls: "accent", spark: rollSpark },
    { label: "Sortino", raw: p.sortino, fmt: "signed", cls: "accent", spark: rollSpark },
    { label: "盈亏比", raw: Number.isFinite(plNum) ? plNum : null, fmt: "ratio", cls: Number.isFinite(plNum) ? "accent" : "" },
    { label: "交易数", raw: Number.isFinite(Number(nTrades)) ? Number(nTrades) : null, fmt: "int", cls: "" },
    { label: "胜率", raw: winRate != null ? Number(winRate) : null, fmt: "winrate", cls: "" },
  ];

  // 签名守卫：数值/焦点/资金曲线未变则不重建，避免每次轮询重播动画
  const sig = [focus, btEquitySig, ...cards.map((c) => c.raw)].join("|");
  if (sig === btPortfolioSig) {
    if ($("btPortfolioHint")) $("btPortfolioHint").textContent = focus ? `${focus} 回测绩效` : "回测绩效";
    return;
  }
  btPortfolioSig = sig;

  grid.innerHTML = cards
    .map((c) => {
      const cardCls = c.cls === "pos" || c.cls === "neg" ? c.cls : "";
      const finite = c.raw != null && Number.isFinite(c.raw);
      const finalText = finite ? METRIC_FMT[c.fmt](c.raw) : "—";
      const countAttr = finite ? ` data-count="${c.raw}" data-fmt="${c.fmt}"` : "";
      const spark = c.spark ? `<div class="metric-spark">${c.spark}</div>` : "";
      return `
    <div class="metric-card ${cardCls}">
      <div class="metric-label">${c.label}</div>
      <div class="metric-value ${c.cls}"${countAttr}>${finalText}</div>
      ${spark}
    </div>`;
    })
    .join("");

  runCountUp(grid);

  if ($("btPortfolioHint")) {
    $("btPortfolioHint").textContent = focus ? `${focus} 回测绩效` : "回测绩效";
  }
}

function renderBacktestTable(symbols) {
  const tbody = $("btTableBody");
  if (!tbody) return;
  const rows = Object.entries(symbols);
  if (!rows.length) {
    tbody.innerHTML = '<tr class="empty-row"><td colspan="7">暂无回测结果</td></tr>';
    if ($("btTableHint")) $("btTableHint").textContent = "—";
    return;
  }
  if ($("btTableHint")) $("btTableHint").textContent = rows.length === 1 ? rows[0][0] : `${rows.length} 个品种`;
  tbody.innerHTML = rows
    .map(([sym, d]) => {
      const retCls = (d.total_return || 0) >= 0 ? "pos" : "neg";
      const shCls = (d.sharpe || 0) >= 0 ? "pos" : "neg";
      return `
      <tr>
        <td class="sym-cell">${sym}</td>
        <td class="${retCls}">${fmtPct(d.total_return)}</td>
        <td class="${shCls}">${fmtSigned(d.sharpe)}</td>
        <td>${fmtSigned(d.sortino)}</td>
        <td>${Number.isFinite(Number(d.profit_loss_ratio)) ? Number(d.profit_loss_ratio).toFixed(3) : "—"}</td>
        <td>${d.n_trades ?? "—"}</td>
        <td>${d.win_rate != null ? (d.win_rate * 100).toFixed(1) + "%" : "—"}</td>
      </tr>`;
    })
    .join("");
}

// ═══════════════════════════════════════════════════════════════════
// 交互式资金曲线（HTML / Chart.js）
// ═══════════════════════════════════════════════════════════════════
let equityChart = null;
let rollingChart = null;
let btEquitySig = "";

const EQUITY_COLORS = [
  { hex: "#5eead4", rgb: "94, 234, 212" },
  { hex: "#38bdf8", rgb: "56, 189, 248" },
  { hex: "#818cf8", rgb: "129, 140, 248" },
  { hex: "#fbbf24", rgb: "251, 191, 36" },
  { hex: "#f472b6", rgb: "244, 114, 182" },
  { hex: "#a3e635", rgb: "163, 230, 53" },
];

function verticalGradient(chart, rgb, topAlpha, bottomAlpha) {
  const { ctx, chartArea } = chart;
  if (!chartArea) return `rgba(${rgb}, ${topAlpha})`;
  const g = ctx.createLinearGradient(0, chartArea.top, 0, chartArea.bottom);
  g.addColorStop(0, `rgba(${rgb}, ${topAlpha})`);
  g.addColorStop(0.62, `rgba(${rgb}, ${(topAlpha + bottomAlpha) / 4})`);
  g.addColorStop(1, `rgba(${rgb}, ${bottomAlpha})`);
  return g;
}

const EQUITY_TOOLTIP = {
  backgroundColor: "rgba(8, 12, 20, 0.94)",
  borderColor: "rgba(94, 234, 212, 0.35)",
  borderWidth: 1,
  titleColor: "#e8edf4",
  bodyColor: "#a9bccf",
  titleFont: { family: "'JetBrains Mono'", size: 11 },
  bodyFont: { family: "'JetBrains Mono'", size: 11 },
  padding: 10,
  cornerRadius: 8,
  usePointStyle: true,
};

const EQUITY_OPTIONS = {
  responsive: true,
  maintainAspectRatio: false,
  interaction: { mode: "index", intersect: false },
  animation: { duration: 500, easing: "easeOutQuart" },
  plugins: {
    legend: {
      display: true,
      labels: {
        color: "#a9bccf",
        usePointStyle: true,
        pointStyle: "circle",
        boxWidth: 8,
        boxHeight: 8,
        padding: 14,
        font: { family: "'DM Sans'", size: 12, weight: "600" },
      },
    },
    tooltip: {
      ...EQUITY_TOOLTIP,
      callbacks: {
        label: (c) => ` ${c.dataset.label}: ${Number(c.parsed.y).toFixed(4)}`,
      },
    },
  },
  scales: {
    x: {
      ticks: { color: "#6b7d92", maxTicksLimit: 8, maxRotation: 0, font: { family: "'JetBrains Mono'", size: 10 } },
      grid: { color: "rgba(120,190,235,0.05)" },
      border: { color: "rgba(120,190,235,0.12)" },
    },
    y: {
      ticks: { color: "#6b7d92", font: { family: "'JetBrains Mono'", size: 10 } },
      grid: { color: "rgba(120,190,235,0.05)" },
      border: { color: "rgba(120,190,235,0.12)" },
    },
  },
};

const ROLLING_OPTIONS = {
  responsive: true,
  maintainAspectRatio: false,
  interaction: { mode: "index", intersect: false },
  animation: { duration: 500, easing: "easeOutQuart" },
  spanGaps: false,
  plugins: {
    legend: { display: false },
    tooltip: {
      ...EQUITY_TOOLTIP,
      borderColor: "rgba(251, 191, 36, 0.4)",
      callbacks: {
        label: (c) => {
          const v = c.parsed.y;
          if (v == null || Number.isNaN(v)) return " 滚动夏普: —";
          return ` 滚动夏普: ${Number(v).toFixed(3)}`;
        },
      },
    },
  },
  scales: {
    x: {
      ticks: { color: "#6b7d92", maxTicksLimit: 8, maxRotation: 0, font: { family: "'JetBrains Mono'", size: 10 } },
      grid: { color: "rgba(120,190,235,0.05)" },
      border: { color: "rgba(120,190,235,0.12)" },
    },
    y: {
      ticks: { color: "#6b7d92", font: { family: "'JetBrains Mono'", size: 10 } },
      grid: { color: "rgba(251,191,36,0.06)" },
      border: { color: "rgba(251,191,36,0.18)" },
    },
  },
};

function destroyEquityCharts() {
  if (equityChart) { equityChart.destroy(); equityChart = null; }
  if (rollingChart) { rollingChart.destroy(); rollingChart = null; }
}

function renderEquityStats(name, series) {
  const el = $("btEquityStats");
  if (!el) return;
  const pl = series.profit_loss_ratio;
  const plText = Number.isFinite(Number(pl)) ? Number(pl).toFixed(3) : "—";
  const roll = series.rolling_sharpe || [];
  let lastRoll = null;
  for (let i = roll.length - 1; i >= 0; i--) {
    const v = Number(roll[i]);
    if (Number.isFinite(v)) {
      lastRoll = v;
      break;
    }
  }
  const plNum = Number(pl);
  const cards = [
    { label: "总收益", raw: series.total_return, fmt: "pct", cls: series.total_return >= 0 ? "pos" : "neg" },
    { label: "夏普", raw: series.sharpe, fmt: "signed", cls: "accent" },
    { label: "索提诺", raw: series.sortino, fmt: "signed", cls: "accent" },
    { label: "盈亏比", raw: Number.isFinite(plNum) ? plNum : null, fmt: "ratio", cls: "accent" },
    {
      label: "最新滚动夏普",
      raw: lastRoll,
      fmt: "signed",
      cls: lastRoll == null ? "" : lastRoll >= 0 ? "accent" : "neg",
    },
  ];
  el.innerHTML =
    `<span class="equity-stat-name">${name}</span>` +
    cards
      .map((c) => {
        const finite = c.raw != null && Number.isFinite(c.raw);
        const finalText = finite ? METRIC_FMT[c.fmt](c.raw) : "—";
        const countAttr = finite ? ` data-count="${c.raw}" data-fmt="${c.fmt}"` : "";
        return `
      <div class="equity-stat">
        <span class="equity-stat-label">${c.label}</span>
        <span class="equity-stat-value ${c.cls}"${countAttr}>${finalText}</span>
      </div>`;
      })
      .join("");
  runCountUp(el);
}

function buildEquityChart(labels, symbols, portfolio) {
  const canvas = $("btEquityChart");
  if (!canvas) return;
  const symNames = Object.keys(symbols);
  const multi = symNames.length > 1;
  const datasets = symNames.map((s, i) => {
    const col = EQUITY_COLORS[i % EQUITY_COLORS.length];
    return {
      label: s,
      data: symbols[s].equity,
      borderColor: col.hex,
      borderWidth: multi ? 1.5 : 2.2,
      tension: 0.25,
      pointRadius: 0,
      pointHoverRadius: 4,
      pointHoverBackgroundColor: col.hex,
      pointHoverBorderColor: "#05070d",
      fill: !multi,
      backgroundColor: (ctx) => verticalGradient(ctx.chart, col.rgb, 0.3, 0),
    };
  });
  if (portfolio) {
    datasets.push({
      label: "等权组合",
      data: portfolio.equity,
      borderColor: "#e8edf4",
      borderWidth: 2.4,
      tension: 0.25,
      pointRadius: 0,
      pointHoverRadius: 4,
      pointHoverBackgroundColor: "#e8edf4",
      pointHoverBorderColor: "#05070d",
      fill: true,
      backgroundColor: (ctx) => verticalGradient(ctx.chart, "232, 237, 244", 0.16, 0),
    });
  }
  if (equityChart) equityChart.destroy();
  equityChart = new Chart(canvas.getContext("2d"), {
    type: "line",
    data: { labels, datasets },
    options: EQUITY_OPTIONS,
  });
}

function buildRollingChart(labels, series, windowBars) {
  const canvas = $("btRollingChart");
  if (!canvas) return;
  if (rollingChart) rollingChart.destroy();
  const data = series.rolling_sharpe || [];
  const labelEl = $("btRollingLabel");
  if (labelEl) {
    labelEl.textContent = windowBars
      ? `滚动夏普 · ${windowBars} bars`
      : "滚动夏普 · Rolling Sharpe";
  }
  rollingChart = new Chart(canvas.getContext("2d"), {
    type: "line",
    data: {
      labels,
      datasets: [
        {
          label: "滚动夏普",
          data,
          borderColor: "#fbbf24",
          borderWidth: 1.5,
          tension: 0.2,
          pointRadius: 0,
          pointHoverRadius: 4,
          pointHoverBackgroundColor: "#fbbf24",
          pointHoverBorderColor: "#05070d",
          spanGaps: false,
          fill: {
            target: "origin",
            above: "rgba(251, 191, 36, 0.16)",
            below: "rgba(248, 113, 113, 0.16)",
          },
        },
      ],
    },
    options: ROLLING_OPTIONS,
  });
}

function renderEquity(resp) {
  const live = $("btEquityLive");
  const empty = $("btEquityEmpty");
  const data = resp?.data;
  const symbols = data?.symbols || {};
  const symNames = Object.keys(symbols);

  if (!resp?.available || !symNames.length) {
    if (live) live.hidden = true;
    if (empty) empty.hidden = false;
    destroyEquityCharts();
    btEquitySig = "";
    return;
  }

  const focus = resp.focus_symbol;
  const sig = [focus, data.total_bars, data.n_points, data.rolling_window, symNames.join(",")].join("|") + "|" + btBuster;
  if (sig === btEquitySig && equityChart) return; // 无变化，避免重建闪烁
  btEquitySig = sig;

  if (live) live.hidden = false;
  if (empty) empty.hidden = true;

  const portfolio = data.portfolio || null;
  let mainName, mainSeries;
  if (portfolio) {
    mainName = "等权组合";
    mainSeries = portfolio;
  } else {
    const key = focus && symbols[focus] ? focus : symNames[0];
    mainName = key;
    mainSeries = symbols[key];
  }

  renderEquityStats(mainName, mainSeries);
  buildEquityChart(data.labels, symbols, portfolio);
  buildRollingChart(data.labels, mainSeries, data.rolling_window);

  if ($("btChartsHint")) {
    $("btChartsHint").textContent = `${mainName} · 交互式资金曲线 · 悬停查看数值`;
  }
}

async function startBacktest() {
  if (!selectedStrategyFile) {
    await logClientError("请先选择策略文件");
    return;
  }
  const startBtn = $("btStartBtn");
  if (startBtn) startBtn.disabled = true;
  try {
    const costs = readBacktestCosts();
    const res = await fetchJSON("/api/backtest/start", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        strategy_file: selectedStrategyFile,
        commission_pct: costs.commission_pct,
        slippage_pct: costs.slippage_pct,
      }),
    });
    if (res.strategy_file) renderStrategyFileCard(res.strategy_file);
    await refreshBacktest();
  } catch (e) {
    if ($("btLogHint")) $("btLogHint").textContent = e.message;
    updateBtStartBtn();
  }
}

async function stopBacktest() {
  try {
    await fetchJSON("/api/backtest/stop", { method: "POST" });
    await refreshBacktest();
  } catch (e) {
    if ($("btLogHint")) $("btLogHint").textContent = e.message;
  }
}

// ═══════════════════════════════════════════════════════════════════
// 实时行情分析（信号雷达）
// ═══════════════════════════════════════════════════════════════════
let rtInited = false;
let rtEngineRunning = false;
let rtSources = [];
let rtSourceById = {};
let rtImportedStrategy = null; // {path, name}
let rtGridSig = "";
let rtServerSkew = 0; // server_time - local_now（秒）
let rtCountdownTimer = null;

const RT_DIR = {
  LONG: { label: "↑ 预期上涨", cls: "rt-long", color: "#4ade80" },
  SHORT: { label: "↓ 预期下跌", cls: "rt-short", color: "#f87171" },
  FLAT: { label: "— 先观望", cls: "rt-flat", color: "#7a8a9e" },
};
const RT_STATE_LABEL = {
  pending: "等待首次计算",
  ok: "运行中",
  insufficient: "历史不足",
  error: "错误",
};

/** 把 0~1 强度翻成「把握」白话 */
function rtSizePlain(strength, direction) {
  if (direction === "FLAT" || direction == null) {
    return { size: "没把握" };
  }
  const s = Math.max(0, Math.min(1, Number(strength) || 0));
  let size;
  if (s < 0.2) size = "一点把握";
  else if (s < 0.4) size = "把握不大";
  else if (s < 0.6) size = "一半把握";
  else if (s < 0.8) size = "比较有把握";
  else size = "很有把握";
  return { size };
}

function escHtml(s) {
  return String(s == null ? "" : s).replace(/[&<>"]/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c])
  );
}

function rtClock(ts) {
  if (!ts) return "—";
  return new Date(ts * 1000).toLocaleTimeString();
}

function rtNowSec() {
  return Date.now() / 1000 + rtServerSkew;
}

function rtFmtCountdown(sec) {
  const s = Math.max(0, Math.floor(sec));
  if (s < 60) return `${s}秒`;
  const m = Math.floor(s / 60);
  const rs = s % 60;
  if (m < 60) return `${m}分${String(rs).padStart(2, "0")}秒`;
  const h = Math.floor(m / 60);
  const rm = m % 60;
  if (h < 48) return `${h}小时${rm}分`;
  const d = Math.floor(h / 24);
  return `${d}天${h % 24}小时`;
}

function ensureRtCountdownTimer() {
  if (rtCountdownTimer) return;
  rtCountdownTimer = setInterval(tickRtCountdowns, 1000);
}

function tickRtCountdowns() {
  document.querySelectorAll(".rt-countdown").forEach((el) => {
    if (el.dataset.session === "closed") {
      el.textContent = "休市中";
      return;
    }
    const nxt = Number(el.dataset.nextClose);
    if (!Number.isFinite(nxt) || nxt <= 0) {
      el.textContent = "距离下次判断 —";
      return;
    }
    const left = nxt - rtNowSec();
    el.textContent = left <= 0 ? "即将重新判断…" : `距离下次判断 ${rtFmtCountdown(left)}`;
  });
  const hintCd = $("rtNextHint");
  if (hintCd) {
    if (hintCd.dataset.session === "closed") {
      hintCd.textContent = "休市中";
      return;
    }
    if (hintCd.dataset.nextClose) {
      const nxt = Number(hintCd.dataset.nextClose);
      if (Number.isFinite(nxt) && nxt > 0) {
        const left = nxt - rtNowSec();
        hintCd.textContent =
          left <= 0 ? "即将重新判断" : `距离下次判断 ${rtFmtCountdown(left)}`;
      }
    }
  }
}

async function initRealtimeOnce() {
  if (rtInited) return;
  rtInited = true;
  try {
    const data = await fetchJSON("/api/realtime/sources");
    rtSources = data.sources || [];
    rtSourceById = {};
    rtSources.forEach((s) => (rtSourceById[s.id] = s));
    const sel = $("rtSourceSelect");
    if (sel) {
      sel.innerHTML = rtSources
        .map((s) => `<option value="${s.id}">${escHtml(s.label)}${s.available ? "" : " · 未就绪"}</option>`)
        .join("");
    }
    if (data.min_exposure != null && $("rtThresholdHint")) {
      $("rtThresholdHint").textContent = `无信号阈值 |tanh(因子)| < ${data.min_exposure}`;
    }
    onRtSourceChange();
  } catch (e) {
    await logClientError("加载数据源失败: " + e.message);
  }
  await loadRtStrategies();
  await loadRtFeishuSettings();
}

async function loadRtFeishuSettings() {
  try {
    const data = await fetchJSON("/api/realtime/feishu");
    const en = $("rtFeishuEnabled");
    const wh = $("rtFeishuWebhook");
    const sec = $("rtFeishuSecret");
    if (en) en.checked = !!data.enabled;
    if (wh) wh.value = data.webhook_url || "";
    if (sec) sec.value = data.secret || "";
  } catch (e) {
    const hint = $("rtFeishuHint");
    if (hint) {
      hint.textContent = "加载飞书设置失败: " + e.message;
      hint.classList.add("bad");
    }
  }
}

async function saveRtFeishuSettings() {
  const hint = $("rtFeishuHint");
  const btn = $("rtFeishuSaveBtn");
  if (btn) btn.disabled = true;
  try {
    await fetchJSON("/api/realtime/feishu", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        enabled: !!$("rtFeishuEnabled")?.checked,
        webhook_url: $("rtFeishuWebhook")?.value || "",
        secret: $("rtFeishuSecret")?.value || "",
      }),
    });
    if (hint) {
      hint.textContent = "✓ 已保存，方向转折时会推送到飞书群。";
      hint.classList.remove("bad", "invalid");
      hint.classList.add("valid");
    }
    if (btn) {
      const old = btn.textContent;
      btn.textContent = "已保存";
      setTimeout(() => {
        if (btn.textContent === "已保存") btn.textContent = old || "保存";
      }, 1600);
    }
  } catch (e) {
    if (hint) {
      hint.textContent = "保存失败: " + e.message;
      hint.classList.remove("valid");
      hint.classList.add("bad", "invalid");
    }
  } finally {
    if (btn) btn.disabled = false;
  }
}

async function testRtFeishu() {
  const hint = $("rtFeishuHint");
  const btn = $("rtFeishuTestBtn");
  if (btn) btn.disabled = true;
  try {
    await fetchJSON("/api/realtime/feishu/test", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        webhook_url: $("rtFeishuWebhook")?.value || "",
        secret: $("rtFeishuSecret")?.value || "",
      }),
    });
    if (hint) {
      hint.textContent = "✓ 测试消息已发送，请到飞书群查收。";
      hint.classList.remove("bad", "invalid");
      hint.classList.add("valid");
    }
  } catch (e) {
    if (hint) {
      hint.textContent = "测试失败: " + e.message;
      hint.classList.remove("valid");
      hint.classList.add("bad", "invalid");
    }
  } finally {
    if (btn) btn.disabled = false;
  }
}

function openRtFeishuHelpModal() {
  const modal = $("rtFeishuHelpModal");
  if (modal) modal.hidden = false;
}

function closeRtFeishuHelpModal() {
  const modal = $("rtFeishuHelpModal");
  if (modal) modal.hidden = true;
}

async function loadRtStrategies() {
  const sel = $("rtStrategySelect");
  if (!sel) return;
  let rows = [];
  try {
    const data = await fetchJSON("/api/realtime/strategies");
    rows = data.strategies || [];
  } catch (_) {}
  const opts = ['<option value="">— 选择已保存策略 —</option>'];
  if (rtImportedStrategy) {
    const isym = escHtml(rtImportedStrategy.symbol || "");
    opts.push(
      `<option value="${escHtml(rtImportedStrategy.path)}" data-symbol="${isym}">导入: ${escHtml(rtImportedStrategy.name)}</option>`
    );
  }
  rows.forEach((r) => {
    const score = r.best_score != null ? Number(r.best_score).toFixed(3) : "—";
    const tf = r.timeframe ? ` ${r.timeframe}` : "";
    opts.push(
      `<option value="${escHtml(r.strategy_file)}" data-symbol="${escHtml(r.symbol || "")}">${escHtml(r.symbol)}${tf} · 分数 ${score}</option>`
    );
  });
  const prev = sel.value;
  sel.innerHTML = opts.join("");
  if (rtImportedStrategy) sel.value = rtImportedStrategy.path;
  else if (prev) sel.value = prev;
  onRtStrategyChange();
}

function onRtSourceChange() {
  const src = rtSourceById[$("rtSourceSelect")?.value];
  const tfSel = $("rtTimeframeSelect");
  const presets = $("rtSymbolPresets");
  const hint = $("rtSourceHint");
  if (!src) return;
  if (tfSel) {
    const cur = tfSel.value;
    tfSel.innerHTML = (src.timeframes || []).map((t) => `<option value="${t}">${t}</option>`).join("");
    if (src.timeframes && src.timeframes.includes(cur)) tfSel.value = cur;
    else if (src.timeframes && src.timeframes.includes("1h")) tfSel.value = "1h";
  }
  if (presets) {
    presets.innerHTML = (src.presets || []).map((s) => `<option value="${escHtml(s)}"></option>`).join("");
  }
  if (hint) {
    hint.textContent = `${src.label}：${src.hint || ""}`;
    hint.classList.toggle("bad", !src.available);
  }
}

function rtParseSymbolFromFilename(pathOrName) {
  const name = String(pathOrName || "").split(/[/\\]/).pop() || "";
  let m = name.match(/^best_(.+)\.json$/i);
  if (m) return m[1];
  m = name.match(/^strategy_(.+)_step\d+/i);
  if (m) return m[1];
  return "";
}

function rtApplySymbolFromStrategy(sym) {
  const s = String(sym || "").trim();
  if (!s) return;
  const input = $("rtSymbolInput");
  if (input) input.value = s;
}

function onRtStrategyChange() {
  const sel = $("rtStrategySelect");
  const picked = $("rtStrategyPicked");
  if (!sel || !picked) return;
  const opt = sel.options[sel.selectedIndex];
  picked.textContent = sel.value
    ? `因子来源：${opt ? opt.textContent : sel.value}。信号取最后已收盘 bar。`
    : "因子来源：从已保存策略下拉选择，或「导入策略」选本地 JSON。信号取最后已收盘 bar。";
  if (!sel.value) return;
  const fromOpt = (opt && opt.dataset.symbol) || "";
  const fromImport =
    rtImportedStrategy && sel.value === rtImportedStrategy.path
      ? rtImportedStrategy.symbol || ""
      : "";
  const sym = fromOpt || fromImport || rtParseSymbolFromFilename(sel.value);
  rtApplySymbolFromStrategy(sym);
}

async function rtBrowseStrategy() {
  try {
    const res = await fetchJSON("/api/strategy-file/browse", { method: "POST" });
    if (res.cancelled) return;
    const name = res.filename || res.strategy_file;
    rtImportedStrategy = {
      path: res.strategy_file,
      name,
      symbol: (res.symbol || "").trim() || rtParseSymbolFromFilename(name),
    };
    await loadRtStrategies();
    rtApplySymbolFromStrategy(rtImportedStrategy.symbol);
  } catch (e) {
    await logClientError("导入策略失败: " + e.message);
  }
}

async function rtAddWatch() {
  const source = $("rtSourceSelect")?.value;
  const symbol = ($("rtSymbolInput")?.value || "").trim();
  const timeframe = $("rtTimeframeSelect")?.value;
  const strategy_file = $("rtStrategySelect")?.value;
  const picked = $("rtStrategyPicked");
  if (!symbol) {
    if (picked) { picked.textContent = "请填写品种"; picked.classList.add("bad"); }
    return;
  }
  if (!strategy_file) {
    if (picked) { picked.textContent = "请选择或导入策略因子"; picked.classList.add("bad"); }
    return;
  }
  const btn = $("rtAddBtn");
  if (btn) btn.disabled = true;
  try {
    await fetchJSON("/api/realtime/watch", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ source, symbol, timeframe, strategy_file }),
    });
    if (picked) picked.classList.remove("bad");
    rtEngineRunning = true;
    rtGridSig = "";
    await refreshRealtime();
  } catch (e) {
    if (picked) { picked.textContent = "添加失败: " + e.message; picked.classList.add("bad"); }
  } finally {
    if (btn) btn.disabled = false;
  }
}

async function rtRemoveWatch(id) {
  try {
    await fetchJSON("/api/realtime/unwatch", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ id }),
    });
    rtGridSig = "";
    await refreshRealtime();
  } catch (e) {
    await logClientError("移除监控失败: " + e.message);
  }
}

async function refreshRealtime() {
  let st;
  try {
    st = await fetchJSON("/api/realtime/status", { silent: true });
  } catch (_) {
    return;
  }
  // 有监控项却未在跑时自动拉起（不再提供手动开关）
  if (st.count > 0 && !st.running) {
    try {
      st = await fetchJSON("/api/realtime/start", { method: "POST", silent: true });
    } catch (_) {}
  }
  rtEngineRunning = !!st.running;
  if (typeof st.server_time === "number") {
    rtServerSkew = st.server_time - Date.now() / 1000;
  }

  const nearest = st.nearest_seconds_to_next;
  let nearestClose = null;
  let anyLive = false;
  let anyOk = false;
  for (const w of st.watches || []) {
    if (w.state === "ok") anyOk = true;
    if (w.session_live && w.next_bar_close_at != null) {
      anyLive = true;
      if (nearestClose == null || w.next_bar_close_at < nearestClose) {
        nearestClose = w.next_bar_close_at;
      }
    }
  }

  const hint = $("rtStatusHint");
  if (hint) {
    const base = st.count
      ? `${rtEngineRunning ? "运行中" : "已暂停"} · ${st.count} 项`
      : "暂无监控项";
    if (nearestClose) {
      hint.innerHTML = `${base} · <span id="rtNextHint" data-next-close="${nearestClose}"></span>`;
    } else if (anyOk && !anyLive) {
      hint.innerHTML = `${base} · <span id="rtNextHint" data-session="closed">休市中</span>`;
    } else {
      hint.textContent = base;
    }
  }
  renderRealtimeGrid(st.watches || []);
  ensureRtCountdownTimer();
  tickRtCountdowns();
}

// 半环表盘（180° 上半环，值弧按强度填充）
const RT_ARC_LEN = 150.8; // π * 48
function halfRingGauge(strength, colorHex) {
  const s = Math.max(0, Math.min(1, strength || 0));
  const off = RT_ARC_LEN * (1 - s);
  return `<svg class="rt-gauge-svg" viewBox="0 0 120 74" aria-hidden="true">
    <path class="rt-gauge-track" d="M12 62 A 48 48 0 0 1 108 62" />
    <path class="rt-gauge-val" d="M12 62 A 48 48 0 0 1 108 62"
      style="stroke:${colorHex};stroke-dasharray:${RT_ARC_LEN};stroke-dashoffset:${off.toFixed(1)};" />
  </svg>`;
}

function renderRealtimeGrid(watches) {
  const grid = $("rtGrid");
  if (!grid) return;
  if (!watches.length) {
    grid.innerHTML =
      '<div class="metric-empty">尚无监控项。添加「数据源 + 品种 + 周期 + 因子」后开始实时分析。</div>';
    rtGridSig = "";
    return;
  }

  // 签名：只在信号相关字段变化时重建（避免每次轮询重播动画）
  const sig = watches
    .map((w) =>
      [
        w.id,
        w.state,
        w.direction,
        w.strength,
        w.warn,
        w.message,
        w.last_bar_ts,
        w.updated_at,
        w.session_live ? 1 : 0,
        w.next_bar_close_at || "",
      ].join("~")
    )
    .join("|");
  // 签名未变时仍同步休市/倒计时锚点
  if (sig === rtGridSig) {
    watches.forEach((w) => {
      const el = grid.querySelector(`.rt-card[data-id="${CSS.escape(w.id)}"] .rt-countdown`);
      if (!el) return;
      if (w.session_live && w.next_bar_close_at) {
        el.dataset.session = "";
        el.dataset.nextClose = String(w.next_bar_close_at);
      } else if (w.state === "ok") {
        el.dataset.nextClose = "";
        el.dataset.session = "closed";
      } else {
        el.dataset.nextClose = "";
        el.dataset.session = "";
      }
    });
    return;
  }
  rtGridSig = sig;

  grid.innerHTML = watches
    .map((w) => {
      const dir = w.state === "ok" ? RT_DIR[w.direction] || RT_DIR.FLAT : null;
      const color = dir ? dir.color : "#7a8a9e";
      const strength = w.state === "ok" ? w.strength || 0 : 0;
      const dirKey = w.state === "ok" ? w.direction : null;
      const plain = w.state === "ok" ? rtSizePlain(strength, dirKey) : null;
      const dirLabel = dir ? dir.label : RT_STATE_LABEL[w.state] || w.state;
      const dirCls = dir ? dir.cls : "rt-flat";
      const srcLabel = (rtSourceById[w.source] || {}).label || w.source;
      const factorText = w.factor_value != null ? Number(w.factor_value).toFixed(4) : "—";
      const warn = w.warn ? `<div class="rt-warn" title="${escHtml(w.warn)}">⚠ ${escHtml(w.warn)}</div>` : "";
      const msg =
        w.state !== "ok" && w.message
          ? `<div class="rt-msg">${escHtml(w.message)}</div>`
          : "";
      const sizeText = plain ? plain.size : "—";
      return `
    <div class="rt-card ${dirCls}" data-id="${escHtml(w.id)}">
      <button class="rt-remove" data-remove="${escHtml(w.id)}" title="移除监控">×</button>
      <div class="rt-card-head">
        <span class="rt-sym">${escHtml(w.symbol)}</span>
        <span class="rt-tf">${escHtml(w.timeframe)}</span>
        <span class="rt-src">${escHtml(srcLabel)}</span>
      </div>
      <div class="rt-gauge">
        ${halfRingGauge(strength, color)}
        <div class="rt-gauge-center">
          <div class="rt-strength">${escHtml(sizeText)}</div>
          <div class="rt-dir ${dirCls}">${dirLabel}</div>
        </div>
      </div>
      <div class="rt-meta">
        <span class="rt-meta-item">因子 <b>${factorText}</b></span>
        <span class="rt-meta-item">${escHtml(w.strategy_name)}</span>
      </div>
      <div class="rt-foot">
        <span class="rt-state ${w.state}">${RT_STATE_LABEL[w.state] || w.state}</span>
        <span class="rt-time">更新 ${rtClock(w.updated_at)}</span>
        <span class="rt-countdown"${
          w.session_live && w.next_bar_close_at
            ? ` data-next-close="${w.next_bar_close_at}"`
            : w.state === "ok"
              ? ` data-session="closed"`
              : ""
        }>${
          w.session_live && w.next_bar_close_at
            ? "距离下次判断 …"
            : w.state === "ok"
              ? "休市中"
              : "距离下次判断 —"
        }</span>
      </div>
      ${warn}
      ${msg}
    </div>`;
    })
    .join("");

  runCountUp(grid);
}

function startPolling() {
  if (pollTimer) clearInterval(pollTimer);
  pollTimer = setInterval(() => {
    refreshOverview();
    if (currentPage === "backtest" || btActive) refreshBacktest();
    if (currentPage === "realtime" || rtEngineRunning) refreshRealtime();
  }, 4000);
}

async function init() {
  try {
    await loadConfig();
    await refreshOverview();
  } catch (e) {
    await logClientError("初始化失败: " + e.message);
  }
  $("browseBtn").addEventListener("click", browseDataFile);
  $("startBtn").addEventListener("click", startTraining);
  if ($("retrainBtn")) $("retrainBtn").addEventListener("click", retrainFromScratch);
  $("stopBtn").addEventListener("click", stopTraining);
  $("exportBtn").addEventListener("click", exportStrategy);
  $("exportTrainingBtn").addEventListener("click", exportTraining);
  $("importTrainingBtn").addEventListener("click", triggerImportTraining);
  $("importTrainingFile").addEventListener("change", handleImportTrainingFile);
  $("debugModeCheck").addEventListener("change", (e) => setDebugMode(e.target.checked));
  if ($("aiApiKeyInput")) {
    $("aiApiKeyInput").addEventListener("input", updateAiChannelHint);
    $("aiApiKeyInput").addEventListener("change", updateAiChannelHint);
  }
  if ($("aiAnalyzeBtn")) $("aiAnalyzeBtn").addEventListener("click", runAiAnalyze);
  if ($("aiUnlimitedBtn")) $("aiUnlimitedBtn").addEventListener("click", openUnlimitedModal);
  document.querySelectorAll("[data-close-unlimited]").forEach((el) => {
    el.addEventListener("click", closeUnlimitedModal);
  });
  document.querySelectorAll("[data-close-error]").forEach((el) => {
    el.addEventListener("click", closeErrorPopup);
  });
  if ($("errorModalCopyBtn")) {
    $("errorModalCopyBtn").addEventListener("click", copyErrorPopupDetail);
  }

  // 步骤导航
  document.querySelectorAll(".stepper .step").forEach((btn) => {
    btn.addEventListener("click", () => switchPage(btn.dataset.page));
  });

  // 回测控制
  if ($("btBrowseStrategyBtn")) $("btBrowseStrategyBtn").addEventListener("click", browseStrategyFile);
  if ($("btStartBtn")) $("btStartBtn").addEventListener("click", startBacktest);
  if ($("btStopBtn")) $("btStopBtn").addEventListener("click", stopBacktest);
  ["btCommissionInput", "btSlippageInput"].forEach((id) => {
    const el = $(id);
    if (!el) return;
    el.addEventListener("input", updateBtCostHint);
    el.addEventListener("change", updateBtCostHint);
  });

  // 实时分析控制
  if ($("rtSourceSelect")) $("rtSourceSelect").addEventListener("change", onRtSourceChange);
  if ($("rtStrategySelect")) $("rtStrategySelect").addEventListener("change", onRtStrategyChange);
  if ($("rtBrowseStrategyBtn")) $("rtBrowseStrategyBtn").addEventListener("click", rtBrowseStrategy);
  if ($("rtAddBtn")) $("rtAddBtn").addEventListener("click", rtAddWatch);
  if ($("rtFeishuSaveBtn")) $("rtFeishuSaveBtn").addEventListener("click", saveRtFeishuSettings);
  if ($("rtFeishuTestBtn")) $("rtFeishuTestBtn").addEventListener("click", testRtFeishu);
  if ($("rtFeishuHelpBtn")) $("rtFeishuHelpBtn").addEventListener("click", openRtFeishuHelpModal);
  document.querySelectorAll("[data-close-feishu-help]").forEach((el) => {
    el.addEventListener("click", closeRtFeishuHelpModal);
  });
  if ($("rtGrid")) {
    $("rtGrid").addEventListener("click", (e) => {
      const btn = e.target.closest("[data-remove]");
      if (btn) rtRemoveWatch(btn.dataset.remove);
    });
  }

  startPolling();
}

init();
