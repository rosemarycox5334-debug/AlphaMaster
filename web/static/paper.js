/* paper.js — A股虚拟炒股看板逻辑（复用主站设计系统） */
"use strict";

let equityChart = null;
let polling = false;

const $ = (id) => document.getElementById(id);
const pct = (x) => (x == null || !Number.isFinite(x) ? "—" : (x * 100).toFixed(2) + "%");
const num2 = (x) => (x == null || !Number.isFinite(x) ? "—" : x.toFixed(2));
const money = (x) => (x == null || !Number.isFinite(x) ? "—" : "¥" + Math.round(x).toLocaleString());
const signCls = (x) => (x == null || !Number.isFinite(x) ? "" : x > 0 ? "pos" : x < 0 ? "neg" : "");

// ── 指标卡（复用主站 metric-card 类）──────────────────────────
function renderMetrics(m) {
  const grid = $("pMetricGrid");
  if (!m || Object.keys(m).length === 0) {
    grid.innerHTML = '<div class="metric-empty">运行回放后显示总收益、回撤、夏普、超额 alpha 等指标</div>';
    return;
  }
  const cards = [
    { label: "总收益", val: pct(m.total_return), cls: signCls(m.total_return) },
    { label: "最大回撤", val: pct(m.max_drawdown), cls: "neg" },
    { label: "年化收益", val: pct(m.annual_return), cls: signCls(m.annual_return) },
    { label: "夏普比率", val: num2(m.sharpe), cls: signCls(m.sharpe) },
    { label: "沪深300", val: pct(m.benchmark_return), cls: signCls(m.benchmark_return) },
    { label: "超额 α", val: pct(m.excess_return), cls: signCls(m.excess_return) },
    { label: "期末净值", val: money(m.final_nav), cls: "" },
    { label: "交易日数", val: m.days != null ? String(m.days) : "—", cls: "" },
  ];
  grid.innerHTML = cards
    .map(
      (c) => `
    <div class="metric-card ${c.cls}">
      <div class="metric-label">${c.label}</div>
      <div class="metric-value ${c.cls}">${c.val}</div>
    </div>`
    )
    .join("");
}

// ── 净值曲线（Chart.js，主站青绿主色）────────────────────────
function renderEquity(eq) {
  const hint = $("pChartHint");
  if (!eq || eq.length === 0) {
    hint.textContent = "回放完成后展示账户净值走势";
    if (equityChart) { equityChart.destroy(); equityChart = null; }
    return;
  }
  hint.textContent = `${eq.length} 个交易日 · 起始 ¥1,000,000`;
  const labels = eq.map((x) => x[0]);
  const data = eq.map((x) => x[1]);
  const ctx = $("pEquityChart");
  if (equityChart) equityChart.destroy();
  const grad = ctx.getContext("2d").createLinearGradient(0, 0, 0, 320);
  grad.addColorStop(0, "rgba(94, 234, 212, 0.28)");
  grad.addColorStop(1, "rgba(94, 234, 212, 0.01)");
  equityChart = new Chart(ctx, {
    type: "line",
    data: {
      labels,
      datasets: [
        {
          label: "账户净值",
          data,
          borderColor: "#5eead4",
          backgroundColor: grad,
          borderWidth: 2,
          pointRadius: 0,
          fill: true,
          tension: 0.15,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        x: { ticks: { color: "#7a8a9e", maxTicksLimit: 10 }, grid: { color: "rgba(120,190,235,0.06)" } },
        y: {
          ticks: {
            color: "#7a8a9e",
            callback: (v) => "¥" + (v / 10000).toFixed(0) + "万",
          },
          grid: { color: "rgba(120,190,235,0.06)" },
        },
      },
    },
  });
}

// ── 交易流水（买红卖绿，A股习惯）────────────────────────────
function renderTrades(tr) {
  const body = $("pTradeBody");
  $("pTradeHint").textContent = tr && tr.length ? `共 ${tr.length} 笔，显示最近 100 笔` : "—";
  if (!tr || tr.length === 0) {
    body.innerHTML = '<tr class="empty-row"><td colspan="6">暂无交易记录</td></tr>';
    return;
  }
  body.innerHTML = tr
    .slice(-100)
    .reverse()
    .map((t) => {
      const buy = t.side === "BUY";
      const sideCls = buy ? "pos" : "neg";
      const sideTxt = buy ? "买入" : "卖出";
      return `<tr>
        <td>${t.date}</td>
        <td>${t.code}</td>
        <td class="${sideCls}">${sideTxt}</td>
        <td>${t.price.toFixed(2)}</td>
        <td>${t.shares.toLocaleString()}</td>
        <td>${t.cost.toFixed(2)}</td>
      </tr>`;
    })
    .join("");
}

// ── 状态轮询 ────────────────────────────────────────────────
function setPill(active, label) {
  const pill = $("jobPill");
  $("jobPillText").textContent = label;
  pill.classList.toggle("running", active);
}

async function refresh() {
  try {
    const [m, eq, tr] = await Promise.all([
      fetch("/api/paper/metrics").then((r) => r.json()),
      fetch("/api/paper/equity").then((r) => r.json()),
      fetch("/api/paper/trades").then((r) => r.json()),
    ]);
    renderMetrics(m.metrics || {});
    renderEquity(eq.equity || []);
    renderTrades(tr.trades || []);
  } catch (e) {
    console.error("refresh failed", e);
  }
}

async function poll() {
  if (polling) return;
  polling = true;
  try {
    const st = await fetch("/api/paper/status").then((r) => r.json());
    const active = !!st.active;
    const state = st.job ? st.job.state : "idle";
    const labelMap = { idle: "空闲", running: "回放中…", completed: "已完成", failed: "失败", stopped: "已停止" };
    setPill(active, labelMap[state] || state);
    $("pStartBtn").disabled = active;
    $("pStopBtn").disabled = !active;
    $("pMetricHint").textContent = active ? "回放进行中…" : state === "completed" ? "回放完成" : "尚未运行回放";
    await refresh();
    if (active) setTimeout(() => { polling = false; poll(); }, 2000);
    else polling = false;
  } catch (e) {
    polling = false;
    console.error("poll failed", e);
  }
}

// ── 控制按钮 ────────────────────────────────────────────────
$("pStartBtn").addEventListener("click", async () => {
  const body = JSON.stringify({
    start: $("pStart").value.trim(),
    end: $("pEnd").value.trim(),
    sim_start: $("pSimStart").value.trim(),
  });
  try {
    const res = await fetch("/api/paper/replay/start", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body,
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      alert("启动失败：" + (err.detail || res.status));
      return;
    }
    poll();
  } catch (e) {
    alert("启动失败：" + e.message);
  }
});

$("pStopBtn").addEventListener("click", async () => {
  await fetch("/api/paper/replay/stop", { method: "POST" });
  poll();
});

// 初次加载
poll();
