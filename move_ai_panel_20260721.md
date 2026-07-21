# 移动 AI 分析面板 — 2026-07-21

## Objective
将 AlphaMaster web 端首页（训练页）的「AI 分析当前训练情况」面板移动到「训练说明」面板的上方。

## File Modified
- `D:\cl\AlphaMaster\web\static\index.html`

## Change
在 `page-train` 中调整了 panel 的 DOM 顺序：

### 修改前
1. launch-panel — 启动训练
2. split (chart + log) — 训练曲线 + 训练日志
3. howto-panel — 训练说明
4. strategies-panel — 已保存策略
5. debug-panel — 调试/报错输出
6. ai-panel — AI 分析当前训练情况

### 修改后
1. launch-panel — 启动训练
2. split (chart + log) — 训练曲线 + 训练日志
3. **ai-panel — AI 分析当前训练情况** ← 上移
4. howto-panel — 训练说明
5. strategies-panel — 已保存策略
6. debug-panel — 调试/报错输出

## Method
纯 HTML DOM 顺序调整，将 ai-panel 区块从 debug-panel 之后整体剪切到 howto-panel 之前。无 CSS 或 JS 改动。
