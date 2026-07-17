"""Market-specific execution constraints shared by training and backtests."""
from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor


MARKETS = {"generic", "a_share"}


def normalize_market(value: str | None) -> str:
    market = (value or "generic").strip().lower()
    aliases = {"a股": "a_share", "ashare": "a_share", "cn": "a_share"}
    market = aliases.get(market, market)
    if market not in MARKETS:
        raise ValueError("市场必须是 generic 或 a_share")
    return market


def _limit_ratio(symbol: str, execution_ts: int) -> float:
    """Board limit ratio; historical ST 5% needs unavailable status data."""
    code = symbol.upper().removeprefix("SH").removeprefix("SZ").removeprefix("BJ")
    if code.startswith(("8", "4", "92")):
        return 0.30
    if code.startswith("688"):
        return 0.20
    if code.startswith(("300", "301")):
        # ChiNext changed from 10% to 20% on 2020-08-24.
        return 0.20 if execution_ts >= 1_598_227_200 else 0.10
    return 0.10


def apply_market_constraints(
    desired: Tensor,
    raw_dict: dict[str, Tensor] | None,
    *,
    market: str = "generic",
    symbols: Sequence[str] | None = None,
) -> Tensor:
    """Convert desired signal positions into positions feasible at next-bar open.

    Position index ``t`` represents an order executed on bar ``t+1`` and earning
    the return from open[t+1] to open[t+2], matching AlphaMaster's target_ret.
    """
    market = normalize_market(market)
    if market != "a_share":
        return desired

    target = desired.clamp(min=0.0, max=1.0)
    if not raw_dict or target.ndim != 2:
        return target

    needed = ("time", "open", "high", "low", "close")
    if any(name not in raw_dict for name in needed):
        return target

    device = target.device
    # Execution rules are discrete and non-differentiable.  Keep their small
    # state machine on CPU to avoid one CUDA synchronization per bar.
    target_cpu = target.detach().cpu()
    raw = {name: raw_dict[name].detach().cpu() for name in needed}
    if raw["open"].ndim == 1:
        raw = {name: value.unsqueeze(0) for name, value in raw.items()}

    n_assets, length = target_cpu.shape
    names = list(symbols or [])
    if len(names) < n_assets:
        names.extend([""] * (n_assets - len(names)))

    # This state machine is intentionally non-differentiable: formula scoring
    # already runs under no_grad and execution feasibility is discrete.
    feasible = target_cpu.clone()
    for n in range(n_assets):
        exec_time = torch.cat((raw["time"][n, 1:], raw["time"][n, -1:]))
        execution_days = torch.div(exec_time, 86_400, rounding_mode="floor")
        is_daily = length < 3 or bool(torch.all(execution_days[1:-1] > execution_days[:-2]))

        if is_daily:
            # T+1 is inherent for one-signal-per-day data: signal t executes on
            # day t+1 and the earliest following sell executes on day t+2.
            # Only limit-locked bars need sequential adjustment.
            previous_close = raw["close"][n]
            exec_high = torch.cat((raw["high"][n, 1:], raw["high"][n, -1:]))
            exec_low = torch.cat((raw["low"][n, 1:], raw["low"][n, -1:]))
            exec_close = torch.cat((raw["close"][n, 1:], raw["close"][n, -1:]))
            code = names[n].upper().removeprefix("SH").removeprefix("SZ").removeprefix("BJ")
            if code.startswith(("8", "4", "92")):
                ratios = torch.full_like(exec_close, 0.30)
            elif code.startswith("688"):
                ratios = torch.full_like(exec_close, 0.20)
            elif code.startswith(("300", "301")):
                ratios = torch.where(
                    exec_time >= 1_598_227_200,
                    torch.full_like(exec_close, 0.20),
                    torch.full_like(exec_close, 0.10),
                )
            else:
                ratios = torch.full_like(exec_close, 0.10)
            tol = 0.002
            valid = previous_close > 0
            one_price = valid & ((exec_high - exec_low) / previous_close.clamp_min(1e-12) <= tol)
            change = exec_close / previous_close.clamp_min(1e-12) - 1.0
            locked_events = torch.nonzero(
                one_price & ((change >= ratios - tol) | (change <= -ratios + tol)),
                as_tuple=False,
            ).flatten().tolist()
            for t in locked_events:
                previous = float(feasible[n, t - 1].item()) if t else 0.0
                wanted = float(target_cpu[n, t].item())
                if float(change[t].item()) >= float(ratios[t].item()) - tol and wanted > previous:
                    feasible[n, t] = previous
                elif float(change[t].item()) <= -float(ratios[t].item()) + tol and wanted < previous:
                    feasible[n, t] = previous
            continue

        held = 0.0
        locked_buys: list[tuple[int, float]] = []
        for t in range(length):
            exec_idx = min(t + 1, length - 1)
            exec_day = int(raw["time"][n, exec_idx].item()) // 86_400
            locked_buys = [(day, qty) for day, qty in locked_buys if day >= exec_day]
            locked = sum(qty for _, qty in locked_buys)

            wanted = float(target_cpu[n, t].item())
            previous_close = float(raw["close"][n, max(exec_idx - 1, 0)].item())
            high_px = float(raw["high"][n, exec_idx].item())
            low_px = float(raw["low"][n, exec_idx].item())
            close_px = float(raw["close"][n, exec_idx].item())
            ratio = _limit_ratio(names[n], int(raw["time"][n, exec_idx].item()))
            tol = 0.002
            one_price = previous_close > 0 and (high_px - low_px) / previous_close <= tol
            change = close_px / previous_close - 1.0 if previous_close > 0 else 0.0
            limit_up = one_price and change >= ratio - tol
            limit_down = one_price and change <= -ratio + tol

            if wanted > held:
                if not limit_up:
                    bought = wanted - held
                    held = wanted
                    locked_buys.append((exec_day, bought))
            elif wanted < held:
                sell_floor = min(held, locked)
                if not limit_down:
                    held = max(wanted, sell_floor)

            feasible[n, t] = held
    return feasible.to(device)
