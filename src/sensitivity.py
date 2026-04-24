"""
Programmatic sensitivity sweep — run the backtest multiple times with a single
parameter varied across a range of values, collect summary stats, produce a
chart showing CAGR / MaxDD / Sharpe / Calmar vs parameter value.

Usage from CLI (see backtest.py):
    python backtest.py --sensitivity gold --range 0.10 0.25 --step 0.025
    python backtest.py --sensitivity dbi --range 0.03 0.10 --step 0.01
    python backtest.py --sensitivity options_budget --range 0.0 0.01 --step 0.001
    python backtest.py --sensitivity rebalance_freq --values 1 2 4 12
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, replace as dataclass_replace
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

from . import portfolio as portfolio_cfg
from .data_loader import DataBundle
from .portfolio_model import AssetAllocation, Portfolio
from .rebalance import simulate_portfolio
from .metrics import compute_all, cumulative_wealth


# ============================================================================
# Supported parameters — each with a "setter" that mutates the config dataclasses
# ============================================================================

SUPPORTED_PARAMS = {
    "gold":              "WEIGHTS['gold']",
    "dbi":               "WEIGHTS['dbi']",
    "options_budget":    "OPTIONS.budget_nav_per_year",
    "rebalance_freq":    "REBALANCE.months (per year)",
    "put_write":         "EQUITY['put_write']",
    "nasdaq_top30":      "EQUITY['nasdaq_top30']",
    "momentum":          "EQUITY['momentum']",
    "quality":           "EQUITY['quality']",
}


@dataclass
class SensitivityResult:
    param_name: str
    param_value: float
    cagr: float
    annualized_vol: float
    sharpe: float
    max_drawdown: float
    calmar: float
    ulcer_index: float
    upi: float
    total_return: float


def _apply_param_override(param_name: str, value: float) -> None:
    """
    Mutate the global portfolio config in-place with a single parameter change.
    The rebalance engine reads these at simulation time, so the change takes
    effect on the next simulate_portfolio() call.

    Delta absorption strategy:
      - For macro sleeves (gold, dbi): absorb delta from `cash` (residual sleeve)
        because WEIGHTS["equity"] is a display aggregate not used by the engine
      - For equity-internal changes (put_write, nasdaq_top30, etc.): absorb
        from another equity sleeve, preserving the equity total
      - For non-weight params (options_budget, rebalance_freq): direct mutation

    Raises ValueError if:
      - Value is not a finite number
      - For weight-like params: value is outside [0, 1] or produces a negative
        balance in the absorbing sleeve (cash or alternate equity sleeve)
      - For rebalance_freq: value is not a positive integer divisor of 12
    """
    # Global input validation
    if not isinstance(value, (int, float)) or not np.isfinite(value):
        raise ValueError(f"Sensitivity value must be a finite number, got {value!r}")

    if param_name == "gold":
        if not (0.0 <= value <= 1.0):
            raise ValueError(f"gold weight must be in [0, 1], got {value}")
        delta = value - portfolio_cfg.WEIGHTS["gold"]
        new_cash = portfolio_cfg.WEIGHTS["cash"] - delta
        if new_cash < 0.0:
            raise ValueError(
                f"gold={value} would drive cash negative ({new_cash:.4f}). "
                f"Current cash = {portfolio_cfg.WEIGHTS['cash']:.4f}, "
                f"current gold = {portfolio_cfg.WEIGHTS['gold']:.4f}"
            )
        portfolio_cfg.WEIGHTS["gold"] = value
        portfolio_cfg.WEIGHTS["cash"] = new_cash
    elif param_name == "dbi":
        if not (0.0 <= value <= 1.0):
            raise ValueError(f"dbi weight must be in [0, 1], got {value}")
        delta = value - portfolio_cfg.WEIGHTS["dbi"]
        new_cash = portfolio_cfg.WEIGHTS["cash"] - delta
        if new_cash < 0.0:
            raise ValueError(
                f"dbi={value} would drive cash negative ({new_cash:.4f}). "
                f"Current cash = {portfolio_cfg.WEIGHTS['cash']:.4f}, "
                f"current dbi = {portfolio_cfg.WEIGHTS['dbi']:.4f}"
            )
        portfolio_cfg.WEIGHTS["dbi"] = value
        portfolio_cfg.WEIGHTS["cash"] = new_cash
    elif param_name == "options_budget":
        if not (0.0 <= value <= 1.0):
            raise ValueError(f"options_budget must be in [0, 1] (as fraction of NAV), got {value}")
        portfolio_cfg.OPTIONS.budget_nav_per_year = value
    elif param_name == "rebalance_freq":
        # value must be a positive integer divisor of 12 for evenly-spaced months.
        # Floats that are not exact integers are rejected to avoid silent truncation.
        if not float(value).is_integer():
            raise ValueError(
                f"rebalance_freq must be an integer, got {value} "
                f"(integer truncation would silently change the result)"
            )
        freq = int(value)
        valid_divisors = (1, 2, 3, 4, 6, 12)
        if freq not in valid_divisors:
            raise ValueError(
                f"rebalance_freq must be a divisor of 12 (one of {valid_divisors}), "
                f"got {freq}. Non-divisors would produce unevenly-spaced rebalance months."
            )
        step = 12 // freq
        portfolio_cfg.REBALANCE.months = tuple(range(1, 13, step))[:freq]
    elif param_name in ("put_write", "nasdaq_top30", "momentum", "quality"):
        if not (0.0 <= value <= 1.0):
            raise ValueError(f"{param_name} weight must be in [0, 1], got {value}")
        # Adjust equity breakdown, absorbing delta from another equity sleeve
        # to preserve WEIGHTS["equity"] total
        delta = value - portfolio_cfg.EQUITY[param_name]
        absorb_key = "put_write" if param_name != "put_write" else "nasdaq_top30"
        new_absorb = portfolio_cfg.EQUITY[absorb_key] - delta
        if new_absorb < 0.0:
            raise ValueError(
                f"{param_name}={value} would drive {absorb_key} negative ({new_absorb:.4f}). "
                f"Current {absorb_key} = {portfolio_cfg.EQUITY[absorb_key]:.4f}. "
                f"Reduce the sweep upper bound or use a parameter with more room."
            )
        portfolio_cfg.EQUITY[param_name] = value
        portfolio_cfg.EQUITY[absorb_key] = new_absorb
        # Sanity check: equity total still matches WEIGHTS["equity"]
        equity_total = sum(portfolio_cfg.EQUITY.values())
        if abs(equity_total - portfolio_cfg.WEIGHTS["equity"]) > 0.002:
            raise ValueError(
                f"After applying {param_name}={value}, equity sleeves sum to "
                f"{equity_total:.4f}, expected {portfolio_cfg.WEIGHTS['equity']:.4f}"
            )
    else:
        raise ValueError(f"Unsupported sensitivity parameter: {param_name}. Choose from {list(SUPPORTED_PARAMS)}")


_WEIGHT_PARAMS_PORTFOLIO = {
    "gold", "dbi", "put_write", "nasdaq_top30", "momentum", "quality"
}


def _apply_param_override_on_portfolio(
    portfolio: Portfolio, param_name: str, value: float
) -> Portfolio:
    """Return a NEW Portfolio with ``param_name`` set to ``value`` — without
    mutating the input or any module global. Delta is absorbed by the
    ``cash`` sleeve (analogous to the legacy globals path, which absorbed
    from ``WEIGHTS['cash']``).

    Supported params:
      - weight-like: gold, dbi, put_write, nasdaq_top30, momentum, quality
      - rebalance_freq (positive integer divisor of 12)
      - options_budget (PR7): builds a new OptionsConfig with the override
        and stores it in ``portfolio.options_config``. The caller
        (``run_sensitivity_sweep``) is responsible for actually invoking
        the overlay so the budget change reaches the returns.
    """
    if not isinstance(value, (int, float)) or not np.isfinite(value):
        raise ValueError(f"Sensitivity value must be a finite number, got {value!r}")

    if param_name in _WEIGHT_PARAMS_PORTFOLIO:
        if not (0.0 <= value <= 1.0):
            raise ValueError(f"{param_name} weight must be in [0, 1], got {value}")
        current_weights = {a.key: a.weight for a in portfolio.assets}
        if param_name not in current_weights:
            raise ValueError(
                f"Portfolio {portfolio.name!r} has no asset named {param_name!r}. "
                f"Available assets: {sorted(current_weights)}"
            )
        if "cash" not in current_weights:
            raise ValueError(
                f"Portfolio {portfolio.name!r} has no 'cash' sleeve; cannot "
                f"absorb sensitivity delta. Add an explicit 'cash' allocation "
                f"or sweep a param on the legacy preset (portfolio=None)."
            )
        delta = value - current_weights[param_name]
        new_cash = current_weights["cash"] - delta
        if new_cash < 0.0:
            raise ValueError(
                f"{param_name}={value} would drive cash negative ({new_cash:.4f}). "
                f"Current cash = {current_weights['cash']:.4f}, "
                f"current {param_name} = {current_weights[param_name]:.4f}"
            )
        new_assets = [
            AssetAllocation(
                key=a.key,
                weight=(
                    value if a.key == param_name
                    else new_cash if a.key == "cash"
                    else a.weight
                ),
            )
            for a in portfolio.assets
        ]
        return dataclass_replace(portfolio, assets=new_assets)

    if param_name == "rebalance_freq":
        if not float(value).is_integer():
            raise ValueError(
                f"rebalance_freq must be an integer, got {value}"
            )
        freq = int(value)
        valid_divisors = (1, 2, 3, 4, 6, 12)
        if freq not in valid_divisors:
            raise ValueError(
                f"rebalance_freq must be a divisor of 12 (one of {valid_divisors}), "
                f"got {freq}"
            )
        step = 12 // freq
        new_months = tuple(range(1, 13, step))[:freq]
        return dataclass_replace(portfolio, rebalance_months=new_months)

    if param_name == "options_budget":
        if not (0.0 <= value <= 1.0):
            raise ValueError(
                f"options_budget must be in [0, 1] (as fraction of NAV), got {value}"
            )
        # PR7: build a new OptionsConfig with the budget override. Start
        # from the portfolio's existing options_config when set, otherwise
        # from the OptionsConfig() defaults — never from the live global
        # OPTIONS, so concurrent sweeps remain isolated.
        from .portfolio import OptionsConfig as _OptionsConfig

        base_cfg = portfolio.options_config or _OptionsConfig()
        new_cfg = dataclass_replace(base_cfg, budget_nav_per_year=float(value))
        return dataclass_replace(portfolio, options_config=new_cfg)

    raise ValueError(
        f"Unsupported sensitivity parameter: {param_name}. "
        f"Choose from {sorted(SUPPORTED_PARAMS)}"
    )


def _snapshot_config():
    """Deep-copy of the relevant config dataclasses for restoration after a sweep."""
    return {
        "WEIGHTS": copy.deepcopy(portfolio_cfg.WEIGHTS),
        "EQUITY": copy.deepcopy(portfolio_cfg.EQUITY),
        "OPTIONS": copy.deepcopy(portfolio_cfg.OPTIONS),
        "REBALANCE": copy.deepcopy(portfolio_cfg.REBALANCE),
    }


def _restore_config(snapshot):
    """Restore config from a snapshot."""
    portfolio_cfg.WEIGHTS.clear()
    portfolio_cfg.WEIGHTS.update(snapshot["WEIGHTS"])
    portfolio_cfg.EQUITY.clear()
    portfolio_cfg.EQUITY.update(snapshot["EQUITY"])
    portfolio_cfg.OPTIONS.__dict__.update(snapshot["OPTIONS"].__dict__)
    portfolio_cfg.REBALANCE.__dict__.update(snapshot["REBALANCE"].__dict__)


def run_sensitivity_sweep(
    bundle: DataBundle,
    param_name: str,
    values: List[float],
    portfolio: Optional[Portfolio] = None,
    risk_free_rate: float = 0.02,
    start_nav: float = 100_000.0,
) -> pd.DataFrame:
    """
    Run the portfolio backtest once per value of ``param_name``, collecting
    summary statistics. Returns a DataFrame indexed by ``param_value`` with
    one row per sweep point.

    Two modes, dispatched on ``portfolio``:

    * ``portfolio is None`` (legacy) — mutates ``src.portfolio`` globals
      around each run, then restores them. Identical to pre-PR3 behavior.
    * ``portfolio is not None`` — applies the override to a **copy** of
      the Portfolio per run; NEVER mutates any global. Supports the
      weight-like params (gold, dbi, put_write, nasdaq_top30, momentum,
      quality), ``rebalance_freq``, and (PR7) ``options_budget`` via the
      per-Portfolio ``options_config``.

    When sweeping ``options_budget``, the overlay is included in the
    simulation on BOTH paths — otherwise a budget change wouldn't reach
    the returns. For all other parameters the core portfolio (no overlay)
    is used so the param's allocation impact is isolated.
    """
    if not values:
        raise ValueError(
            "values list is empty — no sweep points to run. "
            "Check --range/--step or --values CLI arguments."
        )

    # When sweeping options_budget (legacy only), include the overlay to
    # actually measure its effect.
    include_overlay = (param_name == "options_budget")

    results: List[SensitivityResult] = []

    if portfolio is None:
        # Legacy path — globals mutation with snapshot/restore
        snapshot = _snapshot_config()
        try:
            for value in values:
                _restore_config(snapshot)  # always start from clean baseline
                _apply_param_override(param_name, value)

                returns = simulate_portfolio(
                    bundle.monthly_returns_eur,
                    bundle.btc_activation_date,
                    apply_ter=True,
                )
                if include_overlay:
                    from .options_overlay import simulate_options_overlay
                    nav_series = cumulative_wealth(returns, start_nav)
                    overlay_returns = simulate_options_overlay(
                        spy_daily=bundle.spy_daily,
                        qqq_daily=bundle.qqq_daily,
                        vix_daily=bundle.vix_daily,
                        rf_daily=bundle.rf_daily,
                        nav_series=nav_series,
                    )
                    returns = returns + overlay_returns.reindex(returns.index).fillna(0.0)

                stats = compute_all("sensitivity_run", returns, risk_free_rate)
                results.append(_stats_to_result(param_name, value, stats))
        finally:
            _restore_config(snapshot)  # always clean up
    else:
        # Generic path — copy-based, no global mutation
        for value in values:
            adjusted = _apply_param_override_on_portfolio(portfolio, param_name, value)
            returns = simulate_portfolio(
                bundle.monthly_returns_eur,
                bundle.btc_activation_date,
                apply_ter=True,
                portfolio=adjusted,
            )
            if include_overlay:
                # PR7: when sweeping options_budget on a custom Portfolio,
                # run the overlay against the adjusted Portfolio's
                # options_config (built by _apply_param_override_on_portfolio
                # with the new budget). Without this, budget changes would
                # produce identical core-only returns and the sweep chart
                # would be a flat line.
                from .options_overlay import simulate_options_overlay

                nav_series = cumulative_wealth(returns, start_nav)
                overlay_returns = simulate_options_overlay(
                    spy_daily=bundle.spy_daily,
                    qqq_daily=bundle.qqq_daily,
                    vix_daily=bundle.vix_daily,
                    rf_daily=bundle.rf_daily,
                    nav_series=nav_series,
                    rebalance_months=adjusted.rebalance_months,
                    options_config=adjusted.options_config,
                )
                returns = returns + overlay_returns.reindex(returns.index).fillna(0.0)
            stats = compute_all("sensitivity_run", returns, risk_free_rate)
            results.append(_stats_to_result(param_name, value, stats))

    df = pd.DataFrame([asdict(r) for r in results]).set_index("param_value")
    return df


def _stats_to_result(param_name: str, value: float, stats) -> SensitivityResult:
    return SensitivityResult(
        param_name=param_name,
        param_value=value,
        cagr=stats.cagr,
        annualized_vol=stats.annualized_vol,
        sharpe=stats.sharpe,
        max_drawdown=stats.max_drawdown,
        calmar=stats.calmar,
        ulcer_index=stats.ulcer_index,
        upi=stats.upi,
        total_return=stats.total_return,
    )


def plot_sensitivity_results(df: pd.DataFrame, param_name: str, path: Path):
    """
    2x2 subplot: CAGR / MaxDD / Sharpe / Calmar vs parameter value.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True)
    fig.suptitle(f"Sensitivity sweep — {param_name}", fontsize=14, fontweight="bold", y=0.995)

    # CAGR (top-left)
    ax = axes[0, 0]
    ax.plot(df.index, df["cagr"] * 100, marker="o", linewidth=2, color="#2ca02c")
    ax.set_title("CAGR", fontweight="bold")
    ax.set_ylabel("CAGR")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=1))
    ax.grid(True, alpha=0.3)

    # Max DD (top-right) — plotted as positive magnitude (|Max DD|) for clarity.
    # compute_all returns max_drawdown as a negative fraction; we take abs()
    # so the axis shows positive %, which matches the "higher bar = worse" intuition.
    ax = axes[0, 1]
    ax.plot(df.index, df["max_drawdown"].abs() * 100, marker="o", linewidth=2, color="#d62728")
    ax.set_title("|Max Drawdown|", fontweight="bold")
    ax.set_ylabel("|Max DD| (magnitude)")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
    ax.grid(True, alpha=0.3)

    # Sharpe (bottom-left)
    ax = axes[1, 0]
    ax.plot(df.index, df["sharpe"], marker="o", linewidth=2, color="#1f77b4")
    ax.set_title("Sharpe ratio", fontweight="bold")
    ax.set_xlabel(f"{param_name} value")
    ax.set_ylabel("Sharpe")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.grid(True, alpha=0.3)

    # Calmar (bottom-right)
    ax = axes[1, 1]
    ax.plot(df.index, df["calmar"], marker="o", linewidth=2, color="#ff7f0e")
    ax.set_title("Calmar ratio (CAGR / |Max DD|)", fontweight="bold")
    ax.set_xlabel(f"{param_name} value")
    ax.set_ylabel("Calmar")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def parse_range_to_values(range_tuple: Optional[tuple], step: Optional[float], values_list: Optional[List[float]]) -> List[float]:
    """
    Convert CLI range/step/values arguments into a list of parameter values.

    Raises ValueError on malformed input:
      - neither --range/--step nor --values provided
      - step <= 0 (would cause numpy to error or loop infinitely)
      - low > high (would silently produce an empty list)
    """
    if values_list is not None and len(values_list) > 0:
        return [float(v) for v in values_list]
    if range_tuple is None or step is None:
        raise ValueError("Either --range LOW HIGH --step N or --values V1 V2 ... must be provided")
    low, high = float(range_tuple[0]), float(range_tuple[1])
    step_f = float(step)
    if step_f <= 0:
        raise ValueError(f"--step must be > 0, got {step_f}")
    if low > high:
        raise ValueError(f"--range: low ({low}) must be <= high ({high})")
    return list(np.arange(low, high + step_f / 2, step_f))
