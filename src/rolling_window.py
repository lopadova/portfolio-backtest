"""
Rolling-window backtest — run the full portfolio simulation on every N-year
rolling window across the full data range, to stress-test conclusions across
different starting dates.

Answers questions like: "If I had started the strategy in 2006, or 2008, or 2011,
what would my 10-year outcome have been?" — giving a distribution of outcomes
rather than a single point estimate.

Usage from CLI (see backtest.py):
    python backtest.py --rolling-window --window-years 10 --step-months 1  # monthly (default)
    python backtest.py --rolling-window --window-years 15 --step-months 12 # annual
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mtick

from typing import Optional

from .data_loader import DataBundle
from .portfolio_model import Portfolio
from .rebalance import simulate_portfolio
from .metrics import compute_all


@dataclass
class WindowResult:
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    cagr: float
    annualized_vol: float
    sharpe: float
    max_drawdown: float
    calmar: float
    ulcer_index: float
    total_return: float


def run_rolling_backtest(
    bundle: DataBundle,
    portfolio: Optional[Portfolio] = None,
    window_years: int = 10,
    step_months: int = 1,
    risk_free_rate: float = 0.02,
) -> pd.DataFrame:
    """
    Run the portfolio simulation on every rolling window of ``window_years``
    years, stepping by ``step_months`` months. Returns a DataFrame indexed by
    window start date with summary statistics.

    ``portfolio`` is forwarded to ``simulate_portfolio`` as the ``portfolio=``
    kw-only arg. When ``None`` (legacy behavior), the engine uses the
    src.portfolio globals (Four Umbrellas preset). When provided, every
    window is simulated against that Portfolio.

    Raises ValueError if:
      - window_years is not a positive integer
      - step_months is not a positive integer (would cause infinite loop)
      - data range is strictly shorter than the window
    """
    # Explicitly reject bool (subclass of int in Python — would silently
    # treat True as 1 and False as 0, contradicting "positive integer")
    if isinstance(window_years, bool) or not isinstance(window_years, int) or window_years <= 0:
        raise ValueError(
            f"window_years must be a positive integer, got {window_years!r}"
        )
    if isinstance(step_months, bool) or not isinstance(step_months, int) or step_months <= 0:
        raise ValueError(
            f"step_months must be a positive integer, got {step_months!r} "
            f"(non-positive would produce an infinite loop)"
        )

    monthly_returns = bundle.monthly_returns_eur
    btc_activation = bundle.btc_activation_date

    window_months = window_years * 12
    total_months = len(monthly_returns)
    # Reject only if data is STRICTLY shorter than the window —
    # total_months == window_months is valid and yields exactly 1 window.
    if total_months < window_months:
        raise ValueError(
            f"Data range ({total_months} months) is shorter than the window "
            f"({window_months} months). Use a shorter window."
        )

    results: List[WindowResult] = []
    start_idx = 0
    while start_idx + window_months <= total_months:
        end_idx = start_idx + window_months
        window_data = monthly_returns.iloc[start_idx:end_idx]
        returns = simulate_portfolio(
            window_data, btc_activation, apply_ter=True, portfolio=portfolio,
        )
        stats = compute_all("window", returns, risk_free_rate)
        results.append(WindowResult(
            start_date=window_data.index[0],
            end_date=window_data.index[-1],
            cagr=stats.cagr,
            annualized_vol=stats.annualized_vol,
            sharpe=stats.sharpe,
            max_drawdown=stats.max_drawdown,
            calmar=stats.calmar,
            ulcer_index=stats.ulcer_index,
            total_return=stats.total_return,
        ))
        start_idx += step_months

    df = pd.DataFrame([asdict(r) for r in results])
    df.set_index("start_date", inplace=True)
    return df


def _describe_step(step_months: int) -> str:
    """Produce a human-readable label for the step cadence (monthly / quarterly / annual / etc)."""
    if step_months == 1:
        return "monthly"
    if step_months == 3:
        return "quarterly"
    if step_months == 6:
        return "semi-annually"
    if step_months == 12:
        return "annually"
    return f"every {step_months} months"


def plot_rolling_window_results(df: pd.DataFrame, window_years: int, path: Path, step_months: int):
    """
    Three-panel chart showing the distribution of CAGR / Max DD / Sharpe
    across all rolling windows.

    step_months: REQUIRED (no default). Passed through from the caller to
    make the chart title honest ("stepped monthly" vs "quarterly" vs etc.).
    Defaulting this silently would reintroduce the hardcoded-cadence bug
    if a caller forgets to pass the actual step used to generate `df`.
    """
    fig, axes = plt.subplots(3, 1, figsize=(13, 11), sharex=True)
    fig.suptitle(
        f"Rolling {window_years}-year window backtest — "
        f"{len(df)} windows, stepped {_describe_step(step_months)}",
        fontsize=13, fontweight="bold", y=0.995,
    )

    # CAGR (top)
    ax = axes[0]
    ax.plot(df.index, df["cagr"] * 100, marker=".", linewidth=1.2, color="#2ca02c", alpha=0.8)
    ax.fill_between(df.index, df["cagr"] * 100, 0, alpha=0.15, color="#2ca02c")
    ax.axhline(df["cagr"].median() * 100, color="#2ca02c", linestyle="--", linewidth=1,
               label=f"Median: {df['cagr'].median() * 100:.2f}%")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=1))
    ax.set_ylabel(f"CAGR over {window_years}Y")
    ax.set_title("CAGR by starting window", fontweight="bold")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    # Max DD (middle)
    ax = axes[1]
    ax.plot(df.index, df["max_drawdown"] * 100, marker=".", linewidth=1.2, color="#d62728", alpha=0.8)
    ax.fill_between(df.index, df["max_drawdown"] * 100, 0, alpha=0.15, color="#d62728")
    ax.axhline(df["max_drawdown"].median() * 100, color="#d62728", linestyle="--", linewidth=1,
               label=f"Median: {df['max_drawdown'].median() * 100:.1f}%")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
    ax.set_ylabel(f"Max DD over {window_years}Y")
    ax.set_title("Max Drawdown by starting window", fontweight="bold")
    ax.legend(loc="lower left")
    ax.grid(True, alpha=0.3)

    # Sharpe (bottom)
    ax = axes[2]
    ax.plot(df.index, df["sharpe"], marker=".", linewidth=1.2, color="#1f77b4", alpha=0.8)
    ax.axhline(df["sharpe"].median(), color="#1f77b4", linestyle="--", linewidth=1,
               label=f"Median: {df['sharpe'].median():.2f}")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_ylabel(f"Sharpe over {window_years}Y")
    ax.set_title("Sharpe ratio by starting window", fontweight="bold")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    axes[-1].set_xlabel("Window start date")

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def summary_statistics(df: pd.DataFrame) -> dict:
    """
    Compute summary statistics over the rolling-window DataFrame.
    Returns a dict with median / worst / best / pct positive, etc.
    """
    return {
        "n_windows": len(df),
        "cagr_median": float(df["cagr"].median()),
        "cagr_worst": float(df["cagr"].min()),
        "cagr_best": float(df["cagr"].max()),
        "cagr_p10": float(df["cagr"].quantile(0.10)),
        "cagr_p90": float(df["cagr"].quantile(0.90)),
        "maxdd_median": float(df["max_drawdown"].median()),
        "maxdd_worst": float(df["max_drawdown"].min()),
        "sharpe_median": float(df["sharpe"].median()),
        "sharpe_worst": float(df["sharpe"].min()),
        "pct_windows_positive_cagr": float((df["cagr"] > 0).mean()),
        "pct_windows_cagr_above_4pct": float((df["cagr"] > 0.04).mean()),
    }
