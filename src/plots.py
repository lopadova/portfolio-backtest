"""
Chart generators for the Portfolio Backtest Engine.

All plots use matplotlib (no seaborn dependency) and save as high-DPI PNGs.
Each function takes the minimum required data, produces one chart, saves,
and closes — keeping memory usage bounded for long runs.

Chart catalog
-------------
- plot_equity_curve           — log-scale wealth curve, all portfolios
- plot_drawdown               — running drawdown from peak
- plot_underwater             — subplots showing time under prior peak per portfolio
- plot_rolling_sharpe         — rolling 3-year Sharpe ratio
- plot_rolling_returns        — rolling 3-year total return
- plot_crisis_zoom            — GFC 2008 / COVID 2020 / 2022 peak-to-trough panels
- plot_annual_returns         — grouped bar chart of yearly returns
- plot_risk_return_scatter    — CAGR vs volatility with Max DD as marker size
- plot_return_distribution    — histogram of monthly returns per portfolio
- plot_metrics_comparison     — grouped bar chart of key metrics
- plot_correlation_heatmap    — correlation between sleeves inside the main portfolio
- plot_monte_carlo_fan        — percentile fan chart from MC simulation
- plot_mc_distribution        — terminal wealth histogram from MC
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd

from .metrics import cumulative_wealth


# ============================================================================
# Shared styling
# ============================================================================

COLORS = {
    "Four Umbrellas":                   "#1f77b4",
    "Four Umbrellas (no options)":      "#17becf",
    "60/40":                            "#ff7f0e",
    "100% S&P 500 TR EUR":              "#d62728",
    "100% SWDA (MSCI World EUR)":       "#2ca02c",
    "All-Weather proxy":                "#9467bd",
}


def _apply_style():
    """Apply a consistent matplotlib style across all charts."""
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": "-",
        "grid.linewidth": 0.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "legend.frameon": False,
        "legend.fontsize": 9,
    })


def _color_for(name: str, default: str = "#666666") -> str:
    return COLORS.get(name, default)


def _setup_ax(ax, title: str, xlabel: str = "", ylabel: str = ""):
    ax.set_title(title, pad=10)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="both", which="major", labelsize=9)


def _save(fig, path: Path):
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ============================================================================
# Chart: Equity curve
# ============================================================================

def plot_equity_curve(returns_dict: Dict[str, pd.Series], path: Path, start_nav: float = 100_000.0):
    """Log-scale equity curve of all portfolios, starting from start_nav."""
    _apply_style()
    fig, ax = plt.subplots(figsize=(13, 6.5))
    for name, returns in returns_dict.items():
        wealth = cumulative_wealth(returns, start_nav)
        ax.plot(wealth.index, wealth.values, label=name, linewidth=1.8, color=_color_for(name))
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"€{x:,.0f}"))
    _setup_ax(
        ax,
        f"Equity curve (log scale) — starting NAV = €{start_nav:,.0f}",
        "Date",
        "Portfolio value (€, log scale)",
    )
    ax.legend(loc="upper left")
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    _save(fig, path)


# ============================================================================
# Chart: Drawdown
# ============================================================================

def plot_drawdown(returns_dict: Dict[str, pd.Series], path: Path):
    """Running drawdown from prior all-time high, all portfolios overlaid."""
    _apply_style()
    fig, ax = plt.subplots(figsize=(13, 6.5))
    for name, returns in returns_dict.items():
        wealth = cumulative_wealth(returns)
        running_max = wealth.cummax()
        drawdown = (wealth - running_max) / running_max * 100.0
        color = _color_for(name)
        ax.fill_between(drawdown.index, drawdown.values, 0, alpha=0.12, color=color)
        ax.plot(drawdown.index, drawdown.values, label=name, linewidth=1.5, color=color)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
    ax.axhline(0, color="black", linewidth=0.8)
    _setup_ax(ax, "Drawdown from prior peak", "Date", "Drawdown")
    ax.legend(loc="lower left")
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    _save(fig, path)


# ============================================================================
# Chart: Underwater periods
# ============================================================================

def plot_underwater(returns_dict: Dict[str, pd.Series], path: Path):
    """
    Per-portfolio underwater visualization: one subplot per portfolio,
    filled area showing time below prior all-time high.
    """
    _apply_style()
    n = len(returns_dict)
    fig, axes = plt.subplots(n, 1, figsize=(13, 1.6 * n + 1.0), sharex=True)
    if n == 1:
        axes = [axes]
    for ax, (name, returns) in zip(axes, returns_dict.items()):
        wealth = cumulative_wealth(returns)
        running_max = wealth.cummax()
        underwater_pct = (wealth - running_max) / running_max * 100.0
        color = _color_for(name)
        ax.fill_between(underwater_pct.index, underwater_pct.values, 0, color=color, alpha=0.6)
        ax.axhline(0, color="black", linewidth=0.6)
        ax.set_ylabel(f"{name}\n(%)", fontsize=8, rotation=0, labelpad=60, va="center", ha="right")
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
        ax.grid(True, alpha=0.25)
        y_min = min(underwater_pct.min() * 1.1, -5)
        ax.set_ylim(y_min, 2)
    axes[-1].set_xlabel("Date")
    axes[-1].xaxis.set_major_locator(mdates.YearLocator(2))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.suptitle("Underwater periods (time below prior all-time high)", fontsize=13, fontweight="bold", y=0.995)
    _save(fig, path)


# ============================================================================
# Chart: Rolling Sharpe
# ============================================================================

def plot_rolling_sharpe(returns_dict: Dict[str, pd.Series], path: Path, window_months: int = 36):
    _apply_style()
    fig, ax = plt.subplots(figsize=(13, 6.5))
    any_data = False
    max_len = max((len(r) for r in returns_dict.values()), default=0)
    for name, returns in returns_dict.items():
        rolling_mean = returns.rolling(window_months).mean() * 12
        rolling_std = returns.rolling(window_months).std() * np.sqrt(12)
        rolling_sharpe = rolling_mean / rolling_std
        if rolling_sharpe.notna().any():
            any_data = True
            ax.plot(rolling_sharpe.index, rolling_sharpe.values, label=name, linewidth=1.5, color=_color_for(name))
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axhline(1.0, color="grey", linewidth=0.5, linestyle="--", alpha=0.5)
    _setup_ax(ax, f"Rolling {window_months}-month Sharpe ratio", "Date", "Sharpe (annualized)")
    if any_data:
        ax.legend(loc="lower left")
        ax.xaxis.set_major_locator(mdates.YearLocator(2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    else:
        ax.text(
            0.5, 0.5,
            f"Insufficient data for a {window_months}-month rolling window.\n"
            f"Available: {max_len} months — need at least {window_months + 1}.",
            transform=ax.transAxes, ha="center", va="center",
            fontsize=12, color="#666666",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#fff3cd", edgecolor="#f0ad4e"),
        )
        ax.set_xticks([])
    _save(fig, path)


# ============================================================================
# Chart: Rolling total return
# ============================================================================

def plot_rolling_returns(returns_dict: Dict[str, pd.Series], path: Path, window_months: int = 36):
    _apply_style()
    fig, ax = plt.subplots(figsize=(13, 6.5))
    any_data = False
    max_len = max((len(r) for r in returns_dict.values()), default=0)
    for name, returns in returns_dict.items():
        wealth = cumulative_wealth(returns)
        rolling_total = wealth / wealth.shift(window_months) - 1.0
        if rolling_total.notna().any():
            any_data = True
            ax.plot(rolling_total.index, rolling_total.values * 100, label=name, linewidth=1.5, color=_color_for(name))
    ax.axhline(0, color="black", linewidth=0.8)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
    _setup_ax(ax, f"Rolling {window_months}-month total return", "Date", "Return over window")
    if any_data:
        ax.legend(loc="lower left")
    else:
        ax.text(
            0.5, 0.5,
            f"Insufficient data for a {window_months}-month rolling window.\n"
            f"Available: {max_len} months — need at least {window_months + 1}.",
            transform=ax.transAxes, ha="center", va="center",
            fontsize=12, color="#666666",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#fff3cd", edgecolor="#f0ad4e"),
        )
        ax.set_xticks([])
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    _save(fig, path)


# ============================================================================
# Chart: Crisis zoom
# ============================================================================

def plot_crisis_zoom(returns_dict: Dict[str, pd.Series], path: Path):
    _apply_style()
    crises = [
        ("2008-01-01", "2009-06-30", "GFC (Jan 2008 – Jun 2009)"),
        ("2020-01-01", "2020-12-31", "COVID (2020)"),
        ("2022-01-01", "2022-12-31", "Stagflation (2022)"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), sharey=False)
    for ax, (start, end, title) in zip(axes, crises):
        ax.set_title(title, fontsize=11, fontweight="bold")
        plotted_any = False
        for name, returns in returns_dict.items():
            sliced = returns.loc[start:end]
            if len(sliced) == 0:
                continue
            wealth = cumulative_wealth(sliced, 100.0)
            ax.plot(wealth.index, wealth.values, label=name, linewidth=1.6, color=_color_for(name))
            plotted_any = True
        ax.axhline(100, color="black", linewidth=0.6, linestyle="--")
        ax.set_ylabel("Indexed (100 at start)")
        ax.grid(True, alpha=0.25)
        ax.tick_params(axis="x", rotation=30, labelsize=8)
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        if not plotted_any:
            ax.text(
                0.5, 0.5,
                "No data in this window",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=11, color="#999999", style="italic",
            )
            ax.set_xticks([])
    # Only emit a legend on the first axis if there's at least one plotted line
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        axes[0].legend(loc="lower left", fontsize=8)
    fig.suptitle("Crisis zoom — portfolio behavior in major stress events", fontsize=13, fontweight="bold", y=1.0)
    _save(fig, path)


# ============================================================================
# Chart: Annual returns (grouped bar chart)
# ============================================================================

def plot_annual_returns(returns_dict: Dict[str, pd.Series], path: Path):
    """
    Grouped bar chart: for each year, one bar per portfolio showing that year's
    total return. Colored positive green / negative red per portfolio tint.
    """
    _apply_style()

    # Compute annual returns (compounded monthly returns within each year)
    annual_by_portfolio: Dict[str, pd.Series] = {}
    for name, returns in returns_dict.items():
        grouped = returns.groupby(returns.index.year).apply(lambda r: (1.0 + r).prod() - 1.0)
        annual_by_portfolio[name] = grouped

    df = pd.DataFrame(annual_by_portfolio)
    years = df.index.values
    n_portfolios = len(df.columns)
    width = 0.8 / max(n_portfolios, 1)

    fig, ax = plt.subplots(figsize=(15, 7))
    for i, name in enumerate(df.columns):
        offset = (i - (n_portfolios - 1) / 2) * width
        values = df[name].values * 100
        color = _color_for(name)
        ax.bar(years + offset, values, width=width, label=name, color=color, alpha=0.85, edgecolor="white", linewidth=0.3)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
    _setup_ax(ax, "Annual returns by portfolio", "Year", "Annual return")
    ax.set_xticks(years)
    ax.tick_params(axis="x", rotation=45, labelsize=9)
    ax.legend(loc="upper left", ncol=2)
    _save(fig, path)


# ============================================================================
# Chart: Risk-return scatter
# ============================================================================

def plot_risk_return_scatter(stats_list, path: Path):
    """
    Scatter: x=annualized vol, y=CAGR. Marker size = |Max Drawdown|.
    Portfolio name labels next to each point.
    """
    _apply_style()
    fig, ax = plt.subplots(figsize=(11, 7.5))
    for s in stats_list:
        if np.isnan(s.cagr) or np.isnan(s.annualized_vol):
            continue
        x = s.annualized_vol * 100
        y = s.cagr * 100
        size = abs(s.max_drawdown) * 2500  # scale for visibility
        color = _color_for(s.name)
        ax.scatter(x, y, s=size, color=color, alpha=0.55, edgecolors="black", linewidth=0.8, label=s.name)
        ax.annotate(
            s.name, (x, y),
            xytext=(8, 8), textcoords="offset points", fontsize=9, fontweight="bold",
        )
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=1))
    ax.axhline(0, color="black", linewidth=0.6)
    ax.axvline(0, color="black", linewidth=0.6)
    _setup_ax(
        ax,
        "Risk-return positioning (marker size ∝ |Max Drawdown|)",
        "Annualized volatility",
        "CAGR (annualized)",
    )
    _save(fig, path)


# ============================================================================
# Chart: Monthly return distribution
# ============================================================================

def plot_return_distribution(returns_dict: Dict[str, pd.Series], path: Path):
    """Overlapped histograms of monthly returns for each portfolio."""
    _apply_style()
    fig, ax = plt.subplots(figsize=(13, 6.5))
    bins = np.linspace(-0.20, 0.20, 81)  # -20% to +20%, 80 bins
    for name, returns in returns_dict.items():
        ax.hist(
            returns.dropna().values,
            bins=bins,
            alpha=0.45,
            label=name,
            color=_color_for(name),
            edgecolor="white",
            linewidth=0.2,
        )
    ax.axvline(0, color="black", linewidth=0.8)
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=0))
    _setup_ax(ax, "Monthly return distribution", "Monthly return", "Frequency (months)")
    ax.legend(loc="upper left")
    _save(fig, path)


# ============================================================================
# Chart: Metrics comparison (grouped bar chart)
# ============================================================================

def plot_metrics_comparison(stats_list, path: Path):
    """
    Side-by-side grouped bar chart: CAGR, |Max DD|, Sharpe, Sortino,
    Calmar, Ulcer Index per portfolio.
    """
    _apply_style()

    metric_keys = ["cagr", "annualized_vol", "max_drawdown", "sharpe", "sortino", "calmar"]
    metric_labels = ["CAGR", "Ann. Vol", "|Max DD|", "Sharpe", "Sortino", "Calmar"]

    # Build matrix: rows = portfolios, columns = metrics
    data = []
    names = []
    for s in stats_list:
        names.append(s.name)
        data.append([
            s.cagr * 100,
            s.annualized_vol * 100,
            abs(s.max_drawdown) * 100,
            s.sharpe,
            s.sortino,
            s.calmar,
        ])
    data = np.array(data)

    # Two subplots: left = % metrics (CAGR, Vol, MaxDD), right = ratios
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    n_portfolios = len(names)
    x_pct = np.arange(3)
    x_ratio = np.arange(3)
    width = 0.8 / max(n_portfolios, 1)

    for i, name in enumerate(names):
        offset = (i - (n_portfolios - 1) / 2) * width
        color = _color_for(name)
        axes[0].bar(x_pct + offset, data[i, :3], width=width, label=name, color=color, alpha=0.85, edgecolor="white")
        axes[1].bar(x_ratio + offset, data[i, 3:], width=width, label=name, color=color, alpha=0.85, edgecolor="white")

    axes[0].set_xticks(x_pct)
    axes[0].set_xticklabels(metric_labels[:3])
    axes[0].yaxis.set_major_formatter(mtick.PercentFormatter(decimals=1))
    axes[0].set_title("Percentage metrics", fontsize=12, fontweight="bold")
    axes[0].legend(loc="upper right", fontsize=8)
    axes[0].axhline(0, color="black", linewidth=0.6)
    axes[0].grid(True, alpha=0.25, axis="y")

    axes[1].set_xticks(x_ratio)
    axes[1].set_xticklabels(metric_labels[3:])
    axes[1].set_title("Ratio metrics", fontsize=12, fontweight="bold")
    axes[1].legend(loc="upper right", fontsize=8)
    axes[1].axhline(0, color="black", linewidth=0.6)
    axes[1].grid(True, alpha=0.25, axis="y")

    fig.suptitle("Portfolio metrics comparison", fontsize=13, fontweight="bold", y=1.02)
    _save(fig, path)


# ============================================================================
# Chart: Correlation heatmap between sleeves
# ============================================================================

def plot_correlation_heatmap(returns_df: pd.DataFrame, path: Path):
    """
    Correlation matrix between portfolio sleeves (columns of returns_df).
    Helps visualize diversification within the portfolio.
    """
    _apply_style()
    corr = returns_df.corr()
    n = len(corr)
    fig, ax = plt.subplots(figsize=(max(9, 0.6 * n + 4), max(7, 0.6 * n + 2)))

    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(corr.index, fontsize=9)

    # Annotate each cell with the correlation value
    for i in range(n):
        for j in range(n):
            value = corr.values[i, j]
            color = "white" if abs(value) > 0.6 else "black"
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", color=color, fontsize=8)

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Correlation", rotation=270, labelpad=18)
    ax.set_title("Sleeve correlation matrix (monthly returns)", fontsize=13, fontweight="bold", pad=15)
    _save(fig, path)


# ============================================================================
# Chart: Monte Carlo fan
# ============================================================================

def plot_monte_carlo_fan(mc_paths: np.ndarray, dates: pd.DatetimeIndex, path: Path, start_nav: float = 100_000.0):
    _apply_style()
    fig, ax = plt.subplots(figsize=(13, 6.5))
    percentiles = [5, 25, 50, 75, 95]
    values = {p: np.percentile(mc_paths * start_nav, p, axis=0) for p in percentiles}

    ax.fill_between(dates, values[5], values[95], color="#1f77b4", alpha=0.15, label="5th–95th percentile")
    ax.fill_between(dates, values[25], values[75], color="#1f77b4", alpha=0.30, label="25th–75th percentile")
    ax.plot(dates, values[50], color="#1f77b4", linewidth=2.2, label="Median (50th)")
    ax.axhline(start_nav, color="black", linewidth=0.6, linestyle="--", label=f"Starting NAV (€{start_nav:,.0f})")

    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"€{x:,.0f}"))
    _setup_ax(
        ax,
        f"Monte Carlo simulation — {mc_paths.shape[0]:,} block-bootstrap paths, {(len(dates)-1)/12:.0f} years",
        "Date",
        "Portfolio value (log scale)",
    )
    ax.legend(loc="upper left")
    _save(fig, path)


# ============================================================================
# Chart: Monte Carlo terminal wealth distribution
# ============================================================================

def plot_mc_scenarios(scenarios, path: Path, start_nav: float = 100_000.0):
    """
    3-bar chart for the user-facing Monte Carlo scenarios:
    Prudente (10°) / Mediana (50°) / Ottimista (90°).

    Each bar shows both the terminal portfolio value in € and the implied
    annualized CAGR as an annotation above.
    """
    _apply_style()
    fig, ax = plt.subplots(figsize=(11, 6.5))
    labels = ["Prudente\n(10° percentile)", "Mediana\n(50° percentile)", "Ottimista\n(90° percentile)"]
    wealth_multipliers = [scenarios.prudente_wealth, scenarios.mediana_wealth, scenarios.ottimista_wealth]
    cagr_values = [scenarios.prudente_cagr, scenarios.mediana_cagr, scenarios.ottimista_cagr]
    terminal_euros = [w * start_nav for w in wealth_multipliers]
    colors = ["#d62728", "#1f77b4", "#2ca02c"]  # red, blue, green

    bars = ax.bar(labels, terminal_euros, color=colors, alpha=0.8, edgecolor="black", linewidth=0.8)

    # Reference line at starting NAV
    ax.axhline(start_nav, color="black", linewidth=1.0, linestyle="--", alpha=0.6, label=f"Starting NAV €{start_nav:,.0f}")

    # Annotate each bar with CAGR and terminal value
    y_max = max(terminal_euros) * 1.15
    ax.set_ylim(0, y_max)
    for bar, terminal, cagr_v in zip(bars, terminal_euros, cagr_values):
        height = bar.get_height()
        cagr_pct = cagr_v * 100
        ax.annotate(
            f"€{terminal:,.0f}\n({cagr_pct:+.1f}% CAGR)",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 5), textcoords="offset points",
            ha="center", va="bottom", fontsize=10, fontweight="bold",
        )

    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"€{x:,.0f}"))
    _setup_ax(
        ax,
        f"Monte Carlo scenarios over {scenarios.n_years:.0f} years",
        "",
        "Terminal portfolio value",
    )
    ax.legend(loc="upper left")
    _save(fig, path)


def plot_mc_distribution(terminal_wealth: np.ndarray, path: Path, start_nav: float = 100_000.0):
    _apply_style()
    fig, ax = plt.subplots(figsize=(13, 6.5))
    terminal_nav = terminal_wealth * start_nav
    ax.hist(terminal_nav, bins=80, color="#1f77b4", alpha=0.75, edgecolor="white", linewidth=0.3)
    median = np.median(terminal_nav)
    p05 = np.percentile(terminal_nav, 5)
    p95 = np.percentile(terminal_nav, 95)
    ax.axvline(median, color="black", linewidth=2.2, label=f"Median: €{median:,.0f}")
    ax.axvline(p05, color="#d62728", linewidth=1.8, linestyle="--", label=f"5th pct: €{p05:,.0f}")
    ax.axvline(p95, color="#2ca02c", linewidth=1.8, linestyle="--", label=f"95th pct: €{p95:,.0f}")
    ax.axvline(start_nav, color="grey", linewidth=1.2, linestyle=":", label=f"Starting NAV: €{start_nav:,.0f}")
    ax.xaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"€{x:,.0f}"))
    _setup_ax(ax, "Terminal wealth distribution (Monte Carlo)", "Terminal portfolio value", "Frequency")
    ax.legend(loc="upper right")
    _save(fig, path)
