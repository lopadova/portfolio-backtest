"""
Markdown report generator — bundles all charts + tables into a single
output/REPORT.md file for easy consumption without a data viewer.

The report is designed to be readable directly in any Markdown viewer
(VS Code, GitHub, Typora, Obsidian, or simply a browser with a Markdown
extension). All images are referenced by relative path so the report
is portable as long as the PNGs stay in the same folder.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from .metrics import PortfolioStats


def _fmt_pct(x, decimals: int = 2) -> str:
    if pd.isna(x):
        return "N/A"
    return f"{x * 100:.{decimals}f}%"


def _fmt_ratio(x, decimals: int = 2) -> str:
    if pd.isna(x):
        return "N/A"
    return f"{x:.{decimals}f}"


def _fmt_nav(x: float) -> str:
    return f"€{x:,.0f}"


def _stats_table_markdown(stats_list: List[PortfolioStats]) -> str:
    """Build a Markdown summary stats table."""
    header = (
        "| Portfolio | CAGR | Vol | Sharpe | Sortino | Max DD | Avg DD | Max DD dur. (mo) | Calmar | Ulcer | UPI | CVaR 5% | Worst Mo | Best Mo | % Pos Mo | Total Return | Longest UW (mo) | Recovery (mo) |\n"
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n"
    )
    rows = []
    for s in stats_list:
        rec = "—" if s.recovery_months is None else str(s.recovery_months)
        rows.append(
            f"| **{s.name}** "
            f"| {_fmt_pct(s.cagr)} "
            f"| {_fmt_pct(s.annualized_vol)} "
            f"| {_fmt_ratio(s.sharpe)} "
            f"| {_fmt_ratio(s.sortino)} "
            f"| {_fmt_pct(s.max_drawdown)} "
            f"| {_fmt_pct(s.average_drawdown)} "
            f"| {s.max_drawdown_duration_months} "
            f"| {_fmt_ratio(s.calmar)} "
            f"| {_fmt_ratio(s.ulcer_index, 1)} "
            f"| {_fmt_ratio(s.upi)} "
            f"| {_fmt_pct(s.cvar_5pct)} "
            f"| {_fmt_pct(s.worst_month)} "
            f"| {_fmt_pct(s.best_month)} "
            f"| {_fmt_pct(s.pct_positive_months, 0)} "
            f"| {_fmt_pct(s.total_return, 0)} "
            f"| {s.longest_underwater_months} "
            f"| {rec} |"
        )
    return header + "\n".join(rows)


def _crisis_table_markdown(returns_dict, crisis_drawdown_fn) -> str:
    """Build a Markdown table with peak-to-trough drawdown per crisis period."""
    crises = [
        ("GFC 2008–2009", "2008-01-01", "2009-06-30"),
        ("COVID 2020", "2020-01-01", "2020-12-31"),
        ("Stagflation 2022", "2022-01-01", "2022-12-31"),
    ]
    header = "| Portfolio | GFC 2008–09 | COVID 2020 | Stagflation 2022 |\n|---|---:|---:|---:|\n"
    rows = []
    for name, returns in returns_dict.items():
        row_values = [f"**{name}**"]
        for _, start, end in crises:
            dd = crisis_drawdown_fn(returns, start, end)
            row_values.append(_fmt_pct(dd))
        rows.append("| " + " | ".join(row_values) + " |")
    return header + "\n".join(rows)


def generate_markdown_report(
    output_dir: Path,
    returns_dict,
    stats_list: List[PortfolioStats],
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    start_nav: float,
    options_enabled: bool,
    monte_carlo_stats=None,
    mc_config: Optional[Dict] = None,
    crisis_drawdown_fn=None,
) -> Path:
    """
    Generate output/REPORT.md, a single-file Markdown bundle of every
    chart + table + numerical result from the backtest.

    Returns the path to the written REPORT.md.
    """
    path = output_dir / "REPORT.md"
    lines: List[str] = []

    # --- Header ---
    lines.append("# Four Umbrellas Portfolio — Backtest Report")
    lines.append("")
    lines.append(f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
    lines.append("")
    lines.append("> **Disclaimer.** This report is for educational and informational purposes only. "
                 "Past performance and hypothetical simulations do not guarantee future results. "
                 "The author is not a registered investment advisor. Nothing here is investment advice. "
                 "See the full disclaimer in the accompanying Medium article.")
    lines.append("")

    # --- Run configuration ---
    lines.append("## Run configuration")
    lines.append("")
    lines.append("| Parameter | Value |")
    lines.append("|---|---|")
    lines.append(f"| Backtest period | {start_date.date()} → {end_date.date()} |")
    years = (end_date - start_date).days / 365.25
    lines.append(f"| Duration | {years:.1f} years |")
    lines.append(f"| Reference starting NAV | {_fmt_nav(start_nav)} |")
    lines.append(f"| Options overlay | {'ENABLED' if options_enabled else 'DISABLED'} |")
    lines.append(f"| Number of portfolios compared | {len(returns_dict)} |")
    lines.append("")

    # --- Summary table ---
    lines.append("## Summary statistics")
    lines.append("")
    lines.append(_stats_table_markdown(stats_list))
    lines.append("")
    lines.append(
        "**Metric glossary.** CAGR = compound annual growth rate (a.k.a. Annual Return). "
        "Vol = annualized volatility of monthly returns. Sharpe / Sortino = risk-adjusted returns "
        "(annualized, ~2% risk-free assumed). Max DD = worst peak-to-trough loss. "
        "Avg DD = average drawdown (mean of underwater values — more stable measure than Max DD). "
        "Max DD dur. = duration in months of the single worst peak-to-trough event. "
        "Calmar = CAGR / |Max DD|. Ulcer Index = RMS drawdown depth (Martin 1989). "
        "UPI = Ulcer Performance Index = (CAGR - RF) / Ulcer Index — drawdown-aware Sharpe analog. "
        "CVaR 5% = average return in the worst 5% of months (Expected Shortfall). "
        "Longest UW = longest consecutive months below prior all-time high. "
        "Recovery = months from worst drawdown bottom to new all-time high."
    )
    lines.append("")

    # --- Equity curve ---
    lines.append("## Equity curve")
    lines.append("")
    lines.append("![Equity curve](equity_curve.png)")
    lines.append("")
    lines.append("Log scale. Each portfolio starts from the reference NAV shown above.")
    lines.append("")

    # --- Drawdown & underwater ---
    lines.append("## Drawdown analysis")
    lines.append("")
    lines.append("### Running drawdown from prior peak")
    lines.append("")
    lines.append("![Drawdown](drawdown.png)")
    lines.append("")
    lines.append("### Underwater periods")
    lines.append("")
    lines.append("![Underwater](underwater.png)")
    lines.append("")
    lines.append("Each subplot shows the time each portfolio spent below its prior all-time high. "
                 "Wider filled areas indicate longer recoveries.")
    lines.append("")

    # --- Crisis zoom ---
    lines.append("## Crisis period zoom")
    lines.append("")
    lines.append("![Crisis zoom](crisis_zoom.png)")
    lines.append("")
    if crisis_drawdown_fn is not None:
        lines.append("### Peak-to-trough drawdowns by crisis")
        lines.append("")
        lines.append(_crisis_table_markdown(returns_dict, crisis_drawdown_fn))
        lines.append("")

    # --- Rolling metrics ---
    lines.append("## Rolling metrics")
    lines.append("")
    lines.append("### Rolling 3-year Sharpe ratio")
    lines.append("")
    lines.append("![Rolling Sharpe](rolling_sharpe.png)")
    lines.append("")
    lines.append("### Rolling 3-year total return")
    lines.append("")
    lines.append("![Rolling returns](rolling_returns.png)")
    lines.append("")

    # --- Annual returns ---
    lines.append("## Annual returns")
    lines.append("")
    lines.append("![Annual returns](annual_returns.png)")
    lines.append("")

    # --- Return distribution ---
    lines.append("## Monthly return distribution")
    lines.append("")
    lines.append("![Return distribution](return_distribution.png)")
    lines.append("")

    # --- Risk-return scatter ---
    lines.append("## Risk-return positioning")
    lines.append("")
    lines.append("![Risk-return scatter](risk_return_scatter.png)")
    lines.append("")
    lines.append("Marker size is proportional to the magnitude of Max Drawdown — larger markers "
                 "indicate portfolios that have experienced deeper drawdowns historically.")
    lines.append("")

    # --- Metrics comparison ---
    lines.append("## Metrics comparison")
    lines.append("")
    lines.append("![Metrics comparison](metrics_comparison.png)")
    lines.append("")

    # --- Correlation heatmap (if exists) ---
    corr_path = output_dir / "correlation_heatmap.png"
    if corr_path.exists():
        lines.append("## Sleeve correlation matrix")
        lines.append("")
        lines.append("![Correlation heatmap](correlation_heatmap.png)")
        lines.append("")
        lines.append("Correlations between the monthly returns of each sleeve inside the Four "
                     "Umbrellas portfolio. Low correlations indicate strong diversification; "
                     "high correlations indicate sleeves that move together.")
        lines.append("")

    # --- Monte Carlo ---
    if monte_carlo_stats is not None:
        lines.append("## Monte Carlo simulation")
        lines.append("")
        lines.append(f"**Configuration.** {monte_carlo_stats.n_paths:,} block-bootstrap paths, "
                     f"{monte_carlo_stats.n_years:.0f}-year horizon.")
        if mc_config is not None and mc_config.get("block_size"):
            lines.append(f"Block size: {mc_config['block_size']} months "
                         f"(preserves short-term autocorrelation of returns).")
        lines.append("")
        lines.append("### Scenari espliciti — Prudente / Mediana / Ottimista")
        lines.append("")
        if getattr(monte_carlo_stats, "scenarios", None) is not None:
            sc = monte_carlo_stats.scenarios
            lines.append("![Monte Carlo scenarios](monte_carlo_scenarios.png)")
            lines.append("")
            lines.append("| Scenario | Percentile | Terminal wealth | Implied CAGR |")
            lines.append("|---|---:|---:|---:|")
            lines.append(f"| **Prudente** | 10° | {_fmt_nav(sc.prudente_wealth * start_nav)} | {_fmt_pct(sc.prudente_cagr)} |")
            lines.append(f"| **Mediana** | 50° | {_fmt_nav(sc.mediana_wealth * start_nav)} | {_fmt_pct(sc.mediana_cagr)} |")
            lines.append(f"| **Ottimista** | 90° | {_fmt_nav(sc.ottimista_wealth * start_nav)} | {_fmt_pct(sc.ottimista_cagr)} |")
            lines.append("")
        lines.append("### Fan chart — percentile wealth paths")
        lines.append("")
        lines.append("![Monte Carlo fan](monte_carlo_fan.png)")
        lines.append("")
        lines.append("### Terminal wealth distribution")
        lines.append("")
        lines.append("![Monte Carlo distribution](monte_carlo_distribution.png)")
        lines.append("")
        lines.append("### Monte Carlo statistics")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("|---|---:|")
        lines.append(f"| Median terminal wealth | {_fmt_nav(monte_carlo_stats.median_terminal * start_nav)} |")
        lines.append(f"| 5th percentile terminal | {_fmt_nav(monte_carlo_stats.pct5_terminal * start_nav)} |")
        lines.append(f"| 25th percentile terminal | {_fmt_nav(monte_carlo_stats.pct25_terminal * start_nav)} |")
        lines.append(f"| 75th percentile terminal | {_fmt_nav(monte_carlo_stats.pct75_terminal * start_nav)} |")
        lines.append(f"| 95th percentile terminal | {_fmt_nav(monte_carlo_stats.pct95_terminal * start_nav)} |")
        lines.append(f"| P(positive nominal return) | {_fmt_pct(monte_carlo_stats.prob_positive_real_return, 1)} |")
        lines.append(f"| P(drawdown > 20%) | {_fmt_pct(monte_carlo_stats.prob_drawdown_gt_20pct, 1)} |")
        lines.append(f"| P(drawdown > 40%) | {_fmt_pct(monte_carlo_stats.prob_drawdown_gt_40pct, 1)} |")
        lines.append(f"| Median max drawdown | {_fmt_pct(monte_carlo_stats.median_max_drawdown)} |")
        lines.append(f"| Worst-5% max drawdown | {_fmt_pct(monte_carlo_stats.pct5_max_drawdown)} |")
        lines.append("")

    # --- Footer ---
    lines.append("---")
    lines.append("")
    lines.append("*End of report. Data and code: `github.com/padosoft/four-umbrellas-backtest`. "
                 "MIT license on the code; data licensing varies by provider — see `data/README.md`.*")
    lines.append("")

    # Write
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
