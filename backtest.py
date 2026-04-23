"""
Main backtest entry point for the Four Umbrellas Portfolio.

Usage:
    python backtest.py                                       # default: 2005-2024
    python backtest.py --start 2008-01-01 --end 2023-12-31   # custom range
    python backtest.py --monte-carlo --mc-paths 10000        # with MC simulation
    python backtest.py --sensitivity gold --range 0.10 0.25 --step 0.025
    python backtest.py --synthetic                           # synthetic data for plumbing check
    python backtest.py --help

Outputs go to ./output/ by default.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from src.data_loader import load_data, DataBundle
from src.rebalance import simulate_portfolio, simulate_benchmark
from src.options_overlay import simulate_options_overlay
from src.metrics import (
    cumulative_wealth, compute_all, crisis_drawdown, stats_to_dataframe,
)
from src.monte_carlo import block_bootstrap, simulate_wealth_paths, compute_mc_stats
from src.plots import (
    plot_equity_curve, plot_drawdown, plot_underwater,
    plot_rolling_sharpe, plot_rolling_returns, plot_crisis_zoom,
    plot_annual_returns, plot_risk_return_scatter, plot_return_distribution,
    plot_metrics_comparison, plot_correlation_heatmap,
    plot_monte_carlo_fan, plot_mc_distribution, plot_mc_scenarios,
)
from src.report import generate_markdown_report
from src.portfolio import (
    BENCHMARKS, REFERENCE_NAV_EUR, OPTIONS,
)


DEFAULT_START = "2005-01-01"
DEFAULT_END = "2024-12-31"
OUTPUT_DIR = Path(__file__).parent / "output"


def parse_args():
    p = argparse.ArgumentParser(description="Four Umbrellas Portfolio — 20-year backtest")
    p.add_argument("--start", type=str, default=DEFAULT_START, help=f"Start date YYYY-MM-DD (default: {DEFAULT_START})")
    p.add_argument("--end", type=str, default=DEFAULT_END, help=f"End date YYYY-MM-DD (default: {DEFAULT_END})")
    p.add_argument("--synthetic", action="store_true", help="Use synthetic data (plumbing check only; NOT real analysis)")
    p.add_argument("--no-options", action="store_true", help="Disable the options overlay layer")
    p.add_argument("--nav", type=float, default=REFERENCE_NAV_EUR, help=f"Starting NAV in EUR (default: {REFERENCE_NAV_EUR})")
    p.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR), help=f"Output directory (default: {OUTPUT_DIR})")

    # Monte Carlo
    p.add_argument("--monte-carlo", action="store_true", help="Run Monte Carlo block bootstrap simulation")
    p.add_argument("--mc-paths", type=int, default=5000, help="Number of MC paths (default: 5000)")
    p.add_argument("--mc-years", type=int, default=0, help="MC simulation horizon in years (0 = auto, uses max available years in data)")
    p.add_argument("--mc-block-size", type=int, default=3, help="MC block size in months (default: 3)")

    # Sensitivity
    p.add_argument("--sensitivity", type=str, choices=["gold", "dbi", "options_budget", "rebalance_freq"],
                   help="Run sensitivity analysis for a parameter")
    p.add_argument("--range", type=float, nargs=2, metavar=("LOW", "HIGH"), help="Parameter range for sensitivity")
    p.add_argument("--step", type=float, help="Parameter step for sensitivity")

    return p.parse_args()


def run_backtest(bundle: DataBundle, enable_options: bool, start_nav: float, args) -> Dict[str, pd.Series]:
    """Run the main portfolio + benchmarks. Returns dict of name -> monthly return series."""
    print(f"\n{'=' * 70}")
    print(f"Running backtest: {bundle.monthly_returns_eur.index[0].date()} to {bundle.monthly_returns_eur.index[-1].date()}")
    print(f"Starting NAV: €{start_nav:,.0f}")
    print(f"Options overlay: {'ENABLED' if enable_options else 'DISABLED'}")
    print(f"{'=' * 70}\n")

    returns_dict: Dict[str, pd.Series] = {}

    # --- Four Umbrellas (no options) ---
    print("Simulating Four Umbrellas (no options overlay) ...")
    base_returns = simulate_portfolio(
        bundle.monthly_returns_eur,
        bundle.btc_activation_date,
        apply_ter=True,
    )
    returns_dict["Four Umbrellas (no options)"] = base_returns

    # --- Four Umbrellas (with options) ---
    if enable_options:
        print("Simulating options overlay ...")
        # Build NAV series from the no-options portfolio as the reference for option sizing
        nav_series = cumulative_wealth(base_returns, start_nav)
        overlay_returns = simulate_options_overlay(
            spy_daily=bundle.spy_daily,
            qqq_daily=bundle.qqq_daily,
            vix_daily=bundle.vix_daily,
            rf_daily=bundle.rf_daily,
            nav_series=nav_series,
        )
        # Align and combine
        overlay_aligned = overlay_returns.reindex(base_returns.index).fillna(0.0)
        umbrellas_returns = base_returns + overlay_aligned
        returns_dict["Four Umbrellas"] = umbrellas_returns

    # --- Benchmarks ---
    for bench_name, bench_weights in BENCHMARKS.items():
        bench_returns = simulate_benchmark(bundle.benchmark_monthly_eur, bench_weights)
        if len(bench_returns) > 0:
            returns_dict[bench_name] = bench_returns
        else:
            print(f"  (skipping benchmark {bench_name}: required data not available)")

    return returns_dict


def print_summary_table(returns_dict: Dict[str, pd.Series], risk_free_rate: float = 0.02):
    """Print a nicely-formatted summary statistics table and return (df, stats_list)."""
    from tabulate import tabulate

    stats_list = [compute_all(name, r, risk_free_rate) for name, r in returns_dict.items()]
    df = stats_to_dataframe(stats_list)

    # Format for display
    display_df = df.copy()
    pct_cols = ["cagr", "annualized_vol", "max_drawdown", "cvar_5pct", "best_month", "worst_month", "pct_positive_months", "total_return"]
    for c in pct_cols:
        display_df[c] = display_df[c].apply(lambda x: f"{x:.2%}" if isinstance(x, (int, float)) and not pd.isna(x) else x)
    ratio_cols = ["sharpe", "sortino", "calmar", "ulcer_index"]
    for c in ratio_cols:
        display_df[c] = display_df[c].apply(lambda x: f"{x:.2f}" if isinstance(x, (int, float)) and not pd.isna(x) else x)

    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)
    # Transpose for readability
    print(tabulate(display_df.T, headers="keys", tablefmt="github"))

    # Crisis period comparison
    print("\n" + "=" * 70)
    print("CRISIS PERIOD DRAWDOWNS (peak-to-trough within window)")
    print("=" * 70)
    crisis_data = []
    for name, returns in returns_dict.items():
        crisis_data.append({
            "Portfolio": name,
            "GFC 2008-2009": f"{crisis_drawdown(returns, '2008-01-01', '2009-06-30'):.2%}",
            "COVID 2020": f"{crisis_drawdown(returns, '2020-01-01', '2020-12-31'):.2%}",
            "Stagflation 2022": f"{crisis_drawdown(returns, '2022-01-01', '2022-12-31'):.2%}",
        })
    crisis_df = pd.DataFrame(crisis_data).set_index("Portfolio")
    print(tabulate(crisis_df, headers="keys", tablefmt="github"))
    return df, stats_list


def save_outputs(
    returns_dict: Dict[str, pd.Series],
    stats_df: pd.DataFrame,
    stats_list,
    output_dir: Path,
    start_nav: float,
    bundle_monthly_eur: pd.DataFrame = None,
):
    """Save all charts, CSVs, and the Markdown report to output directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nWriting outputs to {output_dir.resolve()}")

    # CSVs
    stats_df.to_csv(output_dir / "summary_statistics.csv")
    pd.DataFrame(returns_dict).to_csv(output_dir / "monthly_returns.csv")

    # Core charts
    print("  - equity_curve.png")
    plot_equity_curve(returns_dict, output_dir / "equity_curve.png", start_nav)
    print("  - drawdown.png")
    plot_drawdown(returns_dict, output_dir / "drawdown.png")
    print("  - underwater.png")
    plot_underwater(returns_dict, output_dir / "underwater.png")
    print("  - rolling_sharpe.png")
    plot_rolling_sharpe(returns_dict, output_dir / "rolling_sharpe.png")
    print("  - rolling_returns.png")
    plot_rolling_returns(returns_dict, output_dir / "rolling_returns.png")
    print("  - crisis_zoom.png")
    plot_crisis_zoom(returns_dict, output_dir / "crisis_zoom.png")

    # Additional comparative charts
    print("  - annual_returns.png")
    plot_annual_returns(returns_dict, output_dir / "annual_returns.png")
    print("  - return_distribution.png")
    plot_return_distribution(returns_dict, output_dir / "return_distribution.png")
    print("  - risk_return_scatter.png")
    plot_risk_return_scatter(stats_list, output_dir / "risk_return_scatter.png")
    print("  - metrics_comparison.png")
    plot_metrics_comparison(stats_list, output_dir / "metrics_comparison.png")

    # Optional: correlation heatmap if underlying sleeve data is available
    if bundle_monthly_eur is not None and not bundle_monthly_eur.empty:
        print("  - correlation_heatmap.png")
        plot_correlation_heatmap(bundle_monthly_eur.dropna(how="all"), output_dir / "correlation_heatmap.png")


def run_monte_carlo(returns_dict: Dict[str, pd.Series], args, output_dir: Path):
    """Run and plot Monte Carlo simulation for the main portfolio. Returns MC stats."""
    # Use the Four Umbrellas portfolio (with options if enabled)
    name = "Four Umbrellas" if "Four Umbrellas" in returns_dict else "Four Umbrellas (no options)"
    returns = returns_dict[name]

    # Auto-detect horizon if --mc-years=0: use the full available monthly
    # history from the returns series (no integer truncation, no floor at 1Y).
    if args.mc_years <= 0:
        n_months = int(returns.dropna().shape[0])
        n_years = n_months / 12.0
    else:
        n_years = args.mc_years
        n_months = n_years * 12
    print(f"\nRunning Monte Carlo: {args.mc_paths:,} paths × {n_years:.2f} years ({n_months} months), block size {args.mc_block_size}")
    simulated = block_bootstrap(
        returns, n_paths=args.mc_paths, n_periods=n_months,
        block_size=args.mc_block_size,
    )
    wealth_paths = simulate_wealth_paths(simulated)

    # Stats
    stats = compute_mc_stats(wealth_paths, n_months)
    print(f"\n{'=' * 70}")
    print(f"MONTE CARLO RESULTS — {name}")
    print(f"{'=' * 70}")
    print(f"Paths simulated:               {stats.n_paths:,}")
    print(f"Horizon:                       {stats.n_years:.0f} years")
    print(f"Terminal wealth (starting €{args.nav:,.0f}):")
    print(f"  5th percentile:              €{stats.pct5_terminal * args.nav:,.0f}")
    print(f"  25th percentile:             €{stats.pct25_terminal * args.nav:,.0f}")
    print(f"  50th percentile (median):    €{stats.median_terminal * args.nav:,.0f}")
    print(f"  75th percentile:             €{stats.pct75_terminal * args.nav:,.0f}")
    print(f"  95th percentile:             €{stats.pct95_terminal * args.nav:,.0f}")
    print(f"Probability of positive nominal return: {stats.prob_positive_real_return:.1%}")
    print(f"Probability of drawdown > 20%: {stats.prob_drawdown_gt_20pct:.1%}")
    print(f"Probability of drawdown > 40%: {stats.prob_drawdown_gt_40pct:.1%}")
    print(f"Median max drawdown:           {stats.median_max_drawdown:.2%}")
    print(f"Worst-5% max drawdown:         {stats.pct5_max_drawdown:.2%}")

    # Save stats CSV — uses .to_dict() to flatten nested scenarios into
    # scalar columns (avoids opaque object serialization)
    pd.DataFrame([stats.to_dict()]).to_csv(output_dir / "monte_carlo_stats.csv", index=False)

    # Charts
    dates = pd.date_range(returns.index[-1], periods=n_months + 1, freq="ME")
    print("  - monte_carlo_fan.png")
    plot_monte_carlo_fan(wealth_paths, dates, output_dir / "monte_carlo_fan.png", start_nav=args.nav)
    print("  - monte_carlo_distribution.png")
    plot_mc_distribution(wealth_paths[:, -1], output_dir / "monte_carlo_distribution.png", start_nav=args.nav)
    if stats.scenarios is not None:
        print("  - monte_carlo_scenarios.png")
        plot_mc_scenarios(stats.scenarios, output_dir / "monte_carlo_scenarios.png", start_nav=args.nav)

    return stats


def run_sensitivity(args):
    """Run sensitivity analysis for one parameter."""
    # Stub — full implementation is straightforward extension.
    # For now prints instructions.
    print("Sensitivity analysis — parameter sweeps are implemented by editing")
    print("`src/portfolio.py` and re-running `python backtest.py` with a different")
    print("`--output-dir` per run, then comparing the summary_statistics.csv files.")
    print()
    print("Example for gold weight (modify WEIGHTS['gold'] and WEIGHTS['equity'] to sum correctly):")
    print(f"  for w in {args.range}: modify WEIGHTS['gold'] = w, re-run, collect results")
    print()
    print("A programmatic sensitivity runner is on the roadmap — see GitHub issues.")


def main():
    args = parse_args()

    # Load data
    print(f"Loading data (synthetic={args.synthetic}) ...")
    bundle = load_data(synthetic=args.synthetic)
    start = pd.Timestamp(args.start)
    end = pd.Timestamp(args.end)
    bundle = bundle.slice(start, end)
    print(f"Loaded {len(bundle.monthly_returns_eur)} months of data")

    if args.sensitivity:
        run_sensitivity(args)
        return

    # Main backtest
    enable_options = not args.no_options
    returns_dict = run_backtest(bundle, enable_options, args.nav, args)

    # Summary
    stats_df, stats_list = print_summary_table(returns_dict)

    # Save outputs (charts + CSVs)
    output_dir = Path(args.output_dir)
    save_outputs(
        returns_dict, stats_df, stats_list, output_dir, args.nav,
        bundle_monthly_eur=bundle.monthly_returns_eur,
    )

    # Monte Carlo
    mc_stats = None
    mc_config = None
    if args.monte_carlo:
        mc_stats = run_monte_carlo(returns_dict, args, output_dir)
        mc_config = {"block_size": args.mc_block_size, "paths": args.mc_paths, "years": args.mc_years}

    # Generate unified Markdown report
    print("\nGenerating output/REPORT.md ...")
    report_path = generate_markdown_report(
        output_dir=output_dir,
        returns_dict=returns_dict,
        stats_list=stats_list,
        start_date=bundle.monthly_returns_eur.index[0],
        end_date=bundle.monthly_returns_eur.index[-1],
        start_nav=args.nav,
        options_enabled=enable_options,
        monte_carlo_stats=mc_stats,
        mc_config=mc_config,
        crisis_drawdown_fn=crisis_drawdown,
    )
    print(f"  Report: {report_path.resolve()}")
    print("\nDone.  Open output/REPORT.md in any Markdown viewer (VS Code, GitHub, Typora) to see the full report with all charts and tables inline.")


if __name__ == "__main__":
    main()
