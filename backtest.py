"""
Main CLI entry point for the Portfolio Backtest Engine.

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
import tomllib
from pathlib import Path
from typing import Dict, List

# Run the environment check BEFORE importing any third-party package —
# otherwise users who forget to activate the venv get a raw
# `ModuleNotFoundError` traceback instead of guidance.
from src.env_check import require_runtime_deps, load_dotenv  # stdlib-only
require_runtime_deps(
    ["pandas", "numpy", "matplotlib", "scipy"],
    script_name="backtest.py",
)
# Optional: load API keys from a .env file at the project root if --ai-analysis
# is used. Real env vars take precedence; no-op when .env is absent.
load_dotenv()

import numpy as np
import pandas as pd

from src.data_loader import load_data, DataBundle
from src.portfolio_model import Portfolio, list_available_presets
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


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Portfolio Backtest Engine — multi-asset 20-year backtest")
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
    p.add_argument("--sensitivity", type=str,
                   choices=["gold", "dbi", "options_budget", "rebalance_freq",
                            "put_write", "nasdaq_top30", "momentum", "quality"],
                   help="Run sensitivity analysis for a parameter")
    p.add_argument("--range", type=float, nargs=2, metavar=("LOW", "HIGH"), help="Parameter range for sensitivity")
    p.add_argument("--step", type=float, help="Parameter step for sensitivity")
    p.add_argument("--values", type=float, nargs="+", help="Explicit parameter values (alternative to --range/--step)")

    # Rolling-window backtest (Phase 5)
    p.add_argument("--rolling-window", action="store_true", help="Run rolling-window backtest")
    p.add_argument("--window-years", type=int, default=10, help="Rolling window size in years (default: 10)")
    p.add_argument("--step-months", type=int, default=1, help="Step between windows in months (default: 1 = monthly)")

    # Efficient frontier (Phase 6)
    p.add_argument("--efficient-frontier", action="store_true", help="Compute and plot the efficient frontier")
    p.add_argument("--n-random", type=int, default=50_000, help="Random portfolios for frontier cloud (default: 50k)")

    # Portfolio selection (PR2)
    p.add_argument("--portfolio", type=str, default=None,
        help="Portfolio to run: NAME (resolves to portfolios/<name>.toml), "
             "PATH (ends in .toml or contains a / or \\), or inline JSON starting with '{'. "
             "When omitted, the CLI defaults to the 'four_umbrellas' preset "
             "(argparse default is None; defaulting is applied by "
             "_resolve_portfolio_or_exit()).")
    p.add_argument("--list-portfolios", action="store_true",
        help="List available preset portfolios (portfolios/*.toml) and exit.")

    return p.parse_args(argv)


def run_backtest(
    bundle: DataBundle,
    enable_options: bool,
    start_nav: float,
    args,
    portfolio: Portfolio | None = None,
) -> Dict[str, pd.Series]:
    """Run the main portfolio + benchmarks. Returns dict of name -> monthly return series.

    If ``portfolio`` is None the engine falls back to the legacy Four Umbrellas
    preset (built from module globals) — preserves pre-PR2 behavior exactly.
    Otherwise the generic engine uses the provided Portfolio and the
    ``returns_dict`` keys mirror the portfolio's display name."""
    print(f"\n{'=' * 70}")
    print(f"Running backtest: {bundle.monthly_returns_eur.index[0].date()} to {bundle.monthly_returns_eur.index[-1].date()}")
    print(f"Starting NAV: €{start_nav:,.0f}")
    print(f"Options overlay: {'ENABLED' if enable_options else 'DISABLED'}")
    if portfolio is not None:
        print(f"Portfolio:       {portfolio.name} ({len(portfolio.assets)} assets)")
    print(f"{'=' * 70}\n")

    returns_dict: Dict[str, pd.Series] = {}

    # Determine the label used in returns_dict / charts / reports.
    # For the legacy Four Umbrellas preset the label is kept identical to
    # pre-PR2 output so numeric diffs against main stay zero; for any other
    # portfolio the Portfolio.name is used as-is.
    portfolio_label = (
        portfolio.name if portfolio is not None and portfolio.name != "Four Umbrellas"
        else "Four Umbrellas"
    )

    # --- No-options run ---
    print(f"Simulating {portfolio_label} (no options overlay) ...")
    base_returns = simulate_portfolio(
        bundle.monthly_returns_eur,
        bundle.btc_activation_date,
        apply_ter=True,
        portfolio=portfolio,
    )
    returns_dict[f"{portfolio_label} (no options)"] = base_returns

    # --- With options overlay ---
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
        returns_dict[portfolio_label] = umbrellas_returns

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


def run_sensitivity(args, bundle=None):
    """Run sensitivity analysis for one parameter."""
    from src.sensitivity import run_sensitivity_sweep, plot_sensitivity_results, parse_range_to_values

    if bundle is None:
        bundle = load_data(synthetic=args.synthetic)
        start = pd.Timestamp(args.start)
        end = pd.Timestamp(args.end)
        bundle = bundle.slice(start, end)

    values = parse_range_to_values(args.range, args.step, args.values)
    print(f"\nRunning sensitivity sweep on parameter '{args.sensitivity}':")
    print(f"  Values: {[f'{v:.4f}' for v in values]}")

    df = run_sensitivity_sweep(bundle, args.sensitivity, values)

    output_dir = Path(args.output_dir) / "sensitivity"
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{args.sensitivity}.csv"
    png_path = output_dir / f"{args.sensitivity}.png"

    df.to_csv(csv_path)
    plot_sensitivity_results(df, args.sensitivity, png_path)

    print(f"\nResults written to:")
    print(f"  {csv_path.resolve()}")
    print(f"  {png_path.resolve()}\n")

    # Also print a summary table
    from tabulate import tabulate
    pct_cols = ["cagr", "annualized_vol", "max_drawdown", "total_return"]
    display_df = df.copy()
    for c in pct_cols:
        display_df[c] = display_df[c].apply(lambda x: f"{x:.2%}")
    for c in ["sharpe", "calmar", "ulcer_index", "upi"]:
        display_df[c] = display_df[c].apply(lambda x: f"{x:.2f}")
    print(tabulate(display_df.drop(columns=["param_name"]), headers="keys", tablefmt="github"))


def _print_preset_listing() -> None:
    """Implementation of ``--list-portfolios``: enumerate presets and exit."""
    entries = list_available_presets()
    if not entries:
        print("No presets found in portfolios/ directory.")
        return
    print(f"{'name':<30} {'assets':>6}  description")
    print("-" * 70)
    for e in entries:
        print(f"{e['name']:<30} {e['n_assets']:>6}  {e['notes'] or e['display_name']}")


_DEFAULT_PORTFOLIO_NAME = "four_umbrellas"


def _resolve_portfolio_or_exit(spec: str | None) -> Portfolio:
    """Resolve the ``--portfolio`` CLI spec to a Portfolio, or exit(2) with a
    helpful error on failure. Default is the 'four_umbrellas' preset.

    Catches every expected parser/validation error: ``FileNotFoundError``
    (bad path / missing preset), ``ValueError`` (malformed JSON, bad
    weights, missing required fields), ``tomllib.TOMLDecodeError``
    (corrupt preset file). Anything else is a genuine bug and bubbles up.
    """
    spec = spec if spec is not None else _DEFAULT_PORTFOLIO_NAME
    try:
        return Portfolio.resolve(spec)
    except (FileNotFoundError, ValueError, tomllib.TOMLDecodeError) as e:
        print(f"ERROR loading portfolio spec: {e}", file=sys.stderr)
        sys.exit(2)


def _is_default_portfolio_spec(spec: str | None) -> bool:
    """True iff the CLI spec selects the shipped default preset. Uses the
    spec string (argparse input), not ``portfolio.name``, because display
    names are user-controlled: a malicious or careless custom preset could
    set ``name = "Four Umbrellas"`` and bypass a name-based check."""
    if spec is None:
        return True
    s = spec.strip()
    # Bare name match
    if s == _DEFAULT_PORTFOLIO_NAME:
        return True
    # Explicit path to the shipped preset (relative or absolute)
    if s.endswith(".toml"):
        try:
            resolved = Path(s).resolve()
        except OSError:
            return False
        default_path = (
            Path(__file__).resolve().parent
            / "portfolios"
            / f"{_DEFAULT_PORTFOLIO_NAME}.toml"
        )
        return resolved == default_path
    return False


def _warn_if_custom_portfolio_with_advanced_analysis(args) -> None:
    """PR2 coexistence warning. ``--sensitivity`` / ``--rolling-window`` /
    ``--efficient-frontier`` still read from the src.portfolio globals in
    this phase (PR3 wires them to accept a Portfolio). If the user combines
    a custom portfolio with one of these flags, tell them clearly that the
    advanced analysis will reflect the DEFAULT preset, not their custom
    portfolio, so they're not silently misled.

    Detection is based on the CLI spec (``args.portfolio``), not on the
    Portfolio's display name — see ``_is_default_portfolio_spec``.
    """
    spec = getattr(args, "portfolio", None)
    is_custom = not _is_default_portfolio_spec(spec)
    advanced_requested = any([
        args.sensitivity, args.rolling_window, args.efficient_frontier,
    ])
    if is_custom and advanced_requested:
        print(
            "\nWARNING: --sensitivity / --rolling-window / --efficient-frontier "
            f"currently use the Four Umbrellas preset, not your custom portfolio "
            f"({spec!r}). This limitation will be lifted in PR3. "
            "The main backtest (no-options / options overlay / benchmarks) "
            "DOES use your custom portfolio.\n",
            file=sys.stderr,
        )


def main():
    args = parse_args()

    # Short-circuit: --list-portfolios just prints and exits.
    if args.list_portfolios:
        _print_preset_listing()
        return

    # Resolve the requested portfolio (default: four_umbrellas preset).
    portfolio = _resolve_portfolio_or_exit(args.portfolio)
    _warn_if_custom_portfolio_with_advanced_analysis(args)

    # Load data
    print(f"Loading data (synthetic={args.synthetic}) ...")
    bundle = load_data(synthetic=args.synthetic)
    start = pd.Timestamp(args.start)
    end = pd.Timestamp(args.end)
    bundle = bundle.slice(start, end)
    print(f"Loaded {len(bundle.monthly_returns_eur)} months of data")

    if args.sensitivity:
        run_sensitivity(args, bundle=bundle)
        return

    if args.efficient_frontier:
        from src.efficient_frontier import (
            run_efficient_frontier, plot_efficient_frontier,
            plot_efficient_frontier_interactive,
        )
        print(f"\nRunning efficient frontier analysis ({args.n_random:,} random portfolios + Markowitz)...")
        # Note: the Four Umbrellas reference is now computed inside
        # run_efficient_frontier() on the *same* sleeve universe as the
        # frontier — no need to pass a full-portfolio return series (which
        # would include pension/cash effects that aren't comparable).
        random_eval_df, weights_matrix, key_portfolios, frontier_df, fu_point, asset_names = run_efficient_frontier(
            bundle, n_random=args.n_random,
        )
        output_dir = Path(args.output_dir) / "efficient_frontier"
        output_dir.mkdir(parents=True, exist_ok=True)
        random_eval_df.to_csv(output_dir / "random_portfolios.csv", index=False)
        frontier_df.to_csv(output_dir / "frontier_line.csv", index=False)
        # Static PNG
        plot_efficient_frontier(
            random_eval_df, key_portfolios, frontier_df, fu_point,
            output_dir / "efficient_frontier.png",
        )
        # Interactive Plotly HTML — click/hover any point to see its full allocation
        html_path = output_dir / "efficient_frontier_interactive.html"
        if plot_efficient_frontier_interactive(
            random_eval_df, weights_matrix, asset_names,
            key_portfolios, frontier_df, fu_point, html_path,
        ):
            print(f"  + interactive HTML: {html_path.name} (open in browser to click points)")
        print("\nKey portfolios found:")
        for key, point in key_portfolios.items():
            print(f"  {point.label:15s}: "
                  f"return {point.expected_return*100:6.2f}%, "
                  f"vol {point.volatility*100:5.2f}%, "
                  f"Sharpe {point.sharpe:.3f}")
        if fu_point is not None:
            print(f"  {'Four Umbrellas':15s}: "
                  f"return {fu_point.expected_return*100:6.2f}%, "
                  f"vol {fu_point.volatility*100:5.2f}%, "
                  f"Sharpe {fu_point.sharpe:.3f}")
        print(f"\nOutputs in {output_dir.resolve()}")
        return

    if args.rolling_window:
        from src.rolling_window import run_rolling_backtest, plot_rolling_window_results, summary_statistics
        print(f"\nRunning rolling {args.window_years}-year window backtest (step: {args.step_months} months)...")
        df = run_rolling_backtest(bundle, window_years=args.window_years, step_months=args.step_months)
        output_dir = Path(args.output_dir) / "rolling_window"
        output_dir.mkdir(parents=True, exist_ok=True)
        csv_path = output_dir / f"rolling_{args.window_years}y.csv"
        png_path = output_dir / f"rolling_{args.window_years}y.png"
        df.to_csv(csv_path)
        plot_rolling_window_results(df, args.window_years, png_path, step_months=args.step_months)
        stats = summary_statistics(df)
        print(f"\nSummary across {stats['n_windows']} windows:")
        print(f"  CAGR: median {stats['cagr_median']:.2%}, worst {stats['cagr_worst']:.2%}, best {stats['cagr_best']:.2%}")
        print(f"  Max DD: median {stats['maxdd_median']:.2%}, worst {stats['maxdd_worst']:.2%}")
        print(f"  Sharpe: median {stats['sharpe_median']:.2f}, worst {stats['sharpe_worst']:.2f}")
        print(f"  % windows with positive CAGR: {stats['pct_windows_positive_cagr']:.1%}")
        print(f"  % windows with CAGR > 4%: {stats['pct_windows_cagr_above_4pct']:.1%}")
        print(f"\nOutputs: {csv_path.name}, {png_path.name} in {output_dir.resolve()}")
        return

    # Main backtest
    # Honor the portfolio's own options_overlay flag unless the user explicitly
    # forced it off with --no-options (which overrides everything for backward
    # compat). A preset with options_overlay=false can still opt back in only
    # by editing the preset file, never via CLI — the CLI only has the "off"
    # switch (--no-options).
    enable_options = (not args.no_options) and portfolio.options_overlay
    returns_dict = run_backtest(
        bundle, enable_options, args.nav, args, portfolio=portfolio
    )

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
