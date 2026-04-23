"""
Standalone FIRE calculator CLI — run FIRE Monte Carlo simulations
without the full portfolio backtest.

Usage:
    python fire.py --age 45 --sex M --capital 100000 --contributions 1000 \\
        --fire-age 60 --spending 2500 --inflation 0.02 \\
        --pension --pension-amount 1500 --pension-age 67 --pension-revaluation 0.75 \\
        --tax-on-withdrawals --simulations 1000

    python fire.py --help
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.data_loader import load_data
from src.rebalance import simulate_portfolio
from src.fire import (
    FireConfig,
    run_fire_simulation,
    aggregate_fire_results,
    plot_fire_projection,
    plot_fire_success_probability,
    plot_fire_failure_distribution,
    plot_fire_legacy_distribution,
)


def parse_args():
    p = argparse.ArgumentParser(description="FIRE Monte Carlo simulator — Italian mortality tables")

    # Personal
    p.add_argument("--age", type=int, required=True, help="Current age (years)")
    p.add_argument("--sex", choices=["M", "F"], default="M")
    p.add_argument("--fixed-end-age", type=int, help="Fixed end age (disables mortality sampling)")

    # Wealth
    p.add_argument("--capital", type=float, required=True, help="Initial capital (€)")
    p.add_argument("--contributions", type=float, default=0, help="Periodic contribution (€)")
    p.add_argument("--frequency", choices=["month", "year"], default="month")

    # FIRE
    p.add_argument("--fire-age", type=int, required=True, help="FIRE target age")
    p.add_argument("--spending", type=float, required=True, help="Monthly spending post-FIRE (€)")
    p.add_argument("--inflation", type=float, default=0.02, help="Annual inflation rate (default 2%)")

    # Pension
    p.add_argument("--pension", action="store_true", help="Enable pension")
    p.add_argument("--pension-amount", type=float, default=0, help="Pension monthly amount (€, 12 months/yr)")
    p.add_argument("--pension-age", type=int, default=67, help="Pension start age")
    p.add_argument("--pension-revaluation", type=float, default=1.0,
                   help="Pension inflation-matching coefficient (1.0 = full, 0.75 = 75%, 0 = no adjustment)")

    # Tax
    p.add_argument("--tax-on-withdrawals", action="store_true", help="Apply 26%% CGT on withdrawals")
    p.add_argument("--tax-rate", type=float, default=0.26)

    # Simulation
    p.add_argument("--simulations", type=int, default=1000, help="Number of Monte Carlo simulations")
    p.add_argument("--block-size", type=int, default=3, help="Bootstrap block size in months")
    p.add_argument("--seed", type=int, default=42)

    # Data
    p.add_argument("--synthetic", action="store_true", help="Use synthetic portfolio data (for testing)")
    p.add_argument("--output-dir", default="output/fire")

    return p.parse_args()


def main():
    args = parse_args()

    config = FireConfig(
        current_age=args.age,
        sex=args.sex,
        fixed_end_age=args.fixed_end_age,
        initial_capital=args.capital,
        contribution_amount=args.contributions,
        contribution_frequency=args.frequency,
        fire_age=args.fire_age,
        monthly_spending=args.spending,
        inflation_rate=args.inflation,
        pension_enabled=args.pension,
        pension_monthly_amount=args.pension_amount,
        pension_start_age=args.pension_age,
        pension_revaluation=args.pension_revaluation,
        tax_on_withdrawals=args.tax_on_withdrawals,
        tax_rate=args.tax_rate,
        n_simulations=args.simulations,
        block_size=args.block_size,
        seed=args.seed,
    )

    print("Loading portfolio data ...")
    bundle = load_data(synthetic=args.synthetic)
    print(f"Running portfolio simulation to obtain return series ({len(bundle.monthly_returns_eur)} months)...")
    portfolio_returns = simulate_portfolio(
        bundle.monthly_returns_eur, bundle.btc_activation_date, apply_ter=True,
    )

    print(f"\nRunning {config.n_simulations:,} FIRE simulations ...")
    results = run_fire_simulation(config, portfolio_returns)
    summary = aggregate_fire_results(results, config)

    # Output
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("FIRE simulation summary")
    print("=" * 60)
    print(f"Simulations:           {summary.n_simulations:,}")
    print(f"Success probability:   {summary.probability_success:.1%}")
    print(f"Median survival age:   {summary.median_survival_age}")
    print(f"Median legacy (nom.):  €{summary.median_legacy_nominal:,.0f}")
    print(f"Median legacy (real):  €{summary.median_legacy_real:,.0f}")
    if summary.worst_case_failure_age:
        print(f"Worst failure age:     {summary.worst_case_failure_age}")
    else:
        print(f"Worst failure age:     No failures")

    print(f"\nGenerating charts ...")
    plot_fire_projection(summary, config, output_dir / "fire_projection.png")
    plot_fire_success_probability(results, config, output_dir / "fire_success_probability.png")
    plot_fire_failure_distribution(results, output_dir / "fire_failure_distribution.png")
    plot_fire_legacy_distribution(results, output_dir / "fire_legacy_distribution.png")

    summary.wealth_percentiles_by_age.to_csv(output_dir / "fire_percentiles_by_age.csv")

    print(f"\nOutputs in {output_dir.resolve()}")


if __name__ == "__main__":
    main()
