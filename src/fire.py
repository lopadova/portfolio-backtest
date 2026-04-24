"""
FIRE (Financial Independence, Retire Early) calculator — Monte Carlo engine
with 2-phase model (accumulation → decumulation) and Italian mortality sampling.

Modelled features:
    - Accumulation: periodic contributions (monthly/annual) at portfolio returns
    - FIRE age trigger: switch from contributions to withdrawals
    - Decumulation: inflation-adjusted monthly spending
    - Optional pension with age-triggered start + annual revaluation %
    - Optional CGT on withdrawals (connects to src/tax.py)
    - Mortality-based horizon: each simulation samples death age from ISTAT
    - Success metric: did the portfolio survive until death?

Outputs:
    - Portfolio projection (area chart with percentile bands)
    - Success probability by age
    - Failure age distribution
    - Legacy (nominal + real inflation-adjusted)
    - Stats panel
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

from .monte_carlo import block_bootstrap
from .mortality import sample_death_age


@dataclass
class FireConfig:
    """User inputs for a FIRE simulation."""
    # Personal
    current_age: int
    sex: Literal["M", "F"] = "M"
    fixed_end_age: Optional[int] = None       # if set, use fixed horizon instead of mortality sampling

    # Wealth & contributions
    initial_capital: float = 100_000.0
    contribution_amount: float = 1_000.0
    contribution_frequency: Literal["month", "year"] = "month"

    # FIRE goal
    fire_age: int = 60
    monthly_spending: float = 2_500.0
    inflation_rate: float = 0.02

    # Pension (optional)
    pension_enabled: bool = False
    pension_monthly_amount: float = 0.0       # 12 months/yr (no 13ª)
    pension_start_age: int = 67
    pension_revaluation: float = 1.00         # 1.00 = full inflation match (100% INPS perequazione)

    # Tax
    tax_on_withdrawals: bool = False
    tax_rate: float = 0.26

    # Simulation
    n_simulations: int = 1000
    block_size: int = 3                       # months
    seed: int = 42


@dataclass
class FireSimulationResult:
    """Output of a single FIRE simulation path."""
    death_age: int
    success: bool
    final_wealth_nominal: float
    final_wealth_real: float
    failure_age: Optional[int]                # age at which portfolio hit 0, if failure
    wealth_trajectory: np.ndarray             # monthly portfolio value through the simulation


@dataclass
class FireSummary:
    """Aggregate statistics across all simulations."""
    n_simulations: int
    probability_success: float
    median_survival_age: int
    median_legacy_nominal: float
    median_legacy_real: float
    worst_case_failure_age: Optional[int]
    wealth_percentiles_by_age: pd.DataFrame   # index=age, columns=[p5, p25, p50, p75, p95]


def run_fire_simulation(
    config: FireConfig,
    portfolio_monthly_returns: pd.Series,
) -> list[FireSimulationResult]:
    """
    Run N Monte Carlo simulations of the FIRE plan.

    For each simulation:
        1. Sample a death age from the ISTAT table (unless fixed_end_age given)
        2. Block-bootstrap the required months of portfolio returns
        3. Simulate month-by-month:
             - Before FIRE: add contribution, apply return
             - After FIRE: subtract (spending - pension + tax), apply return
             - If portfolio goes below 0: failure (record age, wealth=0)
        4. Record final wealth (if success) and trajectory

    Returns a list of FireSimulationResult objects.
    """
    # --- Input validation ---
    max_age = 110
    if not (0 <= config.current_age <= max_age):
        raise ValueError(
            f"current_age must be in [0, {max_age}], got {config.current_age}"
        )
    if config.fixed_end_age is not None:
        if not (config.current_age <= config.fixed_end_age <= max_age):
            raise ValueError(
                f"fixed_end_age must be in [current_age={config.current_age}, {max_age}], "
                f"got {config.fixed_end_age}"
            )
    if config.tax_on_withdrawals and not (0.0 <= config.tax_rate < 1.0):
        raise ValueError(
            f"tax_rate must be in [0, 1) when tax_on_withdrawals is enabled, "
            f"got {config.tax_rate}. (tax_rate=1.0 would produce division by zero on gross-up.)"
        )

    max_months = (max_age - config.current_age) * 12

    # Pre-sample death ages (one per simulation)
    if config.fixed_end_age is not None:
        death_ages = np.full(config.n_simulations, config.fixed_end_age, dtype=np.int32)
    else:
        death_ages = sample_death_age(
            current_age=config.current_age,
            sex=config.sex,
            n_samples=config.n_simulations,
            seed=config.seed,
        )

    # Pre-generate return bootstrap paths (one per simulation)
    bootstrap_paths = block_bootstrap(
        portfolio_monthly_returns,
        n_paths=config.n_simulations,
        n_periods=max_months,
        block_size=config.block_size,
        seed=config.seed + 1,
    )

    # Monthly contribution (convert annual → monthly if needed)
    monthly_contribution = (
        config.contribution_amount if config.contribution_frequency == "month"
        else config.contribution_amount / 12.0
    )

    results = []
    for sim_idx in range(config.n_simulations):
        death_age = int(death_ages[sim_idx])
        sim_months = (death_age - config.current_age) * 12
        returns = bootstrap_paths[sim_idx][:sim_months]

        # Trajectory stores sim_months+1 points: index 0 = wealth at current_age
        # (BEFORE any contribution/return) so downstream aggregation can map
        # age→month_idx via (age - current_age) * 12 without an off-by-one shift.
        trajectory = np.zeros(sim_months + 1)
        trajectory[0] = config.initial_capital
        wealth = config.initial_capital
        failure_age = None
        inflation_factor = 1.0
        pension_revaluation_factor = 1.0
        pension_months_elapsed = 0  # months since pension started (for annual revaluation)

        for m in range(sim_months):
            age_years = config.current_age + m / 12.0
            is_accumulation = age_years < config.fire_age
            inflation_factor *= (1.0 + config.inflation_rate) ** (1.0 / 12.0)

            if is_accumulation:
                # Contribution
                wealth += monthly_contribution
            else:
                # Decumulation
                spending_month = config.monthly_spending * inflation_factor
                pension_income = 0.0
                if config.pension_enabled and age_years >= config.pension_start_age:
                    # Apply pension revaluation ONLY after the first full year of
                    # pension payments (pension_months_elapsed >= 12 and on annual
                    # boundaries thereafter). Previously, m % 12 == 0 triggered an
                    # extra revaluation at pension start — an off-by-one year bug.
                    pension_months_elapsed += 1
                    if pension_months_elapsed > 12 and (pension_months_elapsed - 1) % 12 == 0:
                        pension_revaluation_factor *= (1.0 + config.inflation_rate * config.pension_revaluation)
                    pension_income = config.pension_monthly_amount * pension_revaluation_factor
                net_withdrawal = spending_month - pension_income
                if net_withdrawal < 0:
                    # Surplus — reinvest
                    wealth += (-net_withdrawal)
                else:
                    # Withdrawal + optional tax gross-up
                    if config.tax_on_withdrawals:
                        gross_withdrawal = net_withdrawal / (1.0 - config.tax_rate)
                    else:
                        gross_withdrawal = net_withdrawal
                    wealth -= gross_withdrawal

            # Apply return
            wealth *= (1.0 + returns[m])

            if wealth <= 0:
                failure_age = int(age_years)
                wealth = 0.0
                trajectory[m + 1:] = 0.0
                break
            trajectory[m + 1] = wealth

        final_wealth_nominal = wealth
        final_wealth_real = wealth / inflation_factor

        results.append(FireSimulationResult(
            death_age=death_age,
            success=(failure_age is None),
            final_wealth_nominal=final_wealth_nominal,
            final_wealth_real=final_wealth_real,
            failure_age=failure_age,
            wealth_trajectory=trajectory,
        ))

    return results


def aggregate_fire_results(
    results: list[FireSimulationResult],
    config: FireConfig,
) -> FireSummary:
    """Compute summary statistics across all simulations."""
    n = len(results)
    successes = sum(1 for r in results if r.success)
    prob_success = successes / n if n > 0 else 0.0
    death_ages = [r.death_age for r in results]
    legacies_nominal = [r.final_wealth_nominal for r in results if r.success]
    legacies_real = [r.final_wealth_real for r in results if r.success]
    failure_ages = [r.failure_age for r in results if r.failure_age is not None]

    median_survival = int(np.median(death_ages)) if death_ages else 0
    median_legacy_nominal = float(np.median(legacies_nominal)) if legacies_nominal else 0.0
    median_legacy_real = float(np.median(legacies_real)) if legacies_real else 0.0
    worst_case_failure = int(min(failure_ages)) if failure_ages else None

    # Wealth percentiles by age (need to align trajectories)
    max_age = 110
    ages = list(range(config.current_age, max_age + 1))
    # Build a (n_sims, n_ages) matrix: wealth at each age (year-level, taking month 0)
    wealth_matrix = np.zeros((n, len(ages)))
    for i, r in enumerate(results):
        for j, age in enumerate(ages):
            month_idx = (age - config.current_age) * 12
            if month_idx < len(r.wealth_trajectory):
                wealth_matrix[i, j] = r.wealth_trajectory[month_idx]
            elif r.success:
                # Past death age — keep last known wealth
                wealth_matrix[i, j] = r.final_wealth_nominal if j * 12 <= (r.death_age - config.current_age) * 12 else np.nan
            else:
                wealth_matrix[i, j] = np.nan
    # nanpercentile emits "All-NaN slice encountered" RuntimeWarning when a
    # column (age) has zero live simulations — typically beyond the oldest
    # mortality-sample death age. The result for those columns is NaN,
    # which is exactly what we want for the percentile bands chart (no
    # data to plot past that age). Silence the warning on the expected
    # cause; any other RuntimeWarning still propagates.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="All-NaN slice encountered",
            category=RuntimeWarning,
        )
        percentiles_df = pd.DataFrame({
            "age": ages,
            "p5":  np.nanpercentile(wealth_matrix, 5, axis=0),
            "p25": np.nanpercentile(wealth_matrix, 25, axis=0),
            "p50": np.nanpercentile(wealth_matrix, 50, axis=0),
            "p75": np.nanpercentile(wealth_matrix, 75, axis=0),
            "p95": np.nanpercentile(wealth_matrix, 95, axis=0),
        }).set_index("age")

    return FireSummary(
        n_simulations=n,
        probability_success=prob_success,
        median_survival_age=median_survival,
        median_legacy_nominal=median_legacy_nominal,
        median_legacy_real=median_legacy_real,
        worst_case_failure_age=worst_case_failure,
        wealth_percentiles_by_age=percentiles_df,
    )


def plot_fire_projection(summary: FireSummary, config: FireConfig, path: Path):
    """Portfolio projection with percentile bands."""
    fig, ax = plt.subplots(figsize=(13, 7))
    ages = summary.wealth_percentiles_by_age.index
    df = summary.wealth_percentiles_by_age

    ax.fill_between(ages, df["p25"], df["p75"], color="#1f77b4", alpha=0.30, label="25°–75° percentile")
    ax.fill_between(ages, df["p5"], df["p95"], color="#1f77b4", alpha=0.12, label="5°–95° percentile")
    ax.plot(ages, df["p50"], color="#1f77b4", linewidth=2.2, label="Mediana")

    ax.axvline(config.fire_age, color="#d62728", linestyle="--", linewidth=1.2, label=f"FIRE age ({config.fire_age})")
    if config.pension_enabled:
        ax.axvline(config.pension_start_age, color="#2ca02c", linestyle="--", linewidth=1.2,
                   label=f"Pension start ({config.pension_start_age})")

    ax.set_title(
        f"Portfolio projection — {summary.n_simulations:,} simulations ({config.sex})",
        fontsize=13, fontweight="bold",
    )
    ax.set_xlabel("Età")
    ax.set_ylabel("Portafoglio (€)")
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"€{x:,.0f}"))
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_fire_success_probability(results: list[FireSimulationResult], config: FireConfig, path: Path):
    """
    Success probability by age: conditional on being alive at that age, what
    % of simulations still have portfolio > 0?

    BUG-FIX note (Copilot PR #9): previously the denominator was `n` (all
    simulations), which meant paths with death_age < current_iter_age were
    excluded from the numerator and the curve appeared to drop due to
    MORTALITY rather than portfolio failure. The correct reading for a
    retirement-robustness chart is to condition on being alive at each age.
    """
    max_age = 110
    ages = list(range(config.current_age, max_age + 1))
    success_by_age = []
    for age in ages:
        month_idx = (age - config.current_age) * 12
        # Denominator: simulations where this person is still alive at `age`
        alive_sims = [r for r in results if r.death_age >= age]
        if len(alive_sims) == 0:
            # Nobody in the Monte Carlo sample reached this age → not plotted
            success_by_age.append(float("nan"))
            continue
        # Numerator: among the alive, how many had portfolio > 0 at that age
        portfolio_alive = sum(
            1 for r in alive_sims
            if month_idx < len(r.wealth_trajectory) and r.wealth_trajectory[month_idx] > 0
        )
        success_by_age.append(portfolio_alive / len(alive_sims) * 100)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(ages, success_by_age, color="#2ca02c", linewidth=2.2)
    ax.fill_between(ages, success_by_age, 0, alpha=0.20, color="#2ca02c")
    ax.axhline(80, color="#d62728", linestyle="--", linewidth=1, label="80% threshold (robust plan)")
    ax.axvline(config.fire_age, color="#d62728", linestyle=":", linewidth=1, alpha=0.5, label=f"FIRE age ({config.fire_age})")
    ax.set_title("Probabilità di successo per età", fontsize=13, fontweight="bold")
    ax.set_xlabel("Età")
    ax.set_ylabel("% simulazioni con portfolio > 0")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
    ax.set_ylim(0, 105)
    ax.legend(loc="lower left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_fire_failure_distribution(results: list[FireSimulationResult], path: Path):
    """Histogram of failure ages (only non-success simulations)."""
    failures = [r.failure_age for r in results if r.failure_age is not None]
    fig, ax = plt.subplots(figsize=(12, 6))
    if len(failures) == 0:
        ax.text(0.5, 0.5, "Nessun fallimento — tutte le simulazioni hanno avuto successo!",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=14, color="#2ca02c", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.8", facecolor="#d4edda", edgecolor="#2ca02c"))
        ax.set_xticks([])
        ax.set_yticks([])
    else:
        ax.hist(failures, bins=30, color="#d62728", alpha=0.75, edgecolor="white")
        ax.axvline(float(np.median(failures)), color="black", linewidth=2.0,
                   label=f"Età mediana fallimento: {int(np.median(failures))}")
        ax.set_xlabel("Età al fallimento")
        ax.set_ylabel("Numero simulazioni")
        ax.legend()
    ax.set_title(f"Distribuzione età di fallimento ({len(failures)}/{len(results)} fallimenti)",
                 fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_fire_legacy_distribution(results: list[FireSimulationResult], path: Path):
    """Histogram of legacy (final wealth) nominal + real."""
    legacies_nominal = [r.final_wealth_nominal for r in results if r.success]
    legacies_real = [r.final_wealth_real for r in results if r.success]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for ax, data, title, color in [
        (axes[0], legacies_nominal, "Eredità nominale (euro correnti)", "#1f77b4"),
        (axes[1], legacies_real, "Eredità reale (aggiustata per inflazione)", "#2ca02c"),
    ]:
        if len(data) == 0:
            ax.text(0.5, 0.5, "Nessuna simulazione di successo",
                    transform=ax.transAxes, ha="center", va="center", fontsize=12)
        else:
            ax.hist(data, bins=50, color=color, alpha=0.75, edgecolor="white")
            median = np.median(data)
            ax.axvline(median, color="black", linewidth=2.0, label=f"Mediana: €{median:,.0f}")
            ax.legend()
            ax.xaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"€{x:,.0f}"))
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("Valore finale (€)")
        ax.set_ylabel("Numero simulazioni")
        ax.grid(True, alpha=0.3)

    fig.suptitle("Distribuzione dell'eredità", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches="tight", facecolor="white")
    plt.close(fig)
