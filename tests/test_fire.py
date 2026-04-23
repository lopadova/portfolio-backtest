"""
Unit tests for src/mortality.py and src/fire.py.
"""

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for headless/CI runs
import numpy as np
import pandas as pd
import pytest

from src.mortality import (
    load_mortality_table,
    sample_death_age,
    life_expectancy,
)
from src.fire import (
    FireConfig,
    run_fire_simulation,
    aggregate_fire_results,
    plot_fire_projection,
)
from src.data_loader import _generate_synthetic_bundle
from src.rebalance import simulate_portfolio


# ===========================================================================
# Mortality tests
# ===========================================================================

class TestMortalityTable:
    def test_table_loads(self):
        table = load_mortality_table()
        assert isinstance(table, pd.DataFrame)
        assert "age" in table.columns
        assert "qx_male" in table.columns
        assert "qx_female" in table.columns

    def test_ages_cover_0_to_110(self):
        table = load_mortality_table()
        assert table["age"].min() == 0
        assert table["age"].max() == 110

    def test_qx_in_valid_range(self):
        table = load_mortality_table()
        assert (table["qx_male"] >= 0).all()
        assert (table["qx_male"] <= 1).all()
        assert (table["qx_female"] >= 0).all()
        assert (table["qx_female"] <= 1).all()


class TestSexValidation:
    """Regression (Copilot PR #9): invalid sex values must raise, not
    silently fall through to 'female' via an else-branch."""

    def test_invalid_sex_raises_in_life_expectancy(self):
        with pytest.raises(ValueError, match="sex must be one of"):
            life_expectancy(current_age=30, sex="other")
        with pytest.raises(ValueError, match="sex must be one of"):
            life_expectancy(current_age=30, sex="")

    def test_invalid_sex_raises_in_sample_death_age(self):
        with pytest.raises(ValueError, match="sex must be one of"):
            sample_death_age(current_age=30, sex="X", n_samples=10)

    def test_non_string_sex_raises(self):
        with pytest.raises(ValueError, match="must be a string"):
            life_expectancy(current_age=30, sex=None)
        with pytest.raises(ValueError, match="must be a string"):
            life_expectancy(current_age=30, sex=1)

    def test_valid_sex_case_insensitive(self):
        """All four canonical forms should work."""
        for s in ["M", "m", "male", "MALE", "Male", "F", "f", "female", "FEMALE"]:
            le = life_expectancy(current_age=0, sex=s)
            assert le > 0


class TestLifeExpectancy:
    def test_male_life_expectancy_realistic(self):
        le = life_expectancy(current_age=0, sex="M")
        # Italian life expectancy ~81.5 years for males
        assert 78 < le < 84

    def test_female_higher_than_male(self):
        le_m = life_expectancy(current_age=0, sex="M")
        le_f = life_expectancy(current_age=0, sex="F")
        assert le_f > le_m

    def test_older_age_fewer_years(self):
        le_20 = life_expectancy(current_age=20, sex="M")
        le_60 = life_expectancy(current_age=60, sex="M")
        assert le_20 > le_60


class TestSampleDeathAge:
    def test_output_shape(self):
        ages = sample_death_age(current_age=30, sex="M", n_samples=100)
        assert len(ages) == 100

    def test_deterministic_with_seed(self):
        a1 = sample_death_age(current_age=30, sex="M", n_samples=100, seed=42)
        a2 = sample_death_age(current_age=30, sex="M", n_samples=100, seed=42)
        np.testing.assert_array_equal(a1, a2)

    def test_ages_in_valid_range(self):
        ages = sample_death_age(current_age=50, sex="F", n_samples=500)
        assert (ages >= 50).all()
        assert (ages <= 110).all()

    def test_median_matches_life_expectancy(self):
        """Monte Carlo median death age should match analytical life expectancy roughly."""
        ages = sample_death_age(current_age=30, sex="M", n_samples=5000)
        mc_median = np.median(ages)
        analytical_le = life_expectancy(30, "M") + 30
        assert abs(mc_median - analytical_le) < 3  # within 3 years


# ===========================================================================
# FIRE tests
# ===========================================================================

@pytest.fixture
def basic_fire_config():
    return FireConfig(
        current_age=45,
        sex="M",
        initial_capital=100_000.0,
        contribution_amount=1_000.0,
        contribution_frequency="month",
        fire_age=60,
        monthly_spending=2_500.0,
        inflation_rate=0.02,
        n_simulations=50,  # low for speed in tests
        block_size=3,
        seed=42,
    )


@pytest.fixture
def fire_portfolio_returns():
    bundle = _generate_synthetic_bundle()
    return simulate_portfolio(
        bundle.monthly_returns_eur, bundle.btc_activation_date, apply_ter=False,
    )


class TestFireSimulation:
    def test_simulation_returns_results(self, basic_fire_config, fire_portfolio_returns):
        results = run_fire_simulation(basic_fire_config, fire_portfolio_returns)
        assert len(results) == 50
        for r in results:
            assert hasattr(r, "death_age")
            assert hasattr(r, "success")
            assert hasattr(r, "wealth_trajectory")

    def test_aggregate_stats(self, basic_fire_config, fire_portfolio_returns):
        results = run_fire_simulation(basic_fire_config, fire_portfolio_returns)
        summary = aggregate_fire_results(results, basic_fire_config)
        assert summary.n_simulations == 50
        assert 0.0 <= summary.probability_success <= 1.0
        assert summary.median_survival_age >= basic_fire_config.current_age

    def test_fixed_end_age_no_mortality_sampling(self, fire_portfolio_returns):
        config = FireConfig(
            current_age=45, sex="M",
            fixed_end_age=85,
            initial_capital=500_000.0,
            contribution_amount=0.0,
            fire_age=45,  # already in FIRE
            monthly_spending=1_000.0,  # very low → should succeed
            n_simulations=20,
            seed=42,
        )
        results = run_fire_simulation(config, fire_portfolio_returns)
        # All deaths at 85
        for r in results:
            assert r.death_age == 85

    def test_pension_reduces_withdrawals(self, fire_portfolio_returns):
        """With vs without pension → pension version has fewer failures."""
        base_cfg = FireConfig(
            current_age=50, sex="M",
            initial_capital=200_000.0,
            contribution_amount=0,
            fire_age=55,
            monthly_spending=3_000.0,
            n_simulations=30, seed=42,
        )
        # Without pension
        r_nopension = run_fire_simulation(base_cfg, fire_portfolio_returns)
        # With pension
        cfg_with = FireConfig(**{**base_cfg.__dict__,
                                 "pension_enabled": True,
                                 "pension_monthly_amount": 2_000.0,
                                 "pension_start_age": 67})
        r_pension = run_fire_simulation(cfg_with, fire_portfolio_returns)
        fail_no_pension = sum(1 for r in r_nopension if not r.success)
        fail_pension = sum(1 for r in r_pension if not r.success)
        # Pension version should have fewer or equal failures
        assert fail_pension <= fail_no_pension

    def test_chart_creation(self, basic_fire_config, fire_portfolio_returns, tmp_path):
        results = run_fire_simulation(basic_fire_config, fire_portfolio_returns)
        summary = aggregate_fire_results(results, basic_fire_config)
        chart_path = tmp_path / "fire_proj.png"
        plot_fire_projection(summary, basic_fire_config, chart_path)
        assert chart_path.exists()


class TestFireValidation:
    """Regression (Copilot PR #9): run_fire_simulation must validate its
    FireConfig bounds to avoid negative-length trajectories or tax-rate=1.0
    division-by-zero."""

    def test_fixed_end_age_must_be_ge_current_age(self, fire_portfolio_returns):
        config = FireConfig(
            current_age=50, fixed_end_age=45,  # INVALID: end < current
            initial_capital=100_000, fire_age=45,
            monthly_spending=1_000, n_simulations=5,
        )
        with pytest.raises(ValueError, match="fixed_end_age"):
            run_fire_simulation(config, fire_portfolio_returns)

    def test_current_age_out_of_range(self, fire_portfolio_returns):
        config = FireConfig(
            current_age=150, initial_capital=100_000, fire_age=50,
            monthly_spending=1_000, n_simulations=5,
        )
        with pytest.raises(ValueError, match="current_age"):
            run_fire_simulation(config, fire_portfolio_returns)

    def test_tax_rate_one_raises(self, fire_portfolio_returns):
        """tax_rate=1.0 would cause division-by-zero on gross-up."""
        config = FireConfig(
            current_age=45, initial_capital=100_000, fire_age=50,
            monthly_spending=1_000, n_simulations=5,
            tax_on_withdrawals=True, tax_rate=1.0,
        )
        with pytest.raises(ValueError, match="tax_rate"):
            run_fire_simulation(config, fire_portfolio_returns)


class TestTrajectoryIndexing:
    """Regression (Copilot PR #9): wealth_trajectory[0] must be the INITIAL
    wealth at current_age, not end-of-month-1 wealth. The whole
    (age - current_age) * 12 mapping depends on this."""

    def test_trajectory_starts_at_initial_capital(self, fire_portfolio_returns):
        config = FireConfig(
            current_age=45, sex="M",
            initial_capital=100_000.0,
            contribution_amount=500, contribution_frequency="month",
            fire_age=55,
            monthly_spending=2_000, n_simulations=5,
            seed=42,
        )
        results = run_fire_simulation(config, fire_portfolio_returns)
        for r in results:
            # trajectory[0] must equal initial_capital (before any contribution/return)
            assert r.wealth_trajectory[0] == pytest.approx(100_000.0, abs=1e-6)

    def test_trajectory_length_is_sim_months_plus_one(self, fire_portfolio_returns):
        config = FireConfig(
            current_age=45, sex="M",
            fixed_end_age=50,  # 5 years = 60 months
            initial_capital=100_000.0,
            contribution_amount=0, fire_age=45,
            monthly_spending=1_000, n_simulations=3,
            seed=42,
        )
        results = run_fire_simulation(config, fire_portfolio_returns)
        for r in results:
            # sim_months + 1 = 60 + 1 = 61
            assert len(r.wealth_trajectory) == 61
