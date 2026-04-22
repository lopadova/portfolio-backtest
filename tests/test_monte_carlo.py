"""
Unit tests for src/monte_carlo.py — block bootstrap, wealth path simulation,
MC statistics.
"""

import numpy as np
import pandas as pd
import pytest

from src.monte_carlo import (
    block_bootstrap,
    simulate_wealth_paths,
    compute_mc_stats,
)


class TestBlockBootstrap:
    def test_output_shape(self, realistic_monthly_returns):
        paths = block_bootstrap(realistic_monthly_returns, n_paths=100, n_periods=120, block_size=3, seed=42)
        assert paths.shape == (100, 120)

    def test_deterministic_with_seed(self, realistic_monthly_returns):
        p1 = block_bootstrap(realistic_monthly_returns, n_paths=10, n_periods=24, block_size=3, seed=42)
        p2 = block_bootstrap(realistic_monthly_returns, n_paths=10, n_periods=24, block_size=3, seed=42)
        np.testing.assert_array_equal(p1, p2)

    def test_different_seed_different_result(self, realistic_monthly_returns):
        p1 = block_bootstrap(realistic_monthly_returns, n_paths=10, n_periods=24, block_size=3, seed=1)
        p2 = block_bootstrap(realistic_monthly_returns, n_paths=10, n_periods=24, block_size=3, seed=2)
        assert not np.array_equal(p1, p2)

    def test_sampled_values_come_from_history(self, realistic_monthly_returns):
        """Every value in the bootstrap sample must appear in the historical series."""
        hist = set(realistic_monthly_returns.dropna().values)
        paths = block_bootstrap(realistic_monthly_returns, n_paths=5, n_periods=12, block_size=3, seed=42)
        for v in paths.flatten():
            assert v in hist

    def test_error_on_short_history(self):
        """Bootstrap should error if history is shorter than block size."""
        short = pd.Series([0.01, -0.01])
        with pytest.raises(ValueError):
            block_bootstrap(short, n_paths=10, n_periods=12, block_size=5)


class TestSimulateWealthPaths:
    def test_starts_at_one(self):
        rng = np.random.default_rng(42)
        returns = rng.normal(0.005, 0.03, (50, 120))
        wealth = simulate_wealth_paths(returns)
        assert (wealth[:, 0] == 1.0).all()

    def test_shape_is_n_periods_plus_one(self):
        returns = np.zeros((10, 12))
        wealth = simulate_wealth_paths(returns)
        assert wealth.shape == (10, 13)

    def test_zero_returns_stays_at_one(self):
        returns = np.zeros((5, 12))
        wealth = simulate_wealth_paths(returns)
        assert (wealth == 1.0).all()

    def test_constant_positive_compounds(self):
        """+1% monthly for 12 months → terminal ≈ 1.01^12"""
        returns = np.full((1, 12), 0.01)
        wealth = simulate_wealth_paths(returns)
        assert wealth[0, -1] == pytest.approx(1.01 ** 12, rel=1e-6)


class TestMCStats:
    def test_stats_on_deterministic_paths(self):
        """If all paths are identical flat +1%/month, stats are degenerate but well-defined."""
        returns = np.full((100, 60), 0.01)
        wealth = simulate_wealth_paths(returns)
        stats = compute_mc_stats(wealth, n_months=60)
        # All paths identical → all percentiles equal
        assert stats.median_terminal == pytest.approx(stats.pct5_terminal, rel=1e-6)
        assert stats.median_terminal == pytest.approx(stats.pct95_terminal, rel=1e-6)
        # No drawdown in monotone-up series
        assert stats.median_max_drawdown == pytest.approx(0.0, abs=1e-9)

    def test_probability_positive_return(self):
        """50%/50% split between positive and negative terminal wealth."""
        rng = np.random.default_rng(42)
        returns = rng.normal(0.0, 0.10, (1000, 12))
        wealth = simulate_wealth_paths(returns)
        stats = compute_mc_stats(wealth, n_months=12)
        # Roughly 50/50 — wide tolerance because MC noise
        assert 0.30 < stats.prob_positive_real_return < 0.70

    def test_monotone_down_has_high_drawdown_probability(self):
        """Every path -5% per month → everyone has huge drawdowns."""
        returns = np.full((100, 24), -0.05)
        wealth = simulate_wealth_paths(returns)
        stats = compute_mc_stats(wealth, n_months=24)
        # P(DD > 20%) should be ~100%
        assert stats.prob_drawdown_gt_20pct > 0.99
