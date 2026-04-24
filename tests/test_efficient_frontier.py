"""
Unit tests for src/efficient_frontier.py.
"""

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for headless/CI runs
import numpy as np
import pandas as pd
import pytest

from src.efficient_frontier import (
    compute_annual_moments,
    sample_random_portfolios,
    evaluate_portfolios,
    find_key_portfolios,
    markowitz_frontier,
    plot_efficient_frontier,
    plot_efficient_frontier_interactive,
    run_efficient_frontier,
)
from src.data_loader import _generate_synthetic_bundle


class TestAnnualMoments:
    def test_shape_of_moments(self, realistic_monthly_returns):
        # Convert single series to DataFrame
        df = pd.DataFrame({"asset1": realistic_monthly_returns})
        mean, cov = compute_annual_moments(df)
        assert len(mean) == 1
        assert cov.shape == (1, 1)

    def test_annualization_factor(self):
        dates = pd.date_range("2020-01-31", periods=120, freq="ME")
        # Constant 1%/month → annual mean = 12%, annual var ~0
        df = pd.DataFrame({"x": [0.01] * 120}, index=dates)
        mean, cov = compute_annual_moments(df)
        assert mean["x"] == pytest.approx(0.12, abs=1e-9)


class TestRandomSampling:
    def test_shape(self):
        W = sample_random_portfolios(n_assets=5, n_samples=100, seed=42)
        assert W.shape == (100, 5)

    def test_rows_sum_to_one(self):
        W = sample_random_portfolios(n_assets=5, n_samples=100, seed=42)
        sums = W.sum(axis=1)
        assert np.allclose(sums, 1.0)

    def test_weights_non_negative(self):
        W = sample_random_portfolios(n_assets=5, n_samples=100, seed=42)
        assert (W >= 0).all()

    def test_deterministic_with_seed(self):
        W1 = sample_random_portfolios(n_assets=5, n_samples=100, seed=42)
        W2 = sample_random_portfolios(n_assets=5, n_samples=100, seed=42)
        np.testing.assert_array_equal(W1, W2)


class TestEvaluatePortfolios:
    def test_output_shape(self):
        W = sample_random_portfolios(n_assets=3, n_samples=50, seed=42)
        mean = np.array([0.08, 0.05, 0.03])
        cov = np.eye(3) * 0.04
        df = evaluate_portfolios(W, mean, cov)
        assert len(df) == 50
        assert "expected_return" in df.columns
        assert "volatility" in df.columns
        assert "sharpe" in df.columns


class TestKeyPortfolios:
    def test_finds_three_extremes(self):
        rng = np.random.default_rng(42)
        W = rng.dirichlet(np.ones(4), size=1000)
        mean = np.array([0.08, 0.05, 0.03, 0.10])
        cov = np.diag([0.04, 0.02, 0.01, 0.08])
        eval_df = evaluate_portfolios(W, mean, cov)
        keys = find_key_portfolios(W, eval_df, ["A", "B", "C", "D"])
        assert "max_sharpe" in keys
        assert "min_vol" in keys
        assert "max_return" in keys
        # Max Return point should have highest expected_return
        assert keys["max_return"].expected_return == eval_df["expected_return"].max()
        # Min Vol point should have lowest volatility
        assert keys["min_vol"].volatility == eval_df["volatility"].min()


class TestFrontierIntegration:
    def test_end_to_end_runs(self):
        bundle = _generate_synthetic_bundle()
        random_eval, weights_matrix, keys, frontier, fu, asset_names = run_efficient_frontier(
            bundle, n_random=1000,
        )
        assert len(random_eval) == 1000
        assert weights_matrix.shape == (1000, len(asset_names))
        assert "max_sharpe" in keys
        assert len(frontier) > 0  # Markowitz produced at least some points

    def test_chart_created(self, tmp_path):
        bundle = _generate_synthetic_bundle()
        random_eval, weights_matrix, keys, frontier, fu, asset_names = run_efficient_frontier(
            bundle, n_random=500,
        )
        path = tmp_path / "frontier.png"
        plot_efficient_frontier(random_eval, keys, frontier, fu, path)
        assert path.exists()
        assert path.stat().st_size > 5000

    def test_reference_point_on_same_universe(self):
        """
        Regression: the reference portfolio (Four Umbrellas for the default
        preset) must be computed on the SAME sleeve universe and date index
        as the frontier (not a separate full-portfolio simulation).
        """
        bundle = _generate_synthetic_bundle()
        _, _, _, _, fu, asset_names = run_efficient_frontier(bundle, n_random=100)
        assert fu is not None
        # FU weights should reference only sleeves that exist in the frontier universe
        for k in fu.weights:
            assert k in asset_names
        # Weights normalized to ~1.0
        assert abs(sum(fu.weights.values()) - 1.0) < 1e-6
        # Label mentions "liquid sleeves" and "normalized" to avoid confusion
        assert "liquid sleeves" in fu.label

    def test_frontier_points_carry_weights(self):
        """
        Each Markowitz frontier point must include the full w_<asset> columns
        so the interactive chart can show allocation on hover.
        """
        bundle = _generate_synthetic_bundle()
        _, _, _, frontier, _, asset_names = run_efficient_frontier(bundle, n_random=100)
        weight_cols = [c for c in frontier.columns if c.startswith("w_")]
        assert len(weight_cols) == len(asset_names), \
            f"Expected {len(asset_names)} weight columns, got {len(weight_cols)}"
        # Weights on each point should sum to ~1 (long-only fully-invested)
        for _, row in frontier.iterrows():
            total_w = sum(row[c] for c in weight_cols)
            assert abs(total_w - 1.0) < 1e-3


class TestInteractiveChart:
    """Tests for the Plotly-based interactive frontier chart."""

    def test_interactive_chart_created_if_plotly_available(self, tmp_path):
        """
        If plotly is importable, the interactive HTML must be generated.
        If not, the function returns False without raising.
        """
        bundle = _generate_synthetic_bundle()
        random_eval, weights_matrix, keys, frontier, fu, asset_names = run_efficient_frontier(
            bundle, n_random=200,
        )
        path = tmp_path / "interactive.html"
        result = plot_efficient_frontier_interactive(
            random_eval, weights_matrix, asset_names, keys, frontier, fu, path,
        )
        try:
            import plotly  # noqa: F401
            plotly_available = True
        except ImportError:
            plotly_available = False
        if plotly_available:
            assert result is True
            assert path.exists()
            content = path.read_text(encoding="utf-8")
            # Minimum expected content markers
            assert "Efficient Frontier" in content
            assert "Allocation" in content  # hover text template
            assert "plotly" in content.lower()
        else:
            assert result is False
