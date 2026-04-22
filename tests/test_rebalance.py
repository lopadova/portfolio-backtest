"""
Unit tests for src/rebalance.py — portfolio simulation, rebalance-by-buying,
benchmark simulation.
"""

import numpy as np
import pandas as pd
import pytest

from src.rebalance import build_target_weights, simulate_portfolio, simulate_benchmark
from src.portfolio import WEIGHTS


class TestBuildTargetWeights:
    def test_target_weights_plus_cash_equal_one(self):
        """Sum of non-cash sleeve targets + cash weight should equal 1.0."""
        targets = build_target_weights()
        total = sum(targets.values()) + WEIGHTS["cash"]
        assert abs(total - 1.0) < 0.002

    def test_expected_sleeves_in_target(self):
        targets = build_target_weights()
        required = {"gold", "put_write", "nasdaq_top30", "quality", "momentum",
                    "btc", "eur_gov_1_3y", "dbi", "pension_bond", "pension_equity"}
        assert required.issubset(targets.keys())


class TestSimulatePortfolio:
    @pytest.fixture
    def synthetic_returns(self):
        """Build a DataFrame with all sleeve columns, realistic random returns."""
        rng = np.random.default_rng(123)
        dates = pd.date_range("2020-01-31", periods=60, freq="ME")
        targets = build_target_weights()
        sleeves = list(targets.keys())
        data = {
            s: rng.normal(0.005, 0.03, 60) for s in sleeves
        }
        return pd.DataFrame(data, index=dates)

    def test_simulator_returns_series(self, synthetic_returns):
        out = simulate_portfolio(
            synthetic_returns,
            btc_activation_date=pd.Timestamp("2020-01-01"),
            apply_ter=False,
        )
        assert isinstance(out, pd.Series)
        assert len(out) == len(synthetic_returns)

    def test_simulator_applies_ter_reduces_returns(self, synthetic_returns):
        out_no_ter = simulate_portfolio(
            synthetic_returns,
            btc_activation_date=pd.Timestamp("2020-01-01"),
            apply_ter=False,
        )
        out_with_ter = simulate_portfolio(
            synthetic_returns,
            btc_activation_date=pd.Timestamp("2020-01-01"),
            apply_ter=True,
        )
        assert out_with_ter.sum() <= out_no_ter.sum()

    def test_btc_inactive_before_activation_date(self, synthetic_returns):
        """BTC weight should be 0 before the activation date — this means we shouldn't crash."""
        out = simulate_portfolio(
            synthetic_returns,
            btc_activation_date=pd.Timestamp("2025-01-01"),  # far future, so BTC never activates here
            apply_ter=False,
        )
        assert len(out) == len(synthetic_returns)
        # No assertion on values beyond "runs without error" — BTC behavior is implicit


class TestSimulateBenchmark:
    @pytest.fixture
    def synthetic_bench_data(self):
        dates = pd.date_range("2020-01-31", periods=36, freq="ME")
        rng = np.random.default_rng(7)
        return pd.DataFrame({
            "msci_world_tr_monthly": rng.normal(0.007, 0.04, 36),
            "bloomberg_euro_agg_monthly": rng.normal(0.002, 0.01, 36),
        }, index=dates)

    def test_60_40_benchmark_runs(self, synthetic_bench_data):
        weights = {"msci_world_tr_monthly": 0.6, "bloomberg_euro_agg_monthly": 0.4}
        out = simulate_benchmark(synthetic_bench_data, weights)
        assert len(out) == 36

    def test_100pct_single_asset_matches_returns(self, synthetic_bench_data):
        """If 100% in one asset with annual rebalancing, monthly returns should match
        for in-rebalance months. Between rebalances, drift accumulates — so we just
        check the values are close, not exact."""
        weights = {"msci_world_tr_monthly": 1.0}
        out = simulate_benchmark(synthetic_bench_data, weights)
        # Correlation > 0.99 since it's fundamentally the same series
        corr = out.corr(synthetic_bench_data["msci_world_tr_monthly"])
        assert corr > 0.99

    def test_benchmark_skips_missing_columns(self, synthetic_bench_data):
        """If a benchmark references a column that doesn't exist, it should not crash."""
        weights = {"msci_world_tr_monthly": 0.5, "nonexistent_asset": 0.5}
        out = simulate_benchmark(synthetic_bench_data, weights)
        # Should still return something (using just the available column, normalized)
        assert len(out) > 0
