"""
Smoke tests for src/plots.py — verify every chart generator runs without
error and produces a PNG file.

These are intentionally minimal — pixel-perfect chart testing is brittle.
The goal is to catch breakage when the chart APIs change.
"""

import pandas as pd
import pytest
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for CI

from src.metrics import compute_all
from src.plots import (
    plot_equity_curve, plot_drawdown, plot_underwater,
    plot_rolling_sharpe, plot_rolling_returns, plot_crisis_zoom,
    plot_annual_returns, plot_return_distribution,
    plot_risk_return_scatter, plot_metrics_comparison,
    plot_correlation_heatmap,
)


class TestChartSmoke:
    def test_equity_curve(self, tmp_output_dir, sample_returns_dict):
        path = tmp_output_dir / "equity_curve.png"
        plot_equity_curve(sample_returns_dict, path, start_nav=100_000.0)
        assert path.exists() and path.stat().st_size > 1000

    def test_drawdown(self, tmp_output_dir, sample_returns_dict):
        path = tmp_output_dir / "drawdown.png"
        plot_drawdown(sample_returns_dict, path)
        assert path.exists()

    def test_underwater(self, tmp_output_dir, sample_returns_dict):
        path = tmp_output_dir / "underwater.png"
        plot_underwater(sample_returns_dict, path)
        assert path.exists()

    def test_rolling_sharpe(self, tmp_output_dir, sample_returns_dict):
        path = tmp_output_dir / "rolling_sharpe.png"
        plot_rolling_sharpe(sample_returns_dict, path, window_months=36)
        assert path.exists()

    def test_rolling_returns(self, tmp_output_dir, sample_returns_dict):
        path = tmp_output_dir / "rolling_returns.png"
        plot_rolling_returns(sample_returns_dict, path, window_months=36)
        assert path.exists()

    def test_crisis_zoom(self, tmp_output_dir, sample_returns_dict):
        """Note: sample_returns_dict spans 2015-2024, so GFC window is empty — chart should still render."""
        path = tmp_output_dir / "crisis_zoom.png"
        plot_crisis_zoom(sample_returns_dict, path)
        assert path.exists()

    def test_annual_returns(self, tmp_output_dir, sample_returns_dict):
        path = tmp_output_dir / "annual_returns.png"
        plot_annual_returns(sample_returns_dict, path)
        assert path.exists()

    def test_return_distribution(self, tmp_output_dir, sample_returns_dict):
        path = tmp_output_dir / "return_distribution.png"
        plot_return_distribution(sample_returns_dict, path)
        assert path.exists()

    def test_risk_return_scatter(self, tmp_output_dir, sample_returns_dict):
        stats_list = [compute_all(name, r) for name, r in sample_returns_dict.items()]
        path = tmp_output_dir / "risk_return_scatter.png"
        plot_risk_return_scatter(stats_list, path)
        assert path.exists()

    def test_metrics_comparison(self, tmp_output_dir, sample_returns_dict):
        stats_list = [compute_all(name, r) for name, r in sample_returns_dict.items()]
        path = tmp_output_dir / "metrics_comparison.png"
        plot_metrics_comparison(stats_list, path)
        assert path.exists()

    def test_correlation_heatmap(self, tmp_output_dir):
        import numpy as np
        rng = np.random.default_rng(42)
        dates = pd.date_range("2020-01-31", periods=60, freq="M")
        df = pd.DataFrame({
            "gold": rng.normal(0.005, 0.03, 60),
            "equity": rng.normal(0.008, 0.04, 60),
            "bonds": rng.normal(0.002, 0.01, 60),
        }, index=dates)
        path = tmp_output_dir / "correlation_heatmap.png"
        plot_correlation_heatmap(df, path)
        assert path.exists()
