"""
Unit tests for src/rolling_window.py — rolling-window backtest engine.
"""

import pandas as pd
import pytest

from src.rolling_window import (
    run_rolling_backtest,
    plot_rolling_window_results,
    summary_statistics,
)
from src.data_loader import _generate_synthetic_bundle


class TestRollingBacktest:
    def test_returns_dataframe(self):
        bundle = _generate_synthetic_bundle()
        df = run_rolling_backtest(bundle, window_years=5, step_months=12)
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert "cagr" in df.columns

    def test_window_length_correct(self):
        """Each window should span exactly window_years × 12 months."""
        bundle = _generate_synthetic_bundle()
        df = run_rolling_backtest(bundle, window_years=5, step_months=12)
        for start, end in zip(df.index, df["end_date"]):
            months = (end.year - start.year) * 12 + (end.month - start.month) + 1
            assert 59 <= months <= 61  # 5 years × 12 months ± rounding

    def test_step_months_controls_count(self):
        """Smaller step → more windows."""
        bundle = _generate_synthetic_bundle()
        df_monthly = run_rolling_backtest(bundle, window_years=5, step_months=1)
        df_annual = run_rolling_backtest(bundle, window_years=5, step_months=12)
        assert len(df_monthly) > len(df_annual) * 10  # roughly 12x

    def test_raises_on_insufficient_data(self):
        bundle = _generate_synthetic_bundle()
        with pytest.raises(ValueError):
            # 100-year window on 20-year data must error
            run_rolling_backtest(bundle, window_years=100, step_months=1)

    def test_chart_created(self, tmp_path):
        bundle = _generate_synthetic_bundle()
        df = run_rolling_backtest(bundle, window_years=5, step_months=12)
        chart_path = tmp_path / "rolling.png"
        plot_rolling_window_results(df, window_years=5, path=chart_path)
        assert chart_path.exists()
        assert chart_path.stat().st_size > 1000


class TestSummaryStatistics:
    def test_summary_keys_present(self):
        bundle = _generate_synthetic_bundle()
        df = run_rolling_backtest(bundle, window_years=5, step_months=12)
        stats = summary_statistics(df)
        expected_keys = {
            "n_windows", "cagr_median", "cagr_worst", "cagr_best",
            "cagr_p10", "cagr_p90", "maxdd_median", "maxdd_worst",
            "sharpe_median", "sharpe_worst", "pct_windows_positive_cagr",
            "pct_windows_cagr_above_4pct",
        }
        assert expected_keys.issubset(stats.keys())

    def test_percentiles_ordered(self):
        bundle = _generate_synthetic_bundle()
        df = run_rolling_backtest(bundle, window_years=5, step_months=12)
        stats = summary_statistics(df)
        assert stats["cagr_worst"] <= stats["cagr_p10"]
        assert stats["cagr_p10"] <= stats["cagr_median"]
        assert stats["cagr_median"] <= stats["cagr_p90"]
        assert stats["cagr_p90"] <= stats["cagr_best"]
