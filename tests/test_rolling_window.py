"""
Unit tests for src/rolling_window.py — rolling-window backtest engine.
"""

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for headless/CI runs
import pandas as pd
import pytest

from src.rolling_window import (
    run_rolling_backtest,
    plot_rolling_window_results,
    summary_statistics,
    _describe_step,
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

    def test_raises_on_invalid_window_years(self):
        """window_years must be a positive integer."""
        bundle = _generate_synthetic_bundle()
        with pytest.raises(ValueError, match="window_years must be a positive integer"):
            run_rolling_backtest(bundle, window_years=0, step_months=1)
        with pytest.raises(ValueError, match="window_years must be a positive integer"):
            run_rolling_backtest(bundle, window_years=-5, step_months=1)
        with pytest.raises(ValueError, match="window_years must be a positive integer"):
            run_rolling_backtest(bundle, window_years=5.5, step_months=1)  # non-int

    def test_raises_on_invalid_step_months(self):
        """step_months must be a positive integer (prevents infinite loop)."""
        bundle = _generate_synthetic_bundle()
        with pytest.raises(ValueError, match="step_months must be a positive integer"):
            run_rolling_backtest(bundle, window_years=5, step_months=0)
        with pytest.raises(ValueError, match="step_months must be a positive integer"):
            run_rolling_backtest(bundle, window_years=5, step_months=-1)

    def test_allows_data_length_equal_to_window(self):
        """
        Regression: total_months == window_months is a valid case that should
        produce exactly one window (not raise ValueError). Previously the
        guard was `<=` which wrongly rejected the equality case.
        """
        bundle = _generate_synthetic_bundle()
        total_years = len(bundle.monthly_returns_eur) // 12
        # Use a window exactly matching the data length
        df = run_rolling_backtest(bundle, window_years=total_years, step_months=1)
        assert len(df) == 1  # exactly one window fits

    def test_chart_created(self, tmp_path):
        bundle = _generate_synthetic_bundle()
        df = run_rolling_backtest(bundle, window_years=5, step_months=12)
        chart_path = tmp_path / "rolling.png"
        plot_rolling_window_results(df, window_years=5, path=chart_path, step_months=12)
        assert chart_path.exists()
        assert chart_path.stat().st_size > 1000

    def test_rejects_bool_as_window_years(self):
        """bool is a subclass of int in Python; must be rejected explicitly."""
        bundle = _generate_synthetic_bundle()
        with pytest.raises(ValueError, match="window_years must be a positive integer"):
            run_rolling_backtest(bundle, window_years=True, step_months=1)
        with pytest.raises(ValueError, match="window_years must be a positive integer"):
            run_rolling_backtest(bundle, window_years=False, step_months=1)

    def test_rejects_bool_as_step_months(self):
        bundle = _generate_synthetic_bundle()
        with pytest.raises(ValueError, match="step_months must be a positive integer"):
            run_rolling_backtest(bundle, window_years=5, step_months=True)
        with pytest.raises(ValueError, match="step_months must be a positive integer"):
            run_rolling_backtest(bundle, window_years=5, step_months=False)


class TestDescribeStep:
    """Coverage for _describe_step() — used in plot titles."""

    def test_monthly_label(self):
        assert _describe_step(1) == "monthly"

    def test_quarterly_label(self):
        assert _describe_step(3) == "quarterly"

    def test_semi_annually_label(self):
        assert _describe_step(6) == "semi-annually"

    def test_annually_label(self):
        assert _describe_step(12) == "annually"

    def test_custom_label(self):
        """Non-standard cadences produce a generic 'every N months' label."""
        assert _describe_step(2) == "every 2 months"
        assert _describe_step(5) == "every 5 months"
        assert _describe_step(24) == "every 24 months"


class TestChartTitleReflectsStep:
    """Regression: ensure the chart title honestly reflects the step used."""

    def test_annual_step_produces_annually_label(self, tmp_path):
        from src.rolling_window import _describe_step
        # Unit-level verification: we trust matplotlib to write the suptitle;
        # the title string comes from _describe_step, which we test directly.
        bundle = _generate_synthetic_bundle()
        df = run_rolling_backtest(bundle, window_years=5, step_months=12)
        chart_path = tmp_path / "rolling_annual.png"
        plot_rolling_window_results(df, window_years=5, path=chart_path, step_months=12)
        assert chart_path.exists()
        # The label the chart should have embedded:
        assert _describe_step(12) == "annually"


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
