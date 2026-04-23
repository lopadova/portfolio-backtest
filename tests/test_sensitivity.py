"""
Unit tests for src/sensitivity.py — programmatic parameter sweep.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.sensitivity import (
    run_sensitivity_sweep,
    plot_sensitivity_results,
    parse_range_to_values,
    _snapshot_config,
    _restore_config,
    _apply_param_override,
    SUPPORTED_PARAMS,
)
from src.data_loader import _generate_synthetic_bundle
from src import portfolio as portfolio_cfg


class TestParseRangeToValues:
    def test_range_and_step(self):
        values = parse_range_to_values(range_tuple=(0.10, 0.25), step=0.025, values_list=None)
        assert len(values) >= 6
        assert values[0] == pytest.approx(0.10)
        assert values[-1] == pytest.approx(0.25, abs=0.01)

    def test_explicit_values_list(self):
        values = parse_range_to_values(range_tuple=None, step=None, values_list=[1.0, 2.0, 4.0, 12.0])
        assert values == [1.0, 2.0, 4.0, 12.0]

    def test_error_when_nothing_provided(self):
        with pytest.raises(ValueError):
            parse_range_to_values(range_tuple=None, step=None, values_list=None)


class TestApplyOverride:
    def test_gold_override_preserves_total(self):
        snap = _snapshot_config()
        try:
            original_total = sum(portfolio_cfg.WEIGHTS.values())
            _apply_param_override("gold", 0.25)
            new_total = sum(portfolio_cfg.WEIGHTS.values())
            assert abs(original_total - new_total) < 1e-9
            assert portfolio_cfg.WEIGHTS["gold"] == pytest.approx(0.25)
        finally:
            _restore_config(snap)

    def test_options_budget_override(self):
        snap = _snapshot_config()
        try:
            _apply_param_override("options_budget", 0.005)
            assert portfolio_cfg.OPTIONS.budget_nav_per_year == pytest.approx(0.005)
        finally:
            _restore_config(snap)

    def test_rebalance_freq_override(self):
        snap = _snapshot_config()
        try:
            _apply_param_override("rebalance_freq", 4)
            months = portfolio_cfg.REBALANCE.months
            assert len(months) == 4
            assert all(1 <= m <= 12 for m in months)
        finally:
            _restore_config(snap)

    def test_unsupported_param_raises(self):
        with pytest.raises(ValueError):
            _apply_param_override("unknown_param", 0.1)


class TestSweep:
    def test_sweep_returns_dataframe(self):
        bundle = _generate_synthetic_bundle()
        df = run_sensitivity_sweep(bundle, "gold", [0.15, 0.18, 0.20])
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3
        assert "cagr" in df.columns
        assert "max_drawdown" in df.columns

    def test_sweep_restores_config(self):
        bundle = _generate_synthetic_bundle()
        original_gold = portfolio_cfg.WEIGHTS["gold"]
        _ = run_sensitivity_sweep(bundle, "gold", [0.10, 0.25])
        # After the sweep, config must be back to original
        assert portfolio_cfg.WEIGHTS["gold"] == pytest.approx(original_gold)

    def test_sweep_chart_created(self, tmp_path):
        bundle = _generate_synthetic_bundle()
        df = run_sensitivity_sweep(bundle, "dbi", [0.03, 0.05, 0.07])
        chart_path = tmp_path / "dbi_sensitivity.png"
        plot_sensitivity_results(df, "dbi", chart_path)
        assert chart_path.exists()
        assert chart_path.stat().st_size > 1000  # non-trivial PNG


class TestSupportedParams:
    def test_supported_list_documented(self):
        expected = {"gold", "dbi", "options_budget", "rebalance_freq",
                    "put_write", "nasdaq_top30", "momentum", "quality"}
        assert set(SUPPORTED_PARAMS.keys()) == expected
