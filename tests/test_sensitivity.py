"""
Unit tests for src/sensitivity.py — programmatic parameter sweep.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for headless/CI runs
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

    def test_step_must_be_positive(self):
        """step <= 0 should raise ValueError (avoids numpy edge cases / infinite loops)."""
        with pytest.raises(ValueError, match=r"--step must be > 0"):
            parse_range_to_values(range_tuple=(0.10, 0.25), step=0, values_list=None)
        with pytest.raises(ValueError, match=r"--step must be > 0"):
            parse_range_to_values(range_tuple=(0.10, 0.25), step=-0.01, values_list=None)

    def test_low_must_be_le_high(self):
        """low > high should raise (avoids silently empty result from np.arange)."""
        with pytest.raises(ValueError, match=r"must be <= high"):
            parse_range_to_values(range_tuple=(0.25, 0.10), step=0.025, values_list=None)


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

    def test_value_must_be_finite(self):
        """NaN / Inf / non-numeric value should raise ValueError."""
        snap = _snapshot_config()
        try:
            with pytest.raises(ValueError, match=r"finite number"):
                _apply_param_override("gold", float("nan"))
            with pytest.raises(ValueError, match=r"finite number"):
                _apply_param_override("gold", float("inf"))
            with pytest.raises(ValueError, match=r"finite number"):
                _apply_param_override("gold", "not a number")
        finally:
            _restore_config(snap)

    def test_weight_out_of_range_raises(self):
        """Weight-like params must be in [0, 1]."""
        snap = _snapshot_config()
        try:
            with pytest.raises(ValueError, match=r"\[0, 1\]"):
                _apply_param_override("gold", -0.1)
            with pytest.raises(ValueError, match=r"\[0, 1\]"):
                _apply_param_override("gold", 1.1)
        finally:
            _restore_config(snap)

    def test_gold_override_would_drive_cash_negative(self):
        """Sweeping gold too high should raise rather than silently producing cash < 0."""
        snap = _snapshot_config()
        try:
            # Cash default is 0.11; gold default ~0.18. Setting gold=0.5 → delta=0.32 → cash=-0.21
            with pytest.raises(ValueError, match=r"drive cash negative"):
                _apply_param_override("gold", 0.50)
        finally:
            _restore_config(snap)

    def test_rebalance_freq_rejects_non_integer(self):
        """Floats that are not exact integers must be rejected (no silent truncation)."""
        snap = _snapshot_config()
        try:
            with pytest.raises(ValueError, match=r"must be an integer"):
                _apply_param_override("rebalance_freq", 2.5)
        finally:
            _restore_config(snap)

    def test_rebalance_freq_rejects_non_divisor(self):
        """5 is a valid positive integer but not a divisor of 12 → must raise."""
        snap = _snapshot_config()
        try:
            with pytest.raises(ValueError, match=r"divisor of 12"):
                _apply_param_override("rebalance_freq", 5)
            with pytest.raises(ValueError, match=r"divisor of 12"):
                _apply_param_override("rebalance_freq", 7)
        finally:
            _restore_config(snap)

    def test_equity_sleeve_cannot_exceed_absorb_capacity(self):
        """Setting equity sleeve too high should raise rather than drive absorb_key negative."""
        snap = _snapshot_config()
        try:
            # put_write default ~0.131, nasdaq_top30 default ~0.117 (absorb target)
            # Setting put_write=0.30 → delta=0.17 → nasdaq = 0.117 - 0.17 = -0.05 (negative)
            with pytest.raises(ValueError, match=r"drive .* negative"):
                _apply_param_override("put_write", 0.30)
        finally:
            _restore_config(snap)


class TestSweepValidation:
    def test_empty_values_raises(self):
        """run_sensitivity_sweep with empty values list must raise ValueError."""
        bundle = _generate_synthetic_bundle()
        with pytest.raises(ValueError, match=r"empty"):
            run_sensitivity_sweep(bundle, "gold", [])


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


class TestSweepWithCustomPortfolio:
    """PR3: run_sensitivity_sweep accepts a Portfolio and sweeps on a COPY,
    never touching the src.portfolio globals."""

    def _toy_portfolio(self):
        from src.portfolio_model import AssetAllocation, Portfolio
        return Portfolio(
            name="Toy",
            assets=[
                AssetAllocation("gold", 0.3),
                AssetAllocation("quality", 0.5),
                AssetAllocation("cash", 0.2),
            ],
        )

    def test_sweep_on_custom_portfolio_runs(self):
        bundle = _generate_synthetic_bundle()
        df = run_sensitivity_sweep(
            bundle, "gold", [0.10, 0.20, 0.30],
            portfolio=self._toy_portfolio(),
        )
        assert len(df) == 3
        assert list(df.index) == [0.10, 0.20, 0.30]
        # CAGR should move as gold weight changes
        assert df["cagr"].nunique() > 1

    def test_sweep_custom_portfolio_does_not_mutate_globals(self):
        """Critical PR3 invariant: the custom-portfolio sweep path never
        touches the module globals. Unlike the legacy path which snapshots
        and restores, this path simply doesn't write to them."""
        bundle = _generate_synthetic_bundle()
        before_gold = portfolio_cfg.WEIGHTS["gold"]
        before_cash = portfolio_cfg.WEIGHTS["cash"]
        before_quality = portfolio_cfg.EQUITY["quality"]

        run_sensitivity_sweep(
            bundle, "gold", [0.10, 0.30, 0.50],
            portfolio=self._toy_portfolio(),
        )

        assert portfolio_cfg.WEIGHTS["gold"] == before_gold
        assert portfolio_cfg.WEIGHTS["cash"] == before_cash
        assert portfolio_cfg.EQUITY["quality"] == before_quality

    def test_sweep_custom_portfolio_does_not_mutate_input(self):
        """The input Portfolio object is not modified — we operate on copies."""
        bundle = _generate_synthetic_bundle()
        p = self._toy_portfolio()
        original_weights = tuple((a.key, a.weight) for a in p.assets)
        run_sensitivity_sweep(bundle, "gold", [0.10, 0.20], portfolio=p)
        assert tuple((a.key, a.weight) for a in p.assets) == original_weights

    def test_sweep_custom_portfolio_missing_asset_errors(self):
        """Sweeping a param that doesn't correspond to an asset in the
        portfolio raises a clear ValueError naming the missing asset."""
        from src.portfolio_model import AssetAllocation, Portfolio
        bundle = _generate_synthetic_bundle()
        # A portfolio without 'put_write' — sweeping it must fail
        p = Portfolio(
            name="NoPutWrite",
            assets=[
                AssetAllocation("gold", 0.5),
                AssetAllocation("cash", 0.5),
            ],
        )
        with pytest.raises(ValueError, match="no asset named 'put_write'"):
            run_sensitivity_sweep(bundle, "put_write", [0.10, 0.15], portfolio=p)

    def test_sweep_custom_portfolio_without_cash_errors(self):
        """The generic path absorbs the delta into cash. If the portfolio has
        no cash sleeve, the sweep must fail loudly rather than silently
        breaking the 100% weight sum."""
        from src.portfolio_model import AssetAllocation, Portfolio
        bundle = _generate_synthetic_bundle()
        p = Portfolio(
            name="NoCash",
            assets=[
                AssetAllocation("gold", 0.6),
                AssetAllocation("quality", 0.4),
            ],
        )
        with pytest.raises(ValueError, match="no 'cash' sleeve"):
            run_sensitivity_sweep(bundle, "gold", [0.10], portfolio=p)

    def test_sweep_custom_portfolio_options_budget_not_supported(self):
        """options_budget on a custom portfolio is explicitly blocked in PR3
        — PR5 will introduce a per-Portfolio OptionsConfig that enables it."""
        bundle = _generate_synthetic_bundle()
        with pytest.raises(NotImplementedError, match="PR5"):
            run_sensitivity_sweep(
                bundle, "options_budget", [0.003, 0.005],
                portfolio=self._toy_portfolio(),
            )

    def test_sweep_custom_portfolio_rebalance_freq(self):
        """rebalance_freq IS supported on the generic path — rebalance_months
        sits on the Portfolio dataclass so we can vary it without touching
        globals."""
        bundle = _generate_synthetic_bundle()
        df = run_sensitivity_sweep(
            bundle, "rebalance_freq", [1, 2, 4],
            portfolio=self._toy_portfolio(),
        )
        assert len(df) == 3
