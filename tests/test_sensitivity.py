"""
Unit tests for src/sensitivity.py — programmatic parameter sweep.

PR9: the legacy globals-mutating path was retired. All sweeps now go
through the generic copy-based path (``_apply_param_override_on_portfolio``),
with optional ``absorb_from`` to control which sleeve absorbs the weight
delta. CLI auto-defaults absorb_from to a sibling equity sleeve when
running on the default Four Umbrellas preset, preserving the legacy
sensitivity output byte-identically.
"""


import matplotlib
matplotlib.use("Agg")  # non-interactive backend for headless/CI runs
import pandas as pd
import pytest

from src.sensitivity import (
    run_sensitivity_sweep,
    plot_sensitivity_results,
    parse_range_to_values,
    _apply_param_override_on_portfolio,
    SUPPORTED_PARAMS,
)
from src.data_loader import _generate_synthetic_bundle
from src import portfolio as portfolio_cfg
from src.portfolio_model import AssetAllocation, Portfolio
from src.rebalance import build_portfolio_from_globals


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


class TestApplyOverridePortfolio:
    """PR9: the only override entry point. Operates on a Portfolio copy,
    never touches module globals."""

    def _fu(self) -> Portfolio:
        return build_portfolio_from_globals()

    def test_gold_override_absorbs_from_cash_by_default(self):
        p = self._fu()
        new = _apply_param_override_on_portfolio(p, "gold", 0.25)
        weights = {a.key: a.weight for a in new.assets}
        assert weights["gold"] == pytest.approx(0.25)
        # Total preserved
        assert sum(weights.values()) == pytest.approx(sum(a.weight for a in p.assets))

    def test_value_must_be_finite(self):
        p = self._fu()
        with pytest.raises(ValueError, match=r"finite number"):
            _apply_param_override_on_portfolio(p, "gold", float("nan"))
        with pytest.raises(ValueError, match=r"finite number"):
            _apply_param_override_on_portfolio(p, "gold", float("inf"))
        with pytest.raises(ValueError, match=r"finite number"):
            _apply_param_override_on_portfolio(p, "gold", "not a number")

    def test_weight_out_of_range_raises(self):
        p = self._fu()
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            _apply_param_override_on_portfolio(p, "gold", -0.1)
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            _apply_param_override_on_portfolio(p, "gold", 1.1)

    def test_gold_override_would_drive_cash_negative(self):
        p = self._fu()
        # gold default ~0.18, cash ~0.11 -> gold=0.50 -> cash negative
        with pytest.raises(ValueError, match=r"drive cash negative"):
            _apply_param_override_on_portfolio(p, "gold", 0.50)

    def test_rebalance_freq_rejects_non_integer(self):
        p = self._fu()
        with pytest.raises(ValueError, match=r"must be an integer"):
            _apply_param_override_on_portfolio(p, "rebalance_freq", 2.5)

    def test_rebalance_freq_rejects_non_divisor(self):
        p = self._fu()
        with pytest.raises(ValueError, match=r"divisor of 12"):
            _apply_param_override_on_portfolio(p, "rebalance_freq", 5)
        with pytest.raises(ValueError, match=r"divisor of 12"):
            _apply_param_override_on_portfolio(p, "rebalance_freq", 7)

    def test_unsupported_param_raises(self):
        p = self._fu()
        with pytest.raises(ValueError, match=r"Unsupported sensitivity parameter"):
            _apply_param_override_on_portfolio(p, "unknown_param", 0.1)


class TestAbsorbFrom:
    """PR9: ``absorb_from`` selects which sleeve absorbs the weight delta.
    Default ``None`` means cash. Used by the CLI to preserve legacy
    equity-internal absorption on the FU preset."""

    def _fu(self) -> Portfolio:
        return build_portfolio_from_globals()

    def test_absorb_from_sibling_preserves_other_sleeves(self):
        """Sweeping put_write with absorb_from=nasdaq_top30: only those
        two assets change, all others identical."""
        p = self._fu()
        new = _apply_param_override_on_portfolio(
            p, "put_write", 0.20, absorb_from="nasdaq_top30",
        )
        old = {a.key: a.weight for a in p.assets}
        new_w = {a.key: a.weight for a in new.assets}
        assert new_w["put_write"] == pytest.approx(0.20)
        # cash, gold, dbi etc. unchanged
        for key, val in old.items():
            if key not in ("put_write", "nasdaq_top30"):
                assert new_w[key] == val
        # Delta absorbed by nasdaq_top30
        assert new_w["nasdaq_top30"] == pytest.approx(
            old["nasdaq_top30"] - (0.20 - old["put_write"])
        )

    def test_absorb_from_self_rejected(self):
        p = self._fu()
        with pytest.raises(ValueError, match=r"cannot equal the swept parameter"):
            _apply_param_override_on_portfolio(p, "gold", 0.20, absorb_from="gold")

    def test_absorb_from_unknown_asset_rejected(self):
        p = self._fu()
        with pytest.raises(ValueError, match=r"no asset named 'doesnotexist'"):
            _apply_param_override_on_portfolio(
                p, "gold", 0.20, absorb_from="doesnotexist",
            )

    def test_absorb_from_negative_capacity_rejected(self):
        """If the named absorber doesn't have enough weight to take the
        delta, raise rather than silently produce a negative weight."""
        p = self._fu()
        # nasdaq_top30 default ~0.117 — sweeping put_write to 0.30 would
        # drive nasdaq to -0.05.
        with pytest.raises(ValueError, match=r"drive nasdaq_top30 negative"):
            _apply_param_override_on_portfolio(
                p, "put_write", 0.30, absorb_from="nasdaq_top30",
            )

    def test_absorb_from_works_on_custom_portfolio(self):
        """absorb_from also works on user-provided portfolios — sweeping
        gold absorbs from a chosen non-cash sleeve."""
        p = Portfolio(
            name="GoldVsQuality",
            assets=[
                AssetAllocation("gold", 0.30),
                AssetAllocation("quality", 0.50),
                AssetAllocation("cash", 0.20),
            ],
        )
        new = _apply_param_override_on_portfolio(
            p, "gold", 0.40, absorb_from="quality",
        )
        weights = {a.key: a.weight for a in new.assets}
        assert weights["gold"] == pytest.approx(0.40)
        assert weights["quality"] == pytest.approx(0.40)  # absorbed -0.10
        assert weights["cash"] == pytest.approx(0.20)  # unchanged


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

    def test_sweep_does_not_mutate_globals(self):
        """PR9: sweep on the default preset (portfolio=None) goes through
        the copy-based generic path. The src.portfolio module globals
        must remain unchanged after the sweep."""
        bundle = _generate_synthetic_bundle()
        before_gold = portfolio_cfg.WEIGHTS["gold"]
        before_cash = portfolio_cfg.WEIGHTS["cash"]
        before_equity = dict(portfolio_cfg.EQUITY)
        before_options_budget = portfolio_cfg.OPTIONS.budget_nav_per_year
        before_rebalance = tuple(portfolio_cfg.REBALANCE.months)

        _ = run_sensitivity_sweep(bundle, "gold", [0.10, 0.25])
        _ = run_sensitivity_sweep(bundle, "options_budget", [0.001, 0.005])
        _ = run_sensitivity_sweep(
            bundle, "put_write", [0.10, 0.15], absorb_from="nasdaq_top30",
        )
        _ = run_sensitivity_sweep(bundle, "rebalance_freq", [1, 4])

        assert portfolio_cfg.WEIGHTS["gold"] == before_gold
        assert portfolio_cfg.WEIGHTS["cash"] == before_cash
        assert dict(portfolio_cfg.EQUITY) == before_equity
        assert portfolio_cfg.OPTIONS.budget_nav_per_year == before_options_budget
        assert tuple(portfolio_cfg.REBALANCE.months) == before_rebalance

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

    def test_sweep_custom_portfolio_options_budget_supported(self):
        """PR7: options_budget on a custom portfolio is now supported via
        Portfolio.options_config. The sweep must run end-to-end and
        produce a DataFrame with one row per value."""
        bundle = _generate_synthetic_bundle()
        df = run_sensitivity_sweep(
            bundle, "options_budget", [0.001, 0.005, 0.01],
            portfolio=self._toy_portfolio(),
        )
        assert len(df) == 3
        assert list(df.index) == [0.001, 0.005, 0.01]

    def test_sweep_options_budget_does_not_mutate_globals_or_input(self):
        """Critical PR7 invariant: sweeping options_budget on a custom
        Portfolio must not mutate the global OPTIONS, nor the input
        Portfolio's options_config."""
        from src.portfolio import OPTIONS

        bundle = _generate_synthetic_bundle()
        before_global_budget = OPTIONS.budget_nav_per_year
        p = self._toy_portfolio()
        # No options_config on the input -> stays None after the sweep
        assert p.options_config is None

        run_sensitivity_sweep(
            bundle, "options_budget", [0.002, 0.008],
            portfolio=p,
        )
        assert OPTIONS.budget_nav_per_year == before_global_budget
        assert p.options_config is None  # input unchanged

    def test_sweep_options_budget_produces_varying_cagr(self):
        """The sweep must actually be effective — different budget values
        must yield different CAGRs (overlay is included on the generic
        path when sweeping options_budget). A CAGR series with all
        identical values would mean the budget change isn't reaching
        the engine."""
        bundle = _generate_synthetic_bundle()
        df = run_sensitivity_sweep(
            bundle, "options_budget", [0.0001, 0.005, 0.01],
            portfolio=self._toy_portfolio(),
        )
        # At minimum, not all 3 CAGRs should be exactly equal (overlay
        # cost scales with budget so flat market means lower budget
        # = less premium spent = different total return).
        assert df["cagr"].nunique() > 1

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
