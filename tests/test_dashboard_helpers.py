"""Tests for the portfolio-aware helpers in src/dashboard_helpers.py.

These helpers are pure Python (no Streamlit runtime import) so they
exercise cleanly under pytest. The legacy globals-mutation helpers
(snapshot_config / restore_config / apply_macro_weights) were retired
in PR6 together with their dedicated smoke-test file.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.dashboard_helpers import (
    build_portfolio_from_ui_state,
    common_history_years,
    compute_effective_start,
    fire_config_from_ui_state,
    format_asset_start_date,
)
from src.data_catalog import AssetInfo
from src.data_loader import _generate_synthetic_bundle
from src.fire import FireConfig
from src.portfolio_model import AssetAllocation, Portfolio


# ----------------------------- fixtures -------------------------------------


def _toy_catalog() -> dict:
    """Minimal catalog with 3 assets: two with start dates (gold, btc) and
    one with ``start_date=None`` (cash, which doesn't constrain windows)."""
    return {
        "gold": AssetInfo(
            key="gold", display_name="Gold",
            filename="gold_lbma_monthly", category="commodity",
            native_ccy="USD", is_hedged=False, ter=0.0012,
            start_date=pd.Timestamp("2003-01-31"),
            end_date=pd.Timestamp("2024-12-31"),
        ),
        "btc": AssetInfo(
            key="btc", display_name="Bitcoin (USD)",
            filename="btc_usd_daily", category="crypto",
            native_ccy="USD", is_hedged=False, ter=0.0098,
            start_date=pd.Timestamp("2014-09-17"),
            end_date=pd.Timestamp("2024-12-31"),
        ),
        "cash": AssetInfo(
            key="cash", display_name="Cash",
            filename="__cash__", category="cash",
            native_ccy="EUR", is_hedged=True, ter=0.0,
            start_date=None, end_date=None,
        ),
    }


# ---------------------- build_portfolio_from_ui_state -----------------------


class TestBuildPortfolioFromUiState:
    def test_builds_valid(self):
        p = build_portfolio_from_ui_state(
            name="Test",
            assets_state=[
                {"key": "gold", "weight": 0.5},
                {"key": "cash", "weight": 0.5},
            ],
        )
        assert p.name == "Test"
        assert len(p.assets) == 2
        assert p.to_weights_dict() == {"gold": 0.5, "cash": 0.5}

    def test_defaults_match_portfolio_defaults(self):
        p = build_portfolio_from_ui_state(
            "X", [{"key": "gold", "weight": 1.0}],
        )
        assert p.options_overlay is False
        assert p.rebalance_months == (1, 7)
        assert p.transaction_cost_bps == 20.0
        assert p.notes == ""

    def test_overrides_pass_through(self):
        p = build_portfolio_from_ui_state(
            "X",
            [{"key": "gold", "weight": 1.0}],
            options_overlay=True,
            rebalance_months=(1, 4, 7, 10),
            transaction_cost_bps=10.0,
            notes="trial",
        )
        assert p.options_overlay is True
        assert p.rebalance_months == (1, 4, 7, 10)
        assert p.transaction_cost_bps == 10.0
        assert p.notes == "trial"

    def test_validation_error_propagates(self):
        """Sum != 1.0 must bubble up as ValueError (so the UI can surface it
        via st.error). We don't swallow the validator's message."""
        with pytest.raises(ValueError, match="sum to"):
            build_portfolio_from_ui_state(
                "X",
                [{"key": "gold", "weight": 0.5}, {"key": "cash", "weight": 0.3}],
            )

    def test_weight_is_coerced_to_float(self):
        """st.data_editor may hand back ints or numpy scalars for weights —
        build_portfolio_from_ui_state must coerce."""
        p = build_portfolio_from_ui_state(
            "X",
            [{"key": "gold", "weight": 1}],  # int, not float
        )
        assert p.assets[0].weight == 1.0
        assert isinstance(p.assets[0].weight, float)

    def test_empty_assets_errors(self):
        with pytest.raises(ValueError, match="at least one allocation"):
            build_portfolio_from_ui_state("X", [])

    def test_empty_name_errors(self):
        with pytest.raises(ValueError, match="non-empty string"):
            build_portfolio_from_ui_state(
                "", [{"key": "gold", "weight": 1.0}]
            )


# ---------------------- compute_effective_start -----------------------------


class TestComputeEffectiveStart:
    def test_no_constraint_returns_user_start(self):
        """All portfolio assets have data available before user_start."""
        catalog = _toy_catalog()
        p = Portfolio(
            name="P",
            assets=[AssetAllocation("gold", 0.5), AssetAllocation("cash", 0.5)],
        )
        user_start = pd.Timestamp("2020-01-01")
        effective, warning = compute_effective_start(user_start, p, catalog)
        assert effective == user_start
        assert warning is None

    def test_btc_constrains_early_start(self):
        """BTC (start 2014-09) pushes a 2005 user_start forward."""
        catalog = _toy_catalog()
        p = Portfolio(
            name="P",
            assets=[
                AssetAllocation("gold", 0.3),
                AssetAllocation("btc", 0.3),
                AssetAllocation("cash", 0.4),
            ],
        )
        user_start = pd.Timestamp("2005-01-01")
        effective, warning = compute_effective_start(user_start, p, catalog)
        assert effective == pd.Timestamp("2014-09-17")
        assert warning is not None
        assert "Bitcoin" in warning
        assert "2014-09-17" in warning

    def test_cash_does_not_constrain(self):
        """Cash has start_date=None — it must not enter the max() calculation."""
        catalog = _toy_catalog()
        p = Portfolio(
            name="P",
            assets=[AssetAllocation("gold", 0.5), AssetAllocation("cash", 0.5)],
        )
        user_start = pd.Timestamp("2001-01-01")   # before gold's 2003 start
        effective, warning = compute_effective_start(user_start, p, catalog)
        # Only gold constrains. cash must not appear in the warning.
        assert effective == pd.Timestamp("2003-01-31")
        assert warning is not None
        assert "Cash" not in warning
        assert "Gold" in warning

    def test_unknown_asset_in_catalog_ignored(self):
        """Assets not in the catalog (user typo, stale reference) don't crash
        the helper — they're skipped for effective-start calculation. This
        matches current downstream behavior, where unknown asset keys may be
        silently dropped if they are missing from ``monthly_returns``."""
        catalog = _toy_catalog()
        p = Portfolio(
            name="P",
            assets=[
                AssetAllocation("gold", 0.5),
                AssetAllocation("not_in_catalog", 0.3),
                AssetAllocation("cash", 0.2),
            ],
        )
        # Should not raise; effective start driven by gold alone.
        effective, warning = compute_effective_start(
            pd.Timestamp("2000-01-01"), p, catalog,
        )
        assert effective == pd.Timestamp("2003-01-31")
        assert warning is not None

    def test_equal_start_returns_user_start(self):
        """If user_start exactly equals the latest asset start, no warning."""
        catalog = _toy_catalog()
        p = Portfolio(
            name="P",
            assets=[AssetAllocation("gold", 1.0)],
        )
        user_start = pd.Timestamp("2003-01-31")
        effective, warning = compute_effective_start(user_start, p, catalog)
        assert effective == user_start
        assert warning is None


# ---------------------- format_asset_start_date -----------------------------


class TestFormatAssetStartDate:
    def test_with_start_date(self):
        info = _toy_catalog()["gold"]
        # gold starts 2003-01-31 → "Jan 2003"
        assert format_asset_start_date(info) == "Jan 2003"

    def test_btc_september(self):
        info = _toy_catalog()["btc"]
        # btc starts 2014-09-17 → "Sep 2014"
        assert format_asset_start_date(info) == "Sep 2014"

    def test_none_asset_info(self):
        assert format_asset_start_date(None) == "—"

    def test_no_start_date(self):
        info = _toy_catalog()["cash"]
        assert format_asset_start_date(info) == "—"

    def test_all_months_mapped(self):
        """Every month 1-12 maps to a recognized English abbreviation."""
        expected = {
            1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
            7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec",
        }
        for month, abbr in expected.items():
            info = AssetInfo(
                key="x", display_name="x", filename="x",
                category="cash", native_ccy="EUR", is_hedged=False, ter=0.0,
                start_date=pd.Timestamp(year=2020, month=month, day=1),
            )
            assert format_asset_start_date(info) == f"{abbr} 2020"


# ---------------------- common_history_years (PR5) --------------------------


class TestCommonHistoryYears:
    """Post-Copilot-review (PR #19): this helper now computes from the
    ACTUAL bundle (``monthly_returns_eur`` index + per-column
    first_valid_index), not from the catalog's ``AssetInfo.start_date``.
    This guarantees the gate agrees with what ``run_rolling_backtest`` sees.
    """

    def _synthetic_bundle(self):
        """Synthetic bundle: monthly index spans 2005-01-31 → 2024-12-31
        (~20 years) with every sleeve fully populated (no NaN prefix)."""
        return _generate_synthetic_bundle()

    def test_full_synthetic_bundle_yields_20_years(self):
        """Every asset is populated from the first bundle row → common
        history is the full bundle span."""
        bundle = self._synthetic_bundle()
        p = Portfolio(name="P", assets=[AssetAllocation("gold", 1.0)])
        years = common_history_years(bundle, p)
        # Synthetic bundle is 2005-01-31 through 2024-12-31 → ~20 years
        assert 19.5 <= years <= 20.5

    def test_multiple_assets_full_history(self):
        bundle = self._synthetic_bundle()
        p = Portfolio(
            name="P",
            assets=[
                AssetAllocation("gold", 0.4),
                AssetAllocation("quality", 0.4),
                AssetAllocation("momentum", 0.2),
            ],
        )
        years = common_history_years(bundle, p)
        assert 19.5 <= years <= 20.5

    def test_cash_does_not_constrain(self):
        """Adding cash must not shorten the common history — cash is always
        available."""
        bundle = self._synthetic_bundle()
        p_with_cash = Portfolio(
            name="With cash",
            assets=[AssetAllocation("gold", 0.5), AssetAllocation("cash", 0.5)],
        )
        p_without_cash = Portfolio(
            name="No cash",
            assets=[AssetAllocation("gold", 1.0)],
        )
        assert common_history_years(bundle, p_with_cash) == pytest.approx(
            common_history_years(bundle, p_without_cash),
        )

    def test_asset_missing_from_bundle_columns_returns_zero(self):
        """If a portfolio asset key isn't in the bundle columns,
        simulate_portfolio_generic would silently drop it — so the
        walk-forward gate must hard-fail to 0 rather than pretend the
        portfolio has coverage it doesn't."""
        bundle = self._synthetic_bundle()
        p = Portfolio(
            name="P",
            assets=[
                AssetAllocation("gold", 0.5),
                AssetAllocation("not_in_bundle", 0.5),
            ],
        )
        assert common_history_years(bundle, p) == 0.0

    def test_nan_prefix_shortens_common_history(self):
        """Asset with NaN values at the start of the bundle (e.g. BTC
        before activation) constrains common history to the first valid
        date across all assets."""
        bundle = self._synthetic_bundle()
        # Forcibly NaN the first 60 months (5 years) of gold in a COPY
        # so we don't mutate the shared synthetic bundle fixture.
        import copy as _copy
        bundle_patched = _copy.deepcopy(bundle)
        bundle_patched.monthly_returns_eur.iloc[:60, bundle.monthly_returns_eur.columns.get_loc("gold")] = float("nan")

        p = Portfolio(
            name="P",
            assets=[AssetAllocation("gold", 0.5), AssetAllocation("cash", 0.5)],
        )
        years = common_history_years(bundle_patched, p)
        # Bundle is ~20 years; 5 years of leading NaN → ~15 years of common history.
        assert 14.5 <= years <= 15.5

    def test_sliced_bundle_reduces_history(self):
        """Slicing the bundle to a narrower range must reduce common history
        proportionally — proves the helper reads from the actual bundle index,
        not from catalog metadata."""
        bundle = self._synthetic_bundle()
        sliced = bundle.slice(pd.Timestamp("2015-01-31"), pd.Timestamp("2024-12-31"))
        p = Portfolio(name="P", assets=[AssetAllocation("gold", 1.0)])
        years = common_history_years(sliced, p)
        assert 9.5 <= years <= 10.5

    def test_returns_zero_for_empty_bundle(self):
        """Edge case: bundle has no rows. The helper must not crash."""
        bundle = self._synthetic_bundle()
        import copy as _copy
        empty = _copy.deepcopy(bundle)
        empty.monthly_returns_eur = empty.monthly_returns_eur.iloc[0:0]
        p = Portfolio(name="P", assets=[AssetAllocation("gold", 1.0)])
        assert common_history_years(empty, p) == 0.0


# ---------------------- fire_config_from_ui_state (PR5) ---------------------


def _valid_fire_state() -> dict:
    return {
        "fire_current_age": 45,
        "fire_sex": "M",
        "fire_fixed_end_age": None,
        "fire_initial_capital": 100_000.0,
        "fire_contribution_amount": 1_000.0,
        "fire_contribution_frequency": "month",
        "fire_fire_age": 60,
        "fire_monthly_spending": 2_500.0,
        "fire_inflation_rate": 0.02,
        "fire_pension_enabled": True,
        "fire_pension_monthly_amount": 1_500.0,
        "fire_pension_start_age": 67,
        "fire_pension_revaluation": 0.75,
        "fire_tax_on_withdrawals": True,
        "fire_tax_rate": 0.26,
        "fire_n_simulations": 1000,
        "fire_block_size": 3,
        "fire_seed": 42,
    }


class TestFireConfigFromUiState:
    def test_builds_valid_config(self):
        cfg = fire_config_from_ui_state(_valid_fire_state())
        assert isinstance(cfg, FireConfig)
        assert cfg.current_age == 45
        assert cfg.fire_age == 60
        assert cfg.pension_enabled is True
        assert cfg.pension_monthly_amount == 1_500.0

    def test_fire_age_must_exceed_current_age(self):
        state = _valid_fire_state()
        state["fire_fire_age"] = state["fire_current_age"]   # invalid: equal
        with pytest.raises(ValueError, match="strictly greater than current age"):
            fire_config_from_ui_state(state)

    def test_negative_spending_rejected(self):
        state = _valid_fire_state()
        state["fire_monthly_spending"] = -100
        with pytest.raises(ValueError, match="Monthly spending"):
            fire_config_from_ui_state(state)

    def test_inflation_out_of_range(self):
        state = _valid_fire_state()
        state["fire_inflation_rate"] = 0.5  # 50% — out of [0, 0.25]
        with pytest.raises(ValueError, match="Inflation rate"):
            fire_config_from_ui_state(state)

    def test_fixed_end_age_none_treated_as_unset(self):
        state = _valid_fire_state()
        state["fire_fixed_end_age"] = None
        cfg = fire_config_from_ui_state(state)
        assert cfg.fixed_end_age is None

    def test_fixed_end_age_zero_treated_as_unset(self):
        """UI number_input often uses 0 as 'no value set'."""
        state = _valid_fire_state()
        state["fire_fixed_end_age"] = 0
        cfg = fire_config_from_ui_state(state)
        assert cfg.fixed_end_age is None

    def test_pension_disabled_still_builds(self):
        state = _valid_fire_state()
        state["fire_pension_enabled"] = False
        cfg = fire_config_from_ui_state(state)
        assert cfg.pension_enabled is False
        # Pension amount still in the dataclass (the simulator ignores it
        # when pension_enabled=False), just preserves what the UI had.

    def test_tax_rate_bounds(self):
        state = _valid_fire_state()
        state["fire_tax_rate"] = 1.5
        with pytest.raises(ValueError, match="Tax rate"):
            fire_config_from_ui_state(state)
