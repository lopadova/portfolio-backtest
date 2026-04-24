"""Tests for the PR4 portfolio-aware helpers in src/dashboard_helpers.py.

These helpers are pure Python (no Streamlit runtime import) so they
exercise cleanly under pytest. The legacy globals-mutation helpers
(snapshot/restore/apply_macro_weights) are covered by
``tests/test_dashboard_smoke.py`` and intentionally not re-tested here.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.dashboard_helpers import (
    build_portfolio_from_ui_state,
    compute_effective_start,
    format_asset_start_date,
)
from src.data_catalog import AssetInfo
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
        # gold starts 2003-01-31 → "Gen 2003"
        assert format_asset_start_date(info) == "Gen 2003"

    def test_btc_september(self):
        info = _toy_catalog()["btc"]
        # btc starts 2014-09-17 → "Set 2014"
        assert format_asset_start_date(info) == "Set 2014"

    def test_none_asset_info(self):
        assert format_asset_start_date(None) == "—"

    def test_no_start_date(self):
        info = _toy_catalog()["cash"]
        assert format_asset_start_date(info) == "—"

    def test_all_months_mapped(self):
        """Every month 1-12 maps to a recognized Italian abbreviation."""
        expected = {
            1: "Gen", 2: "Feb", 3: "Mar", 4: "Apr", 5: "Mag", 6: "Giu",
            7: "Lug", 8: "Ago", 9: "Set", 10: "Ott", 11: "Nov", 12: "Dic",
        }
        for month, abbr in expected.items():
            info = AssetInfo(
                key="x", display_name="x", filename="x",
                category="cash", native_ccy="EUR", is_hedged=False, ter=0.0,
                start_date=pd.Timestamp(year=2020, month=month, day=1),
            )
            assert format_asset_start_date(info) == f"{abbr} 2020"
