"""Consistency tests for portfolios/four_umbrellas.toml.

The preset MUST remain bit-equivalent to the legacy globals in
src/portfolio.py (WEIGHTS / EQUITY / CRYPTO / BONDS / EM_SATELLITES /
PENSION / REBALANCE / OPTIONS / TER_ANNUAL / HEDGED). When the legacy
globals are finally retired (PR3+), this test file becomes the sole
guardian of the Four Umbrellas preset's invariants.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.data_catalog import load_catalog
from src.portfolio_model import Portfolio

PRESET_PATH = (
    Path(__file__).resolve().parent.parent.parent / "portfolios" / "four_umbrellas.toml"
)


@pytest.fixture(scope="module")
def preset() -> Portfolio:
    return Portfolio.from_toml(PRESET_PATH)


class TestPresetStructure:
    def test_file_ships_with_repo(self):
        assert PRESET_PATH.is_file(), f"preset not found at {PRESET_PATH}"

    def test_loads_and_validates(self, preset: Portfolio):
        # Validation runs inside from_toml; a failure here means the preset's
        # weights drifted from the ±0.002 tolerance.
        preset.validate()

    def test_name(self, preset: Portfolio):
        assert preset.name == "Four Umbrellas"

    def test_options_overlay_enabled(self, preset: Portfolio):
        """Historical behavior: default backtest includes the options overlay."""
        assert preset.options_overlay is True

    def test_rebalance_months(self, preset: Portfolio):
        assert preset.rebalance_months == (1, 7)

    def test_transaction_cost(self, preset: Portfolio):
        assert preset.transaction_cost_bps == pytest.approx(20.0)


class TestPresetMatchesLegacyGlobals:
    """Every numeric value in the preset must match the corresponding value
    in src.portfolio module-level globals. This is the guardrail that keeps
    the preset and the legacy engine path consistent during PR2 coexistence."""

    def test_non_cash_weights_match_build_target_weights(self, preset: Portfolio):
        """to_legacy_target_weights() (preset minus cash) must equal the
        legacy build_target_weights() output, key-by-key, to tight tolerance."""
        from src.rebalance import build_target_weights

        legacy = build_target_weights()
        preset_flat = preset.to_legacy_target_weights()
        assert set(preset_flat.keys()) == set(legacy.keys()), (
            f"keys differ:\n"
            f"  only in preset: {set(preset_flat) - set(legacy)}\n"
            f"  only in legacy: {set(legacy) - set(preset_flat)}"
        )
        for k, legacy_w in legacy.items():
            assert preset_flat[k] == pytest.approx(legacy_w, rel=1e-6, abs=1e-9), (
                f"weight mismatch for {k!r}: preset {preset_flat[k]} vs legacy {legacy_w}"
            )

    def test_cash_matches_WEIGHTS(self, preset: Portfolio):
        from src.portfolio import WEIGHTS

        assert preset.cash_weight() == pytest.approx(WEIGHTS["cash"], rel=1e-6)

    def test_rebalance_matches_REBALANCE(self, preset: Portfolio):
        from src.portfolio import REBALANCE

        assert preset.rebalance_months == tuple(REBALANCE.months)
        assert preset.transaction_cost_bps == pytest.approx(REBALANCE.transaction_cost_bps)

    def test_options_flag_matches_OPTIONS(self, preset: Portfolio):
        from src.portfolio import OPTIONS

        assert preset.options_overlay == OPTIONS.enabled


class TestPresetAssetsInCatalog:
    """Every asset referenced by the preset must exist in data/catalog.toml,
    and its ter + is_hedged must match the legacy TER_ANNUAL / HEDGED dicts."""

    def test_all_preset_assets_have_catalog_entry(self, preset: Portfolio):
        catalog = load_catalog()
        missing = [a.key for a in preset.assets if a.key not in catalog]
        assert missing == [], f"preset references assets not in catalog: {missing}"

    def test_ter_matches_legacy(self, preset: Portfolio):
        from src.portfolio import TER_ANNUAL

        catalog = load_catalog()
        for a in preset.assets:
            if a.key == "cash":
                continue  # cash has no TER
            info = catalog[a.key]
            # Legacy TER_ANNUAL is the single source of truth the engine already
            # uses; the catalog value must match.
            assert info.ter == pytest.approx(TER_ANNUAL[a.key]), (
                f"TER mismatch for {a.key}: catalog {info.ter} vs legacy {TER_ANNUAL[a.key]}"
            )

    def test_hedged_matches_legacy(self, preset: Portfolio):
        from src.portfolio import HEDGED

        catalog = load_catalog()
        for a in preset.assets:
            if a.key == "cash":
                continue
            if a.key in HEDGED:
                assert catalog[a.key].is_hedged == HEDGED[a.key], (
                    f"HEDGED mismatch for {a.key}"
                )
