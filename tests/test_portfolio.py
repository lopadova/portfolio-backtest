"""
Unit tests for src/portfolio.py — configuration integrity.

These tests enforce the invariants that must hold across any modification
of the portfolio configuration. A PR that breaks one of these tests is
almost certainly wrong.
"""

import pytest

from src.portfolio import (
    WEIGHTS, EQUITY, CRYPTO, BONDS, EM_SATELLITES, PENSION,
    HEDGED, SYMBOL_MAP, TER_ANNUAL,
    OPTIONS, CASH, REBALANCE, BENCHMARKS,
)


# ---------------------------------------------------------------------------
# Macro weights integrity
# ---------------------------------------------------------------------------

class TestMacroWeights:
    def test_weights_sum_to_one(self):
        """Total NAV allocation must sum to exactly 1.0 (100%)."""
        total = sum(WEIGHTS.values())
        assert abs(total - 1.0) < 1e-9, f"WEIGHTS sum to {total}, not 1.0"

    def test_all_weights_non_negative(self):
        for sleeve, w in WEIGHTS.items():
            assert w >= 0.0, f"{sleeve} has negative weight {w}"

    def test_expected_macro_blocks_present(self):
        required = {"pension", "gold", "equity", "crypto", "bonds", "em_sat", "dbi", "cash"}
        assert required.issubset(WEIGHTS.keys())


# ---------------------------------------------------------------------------
# Equity sleeve integrity
# ---------------------------------------------------------------------------

class TestEquitySleeve:
    def test_equity_sums_to_weights_equity(self):
        """EQUITY must sum to WEIGHTS['equity'] within small rounding tolerance."""
        total = sum(EQUITY.values())
        expected = WEIGHTS["equity"]
        assert abs(total - expected) < 0.002, \
            f"EQUITY sums to {total}, expected {expected} (WEIGHTS['equity'])"

    def test_expected_equity_positions_present(self):
        required = {"put_write", "nasdaq_top30", "hc_us_hedged", "hc_world",
                    "quality", "momentum", "ex_usa"}
        assert required.issubset(EQUITY.keys())

    def test_no_negative_equity_weights(self):
        for pos, w in EQUITY.items():
            assert w >= 0.0, f"{pos} has negative weight {w}"


# ---------------------------------------------------------------------------
# Sub-sleeves (crypto, bonds, EM)
# ---------------------------------------------------------------------------

class TestSubSleeves:
    def test_crypto_sums_to_macro(self):
        total = sum(CRYPTO.values())
        assert abs(total - WEIGHTS["crypto"]) < 1e-9

    def test_bonds_sums_to_macro(self):
        total = sum(BONDS.values())
        assert abs(total - WEIGHTS["bonds"]) < 1e-9

    def test_em_sums_to_macro(self):
        total = sum(EM_SATELLITES.values())
        assert abs(total - WEIGHTS["em_sat"]) < 1e-9

    def test_pension_sums_to_macro(self):
        total = sum(PENSION.values())
        assert abs(total - WEIGHTS["pension"]) < 1e-9


# ---------------------------------------------------------------------------
# Hedging flags
# ---------------------------------------------------------------------------

class TestHedgingFlags:
    def test_every_equity_has_hedging_flag(self):
        for pos in EQUITY:
            assert pos in HEDGED, f"{pos} missing from HEDGED dict"

    def test_hedging_flag_is_bool(self):
        for pos, h in HEDGED.items():
            assert isinstance(h, bool), f"HEDGED[{pos}] = {h} is not bool"

    def test_put_write_is_hedged(self):
        """Defensive put/write should be EUR hedged (strategy is low-vol; FX noise would pollute it)."""
        assert HEDGED["put_write"] is True

    def test_nasdaq_is_unhedged(self):
        """Concentrated Nasdaq is deliberately unhedged (US exposure is intentional)."""
        assert HEDGED["nasdaq_top30"] is False

    def test_gold_is_unhedged(self):
        """Gold is unhedged to capture USD flight-to-safety in crises."""
        assert HEDGED["gold"] is False


# ---------------------------------------------------------------------------
# Symbol map and TER
# ---------------------------------------------------------------------------

class TestSymbolMap:
    def test_every_sleeve_has_symbol(self):
        """Every entry in EQUITY/CRYPTO/BONDS/EM etc. must have a CSV symbol."""
        for pos in EQUITY:
            assert pos in SYMBOL_MAP, f"{pos} missing from SYMBOL_MAP"
        for pos in CRYPTO:
            assert pos in SYMBOL_MAP, f"{pos} missing from SYMBOL_MAP"
        for pos in BONDS:
            assert pos in SYMBOL_MAP, f"{pos} missing from SYMBOL_MAP"
        for pos in EM_SATELLITES:
            assert pos in SYMBOL_MAP, f"{pos} missing from SYMBOL_MAP"
        assert "gold" in SYMBOL_MAP
        assert "dbi" in SYMBOL_MAP

    def test_symbols_are_strings(self):
        for k, v in SYMBOL_MAP.items():
            assert isinstance(v, str), f"SYMBOL_MAP[{k}] = {v} is not a string"


class TestTER:
    def test_ter_in_reasonable_range(self):
        """All TERs should be between 0 and 3% (sanity)."""
        for pos, ter in TER_ANNUAL.items():
            assert 0.0 <= ter <= 0.03, f"{pos} TER {ter} out of range"


# ---------------------------------------------------------------------------
# Options configuration
# ---------------------------------------------------------------------------

class TestOptionsConfig:
    def test_default_options_config_enabled(self):
        assert OPTIONS.enabled is True

    def test_budget_is_small_fraction(self):
        assert 0.0 < OPTIONS.budget_nav_per_year <= 0.01, \
            "Options budget should be < 1% NAV/year"

    def test_strikes_are_otm(self):
        assert OPTIONS.long_strike_pct < 1.0, "Long put must be OTM"
        assert OPTIONS.short_strike_pct < OPTIONS.long_strike_pct, \
            "Short put must be deeper OTM than long put"

    def test_take_profit_multiples_ordered(self):
        assert OPTIONS.take_profit_partial_multiple < OPTIONS.take_profit_full_multiple, \
            "Partial TP multiple must be less than full TP multiple"

    def test_spy_qqq_split_sums_to_one(self):
        s = sum(OPTIONS.spy_qqq_split)
        assert abs(s - 1.0) < 1e-9, f"SPY/QQQ split sums to {s}"


# ---------------------------------------------------------------------------
# Cash configuration
# ---------------------------------------------------------------------------

class TestCashConfig:
    def test_floor_below_comfort_below_target_below_upper(self):
        assert CASH.floor_hard < CASH.comfort <= CASH.target < CASH.upper_band

    def test_floor_matches_article(self):
        """The article specifies an 8% hard floor."""
        assert CASH.floor_hard == 0.08


# ---------------------------------------------------------------------------
# Rebalance configuration
# ---------------------------------------------------------------------------

class TestRebalanceConfig:
    def test_months_valid(self):
        for m in REBALANCE.months:
            assert 1 <= m <= 12

    def test_transaction_cost_reasonable(self):
        assert 0 <= REBALANCE.transaction_cost_bps <= 100, \
            "Transaction cost should be 0-100 bps"

    def test_band_relative_pct_reasonable(self):
        assert 0.0 < REBALANCE.band_relative_pct <= 1.0


# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------

class TestBenchmarks:
    def test_each_benchmark_weights_sum_to_one(self):
        """Each benchmark portfolio must have weights summing to 1.0."""
        for name, weights in BENCHMARKS.items():
            total = sum(weights.values())
            assert abs(total - 1.0) < 1e-9, \
                f"Benchmark {name} weights sum to {total}"

    def test_expected_benchmarks_present(self):
        required = {"60/40", "100% S&P 500 TR EUR", "100% SWDA (MSCI World EUR)", "All-Weather proxy"}
        assert required.issubset(BENCHMARKS.keys())
