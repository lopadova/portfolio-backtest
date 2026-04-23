"""
Unit tests for src/tax.py — Italian capital-gains model with minusvalenze
carryforward.
"""

import pandas as pd
import pytest

from src.tax import TaxLedger


@pytest.fixture
def ledger():
    return TaxLedger()


class TestBasicRecording:
    def test_gain_generates_tax(self, ledger):
        event = ledger.record_sale(
            date=pd.Timestamp("2024-03-15"),
            sleeve="quality",
            proceeds=11_000,
            cost_basis=10_000,
        )
        # 1000 gain × 26% = 260
        assert event.realized_pl == pytest.approx(1000.0)
        assert event.tax_due == pytest.approx(260.0)
        assert event.rate_applied == pytest.approx(0.26)

    def test_loss_adds_to_bucket(self, ledger):
        event = ledger.record_sale(
            date=pd.Timestamp("2024-03-15"),
            sleeve="nasdaq_top30",
            proceeds=9_000,
            cost_basis=10_000,
        )
        assert event.realized_pl == pytest.approx(-1000.0)
        assert event.tax_due == 0.0
        assert ledger.outstanding_losses() == pytest.approx(1000.0)


class TestLossOffset:
    def test_loss_offsets_future_gain(self, ledger):
        # Loss in 2023
        ledger.record_sale(
            pd.Timestamp("2023-05-01"), "quality", 8_500, 10_000,
        )
        # Gain in 2024 that fully offsets
        event = ledger.record_sale(
            pd.Timestamp("2024-05-01"), "quality", 11_200, 10_000,
        )
        # Gain was 1200; loss was 1500; offset 1200 → no tax, 300 left in bucket
        assert event.tax_due == 0.0
        assert event.offset_used == pytest.approx(1200.0)
        assert ledger.outstanding_losses() == pytest.approx(300.0)

    def test_partial_offset(self, ledger):
        # Small loss: 500
        ledger.record_sale(
            pd.Timestamp("2023-01-15"), "gold", 9_500, 10_000,
        )
        # Large gain: 2000
        event = ledger.record_sale(
            pd.Timestamp("2024-01-15"), "gold", 12_000, 10_000,
        )
        # 2000 - 500 offset = 1500 taxable × 0.26 = 390
        assert event.offset_used == pytest.approx(500.0)
        assert event.tax_due == pytest.approx(390.0)
        assert ledger.outstanding_losses() == 0.0

    def test_fifo_order(self, ledger):
        # Older loss (2022) first, newer loss (2024) second
        ledger.record_sale(pd.Timestamp("2022-01-01"), "quality", 7_000, 10_000)
        ledger.record_sale(pd.Timestamp("2024-01-01"), "quality", 8_500, 10_000)
        # Now a small gain in 2024 that only partially consumes the bucket
        event = ledger.record_sale(pd.Timestamp("2024-06-01"), "quality", 11_000, 10_000)
        # Gain 1000 should consume from the 2022 lot first (FIFO)
        assert event.offset_used == pytest.approx(1000.0)
        assert ledger.outstanding_losses() == pytest.approx(3500.0)  # 2000 remaining 2022 + 1500 2024


class TestOutOfOrderInsertion:
    """Regression (Copilot PR #8): full-scan purge + FIFO re-sort must handle
    loss lots that are NOT inserted in chronological order (e.g., a backfilled
    late trade whose `year` is older than an already-present lot)."""

    def test_out_of_order_losses_still_purged(self, ledger):
        # Insert a RECENT loss first (2024), then a BACKFILLED older one (2020)
        ledger.record_sale(pd.Timestamp("2024-06-01"), "quality", 9_500, 10_000)   # -500 loss @ 2024
        ledger.record_sale(pd.Timestamp("2020-06-01"), "nasdaq_top30", 9_000, 10_000)  # -1000 loss @ 2020 (backfilled)
        # Purge at end of 2025 — the 2020 lot has expires_year=2025 so it expires, the 2024 lot survives
        expired = ledger.purge_expired(current_year=2025)
        assert expired == pytest.approx(1000.0)  # the 2020 lot (1000)
        assert ledger.outstanding_losses() == pytest.approx(500.0)  # the 2024 lot survives

    def test_out_of_order_losses_consumed_oldest_first(self, ledger):
        """After an out-of-order insertion, a subsequent gain must offset
        the oldest-by-year lot first (FIFO), not the insertion-order first."""
        # Insert 2024 loss first, then backfill a 2021 loss
        ledger.record_sale(pd.Timestamp("2024-01-01"), "quality", 8_500, 10_000)   # -1500 @ 2024
        ledger.record_sale(pd.Timestamp("2021-01-01"), "quality", 9_000, 10_000)   # -1000 @ 2021
        # A 2024 gain of 1000 should consume from the 2021 lot (oldest by year) first
        event = ledger.record_sale(pd.Timestamp("2024-06-01"), "quality", 11_000, 10_000)
        assert event.offset_used == pytest.approx(1000.0)  # consumed the 2021 lot fully
        # Remaining should be the 2024 lot untouched (1500)
        assert ledger.outstanding_losses() == pytest.approx(1500.0)


class TestCarryforwardExpiration:
    def test_loss_expires_after_4_years(self, ledger):
        # Loss in 2020 → expires after 2024 (i.e., not usable in 2025+)
        ledger.record_sale(pd.Timestamp("2020-06-01"), "quality", 9_000, 10_000)
        # Purge at end of 2025
        expired = ledger.purge_expired(current_year=2025)
        assert expired == pytest.approx(1000.0)
        assert ledger.outstanding_losses() == 0.0

    def test_loss_still_valid_within_window(self, ledger):
        ledger.record_sale(pd.Timestamp("2020-06-01"), "quality", 9_000, 10_000)
        # Within 4-year window (end of 2023)
        expired = ledger.purge_expired(current_year=2023)
        assert expired == 0.0
        assert ledger.outstanding_losses() == pytest.approx(1000.0)

    def test_gain_uses_loss_before_expiration(self, ledger):
        # Loss in 2020 (expires end of 2024), gain in 2024 — should offset
        ledger.record_sale(pd.Timestamp("2020-06-01"), "quality", 9_000, 10_000)
        event = ledger.record_sale(pd.Timestamp("2024-06-01"), "quality", 11_000, 10_000)
        assert event.offset_used == pytest.approx(1000.0)
        assert event.tax_due == 0.0


class TestGovBondRate:
    def test_gov_bond_uses_12_5_percent(self, ledger):
        event = ledger.record_sale(
            pd.Timestamp("2024-03-15"), "eur_gov_7_10y", 11_000, 10_000,
        )
        assert event.rate_applied == pytest.approx(0.125)
        assert event.tax_due == pytest.approx(125.0)  # 1000 × 0.125


class TestPensionExempt:
    def test_pension_gains_not_taxed(self, ledger):
        event = ledger.record_sale(
            pd.Timestamp("2024-03-15"), "pension_bond", 11_000, 10_000,
        )
        assert event.tax_due == 0.0
        assert event.rate_applied == 0.0

    def test_pension_exempt_off(self):
        ledger = TaxLedger(pension_exempt=False)
        event = ledger.record_sale(
            pd.Timestamp("2024-03-15"), "pension_bond", 11_000, 10_000,
        )
        # pension_bond is in gov_bond_sleeves → 12.5% applies
        assert event.tax_due == pytest.approx(125.0)


class TestSummary:
    def test_summary_aggregates_correctly(self, ledger):
        ledger.record_sale(pd.Timestamp("2024-01-01"), "quality", 11_000, 10_000)  # +1000 gain, 260 tax
        ledger.record_sale(pd.Timestamp("2024-02-01"), "nasdaq_top30", 8_500, 10_000)  # -1500 loss
        ledger.record_sale(pd.Timestamp("2024-03-01"), "quality", 11_000, 10_000)  # +1000 gain, offset 1000, 0 tax
        s = ledger.summary()
        assert s["n_events"] == 3
        assert s["total_tax_paid"] == pytest.approx(260.0)
        assert s["outstanding_loss_bucket"] == pytest.approx(500.0)
