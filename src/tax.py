"""
Italian tax modeling — simplified capital-gains simulator for backtest runs.

Rules modeled:
    * 26% capital-gains tax (CGT) on realized plusvalenze from ETF / crypto sales
    * 12.5% optional rate on whitelist government bonds (configurable per sleeve)
    * "Zainetto fiscale" (loss bucket) — realized losses (minusvalenze) carry
      forward and can offset future realized gains. Default 4-year expiration
      (anno di realizzo + 4)
    * FIFO compensation: oldest losses used first to avoid expiration waste
    * Pension block excluded during accumulation (regulatory — taxed at exit)

Important caveats (embedded in the disclaimer):
    * This is a *simplified* model. Real Italian taxation has edge cases:
      ETF dividend distributions ("redditi di capitale") cannot offset
      losses in the same bucket; imposta di bollo (0.20% annually);
      obbligazioni di Stato whitelist vs non-whitelist, etc.
    * This model does NOT replace a commercialista. It provides a
      realistic CGT drag estimate for backtest purposes only.

References:
    * TUIR art. 67-68 (redditi diversi finanziari)
    * D.L. 66/2014 (aumento aliquota al 26%)
    * Guida Agenzia delle Entrate — "Tassazione degli strumenti finanziari"
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional

import pandas as pd


@dataclass
class LossLot:
    """A realized loss available to offset future gains."""
    amount: float         # positive value (loss as magnitude)
    year: int             # year in which the loss was realized
    expires_year: int     # year in which the loss expires (exclusive)


@dataclass
class TaxEvent:
    """A record of a single taxable event (for audit / reporting)."""
    date: pd.Timestamp
    sleeve: str
    proceeds: float
    cost_basis: float
    realized_pl: float           # positive = gain, negative = loss
    offset_used: float = 0.0     # how much loss carry-forward offset this event
    tax_due: float = 0.0         # computed tax (always >= 0)
    rate_applied: float = 0.0    # 0.26 or 0.125


class TaxLedger:
    """
    Accumulates realized P&L per sleeve and applies Italian CGT rules.

    Usage:
        ledger = TaxLedger(
            capital_gains_rate=0.26,
            gov_bond_rate=0.125,
            loss_carryforward_years=4,
        )
        ledger.record_sale(date, sleeve, proceeds, cost_basis)
        tax_due_this_year = ledger.tax_accrued_for_year(year)
        ledger.purge_expired(current_year=current_year)

    Assumptions:
        * Lot tracking is at the *sleeve* level, not individual security.
        * Cost basis is carried as a single running average per sleeve (the
          portfolio does not hold individual shares — it holds proportional
          stakes). This is a simplification vs real lot-by-lot accounting.
        * Loss lots are purged by full scan (no assumption that they are
          inserted in chronological order — see `_purge_expired_internal`).
    """

    def __init__(
        self,
        capital_gains_rate: float = 0.26,
        gov_bond_rate: float = 0.125,
        gov_bond_sleeves: Optional[set] = None,
        loss_carryforward_years: int = 4,
        pension_exempt: bool = True,
    ):
        self.cgt_rate = capital_gains_rate
        self.gov_bond_rate = gov_bond_rate
        self.gov_bond_sleeves = gov_bond_sleeves or {
            "eur_gov_direct", "eur_gov_1_3y", "eur_gov_7_10y",
            "pension_bond",
        }
        self.carryforward_years = loss_carryforward_years
        self.pension_exempt = pension_exempt

        self.loss_bucket: Deque[LossLot] = deque()
        self.events: List[TaxEvent] = []
        self.tax_by_year: Dict[int, float] = {}

    def _rate_for_sleeve(self, sleeve: str) -> float:
        """Return the tax rate applicable to this sleeve."""
        if sleeve in self.gov_bond_sleeves:
            return self.gov_bond_rate
        return self.cgt_rate

    def _is_tax_exempt(self, sleeve: str) -> bool:
        """Pension sleeves are exempt during accumulation (taxed at exit)."""
        return self.pension_exempt and sleeve.startswith("pension_")

    def record_sale(
        self,
        date: pd.Timestamp,
        sleeve: str,
        proceeds: float,
        cost_basis: float,
    ) -> TaxEvent:
        """
        Record a realized-P&L event from a sale.
        proceeds and cost_basis are in the same currency units.
        Returns the TaxEvent recorded.
        """
        year = date.year
        realized_pl = proceeds - cost_basis

        if self._is_tax_exempt(sleeve):
            event = TaxEvent(
                date=date, sleeve=sleeve,
                proceeds=proceeds, cost_basis=cost_basis,
                realized_pl=realized_pl,
                tax_due=0.0, rate_applied=0.0,
            )
            self.events.append(event)
            return event

        rate = self._rate_for_sleeve(sleeve)

        if realized_pl < 0:
            # Loss — add to bucket
            self.loss_bucket.append(LossLot(
                amount=-realized_pl,
                year=year,
                expires_year=year + self.carryforward_years + 1,
            ))
            event = TaxEvent(
                date=date, sleeve=sleeve,
                proceeds=proceeds, cost_basis=cost_basis,
                realized_pl=realized_pl,
                offset_used=0.0, tax_due=0.0, rate_applied=rate,
            )
        else:
            # Gain — offset against oldest losses first (FIFO).
            # We explicitly sort the bucket by year (then expires_year) so the
            # FIFO invariant holds even if record_sale() was called out of
            # chronological order (e.g. backfilled trades). Surviving lots
            # are written back as a deque for O(1) popleft() in consumption.
            remaining_gain = realized_pl
            offset_total = 0.0
            self._purge_expired_internal(year)
            ordered = sorted(self.loss_bucket, key=lambda l: (l.year, l.expires_year))
            self.loss_bucket = deque(ordered)
            while remaining_gain > 0 and self.loss_bucket:
                lot = self.loss_bucket[0]
                if lot.amount <= remaining_gain:
                    offset_total += lot.amount
                    remaining_gain -= lot.amount
                    self.loss_bucket.popleft()
                else:
                    offset_total += remaining_gain
                    lot.amount -= remaining_gain
                    remaining_gain = 0.0
            taxable_gain = remaining_gain
            tax_due = taxable_gain * rate
            event = TaxEvent(
                date=date, sleeve=sleeve,
                proceeds=proceeds, cost_basis=cost_basis,
                realized_pl=realized_pl,
                offset_used=offset_total, tax_due=tax_due, rate_applied=rate,
            )
            self.tax_by_year[year] = self.tax_by_year.get(year, 0.0) + tax_due

        self.events.append(event)
        return event

    def _purge_expired_internal(self, current_year: int) -> float:
        """
        Remove ALL loss lots whose expires_year is <= current_year.

        Uses a full scan of the bucket (not just `loss_bucket[0]`) so that
        out-of-order insertions — e.g. a backfilled or late-reported trade —
        cannot leave expired lots stranded in the middle of the deque. The
        scan is O(N) per call but N is bounded (one entry per realized loss
        within the 4-year rollover, typically a handful).

        Returns the total amount of losses expired (for reporting).
        """
        expired_total = 0.0
        surviving = deque()
        for lot in self.loss_bucket:
            if lot.expires_year <= current_year:
                expired_total += lot.amount
            else:
                surviving.append(lot)
        self.loss_bucket = surviving
        return expired_total

    def purge_expired(self, current_year: int) -> float:
        """Public wrapper for _purge_expired_internal."""
        return self._purge_expired_internal(current_year)

    def tax_accrued_for_year(self, year: int) -> float:
        return self.tax_by_year.get(year, 0.0)

    def total_tax_paid(self) -> float:
        return sum(self.tax_by_year.values())

    def outstanding_losses(self) -> float:
        return sum(lot.amount for lot in self.loss_bucket)

    def summary(self) -> Dict[str, float]:
        return {
            "n_events": len(self.events),
            "total_tax_paid": self.total_tax_paid(),
            "outstanding_loss_bucket": self.outstanding_losses(),
            "total_realized_gains": sum(e.realized_pl for e in self.events if e.realized_pl > 0),
            "total_realized_losses": sum(e.realized_pl for e in self.events if e.realized_pl < 0),
            "total_offset_used": sum(e.offset_used for e in self.events),
        }
