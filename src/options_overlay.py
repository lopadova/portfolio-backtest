"""
Options overlay simulator — semi-annual SPY/QQQ put spreads with 2×/3× take-profit.

Methodology: Black-Scholes + VIX (skew-adjusted) approximation. See Israelov & Nielsen
(2015) "Still Not Cheap: Portfolio Protection in Calm Markets", AQR, for the standard
practitioner justification of this approach.

The simulator:
1. On each rebalance date (default Jan/Jul), buys put spreads on SPY (70%) and QQQ (30%)
   up to a semi-annual budget of 0.15% of NAV.
2. Daily, checks if the spread has appreciated to 2× or 3× the premium paid; if so,
   closes 50% or 100% of the position respectively.
3. At expiry, any remaining position is settled at intrinsic value.
4. Returns a monthly P&L series to be added to the main portfolio return.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import norm

from .portfolio import OPTIONS


# ============================================================================
# Black-Scholes put pricing
# ============================================================================

def bs_put_price(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.015) -> float:
    """
    Black-Scholes European put price.

    S: spot
    K: strike
    T: time to expiration in years
    r: risk-free rate (annualized, decimal)
    sigma: implied volatility (annualized, decimal)
    q: dividend yield (annualized, decimal)
    """
    if T <= 0.0 or sigma <= 0.0:
        return max(K - S, 0.0)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return float(K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1))


def bs_put_spread_price(
    S: float, K_long: float, K_short: float, T: float, r: float, sigma: float, q: float = 0.015
) -> float:
    """Debit put spread value: long put at K_long, short put at K_short (K_long > K_short)."""
    return bs_put_price(S, K_long, T, r, sigma, q) - bs_put_price(S, K_short, T, r, sigma, q)


# ============================================================================
# Implied vol — VIX + skew adjustment
# ============================================================================

def iv_from_vix(vix: float, moneyness: float, skew_adjustment: float = None) -> float:
    """
    Approximate the implied vol for an OTM put at the given moneyness
    (K / S, typically < 1.0) from the VIX.

    Simple model: OTM put IV ≈ VIX × skew_adjustment × (1 + |moneyness_gap| * skew_slope).

    For the portfolio's strikes (K/S ≈ 0.87 and 0.70), OTM put IV is typically
    15-25% higher than ATM IV due to the well-documented volatility skew.
    """
    if skew_adjustment is None:
        skew_adjustment = OPTIONS.iv_skew_adjustment
    vix_decimal = vix / 100.0
    moneyness_gap = max(0.0, 1.0 - moneyness)  # how far OTM in decimal
    skew_slope = 0.50  # per unit of 1 - K/S (empirical rough calibration)
    return vix_decimal * skew_adjustment * (1.0 + skew_slope * moneyness_gap)


# ============================================================================
# Position tracking
# ============================================================================

@dataclass
class Position:
    open_date: pd.Timestamp
    expiry_date: pd.Timestamp
    spot_at_open: float
    K_long: float
    K_short: float
    contracts: float           # in "units" (fractional for simulation; each unit = 1 contract of 100 shares)
    premium_paid: float        # total cost of opening (per unit, so total = contracts * premium_paid)
    underlying: str            # "SPY" or "QQQ"
    closed_fraction: float = 0.0  # 0.0 = fully open, 0.5 = half closed, 1.0 = fully closed
    pnl_realized: float = 0.0  # cumulative cash flow from partial closes


# ============================================================================
# Main simulator
# ============================================================================

def simulate_options_overlay(
    spy_daily: pd.Series,
    qqq_daily: pd.Series,
    vix_daily: pd.Series,
    rf_daily: pd.Series,
    nav_series: pd.Series,          # monthly NAV, indexed by EOM date
    rebalance_months: Tuple[int, ...] = (1, 7),
) -> pd.Series:
    """
    Main driver. Returns a monthly P&L series (in EUR-approximate terms — since
    SPY/QQQ are USD, we treat the $ P&L as if converted at end-of-month EUR/USD;
    for simplicity in this simulation we do not re-convert — the overlay notional
    is small enough that the FX approximation is 2nd-order).

    The returned P&L is the net cash flow during each month divided by NAV at
    start of month, so it can be added to the main portfolio return.
    """
    cfg = OPTIONS
    commission = cfg.commission_per_contract
    budget_per_half = cfg.budget_nav_per_year / 2.0  # 0.15% NAV per half-year

    # Determine roll dates — first trading day of each month in rebalance_months
    spy_index = spy_daily.index
    roll_dates: List[pd.Timestamp] = []
    for year in range(spy_index.year.min(), spy_index.year.max() + 1):
        for month in rebalance_months:
            month_dates = spy_index[(spy_index.year == year) & (spy_index.month == month)]
            if len(month_dates) > 0:
                roll_dates.append(month_dates[0])

    positions: List[Position] = []
    pnl_by_day: pd.Series = pd.Series(0.0, index=spy_daily.index)

    for today in spy_daily.index:
        spy = spy_daily.loc[today]
        qqq = qqq_daily.loc[today]
        vix = vix_daily.loc[today] if today in vix_daily.index else vix_daily.asof(today)
        rf = rf_daily.loc[today] if today in rf_daily.index else rf_daily.asof(today)

        # --- Mark-to-market all open positions (for 2x/3x check) ---
        for p in positions:
            if p.closed_fraction >= 1.0:
                continue
            T = max((p.expiry_date - today).days / 365.0, 1e-6)
            if T <= 0:
                continue
            spot = spy if p.underlying == "SPY" else qqq
            moneyness_long = p.K_long / spot
            iv = iv_from_vix(vix, moneyness_long)
            current_value = bs_put_spread_price(spot, p.K_long, p.K_short, T, rf, iv)
            ratio = current_value / p.premium_paid if p.premium_paid > 0 else 0.0
            # Take-profit 2×: close 50% if not already done
            if ratio >= cfg.take_profit_partial_multiple and p.closed_fraction < 0.5:
                fraction_to_close = 0.5 - p.closed_fraction
                cash_in = fraction_to_close * p.contracts * current_value * 100.0  # 100 = contract multiplier
                p.pnl_realized += cash_in
                p.closed_fraction = 0.5
                pnl_by_day.loc[today] += cash_in
            # Take-profit 3×: close 100%
            if ratio >= cfg.take_profit_full_multiple and p.closed_fraction < 1.0:
                fraction_to_close = 1.0 - p.closed_fraction
                cash_in = fraction_to_close * p.contracts * current_value * 100.0
                p.pnl_realized += cash_in
                p.closed_fraction = 1.0
                pnl_by_day.loc[today] += cash_in

        # --- Settle expired positions at intrinsic value ---
        for p in positions:
            if p.closed_fraction < 1.0 and today >= p.expiry_date:
                spot = spy if p.underlying == "SPY" else qqq
                # Payoff = max(K_long - S, 0) - max(K_short - S, 0), capped at (K_long - K_short)
                payoff_long = max(p.K_long - spot, 0.0)
                payoff_short = max(p.K_short - spot, 0.0)
                payoff_per_unit = payoff_long - payoff_short
                remaining = 1.0 - p.closed_fraction
                cash_in = remaining * p.contracts * payoff_per_unit * 100.0
                p.pnl_realized += cash_in
                p.closed_fraction = 1.0
                pnl_by_day.loc[today] += cash_in

        # --- Open new positions on roll dates ---
        if today in roll_dates:
            # Find NAV at the start of this half-year
            nav_at_roll = _nav_at_or_before(nav_series, today)
            if nav_at_roll is None or nav_at_roll <= 0:
                continue
            budget = budget_per_half * nav_at_roll  # in EUR
            # Approximate USD budget (NAV is in EUR; options denominated in USD)
            # Conservative: treat budget as USD directly (small distortion for small overlay)
            budget_usd = budget  # simplification

            # Expiry date
            expiry_date = today + pd.DateOffset(months=cfg.tenor_months)

            # Allocate budget 70% SPY / 30% QQQ
            for underlying, alloc in (("SPY", cfg.spy_qqq_split[0]), ("QQQ", cfg.spy_qqq_split[1])):
                budget_leg = budget_usd * alloc
                spot = spy if underlying == "SPY" else qqq
                K_long = spot * cfg.long_strike_pct
                K_short = spot * cfg.short_strike_pct
                T = cfg.tenor_months / 12.0
                iv = iv_from_vix(vix, cfg.long_strike_pct)
                premium_per_unit = bs_put_spread_price(spot, K_long, K_short, T, rf, iv)
                premium_per_contract_cash = premium_per_unit * 100.0 + commission  # 100 shares per contract + commission
                if premium_per_contract_cash <= 0:
                    continue
                n_contracts = budget_leg / premium_per_contract_cash
                if n_contracts <= 0:
                    continue
                positions.append(
                    Position(
                        open_date=today,
                        expiry_date=expiry_date,
                        spot_at_open=spot,
                        K_long=K_long,
                        K_short=K_short,
                        contracts=n_contracts,
                        premium_paid=premium_per_unit,
                        underlying=underlying,
                    )
                )
                pnl_by_day.loc[today] -= n_contracts * premium_per_contract_cash  # outflow

    # Monthly aggregation — convert to NAV-relative return
    monthly_pnl = pnl_by_day.resample("ME").sum()
    # Align to nav_series index
    monthly_pnl = monthly_pnl.reindex(nav_series.index).fillna(0.0)
    # Return as fraction of beginning-of-month NAV (shift NAV by 1)
    nav_shifted = nav_series.shift(1).bfill()
    monthly_return = monthly_pnl / nav_shifted.replace(0.0, np.nan)
    return monthly_return.fillna(0.0)


def _nav_at_or_before(nav_series: pd.Series, date: pd.Timestamp):
    """Find the most recent NAV value at or before `date`."""
    try:
        valid = nav_series.loc[:date]
        if len(valid) == 0:
            return None
        return valid.iloc[-1]
    except KeyError:
        return None
