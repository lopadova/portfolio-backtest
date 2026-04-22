"""
Rebalancing engine — computes monthly portfolio returns with semi-annual
rebalancing back to target weights, including transaction costs.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from .portfolio import (
    WEIGHTS, EQUITY, CRYPTO, BONDS, EM_SATELLITES, PENSION,
    TER_ANNUAL, SYMBOL_MAP, REBALANCE,
)


def build_target_weights() -> Dict[str, float]:
    """
    Flatten all the nested portfolio structures into a single dict of
    sleeve -> target weight in total NAV.
    """
    target: Dict[str, float] = {}
    target["pension_bond"] = PENSION["bond_like"]
    target["pension_equity"] = PENSION["equity_like"]
    target["gold"] = WEIGHTS["gold"]
    for k, v in EQUITY.items():
        target[k] = v
    for k, v in CRYPTO.items():
        target[k] = v
    for k, v in BONDS.items():
        target[k] = v
    for k, v in EM_SATELLITES.items():
        target[k] = v
    target["dbi"] = WEIGHTS["dbi"]
    # Cash is the residual; computed dynamically
    # Sanity check: sum of everything except cash + cash_target = 1.0
    non_cash_sum = sum(target.values())
    assert abs(non_cash_sum + WEIGHTS["cash"] - 1.0) < 0.002, \
        f"Weights error: non-cash sum = {non_cash_sum}, expected {1 - WEIGHTS['cash']}"
    return target


def simulate_portfolio(
    monthly_returns: pd.DataFrame,
    btc_activation_date: pd.Timestamp,
    apply_ter: bool = True,
    rebalance_months: tuple = None,
    transaction_cost_bps: float = None,
    verbose: bool = False,
) -> pd.Series:
    """
    Run the rebalancing simulation.

    monthly_returns: DataFrame indexed by EOM date, columns = sleeve keys
                     matching build_target_weights()
    btc_activation_date: BTC weight is 0 before this date, target weight after

    Returns: monthly portfolio return series (decimals)
    """
    if rebalance_months is None:
        rebalance_months = REBALANCE.months
    if transaction_cost_bps is None:
        transaction_cost_bps = REBALANCE.transaction_cost_bps

    target_weights = build_target_weights()

    # Only keep sleeves for which we have return data
    available_sleeves = [s for s in target_weights if s in monthly_returns.columns]

    # Start weights = targets (with BTC handled specially)
    current_weights = pd.Series(0.0, index=available_sleeves)
    cash_weight = WEIGHTS["cash"]

    for s in available_sleeves:
        current_weights[s] = target_weights[s]
    # If BTC not yet active at start, shift its weight to cash
    if "btc" in available_sleeves and monthly_returns.index[0] < btc_activation_date:
        cash_weight += current_weights["btc"]
        current_weights["btc"] = 0.0

    # Monthly TER deduction (per sleeve)
    monthly_ter = pd.Series({
        s: TER_ANNUAL.get(s, 0.0) / 12.0 for s in available_sleeves
    })

    portfolio_returns: List[float] = []
    dates: List[pd.Timestamp] = []

    for i, date in enumerate(monthly_returns.index):
        # Get this month's sleeve returns
        month_returns = monthly_returns.loc[date, available_sleeves].fillna(0.0)

        # Apply TER
        if apply_ter:
            month_returns = month_returns - monthly_ter

        # BTC activation
        if "btc" in available_sleeves and date >= btc_activation_date and current_weights["btc"] == 0.0:
            # First month BTC activates — move btc target weight from cash to btc
            # (simulates buying BTC)
            btc_target = target_weights["btc"]
            current_weights["btc"] = btc_target
            cash_weight -= btc_target
            if verbose:
                print(f"BTC activated on {date.date()}, weight {btc_target:.4f}")

        # Portfolio return this month = sum of weight * return + cash * 0 (cash earns nothing in base model)
        portfolio_return = float((current_weights * month_returns).sum())
        # Cash earns zero in base model (conservative)
        # Could add a cash yield here: portfolio_return += cash_weight * monthly_cash_rate

        # Update weights post-return (drift)
        growth_factors = 1.0 + month_returns
        new_weights = current_weights * growth_factors
        new_cash_weight = cash_weight  # cash unchanged
        total = new_weights.sum() + new_cash_weight
        # Normalize (should be ~ 1 + portfolio_return)
        new_weights = new_weights / total
        new_cash_weight = new_cash_weight / total
        current_weights = new_weights
        cash_weight = new_cash_weight

        # Rebalance if this month is a rebalance month
        if date.month in rebalance_months:
            # Target weights (cash goes to target too)
            target_vec = pd.Series(target_weights).reindex(available_sleeves).fillna(0.0)
            if "btc" in available_sleeves and date < btc_activation_date:
                target_vec["btc"] = 0.0
                # Extra weight goes to cash
                target_cash = WEIGHTS["cash"] + target_weights["btc"]
            else:
                target_cash = WEIGHTS["cash"]

            # Compute trade magnitude (sum of absolute deviations / 2)
            deviation = (current_weights - target_vec).abs().sum() + abs(cash_weight - target_cash)
            transaction_cost = (transaction_cost_bps / 10000.0) * (deviation / 2.0)
            portfolio_return -= transaction_cost

            current_weights = target_vec.copy()
            cash_weight = target_cash

        portfolio_returns.append(portfolio_return)
        dates.append(date)

    return pd.Series(portfolio_returns, index=pd.Index(dates), name="four_umbrellas")


def simulate_benchmark(
    monthly_returns: pd.DataFrame,
    weights: Dict[str, float],
    rebalance_annual: bool = True,
) -> pd.Series:
    """
    Simulate a simple benchmark portfolio (e.g., 60/40) with annual rebalancing.
    """
    available = [k for k in weights if k in monthly_returns.columns]
    if len(available) == 0:
        return pd.Series(dtype=float)

    # Normalize weights over available columns
    weights_avail = pd.Series({k: weights[k] for k in available})
    weights_avail = weights_avail / weights_avail.sum()

    current = weights_avail.copy()
    returns_out = []
    dates = []

    for date in monthly_returns.index:
        r = monthly_returns.loc[date, available].fillna(0.0)
        portfolio_r = float((current * r).sum())
        # Drift
        grown = current * (1.0 + r)
        current = grown / grown.sum()
        # Annual rebalance in January
        if rebalance_annual and date.month == 1:
            current = weights_avail.copy()
        returns_out.append(portfolio_r)
        dates.append(date)

    return pd.Series(returns_out, index=pd.Index(dates))
