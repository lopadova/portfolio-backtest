"""
Rebalancing engine — computes monthly portfolio returns with semi-annual
rebalancing back to target weights, including transaction costs.
"""

from __future__ import annotations

import re
from dataclasses import replace
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .portfolio import (
    WEIGHTS, EQUITY, CRYPTO, BONDS, EM_SATELLITES, PENSION,
    TER_ANNUAL, REBALANCE,
)
from .portfolio_model import AssetAllocation, Portfolio


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


_LEGACY_SERIES_NAME = "four_umbrellas"


def _slugify(name: str) -> str:
    """Lowercase, replace spaces with underscores, strip non-identifier chars.
    Used for the returned ``pd.Series.name`` in the generic engine."""
    s = name.strip().lower().replace(" ", "_")
    return re.sub(r"[^a-z0-9_]+", "", s) or "portfolio"


def build_portfolio_from_globals() -> Portfolio:
    """Construct a Portfolio from the legacy src.portfolio module globals
    (WEIGHTS / EQUITY / CRYPTO / BONDS / EM_SATELLITES / PENSION / REBALANCE /
    OPTIONS). Used by the simulate_portfolio() shim so pre-PR2 callers keep
    getting bit-identical results without a code change.

    When the globals are retired (PR3+), this function becomes the last
    place that reads them and can be deleted together with the globals.
    """
    from .portfolio import OPTIONS  # local import avoids a circular ref at module load

    target = build_target_weights()  # dict sleeve -> weight, excluding cash
    assets = [AssetAllocation(key=k, weight=float(v)) for k, v in target.items()]
    assets.append(AssetAllocation(key="cash", weight=float(WEIGHTS["cash"])))
    return Portfolio(
        name="Four Umbrellas",
        assets=assets,
        options_overlay=bool(OPTIONS.enabled),
        rebalance_months=tuple(REBALANCE.months),
        transaction_cost_bps=float(REBALANCE.transaction_cost_bps),
        notes="Default preset, built from the src.portfolio legacy globals.",
    )


def simulate_portfolio_generic(
    monthly_returns: pd.DataFrame,
    portfolio: Portfolio,
    ter_by_key: Optional[Dict[str, float]] = None,
    btc_activation_date: Optional[pd.Timestamp] = None,
    verbose: bool = False,
) -> pd.Series:
    """
    Run the rebalancing simulation for an arbitrary :class:`Portfolio`.

    Parameters
    ----------
    monthly_returns :
        DataFrame indexed by EOM date; columns are asset keys. Any asset
        present in ``portfolio`` but missing from ``monthly_returns.columns``
        is silently dropped (same behavior as the legacy engine).
    portfolio :
        The Portfolio to simulate. Must pass :meth:`Portfolio.validate`.
    ter_by_key :
        Per-asset annual TER (decimal). Any asset not in the dict is assumed
        zero TER. Pass ``None`` or an empty dict to disable TER entirely.
    btc_activation_date :
        Date after which the asset whose key is ``"btc"`` starts receiving
        its target weight. Before that date, its weight is parked in cash.
        Pass ``None`` if the portfolio has no BTC or no activation logic.

    Returns
    -------
    pd.Series of monthly portfolio returns, named after the portfolio slug.
    """
    portfolio.validate()
    rebalance_months = tuple(portfolio.rebalance_months)
    transaction_cost_bps = float(portfolio.transaction_cost_bps)

    # Flat non-cash target (matches legacy build_target_weights shape)
    target_weights = portfolio.to_legacy_target_weights()
    cash_target_weight = portfolio.cash_weight()

    # Only keep assets for which we have return data
    available = [k for k in target_weights if k in monthly_returns.columns]

    # Start weights = targets
    current_weights = pd.Series(0.0, index=available)
    cash_weight = cash_target_weight
    for k in available:
        current_weights[k] = target_weights[k]

    # BTC not yet active → park weight in cash (same convention as legacy)
    if (
        "btc" in available
        and btc_activation_date is not None
        and monthly_returns.index[0] < btc_activation_date
    ):
        cash_weight += current_weights["btc"]
        current_weights["btc"] = 0.0

    # Monthly TER per asset
    ter = ter_by_key or {}
    monthly_ter = pd.Series({k: ter.get(k, 0.0) / 12.0 for k in available})

    portfolio_returns: List[float] = []
    dates: List[pd.Timestamp] = []

    for date in monthly_returns.index:
        month_returns = monthly_returns.loc[date, available].fillna(0.0)
        # Applying TER unconditionally: callers that want to skip TER pass
        # ter_by_key=None or {}, which makes monthly_ter a series of zeros.
        month_returns = month_returns - monthly_ter

        # BTC activation
        if (
            "btc" in available
            and btc_activation_date is not None
            and date >= btc_activation_date
            and current_weights["btc"] == 0.0
        ):
            btc_target = target_weights["btc"]
            current_weights["btc"] = btc_target
            cash_weight -= btc_target
            if verbose:
                print(f"BTC activated on {date.date()}, weight {btc_target:.4f}")

        portfolio_return = float((current_weights * month_returns).sum())
        # Cash earns 0% in conservative base model.

        # Drift
        growth_factors = 1.0 + month_returns
        new_weights = current_weights * growth_factors
        new_cash_weight = cash_weight
        total = new_weights.sum() + new_cash_weight
        current_weights = new_weights / total
        cash_weight = new_cash_weight / total

        # Rebalance
        if date.month in rebalance_months:
            target_vec = pd.Series(target_weights).reindex(available).fillna(0.0)
            if (
                "btc" in available
                and btc_activation_date is not None
                and date < btc_activation_date
            ):
                target_vec["btc"] = 0.0
                target_cash = cash_target_weight + target_weights["btc"]
            else:
                target_cash = cash_target_weight

            deviation = (current_weights - target_vec).abs().sum() + abs(
                cash_weight - target_cash
            )
            transaction_cost = (transaction_cost_bps / 10000.0) * (deviation / 2.0)
            portfolio_return -= transaction_cost

            current_weights = target_vec.copy()
            cash_weight = target_cash

        portfolio_returns.append(portfolio_return)
        dates.append(date)

    return pd.Series(
        portfolio_returns, index=pd.Index(dates), name=_slugify(portfolio.name)
    )


def simulate_portfolio(
    monthly_returns: pd.DataFrame,
    btc_activation_date: pd.Timestamp,
    apply_ter: bool = True,
    rebalance_months: tuple = None,
    transaction_cost_bps: float = None,
    verbose: bool = False,
    *,
    portfolio: Optional[Portfolio] = None,
) -> pd.Series:
    """
    Backward-compatible shim around :func:`simulate_portfolio_generic`.

    With no ``portfolio`` argument, it reconstructs one from the legacy
    module globals (Four Umbrellas preset) and runs the generic engine.
    The returned series is named ``"four_umbrellas"`` to match pre-PR2
    behavior. Passing an explicit ``portfolio`` uses its name slug instead.
    ``rebalance_months`` / ``transaction_cost_bps`` kwargs override the
    portfolio's own settings (applied to a copy — no mutation).
    """
    if portfolio is None:
        portfolio = build_portfolio_from_globals()
        preserve_legacy_name = True
    else:
        preserve_legacy_name = False

    # Apply per-call overrides to a COPY so the caller's Portfolio isn't mutated.
    overrides = {}
    if rebalance_months is not None:
        overrides["rebalance_months"] = tuple(rebalance_months)
    if transaction_cost_bps is not None:
        overrides["transaction_cost_bps"] = float(transaction_cost_bps)
    if overrides:
        portfolio = replace(portfolio, **overrides)

    ter_by_key = dict(TER_ANNUAL) if apply_ter else {}

    series = simulate_portfolio_generic(
        monthly_returns,
        portfolio,
        ter_by_key=ter_by_key,
        btc_activation_date=btc_activation_date,
        verbose=verbose,
    )
    if preserve_legacy_name:
        series.name = _LEGACY_SERIES_NAME
    return series


def simulate_benchmark(
    monthly_returns: pd.DataFrame,
    weights: Dict[str, float],
    rebalance_annual: bool = True,
    min_coverage: float = 0.80,
) -> pd.Series:
    """
    Simulate a simple benchmark portfolio (e.g., 60/40) with annual rebalancing.

    min_coverage: minimum fraction of total weight that must be represented
    in monthly_returns.columns. If less than this fraction is available, the
    benchmark is skipped (returns empty Series). Default 80% — means we allow
    up to 20% of the portfolio weight to be on missing data and normalize
    the rest. This prevents artificially scaling up a tiny remainder.
    Must be a finite number in [0.0, 1.0]; ValueError is raised otherwise.
    """
    # Validate min_coverage is a finite number in [0, 1]
    if not isinstance(min_coverage, (int, float)) or not np.isfinite(min_coverage):
        raise ValueError(
            f"min_coverage must be a finite number in [0, 1], got {min_coverage!r}"
        )
    if not (0.0 <= min_coverage <= 1.0):
        raise ValueError(
            f"min_coverage must be in [0.0, 1.0], got {min_coverage}"
        )

    available = [k for k in weights if k in monthly_returns.columns]
    if len(available) == 0:
        return pd.Series(dtype=float)

    # Coverage check: sum of weights for which we have data
    coverage = sum(weights[k] for k in available)
    if coverage < min_coverage:
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
