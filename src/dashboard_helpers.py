"""
Streamlit-free helpers shared between `streamlit_app.py` and its tests.

Two families of helpers live here:

1. **Legacy globals management** (pre-PR4): `snapshot_config`, `restore_config`,
   `apply_macro_weights`. The old gold/DBi-slider UI mutated
   src.portfolio globals; these helpers snapshotted / restored them so CLI
   runs in the same process weren't contaminated. Retained for backwards
   compat of `tests/test_dashboard_smoke.py`; the new Streamlit UI (PR4+)
   never mutates globals.

2. **Portfolio-aware helpers** (PR4): `build_portfolio_from_ui_state`,
   `compute_effective_start`, `format_asset_start_date`. Pure functions
   consumed by the new Impostazioni tab and trivially testable without a
   Streamlit runtime.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from . import portfolio as portfolio_cfg
from .data_catalog import AssetInfo
from .portfolio_model import AssetAllocation, Portfolio

_ITALIAN_MONTH_ABBR = (
    "Gen", "Feb", "Mar", "Apr", "Mag", "Giu",
    "Lug", "Ago", "Set", "Ott", "Nov", "Dic",
)


def snapshot_config() -> Dict[str, Any]:
    """
    Return a deep-copied snapshot of the mutable portfolio configuration.
    Safe to call multiple times; each snapshot is independent.
    """
    return {
        "WEIGHTS": copy.deepcopy(portfolio_cfg.WEIGHTS),
        "EQUITY": copy.deepcopy(portfolio_cfg.EQUITY),
        "OPTIONS": copy.deepcopy(portfolio_cfg.OPTIONS),
        "REBALANCE": copy.deepcopy(portfolio_cfg.REBALANCE),
    }


def restore_config(snapshot: Dict[str, Any]) -> None:
    """
    Restore the portfolio configuration from a previously-taken snapshot.
    Mutates the module globals in place so existing references stay valid.
    """
    portfolio_cfg.WEIGHTS.clear()
    portfolio_cfg.WEIGHTS.update(snapshot["WEIGHTS"])
    portfolio_cfg.EQUITY.clear()
    portfolio_cfg.EQUITY.update(snapshot["EQUITY"])
    portfolio_cfg.OPTIONS.__dict__.update(snapshot["OPTIONS"].__dict__)
    portfolio_cfg.REBALANCE.__dict__.update(snapshot["REBALANCE"].__dict__)


def apply_macro_weights(gold_pct: float, dbi_pct: float) -> None:
    """
    Apply macro weight changes (gold and DBi) and re-derive cash as the
    residual so WEIGHTS always sums to exactly 1.0.

    Raises ValueError if the resulting cash weight would be negative.

    Args:
        gold_pct: target gold weight as a fraction [0, 1] (not percentage).
        dbi_pct: target DBi weight as a fraction [0, 1].

    .. deprecated:: PR4
        The new Streamlit UI builds a :class:`Portfolio` dataclass instead
        of mutating globals. This helper is retained so the pre-PR4 test
        suite keeps passing; new callers should use
        :func:`build_portfolio_from_ui_state`.
    """
    if not (0.0 <= gold_pct <= 1.0):
        raise ValueError(f"gold_pct must be in [0, 1], got {gold_pct}")
    if not (0.0 <= dbi_pct <= 1.0):
        raise ValueError(f"dbi_pct must be in [0, 1], got {dbi_pct}")

    portfolio_cfg.WEIGHTS["gold"] = gold_pct
    portfolio_cfg.WEIGHTS["dbi"] = dbi_pct
    non_cash_sum = sum(v for k, v in portfolio_cfg.WEIGHTS.items() if k != "cash")
    new_cash = 1.0 - non_cash_sum
    if new_cash < -1e-9:
        raise ValueError(
            f"gold={gold_pct} + dbi={dbi_pct} leave negative cash ({new_cash:.4f}). "
            f"Other non-cash weights sum to {non_cash_sum - gold_pct - dbi_pct:.4f}."
        )
    portfolio_cfg.WEIGHTS["cash"] = max(0.0, new_cash)


# ============================================================================
# PR4 — portfolio-aware helpers (new Streamlit UI)
# ============================================================================

def build_portfolio_from_ui_state(
    name: str,
    assets_state: List[Dict[str, Any]],
    options_overlay: bool = False,
    rebalance_months: Tuple[int, ...] = (1, 7),
    transaction_cost_bps: float = 20.0,
    notes: str = "",
) -> Portfolio:
    """Construct and validate a :class:`Portfolio` from the Streamlit
    session-state shape produced by the Impostazioni tab.

    ``assets_state`` is a list of dicts as rendered by ``st.data_editor``
    — each must have at least ``key`` (str) and ``weight`` (float,
    expressed as a **fraction** in [0, 1], not a percentage).

    Propagates ``Portfolio.validate`` errors (sum ≠ 1.0, duplicates,
    negatives, empty fields) as ``ValueError`` with unchanged messages so
    the UI can surface them directly in ``st.error``.

    This is the single gateway between the UI's session_state and the
    engine — no globals are touched.
    """
    allocations = [
        AssetAllocation(key=str(row["key"]), weight=float(row["weight"]))
        for row in assets_state
    ]
    portfolio = Portfolio(
        name=name,
        assets=allocations,
        options_overlay=bool(options_overlay),
        rebalance_months=tuple(int(m) for m in rebalance_months),
        transaction_cost_bps=float(transaction_cost_bps),
        notes=notes,
    )
    portfolio.validate()
    return portfolio


def compute_effective_start(
    user_start: pd.Timestamp,
    portfolio: Portfolio,
    catalog: Dict[str, AssetInfo],
) -> Tuple[pd.Timestamp, Optional[str]]:
    """Return ``(effective_start, warning_text_or_None)``.

    Looks up each portfolio asset in ``catalog`` and finds the **latest**
    ``start_date`` across the portfolio's assets (i.e. the earliest date
    from which ALL assets have data). If that latest-start is after
    ``user_start``, returns the latest-start as the effective start plus a
    warning string identifying the constraining asset. Otherwise returns
    the user's own start with no warning.

    Assets whose catalog entry has ``start_date=None`` (cash, or a CSV
    that hasn't been downloaded yet) are ignored — they don't constrain
    the window; the engine drops them for that period if unavailable.

    Used by the Impostazioni tab to tell the user "your selected period
    starts before BTC has data; the simulation will actually start at
    2014-09".
    """
    constraining_key: Optional[str] = None
    constraining_start: Optional[pd.Timestamp] = None
    for alloc in portfolio.assets:
        info = catalog.get(alloc.key)
        if info is None or info.start_date is None:
            continue
        if constraining_start is None or info.start_date > constraining_start:
            constraining_start = info.start_date
            constraining_key = alloc.key

    if constraining_start is None or constraining_start <= user_start:
        return user_start, None

    display_name = catalog[constraining_key].display_name if constraining_key else constraining_key
    warning = (
        f"Il periodo parte prima della disponibilità di \"{display_name}\": "
        f"effective start = {constraining_start.date().isoformat()}"
    )
    return constraining_start, warning


def format_asset_start_date(info: Optional[AssetInfo]) -> str:
    """Compact Italian-language start-date label for the asset picker table.

    Returns:
      - ``"—"`` when ``info`` is ``None`` OR has no ``start_date``
        (cash, or a CSV not yet in data/raw/)
      - ``"Gen 2003"`` style otherwise (Italian month abbreviation + year)
    """
    if info is None or info.start_date is None:
        return "—"
    d = info.start_date
    return f"{_ITALIAN_MONTH_ABBR[d.month - 1]} {d.year}"
