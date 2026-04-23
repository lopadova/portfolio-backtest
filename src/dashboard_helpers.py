"""
Streamlit-free helpers shared between `streamlit_app.py` and its tests.

These functions snapshot and restore the mutable portfolio configuration
(WEIGHTS, EQUITY, OPTIONS, REBALANCE) so that UI-driven parameter changes
do not leak into subsequent CLI runs in the same Python process. Kept in
a dedicated module so tests can import and exercise the REAL implementation
(rather than re-implementing the logic inside the test file, which would
silently drift if the production code changes).
"""

from __future__ import annotations

import copy
from typing import Any, Dict

from . import portfolio as portfolio_cfg


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
