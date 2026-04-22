"""
Shared pytest fixtures for the test suite.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Make src importable
import sys
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


@pytest.fixture
def deterministic_monthly_returns() -> pd.Series:
    """
    A 12-month return series of constant +1% — useful for CAGR / Sharpe
    sanity checks where the answer is analytically known.
    """
    dates = pd.date_range("2020-01-31", periods=12, freq="ME")
    return pd.Series([0.01] * 12, index=dates)


@pytest.fixture
def drawdown_recovery_returns() -> pd.Series:
    """
    A series that rises, then drops 50%, then fully recovers over 6 months.
    Used for Max DD / underwater / recovery tests.
    """
    dates = pd.date_range("2020-01-31", periods=15, freq="ME")
    # +10% each for 3 months, then -50% once, then recover by +19.05% × 6
    returns = [0.10, 0.10, 0.10, -0.50] + [0.19005] * 6 + [0.0] * 5
    # Trim to 15 items
    return pd.Series(returns[:15], index=dates)


@pytest.fixture
def zero_returns() -> pd.Series:
    """All-zero return series (flat portfolio)."""
    dates = pd.date_range("2020-01-31", periods=24, freq="ME")
    return pd.Series([0.0] * 24, index=dates)


@pytest.fixture
def realistic_monthly_returns() -> pd.Series:
    """A 10-year series of random-walk monthly returns, seeded for determinism."""
    rng = np.random.default_rng(42)
    dates = pd.date_range("2015-01-31", periods=120, freq="ME")
    returns = rng.normal(0.006, 0.035, 120)
    return pd.Series(returns, index=dates)


@pytest.fixture
def sample_returns_dict(realistic_monthly_returns) -> dict:
    """Small returns_dict with two portfolios for comparative tests."""
    dates = realistic_monthly_returns.index
    rng = np.random.default_rng(43)
    benchmark = pd.Series(rng.normal(0.005, 0.045, 120), index=dates)
    return {
        "Four Umbrellas": realistic_monthly_returns,
        "100% S&P 500 TR EUR": benchmark,
    }


@pytest.fixture
def tmp_output_dir(tmp_path) -> Path:
    """Pytest's tmp_path as the output directory for chart/report tests."""
    out = tmp_path / "output"
    out.mkdir()
    return out
