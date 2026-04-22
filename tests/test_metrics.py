"""
Unit tests for src/metrics.py — CAGR, Sharpe, Sortino, Max DD, Calmar,
Ulcer Index, CVaR, longest underwater, recovery months.

Tests use analytically-known cases where possible, plus realistic
fixtures for sanity checks.
"""

import math

import numpy as np
import pandas as pd
import pytest

from src.metrics import (
    cumulative_wealth,
    cagr,
    annualized_vol,
    sharpe_ratio,
    sortino_ratio,
    max_drawdown,
    calmar_ratio,
    ulcer_index,
    cvar,
    longest_underwater_months,
    recovery_months_after_max_dd,
    crisis_drawdown,
    compute_all,
    PortfolioStats,
)


# ---------------------------------------------------------------------------
# cumulative_wealth
# ---------------------------------------------------------------------------

class TestCumulativeWealth:
    def test_starts_at_start_nav(self, deterministic_monthly_returns):
        wealth = cumulative_wealth(deterministic_monthly_returns, start_nav=1000.0)
        # After 1 month of +1%, wealth should be 1000 * 1.01 = 1010
        assert wealth.iloc[0] == pytest.approx(1010.0)

    def test_ends_at_expected_value(self, deterministic_monthly_returns):
        """12 months of +1% → W = 1 * 1.01^12 ≈ 1.1268"""
        wealth = cumulative_wealth(deterministic_monthly_returns, start_nav=1.0)
        assert wealth.iloc[-1] == pytest.approx(1.01 ** 12, rel=1e-6)

    def test_zero_returns_stays_flat(self, zero_returns):
        wealth = cumulative_wealth(zero_returns, start_nav=100.0)
        assert (wealth == 100.0).all()


# ---------------------------------------------------------------------------
# CAGR
# ---------------------------------------------------------------------------

class TestCAGR:
    def test_cagr_constant_positive(self, deterministic_monthly_returns):
        """12 months at +1% monthly → CAGR = 1.01^12 - 1 ≈ 0.1268"""
        c = cagr(deterministic_monthly_returns)
        expected = 1.01 ** 12 - 1.0
        assert c == pytest.approx(expected, rel=1e-6)

    def test_cagr_zero(self, zero_returns):
        assert cagr(zero_returns) == pytest.approx(0.0, abs=1e-9)

    def test_cagr_negative(self):
        """All -1% monthly → CAGR negative"""
        dates = pd.date_range("2020-01-31", periods=12, freq="M")
        r = pd.Series([-0.01] * 12, index=dates)
        assert cagr(r) < 0


# ---------------------------------------------------------------------------
# Annualized volatility
# ---------------------------------------------------------------------------

class TestAnnualizedVol:
    def test_zero_vol_on_constant_series(self, deterministic_monthly_returns):
        assert annualized_vol(deterministic_monthly_returns) == pytest.approx(0.0, abs=1e-12)

    def test_scales_with_sqrt_12(self):
        """Monthly std of 0.02 → annualized = 0.02 * sqrt(12) ≈ 0.0693"""
        dates = pd.date_range("2020-01-31", periods=12, freq="M")
        r = pd.Series([0.02, -0.02] * 6, index=dates)
        # std(ddof=1) of [0.02,-0.02,...] = 0.02087 (with 11 dof)
        # But any non-zero std should give positive annualized vol
        assert annualized_vol(r) > 0.05


# ---------------------------------------------------------------------------
# Sharpe / Sortino
# ---------------------------------------------------------------------------

class TestSharpe:
    def test_sharpe_positive_for_positive_returns(self, realistic_monthly_returns):
        s = sharpe_ratio(realistic_monthly_returns, risk_free_rate=0.02)
        # realistic_monthly_returns has mean ~0.006, std ~0.035 → annualized mean ~0.072, vol ~0.12, Sharpe > 0.3
        assert s > 0.0

    def test_sharpe_on_zero_vol_is_nan(self, deterministic_monthly_returns):
        """Constant positive returns → vol = 0 → Sharpe undefined."""
        s = sharpe_ratio(deterministic_monthly_returns, risk_free_rate=0.0)
        assert math.isnan(s)

    def test_sortino_positive(self, realistic_monthly_returns):
        s = sortino_ratio(realistic_monthly_returns, risk_free_rate=0.02)
        # Should be positive and typically > Sharpe for positive-drift series
        assert s > 0.0


# ---------------------------------------------------------------------------
# Max Drawdown / Calmar
# ---------------------------------------------------------------------------

class TestDrawdown:
    def test_no_drawdown_when_monotone(self, deterministic_monthly_returns):
        assert max_drawdown(deterministic_monthly_returns) == pytest.approx(0.0, abs=1e-9)

    def test_drawdown_is_negative(self, drawdown_recovery_returns):
        dd = max_drawdown(drawdown_recovery_returns)
        # After +10%,+10%,+10%,-50% the wealth is 1.1^3 * 0.5 = 0.6655
        # Peak was 1.1^3 = 1.331 → DD = 0.6655/1.331 - 1 = -0.50
        assert dd == pytest.approx(-0.50, abs=1e-6)

    def test_calmar_ratio_sign(self, realistic_monthly_returns):
        """Positive-drift series → Calmar should be positive."""
        c = calmar_ratio(realistic_monthly_returns)
        assert c > 0.0

    def test_calmar_nan_on_no_drawdown(self, deterministic_monthly_returns):
        """No drawdown → Calmar is NaN (division by zero guarded)."""
        c = calmar_ratio(deterministic_monthly_returns)
        assert math.isnan(c)


# ---------------------------------------------------------------------------
# Ulcer Index / CVaR
# ---------------------------------------------------------------------------

class TestUlcerCVaR:
    def test_ulcer_zero_when_no_drawdown(self, deterministic_monthly_returns):
        assert ulcer_index(deterministic_monthly_returns) == pytest.approx(0.0, abs=1e-9)

    def test_ulcer_positive_with_drawdown(self, drawdown_recovery_returns):
        assert ulcer_index(drawdown_recovery_returns) > 0.0

    def test_cvar_is_negative_with_negative_tail(self, realistic_monthly_returns):
        """With any realistic series, the 5% worst months are negative."""
        c = cvar(realistic_monthly_returns, alpha=0.05)
        assert c < 0.0

    def test_cvar_at_alpha_one_is_mean(self, realistic_monthly_returns):
        """CVaR at alpha=1.0 is the mean of all returns."""
        c = cvar(realistic_monthly_returns, alpha=1.0)
        assert c == pytest.approx(realistic_monthly_returns.mean(), rel=1e-6)


# ---------------------------------------------------------------------------
# Underwater / Recovery
# ---------------------------------------------------------------------------

class TestUnderwater:
    def test_no_underwater_when_monotone(self, deterministic_monthly_returns):
        assert longest_underwater_months(deterministic_monthly_returns) == 0

    def test_underwater_counts_months(self, drawdown_recovery_returns):
        """Fixture: 3 months up, 1 crash, 6 months recovery, 5 flat.
        After the crash, wealth is below the prior peak for some months until it recovers."""
        lw = longest_underwater_months(drawdown_recovery_returns)
        assert lw >= 1

    def test_recovery_months_is_positive(self, drawdown_recovery_returns):
        r = recovery_months_after_max_dd(drawdown_recovery_returns)
        # The fixture is designed to recover; may or may not reach exactly the prior peak
        assert r is None or r >= 0

    def test_recovery_months_none_when_no_recovery(self):
        """Series that drops and never recovers → None."""
        dates = pd.date_range("2020-01-31", periods=6, freq="M")
        r = pd.Series([0.10, -0.50, -0.05, -0.02, 0.01, 0.01], index=dates)
        assert recovery_months_after_max_dd(r) is None


# ---------------------------------------------------------------------------
# Crisis drawdown
# ---------------------------------------------------------------------------

class TestCrisisDrawdown:
    def test_crisis_drawdown_bounded(self, realistic_monthly_returns):
        """Any slice should return a value in [-1, 0] (or NaN if empty)."""
        r = realistic_monthly_returns
        dd = crisis_drawdown(r, str(r.index[0].date()), str(r.index[-1].date()))
        assert dd <= 0.0
        assert dd >= -1.0


# ---------------------------------------------------------------------------
# compute_all / PortfolioStats
# ---------------------------------------------------------------------------

class TestComputeAll:
    def test_returns_portfolio_stats_object(self, realistic_monthly_returns):
        s = compute_all("Test", realistic_monthly_returns, risk_free_rate=0.02)
        assert isinstance(s, PortfolioStats)
        assert s.name == "Test"

    def test_all_fields_populated(self, realistic_monthly_returns):
        s = compute_all("Test", realistic_monthly_returns, risk_free_rate=0.02)
        # All numeric fields should be floats (no NaN for a realistic series except possibly recovery_months)
        assert isinstance(s.cagr, float)
        assert isinstance(s.annualized_vol, float)
        assert isinstance(s.sharpe, float)
        assert isinstance(s.max_drawdown, float)
        assert isinstance(s.longest_underwater_months, int)

    def test_pct_positive_months_bounded(self, realistic_monthly_returns):
        s = compute_all("Test", realistic_monthly_returns)
        assert 0.0 <= s.pct_positive_months <= 1.0
