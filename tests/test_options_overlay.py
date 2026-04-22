"""
Unit tests for src/options_overlay.py — Black-Scholes pricing, put spread
payoff, VIX-based implied vol, and the overall simulator.
"""

import math

import numpy as np
import pandas as pd
import pytest

from src.options_overlay import (
    bs_put_price,
    bs_put_spread_price,
    iv_from_vix,
    simulate_options_overlay,
)


# ---------------------------------------------------------------------------
# Black-Scholes put pricing
# ---------------------------------------------------------------------------

class TestBSPutPrice:
    def test_atm_put_known_value(self):
        """
        Hull textbook-style test: S=100, K=100, T=1, r=0.05, sigma=0.20, q=0
        → Put price ≈ 5.57 (computed analytically).
        """
        p = bs_put_price(S=100, K=100, T=1.0, r=0.05, sigma=0.20, q=0.0)
        assert p == pytest.approx(5.57, abs=0.05)

    def test_deep_otm_put_near_zero(self):
        """Strike far below spot, short time → put ~ 0."""
        p = bs_put_price(S=100, K=50, T=0.5, r=0.03, sigma=0.20, q=0.0)
        assert p < 0.01

    def test_deep_itm_put_approaches_intrinsic(self):
        """Strike far above spot, short time → put ~ K - S (discounted)."""
        p = bs_put_price(S=50, K=100, T=0.01, r=0.0, sigma=0.20, q=0.0)
        # Intrinsic value is K - S = 50; p should be very close
        assert p == pytest.approx(50.0, abs=0.5)

    def test_zero_time_is_intrinsic(self):
        """At expiry, put = max(K - S, 0)."""
        p = bs_put_price(S=90, K=100, T=0.0, r=0.05, sigma=0.20, q=0.0)
        assert p == pytest.approx(10.0, abs=1e-6)

    def test_zero_time_otm_is_zero(self):
        p = bs_put_price(S=100, K=90, T=0.0, r=0.05, sigma=0.20, q=0.0)
        assert p == pytest.approx(0.0, abs=1e-6)

    def test_higher_vol_gives_higher_price(self):
        """Put price is monotonic increasing in volatility."""
        p_low = bs_put_price(S=100, K=95, T=0.5, r=0.03, sigma=0.15, q=0.0)
        p_high = bs_put_price(S=100, K=95, T=0.5, r=0.03, sigma=0.30, q=0.0)
        assert p_high > p_low


# ---------------------------------------------------------------------------
# Put spread pricing
# ---------------------------------------------------------------------------

class TestPutSpreadPrice:
    def test_positive_for_atm_otm_combo(self):
        """Long ATM put, short deep OTM put → spread is positive debit."""
        spread = bs_put_spread_price(S=100, K_long=100, K_short=70, T=0.5, r=0.03, sigma=0.20, q=0.0)
        assert spread > 0

    def test_equal_strikes_give_zero(self):
        """If K_long == K_short, spread value = 0."""
        spread = bs_put_spread_price(S=100, K_long=90, K_short=90, T=0.5, r=0.03, sigma=0.20, q=0.0)
        assert spread == pytest.approx(0.0, abs=1e-6)

    def test_spread_less_than_long_put_alone(self):
        """Debit spread value must be less than a long put alone (short put subtracts value)."""
        long_only = bs_put_price(S=100, K=95, T=0.5, r=0.03, sigma=0.20, q=0.0)
        spread = bs_put_spread_price(S=100, K_long=95, K_short=70, T=0.5, r=0.03, sigma=0.20, q=0.0)
        assert spread < long_only


# ---------------------------------------------------------------------------
# IV from VIX
# ---------------------------------------------------------------------------

class TestIVFromVIX:
    def test_otm_iv_higher_than_atm(self):
        """Skew adjustment: OTM put IV should exceed ATM IV."""
        vix = 20.0
        atm = iv_from_vix(vix, moneyness=1.0)
        otm = iv_from_vix(vix, moneyness=0.85)
        assert otm > atm

    def test_iv_in_decimal(self):
        """For VIX=20, IV returned should be ~0.20 (decimal, not percent)."""
        iv = iv_from_vix(20.0, moneyness=1.0)
        assert 0.15 < iv < 0.30

    def test_zero_vix_zero_iv(self):
        assert iv_from_vix(0.0, moneyness=0.9) == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# End-to-end simulator smoke test
# ---------------------------------------------------------------------------

class TestSimulatorSmoke:
    @pytest.fixture
    def fake_market_data(self):
        """Minimal fake data for a 12-month overlay run."""
        daily_idx = pd.date_range("2023-01-03", "2023-12-29", freq="B")
        monthly_idx = pd.date_range("2023-01-31", "2023-12-31", freq="M")

        # Flat market
        spy = pd.Series(400.0, index=daily_idx)
        qqq = pd.Series(300.0, index=daily_idx)
        vix = pd.Series(18.0, index=daily_idx)
        rf = pd.Series(0.04, index=daily_idx)
        nav = pd.Series(100_000.0, index=monthly_idx)
        return dict(spy=spy, qqq=qqq, vix=vix, rf=rf, nav=nav)

    def test_simulator_returns_series_indexed_by_nav(self, fake_market_data):
        d = fake_market_data
        out = simulate_options_overlay(
            spy_daily=d["spy"],
            qqq_daily=d["qqq"],
            vix_daily=d["vix"],
            rf_daily=d["rf"],
            nav_series=d["nav"],
            rebalance_months=(1, 7),
        )
        assert isinstance(out, pd.Series)
        assert out.index.equals(d["nav"].index)

    def test_simulator_premium_outflow_sums_negative_initially(self, fake_market_data):
        """On a flat market, total overlay P&L should be negative (premium lost, no payoff)."""
        d = fake_market_data
        out = simulate_options_overlay(
            spy_daily=d["spy"],
            qqq_daily=d["qqq"],
            vix_daily=d["vix"],
            rf_daily=d["rf"],
            nav_series=d["nav"],
            rebalance_months=(1, 7),
        )
        # Flat market: we pay premium at roll dates and get zero back at expiry
        assert out.sum() <= 0.0
