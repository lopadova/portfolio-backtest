"""
Portfolio configuration — the Four Umbrellas Portfolio.

All weights are on total NAV. The structure mirrors the article:

    Total NAV (100%) =
        Pension 11%
        + Liquid sleeve 73%  (gold 18.25, equity 46.72, crypto 3.65, bonds 3.65, EM sat 0.73)
        + DBi Managed Futures 5%
        + Cash 11%

    Options overlay = 0.30% NAV/year flow (not an asset; modeled as ongoing cost + payoff)

Every dict in this file is meant to be edited for sensitivity analysis.
"""

from dataclasses import dataclass, field
from typing import Dict, Tuple


# ============================================================================
# Main portfolio weights (sum to 1.00 excluding the options overlay flow)
# ============================================================================

WEIGHTS: Dict[str, float] = {
    "pension":  0.11,      # Locked pension block — not actively rebalanced
    "gold":     0.1825,    # Physical gold, unhedged
    "equity":   0.4672,    # Factor-tilted equity core — breakdown in EQUITY below
    "crypto":   0.0365,    # BTC + diversified basket
    "bonds":    0.0365,    # Euro gov short + medium duration
    "em_sat":   0.0073,    # India + China satellites
    "dbi":      0.05,      # Managed futures (crisis alpha)
    "cash":     0.11,      # Cash buffer
}

# Sanity check
assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-9, \
    f"Portfolio weights must sum to 1.00 exactly, got {sum(WEIGHTS.values())}"


# ============================================================================
# Equity sleeve internal breakdown (sums to WEIGHTS["equity"] = 0.4672)
# ============================================================================

EQUITY: Dict[str, float] = {
    "put_write":    0.131,    # Defensive US put-write ETF, EUR hedged
    "nasdaq_top30": 0.117,    # Concentrated Nasdaq top 30
    "hc_us_hedged": 0.022,    # S&P 500 Healthcare, EUR hedged
    "hc_world":     0.051,    # MSCI World Healthcare, unhedged
    "quality":      0.073,    # Quality / Developed Aristocrats, unhedged
    "momentum":     0.058,    # MSCI World Momentum, unhedged
    "ex_usa":       0.015,    # MSCI World ex-USA, unhedged
}

# Sanity check (allow 0.1% rounding tolerance since article percentages are rounded)
assert abs(sum(EQUITY.values()) - WEIGHTS["equity"]) < 0.001, \
    f"Equity sleeve must sum to {WEIGHTS['equity']}, got {sum(EQUITY.values())}"


# ============================================================================
# Crypto sleeve internal breakdown (sums to WEIGHTS["crypto"] = 0.0365)
# ============================================================================

CRYPTO: Dict[str, float] = {
    "btc":           0.80 * WEIGHTS["crypto"],   # 2.92% NAV
    "crypto_basket": 0.20 * WEIGHTS["crypto"],   # 0.73% NAV
}


# ============================================================================
# Bonds sleeve internal breakdown
# ============================================================================

BONDS: Dict[str, float] = {
    "eur_gov_direct": 0.10 * WEIGHTS["bonds"],   # direct holdings; modeled as short-gov proxy
    "eur_gov_1_3y":   0.45 * WEIGHTS["bonds"],
    "eur_gov_7_10y":  0.45 * WEIGHTS["bonds"],
}


# ============================================================================
# EM satellites
# ============================================================================

EM_SATELLITES: Dict[str, float] = {
    "india": 0.50 * WEIGHTS["em_sat"],
    "china": 0.50 * WEIGHTS["em_sat"],
}


# ============================================================================
# Pension block internal composition (fixed, regulatory)
# ============================================================================

PENSION: Dict[str, float] = {
    "bond_like":   0.88 * WEIGHTS["pension"],   # 9.68% NAV
    "equity_like": 0.12 * WEIGHTS["pension"],   # 1.32% NAV
}


# ============================================================================
# Hedging flags — True means return is in EUR (hedged), False means USD (unhedged)
# ============================================================================

HEDGED: Dict[str, bool] = {
    "put_write":    True,
    "nasdaq_top30": False,
    "hc_us_hedged": True,
    "hc_world":     False,
    "quality":      False,
    "momentum":     False,
    "ex_usa":       False,
    "gold":         False,
    "btc":          False,
    "crypto_basket": False,
    "eur_gov_direct": True,      # EUR denominated natively
    "eur_gov_1_3y":   True,
    "eur_gov_7_10y":  True,
    "india":          False,
    "china":          False,
    "dbi":            False,
}


# ============================================================================
# Symbol map — which CSV file drives each sleeve's returns
# ============================================================================

SYMBOL_MAP: Dict[str, str] = {
    # equity sleeves → index proxy monthly
    "put_write":      "cboe_put_index_monthly",
    "nasdaq_top30":   "nasdaq100_tr_monthly",
    "hc_us_hedged":   "sp500_healthcare_tr_monthly",
    "hc_world":       "msci_world_healthcare_monthly",
    "quality":        "msci_world_quality_monthly",
    "momentum":       "msci_world_momentum_monthly",
    "ex_usa":         "msci_world_ex_usa_monthly",
    # other sleeves
    "gold":           "gold_lbma_monthly",
    "btc":            "btc_usd_daily",             # daily → resampled monthly
    "crypto_basket":  "crypto_basket_monthly",
    "eur_gov_direct": "iboxx_eurgov_1_3y_monthly",  # proxy
    "eur_gov_1_3y":   "iboxx_eurgov_1_3y_monthly",
    "eur_gov_7_10y":  "iboxx_eurgov_7_10y_monthly",
    "india":          "india_ftse_monthly",
    "china":          "china_ftse_monthly",
    "dbi":            "sg_cta_index_monthly",
    # pension
    "pension_bond":   "iboxx_eurgov_7_10y_monthly",  # proxy
    "pension_equity": "msci_europe_monthly",
}


# ============================================================================
# TER (annual expense ratios, decimal) — approximate for real ETFs
# ============================================================================

TER_ANNUAL: Dict[str, float] = {
    "put_write":     0.0029,
    "nasdaq_top30":  0.0020,
    "hc_us_hedged":  0.0020,
    "hc_world":      0.0025,
    "quality":       0.0035,
    "momentum":      0.0030,
    "ex_usa":        0.0015,
    "gold":          0.0012,
    "btc":           0.0098,
    "crypto_basket": 0.0250,
    "eur_gov_direct": 0.0,
    "eur_gov_1_3y":  0.0020,
    "eur_gov_7_10y": 0.0020,
    "india":         0.0019,
    "china":         0.0019,
    "dbi":           0.0085,
    "pension_bond":  0.0050,  # typical Italian pension fund TER
    "pension_equity": 0.0050,
}


# ============================================================================
# Options overlay parameters
# ============================================================================

@dataclass
class OptionsConfig:
    enabled: bool = True
    budget_nav_per_year: float = 0.003        # 0.30% NAV/year (hard cap)
    hedge_ratio_of_equity: float = 0.30       # 30% of equity sleeve notional
    spy_qqq_split: Tuple[float, float] = (0.70, 0.30)
    long_strike_pct: float = 0.87             # long put at ~13% OTM (spot * 0.87)
    short_strike_pct: float = 0.70            # short put at ~30% OTM
    tenor_months: int = 6
    take_profit_partial_multiple: float = 2.0
    take_profit_full_multiple: float = 3.0
    commission_per_contract: float = 1.0      # USD per contract
    iv_skew_adjustment: float = 1.15          # OTM put IV ≈ 1.15 × ATM VIX


OPTIONS = OptionsConfig()


# ============================================================================
# Cash management
# ============================================================================

@dataclass
class CashConfig:
    target: float = 0.11
    floor_hard: float = 0.08
    comfort: float = 0.10
    upper_band: float = 0.14


CASH = CashConfig()


# ============================================================================
# Rebalancing
# ============================================================================

@dataclass
class RebalanceConfig:
    months: Tuple[int, ...] = (1, 7)          # January, July
    transaction_cost_bps: float = 20.0        # 0.20% round-trip
    band_relative_pct: float = 0.20           # ±20% of target weight
    band_minimum_absolute: float = 0.005      # ±0.5% NAV minimum


REBALANCE = RebalanceConfig()


# ============================================================================
# Dynamic PAC on drawdowns (optional; used for monthly-flow simulation)
# ============================================================================

PAC_DRAWDOWN_MULTIPLIERS: Dict[Tuple[float, float], float] = {
    # (drawdown_lower, drawdown_upper): multiplier
    (-0.10, -0.05): 2.0,
    (-0.20, -0.15): 3.0,
    (-1.00, -0.40): 100.0,   # "opportunity mode"; capped by available flow
}


# ============================================================================
# Benchmarks — portfolios against which we compare the Four Umbrellas
# ============================================================================

BENCHMARKS: Dict[str, Dict[str, float]] = {
    "60/40": {
        "msci_world_tr_monthly":    0.60,
        "bloomberg_euro_agg_monthly": 0.40,
    },
    "100% S&P 500 TR EUR": {
        "sp500_tr_monthly": 1.00,
    },
    "100% SWDA (MSCI World EUR)": {
        "msci_world_tr_monthly": 1.00,
    },
    "All-Weather proxy": {
        "msci_world_tr_monthly":       0.30,
        "iboxx_eurgov_7_10y_monthly":  0.40,
        "iboxx_eurgov_1_3y_monthly":   0.15,
        "gold_lbma_monthly":           0.075,
        "sg_cta_index_monthly":        0.075,  # commodity-trend proxy
    },
    # Phase 3: classical allocator portfolios.
    # Missing-data policy (see simulate_benchmark `min_coverage`, default 0.80):
    #   - if ≥ 80% of the portfolio's weight has data available, the benchmark
    #     runs with renormalized weights (remaining sleeves scaled to sum=1.0)
    #   - if coverage falls below 80%, the benchmark is skipped entirely with
    #     a log message (no crash, no silently distorted weights)
    "Golden Butterfly (Tyler)": {
        # 20/20/20/20/20: US large + US SCV + LT Treas + ST Treas + Gold
        "sp500_tr_monthly":           0.20,
        "sp600_scv_tr_monthly":       0.20,   # S&P 600 Small-Cap Value
        "iboxx_eurgov_7_10y_monthly": 0.20,   # LT treasuries (EU proxy)
        "iboxx_eurgov_1_3y_monthly":  0.20,   # ST treasuries (EU proxy)
        "gold_lbma_monthly":          0.20,
    },
    "Harry Browne Permanent Portfolio": {
        # 25/25/25/25: Stocks + LT Treas + Cash + Gold
        "msci_world_tr_monthly":      0.25,
        "iboxx_eurgov_7_10y_monthly": 0.25,
        "iboxx_eurgov_1_3y_monthly":  0.25,   # cash proxy (short-duration gov)
        "gold_lbma_monthly":          0.25,
    },
    "Dalio All-Weather (official)": {
        # Dalio's published All-Weather weights: 30/40/15/7.5/7.5
        "msci_world_tr_monthly":      0.30,
        "iboxx_eurgov_7_10y_monthly": 0.40,   # long-term
        "iboxx_eurgov_1_3y_monthly":  0.15,   # intermediate-term proxy
        "gold_lbma_monthly":          0.075,
        "sg_cta_index_monthly":       0.075,  # commodities proxy
    },
    "Swensen Lazy": {
        # Yale CIO David Swensen "Lazy Portfolio": 30/15/5/20/15/15
        "sp500_tr_monthly":           0.30,   # domestic (US) equity
        "msci_world_ex_usa_monthly":  0.15,   # international developed
        "msci_emerging_tr_monthly":   0.05,   # emerging markets
        "ftse_nareit_tr_monthly":     0.20,   # REITs
        "iboxx_eurgov_7_10y_monthly": 0.15,   # long treasuries
        "us_tips_tr_monthly":         0.15,   # TIPS (inflation-protected)
    },
}


# ============================================================================
# Italian tax modeling (Phase 7) — simplified model for backtest CGT drag
# ============================================================================

@dataclass
class TaxConfig:
    """
    Italian capital-gains tax configuration.

    See src/tax.py for implementation details and caveats. This is a
    simplified model and does NOT replace professional tax advice.
    """
    enabled: bool = False                 # default OFF (preserves backward compat)
    capital_gains_rate: float = 0.26      # 26% on most financial instruments
    gov_bond_rate: float = 0.125          # 12.5% on whitelist gov bonds
    loss_carryforward_years: int = 4      # "zainetto fiscale" — 4 anni di carry-forward
    pension_exempt: bool = True           # pensione non tassata durante accumulo
    apply_to_rebalance_sales: bool = True # apply CGT to rebalance-triggered sales


TAX = TaxConfig()


# ============================================================================
# Reference NAV (pure cosmetic — the backtest is scale-invariant)
# ============================================================================

REFERENCE_NAV_EUR = 100_000.0
