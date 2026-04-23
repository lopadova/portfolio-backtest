"""
Data loader — reads CSVs from data/raw/, aligns dates, handles currencies.

Expected CSV format:
    monthly: date,value          (date = YYYY-MM-DD end-of-month)
    daily:   date,close          (date = YYYY-MM-DD trading day)

All monthly series are aligned to end-of-month dates. Daily series follow
the SPY trading calendar.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from .portfolio import SYMBOL_MAP, HEDGED

DATA_RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


@dataclass
class DataBundle:
    """Container for all aligned time-series used by the backtest."""
    monthly_returns_usd: pd.DataFrame   # columns = sleeve keys; EUR where natively EUR
    monthly_returns_eur: pd.DataFrame   # all in EUR, after FX conversion
    spy_daily: pd.Series                # for options overlay
    qqq_daily: pd.Series                # for options overlay
    vix_daily: pd.Series                # for options overlay
    eurusd_daily: pd.Series             # EUR per 1 USD
    rf_daily: pd.Series                 # 3M Treasury daily, as annualized yield
    benchmark_monthly_eur: pd.DataFrame  # benchmark series for comparison
    btc_activation_date: pd.Timestamp   # date BTC enters the portfolio

    def slice(self, start: pd.Timestamp, end: pd.Timestamp) -> "DataBundle":
        """Return a date-sliced copy."""
        return DataBundle(
            monthly_returns_usd=self.monthly_returns_usd.loc[start:end],
            monthly_returns_eur=self.monthly_returns_eur.loc[start:end],
            spy_daily=self.spy_daily.loc[start:end],
            qqq_daily=self.qqq_daily.loc[start:end],
            vix_daily=self.vix_daily.loc[start:end],
            eurusd_daily=self.eurusd_daily.loc[start:end],
            rf_daily=self.rf_daily.loc[start:end],
            benchmark_monthly_eur=self.benchmark_monthly_eur.loc[start:end],
            btc_activation_date=self.btc_activation_date,
        )


def _read_monthly(filename: str) -> pd.Series:
    """Read a monthly CSV and return a pd.Series of levels indexed by date."""
    path = DATA_RAW_DIR / f"{filename}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing data file: {path}\n"
            f"See data/README.md for download instructions."
        )
    df = pd.read_csv(path, parse_dates=["date"])
    # Accept either `value` or `close` column
    value_col = "value" if "value" in df.columns else "close"
    s = df.set_index("date")[value_col].sort_index()
    # Ensure end-of-month alignment (period "M" is still a valid PERIOD alias,
    # but to_timestamp needs "ME" on pandas 2.2+; using how="end" is version-safe)
    s.index = s.index.to_period("M").to_timestamp(how="end").normalize()
    return s


def _read_daily(filename: str) -> pd.Series:
    """Read a daily CSV and return a pd.Series of levels."""
    path = DATA_RAW_DIR / f"{filename}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing data file: {path}\n"
            f"See data/README.md for download instructions."
        )
    df = pd.read_csv(path, parse_dates=["date"])
    value_col = "close" if "close" in df.columns else "value"
    return df.set_index("date")[value_col].sort_index()


def _levels_to_returns(levels: pd.Series) -> pd.Series:
    """Period-over-period simple returns; first period is NaN."""
    return levels.pct_change()


def _convert_usd_return_to_eur(
    usd_returns: pd.Series,
    eurusd_end_of_period: pd.Series,
) -> pd.Series:
    """
    Convert USD-denominated returns to EUR-denominated returns.

    eurusd_end_of_period: EUR per 1 USD, aligned to the return's period-end date.

    If USD asset returned r_usd and EUR/USD moved from fx_0 to fx_1:
        r_eur = (1 + r_usd) * (fx_1 / fx_0) - 1
    """
    fx_aligned = eurusd_end_of_period.reindex(usd_returns.index).ffill()
    fx_change = fx_aligned / fx_aligned.shift(1)
    return (1.0 + usd_returns) * fx_change - 1.0


def load_data(synthetic: bool = False) -> DataBundle:
    """
    Main entry point. Loads every required CSV, aligns dates, converts currencies.
    """
    if synthetic:
        return _generate_synthetic_bundle()

    # --- FX ---
    eurusd_daily = _read_daily("eurusd_daily")
    # Some FRED series express USD/EUR; if values < 0.5 we assume they are EUR/USD already.
    # EUR/USD (how many USD per 1 EUR) from FRED DEXUSEU is ~1.05-1.50.
    # We normalize to "EUR per 1 USD" (reciprocal) for the formulas in this module.
    if eurusd_daily.median() > 1.0:
        # Data is USD per 1 EUR; invert
        eurusd_daily = 1.0 / eurusd_daily
    eurusd_monthly = eurusd_daily.resample("ME").last()

    # --- Daily series for options overlay ---
    spy_daily = _read_daily("spy_daily")
    qqq_daily = _read_daily("qqq_daily")
    vix_daily = _read_daily("vix_daily")
    rf_daily = _read_daily("treasury_3m_daily")
    # FRED 3M T-bill is quoted in %; convert to decimal
    if rf_daily.median() > 1.0:
        rf_daily = rf_daily / 100.0

    # --- Monthly levels per sleeve ---
    sleeve_levels: Dict[str, pd.Series] = {}

    for sleeve, filename in SYMBOL_MAP.items():
        if sleeve == "btc":
            # BTC is daily; resample to monthly end
            btc_daily = _read_daily(filename)
            sleeve_levels[sleeve] = btc_daily.resample("ME").last()
        else:
            sleeve_levels[sleeve] = _read_monthly(filename)

    # --- Compute returns per sleeve (local currency) ---
    sleeve_returns_local = pd.DataFrame(
        {k: _levels_to_returns(v) for k, v in sleeve_levels.items()}
    )

    # --- Convert to EUR where unhedged ---
    sleeve_returns_eur = sleeve_returns_local.copy()
    for sleeve, is_hedged in HEDGED.items():
        if sleeve not in sleeve_returns_local.columns:
            continue
        if not is_hedged:
            sleeve_returns_eur[sleeve] = _convert_usd_return_to_eur(
                sleeve_returns_local[sleeve], eurusd_monthly
            )

    # --- Benchmark series ---
    benchmark_monthly_eur = pd.DataFrame(index=sleeve_returns_eur.index)

    # Load benchmark components
    bench_files = {
        "msci_world_tr_monthly":      ("unhedged", "msci_world_tr_monthly"),
        "sp500_tr_monthly":           ("unhedged", "sp500_tr_monthly"),
        "bloomberg_euro_agg_monthly": ("eur",      "bloomberg_euro_agg_monthly"),
        "iboxx_eurgov_7_10y_monthly": ("eur",      "iboxx_eurgov_7_10y_monthly"),
        "iboxx_eurgov_1_3y_monthly":  ("eur",      "iboxx_eurgov_1_3y_monthly"),
        "gold_lbma_monthly":          ("unhedged", "gold_lbma_monthly"),
        "sg_cta_index_monthly":       ("unhedged", "sg_cta_index_monthly"),
        # Phase 3: classical allocator benchmarks (loaded if present, skipped otherwise)
        "msci_world_ex_usa_monthly":  ("unhedged", "msci_world_ex_usa_monthly"),
        "msci_emerging_tr_monthly":   ("unhedged", "msci_emerging_tr_monthly"),
        "sp600_scv_tr_monthly":       ("unhedged", "sp600_scv_tr_monthly"),
        "ftse_nareit_tr_monthly":     ("unhedged", "ftse_nareit_tr_monthly"),
        "us_tips_tr_monthly":         ("unhedged", "us_tips_tr_monthly"),
    }
    for key, (currency, filename) in bench_files.items():
        try:
            levels = _read_monthly(filename)
            returns_local = _levels_to_returns(levels)
            if currency == "unhedged":
                benchmark_monthly_eur[key] = _convert_usd_return_to_eur(
                    returns_local, eurusd_monthly
                )
            else:
                benchmark_monthly_eur[key] = returns_local
        except FileNotFoundError:
            print(f"Warning: benchmark data missing ({filename}); benchmark portfolios using this file will be skipped.")

    # --- Determine BTC activation date ---
    btc_valid = sleeve_returns_eur["btc"].dropna()
    btc_activation = btc_valid.index.min() if len(btc_valid) > 0 else pd.Timestamp("2099-01-01")

    return DataBundle(
        monthly_returns_usd=sleeve_returns_local,
        monthly_returns_eur=sleeve_returns_eur,
        spy_daily=spy_daily,
        qqq_daily=qqq_daily,
        vix_daily=vix_daily,
        eurusd_daily=eurusd_daily,
        rf_daily=rf_daily,
        benchmark_monthly_eur=benchmark_monthly_eur,
        btc_activation_date=btc_activation,
    )


# ============================================================================
# Synthetic data generator (for plumbing verification only)
# ============================================================================

def _generate_synthetic_bundle() -> DataBundle:
    """
    Generate a synthetic 20-year dataset (2005-2024) so users can verify
    the pipeline end-to-end with output that is structurally representative
    of a real run (rolling windows populated, crisis zooms non-empty).

    NOT FOR REAL ANALYSIS — the underlying returns are pure random-walks
    calibrated to rough historical means/vols but with no real correlation
    structure, no regime dependence, and no actual crisis drawdowns.
    """
    np.random.seed(42)
    dates_monthly = pd.date_range("2005-01-31", "2024-12-31", freq="ME")
    dates_daily = pd.date_range("2005-01-03", "2024-12-31", freq="B")

    # Generate random-walk returns for each sleeve, plausible but fake
    params = {
        "put_write":      (0.005, 0.02),
        "nasdaq_top30":   (0.010, 0.05),
        "hc_us_hedged":   (0.007, 0.03),
        "hc_world":       (0.007, 0.03),
        "quality":        (0.008, 0.035),
        "momentum":       (0.009, 0.04),
        "ex_usa":         (0.006, 0.035),
        "gold":           (0.004, 0.03),
        "btc":            (0.015, 0.18),
        "crypto_basket":  (0.012, 0.15),
        "eur_gov_direct": (0.002, 0.005),
        "eur_gov_1_3y":   (0.002, 0.006),
        "eur_gov_7_10y":  (0.003, 0.015),
        "india":          (0.008, 0.045),
        "china":          (0.005, 0.055),
        "dbi":            (0.005, 0.025),
        "pension_bond":   (0.003, 0.010),
        "pension_equity": (0.006, 0.035),
    }
    returns = pd.DataFrame(index=dates_monthly)
    for k, (mu, sigma) in params.items():
        returns[k] = np.random.normal(mu, sigma, size=len(dates_monthly))

    benchmark = pd.DataFrame(index=dates_monthly)
    bench_params = {
        "msci_world_tr_monthly":      (0.008, 0.04),
        "sp500_tr_monthly":           (0.009, 0.045),
        "bloomberg_euro_agg_monthly": (0.002, 0.012),
        "iboxx_eurgov_7_10y_monthly": (0.003, 0.015),
        "iboxx_eurgov_1_3y_monthly":  (0.002, 0.006),
        "gold_lbma_monthly":          (0.004, 0.03),
        "sg_cta_index_monthly":       (0.005, 0.025),
        # Phase 3 additions for synthetic mode
        "msci_world_ex_usa_monthly":  (0.006, 0.035),
        "msci_emerging_tr_monthly":   (0.008, 0.055),
        "sp600_scv_tr_monthly":       (0.010, 0.055),
        "ftse_nareit_tr_monthly":     (0.007, 0.045),
        "us_tips_tr_monthly":         (0.003, 0.013),
    }
    for k, (mu, sigma) in bench_params.items():
        benchmark[k] = np.random.normal(mu, sigma, size=len(dates_monthly))

    spy_daily = pd.Series(100.0 * np.cumprod(1 + np.random.normal(0.0005, 0.012, len(dates_daily))), index=dates_daily)
    qqq_daily = pd.Series(200.0 * np.cumprod(1 + np.random.normal(0.0006, 0.015, len(dates_daily))), index=dates_daily)
    vix_daily = pd.Series(15.0 + 5.0 * np.abs(np.random.randn(len(dates_daily))), index=dates_daily)
    eurusd_daily = pd.Series(0.92 + 0.05 * np.random.randn(len(dates_daily)).cumsum() / 50.0, index=dates_daily)
    rf_daily = pd.Series(0.045 * np.ones(len(dates_daily)), index=dates_daily)

    return DataBundle(
        monthly_returns_usd=returns,
        monthly_returns_eur=returns,  # identical in synthetic (simplification)
        spy_daily=spy_daily,
        qqq_daily=qqq_daily,
        vix_daily=vix_daily,
        eurusd_daily=eurusd_daily,
        rf_daily=rf_daily,
        benchmark_monthly_eur=benchmark,
        btc_activation_date=dates_monthly[0],
    )


def validate() -> bool:
    """CLI validation utility."""
    missing = []
    for sleeve, filename in SYMBOL_MAP.items():
        path = DATA_RAW_DIR / f"{filename}.csv"
        if not path.exists():
            missing.append(f"  missing: {filename}.csv (for sleeve: {sleeve})")
    for fname in ["spy_daily", "qqq_daily", "vix_daily", "eurusd_daily", "treasury_3m_daily"]:
        path = DATA_RAW_DIR / f"{fname}.csv"
        if not path.exists():
            missing.append(f"  missing: {fname}.csv")

    if missing:
        print("Data validation FAILED:")
        for m in missing:
            print(m)
        print("\nSee data/README.md for download instructions.")
        return False

    print("Data validation OK — all expected files present.")
    return True


if __name__ == "__main__":
    if "--validate" in sys.argv:
        sys.exit(0 if validate() else 1)
    elif "--generate-synthetic" in sys.argv:
        bundle = _generate_synthetic_bundle()
        print("Synthetic bundle generated (in-memory only).")
        print("Monthly returns preview:")
        print(bundle.monthly_returns_eur.head())
