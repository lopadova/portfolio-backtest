"""
fetch_data.py — auto-download what is legally/practically fetchable.

This script fetches:
  * Daily: SPY, QQQ, VIX, BTC-USD from Yahoo Finance (personal use)
  * Daily: EUR/USD (DEXUSEU) and 3M Treasury (DGS3MO) from FRED (public domain)
  * Monthly (computed): SP500 TR, Nasdaq 100 TR from SPY/QQQ + dividend data

It DOES NOT fetch MSCI, CBOE, SG, LBMA, S&P, FTSE data — those are
licensing-protected and must be downloaded manually per data/README.md.
For those, this script prints explicit instructions.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

try:
    import yfinance as yf
    import pandas_datareader.data as pdr
except ImportError:
    print("Missing dependencies. Run: pip install -r requirements.txt")
    sys.exit(1)


DATA_RAW = Path(__file__).parent / "data" / "raw"
DATA_RAW.mkdir(parents=True, exist_ok=True)

START_DATE = "2005-01-01"
END_DATE = None  # today


def save_daily_close(ticker: str, filename: str, start=START_DATE, end=END_DATE):
    """Download daily close via yfinance and save as date,close CSV."""
    print(f"  Fetching {ticker} from Yahoo Finance ...")
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=False)
    if df.empty:
        print(f"    WARNING: no data returned for {ticker}")
        return
    # yfinance returns MultiIndex sometimes; flatten
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    close = df["Adj Close"] if "Adj Close" in df.columns else df["Close"]
    out = pd.DataFrame({"date": close.index, "close": close.values})
    out.to_csv(DATA_RAW / f"{filename}.csv", index=False)
    print(f"    -> {filename}.csv ({len(out)} rows, {out['date'].iloc[0].date()} to {out['date'].iloc[-1].date()})")


def save_fred_series(series_id: str, filename: str, start=START_DATE):
    """Download a FRED series and save as date,close CSV."""
    print(f"  Fetching {series_id} from FRED ...")
    try:
        s = pdr.DataReader(series_id, "fred", start=start)
    except Exception as e:
        print(f"    ERROR fetching {series_id}: {e}")
        return
    df = s.reset_index()
    df.columns = ["date", "close"]
    df.dropna(inplace=True)
    df.to_csv(DATA_RAW / f"{filename}.csv", index=False)
    print(f"    -> {filename}.csv ({len(df)} rows)")


def compute_sp500_tr_monthly():
    """Compute SP500 total return monthly from SPY dividends + price."""
    spy_csv = DATA_RAW / "spy_daily.csv"
    if not spy_csv.exists():
        print("  Skipping SP500 TR computation — spy_daily.csv missing")
        return
    print("  Computing SP500 TR monthly from SPY Adj Close ...")
    df = pd.read_csv(spy_csv, parse_dates=["date"])
    df.set_index("date", inplace=True)
    monthly = df["close"].resample("M").last()
    out = pd.DataFrame({"date": monthly.index, "value": monthly.values})
    out.to_csv(DATA_RAW / "sp500_tr_monthly.csv", index=False)
    print(f"    -> sp500_tr_monthly.csv ({len(out)} rows)")


def compute_nasdaq100_tr_monthly():
    """Compute Nasdaq 100 TR monthly from QQQ Adj Close."""
    qqq_csv = DATA_RAW / "qqq_daily.csv"
    if not qqq_csv.exists():
        print("  Skipping Nasdaq 100 TR computation — qqq_daily.csv missing")
        return
    print("  Computing Nasdaq 100 TR monthly from QQQ Adj Close ...")
    df = pd.read_csv(qqq_csv, parse_dates=["date"])
    df.set_index("date", inplace=True)
    monthly = df["close"].resample("M").last()
    out = pd.DataFrame({"date": monthly.index, "value": monthly.values})
    out.to_csv(DATA_RAW / "nasdaq100_tr_monthly.csv", index=False)
    print(f"    -> nasdaq100_tr_monthly.csv ({len(out)} rows)")


def print_manual_instructions():
    print("""
============================================================
MANUAL DOWNLOAD REQUIRED for the following files:
============================================================

1. MSCI indices — register free at https://www.msci.com/end-of-day-data-search
   Required (monthly Net TR, USD unless noted, YYYY-MM-DD):
     * msci_world_tr_monthly.csv             (MSCI World Net TR, EUR)
     * msci_world_ex_usa_monthly.csv         (MSCI World ex-USA Net TR, USD)
     * msci_world_momentum_monthly.csv       (MSCI World Momentum Net TR, USD)
     * msci_world_quality_monthly.csv        (MSCI World Quality Net TR, USD)
     * msci_world_healthcare_monthly.csv     (MSCI World Healthcare Net TR, USD)
     * msci_europe_monthly.csv               (MSCI Europe Net TR, EUR)
   Format each CSV as two columns: date,value

2. CBOE PUT Index — https://www.cboe.com/us/indices/dashboard/PUT/
   Save as: cboe_put_index_monthly.csv
   Columns: date,value

3. SG CTA Index — https://wholesale.banking.societegenerale.com/en/prime-services-indices/
   Free registration; look for "SG CTA Index" monthly returns.
   Save as: sg_cta_index_monthly.csv
   Columns: date,value

4. S&P 500 Healthcare Total Return — https://www.spglobal.com/spdji/
   Save as: sp500_healthcare_tr_monthly.csv

5. LBMA Gold PM — https://www.lbma.org.uk/prices-and-data/precious-metal-prices
   Download historical monthly prices (USD).
   Save as: gold_lbma_monthly.csv

6. iBoxx Euro Sovereigns 1-3y and 7-10y total return — S&P Global / IHS Markit
   Alternative proxies via Yahoo: IBGS (1-3y), IEGA (7-10y)
   Save as: iboxx_eurgov_1_3y_monthly.csv, iboxx_eurgov_7_10y_monthly.csv

7. Bloomberg Euro Aggregate TR — or iShares Core € Aggregate Bond proxy (ticker IEAG).
   Save as: bloomberg_euro_agg_monthly.csv

8. FTSE India and FTSE China TR monthly — https://www.ftserussell.com/
   Save as: india_ftse_monthly.csv, china_ftse_monthly.csv

9. Crypto basket (optional, can leave empty if unavailable):
   CoinMarketCap CMC200 or similar top-10 index.
   Save as: crypto_basket_monthly.csv

After downloading, validate with:
    python -m src.data_loader --validate

""")


def main():
    print("=" * 60)
    print("Four Umbrellas backtest — automated data fetch")
    print("=" * 60)
    print(f"\nOutput directory: {DATA_RAW.resolve()}\n")

    # Daily from Yahoo
    save_daily_close("SPY",    "spy_daily")
    save_daily_close("QQQ",    "qqq_daily")
    save_daily_close("^VIX",   "vix_daily")
    save_daily_close("BTC-USD", "btc_usd_daily", start="2014-09-17")

    # Daily from FRED (public)
    save_fred_series("DEXUSEU", "eurusd_daily")   # USD per 1 EUR
    save_fred_series("DGS3MO",  "treasury_3m_daily")  # 3M Treasury yield (%)

    # Computed series
    compute_sp500_tr_monthly()
    compute_nasdaq100_tr_monthly()

    # Manual instructions for the rest
    print_manual_instructions()


if __name__ == "__main__":
    main()
