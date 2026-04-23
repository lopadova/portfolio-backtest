# Data sourcing guide

This directory is **empty on purpose**. The raw time-series data used by the backtest is subject to index providers' licensing terms and is not redistributed here. You must source it yourself — all sources are free with registration.

## Required files

Place every CSV listed below in `data/raw/` with **exactly the filename shown**. The file format expected by the loader is documented in each row (typically two columns: `date, value` or `date, close`).

| Filename | What | Source | Licensing note | Needed for |
|---|---|---|---|---|
| `spy_daily.csv` | SPY daily close (USD) | Yahoo Finance via `fetch_data.py` | Personal use ok | Options overlay simulation + US equity benchmark |
| `qqq_daily.csv` | QQQ daily close (USD) | Yahoo Finance via `fetch_data.py` | Personal use ok | Options overlay (QQQ leg) |
| `vix_daily.csv` | VIX daily close | Yahoo Finance via `fetch_data.py` | Public CBOE data | Options overlay IV parameterization |
| `btc_usd_daily.csv` | Bitcoin USD daily | Yahoo Finance via `fetch_data.py` | Personal use ok | Crypto sleeve (from 2014) |
| `eurusd_daily.csv` | EUR/USD FX daily | FRED via `fetch_data.py` | Public (FRED) | FX conversion |
| `treasury_3m_daily.csv` | 3-month US Treasury yield | FRED via `fetch_data.py` | Public (FRED) | Risk-free rate for BS |
| `sp500_tr_monthly.csv` | S&P 500 Total Return (USD) monthly | `fetch_data.py` from Yahoo + computation, OR from [S&P Global](https://www.spglobal.com/spdji/en/indices/equity/sp-500/) (free registration) | S&P licensed — personal use ok, redistribution no | 100% S&P benchmark |
| `msci_world_tr_monthly.csv` | MSCI World Net TR (EUR) monthly | [MSCI index data](https://www.msci.com/end-of-day-data-search) (free registration) | MSCI licensed | 100% SWDA benchmark + 60/40 equity leg |
| `msci_world_ex_usa_monthly.csv` | MSCI World ex-USA Net TR (USD) monthly | MSCI | MSCI licensed | ex-USA sleeve |
| `msci_world_momentum_monthly.csv` | MSCI World Momentum Net TR (USD) monthly | MSCI | MSCI licensed | Momentum sleeve |
| `msci_world_quality_monthly.csv` | MSCI World Quality Net TR (USD) monthly | MSCI | MSCI licensed | Quality sleeve |
| `msci_world_healthcare_monthly.csv` | MSCI World Healthcare Net TR (USD) monthly | MSCI | MSCI licensed | World healthcare sleeve |
| `sp500_healthcare_tr_monthly.csv` | S&P 500 Healthcare Total Return (USD) monthly | [S&P Global](https://www.spglobal.com/spdji/en/indices/equity/sp-500-health-care-sector/) | S&P licensed | US healthcare sleeve |
| `nasdaq100_tr_monthly.csv` | Nasdaq 100 Total Return (USD) monthly | Nasdaq, or computed from `QQQ` via dividends | Personal use ok | Nasdaq sleeve proxy |
| `gold_lbma_monthly.csv` | LBMA Gold PM USD monthly | [LBMA Gold Prices](https://www.lbma.org.uk/prices-and-data/precious-metal-prices) | LBMA — free for personal use | Gold sleeve |
| `cboe_put_index_monthly.csv` | CBOE PUT Index (PutWrite TR) monthly | [CBOE Indices](https://www.cboe.com/us/indices/dashboard/PUT/) (free registration) | CBOE licensed | Defensive Put/Write sleeve |
| `sg_cta_index_monthly.csv` | SG CTA Index monthly | [SG Prime Services Indices](https://wholesale.banking.societegenerale.com/en/prime-services-indices/) (free registration) | SG licensed | DBi / Managed futures sleeve |
| `iboxx_eurgov_1_3y_monthly.csv` | iBoxx € Sovereigns 1–3y TR monthly | S&P Global / IHS Markit — or proxy via IEAA/IBGS ETF (Yahoo) | Licensed / personal use | Bond short sleeve |
| `iboxx_eurgov_7_10y_monthly.csv` | iBoxx € Sovereigns 7–10y TR monthly | Same | Same | Bond medium sleeve |
| `bloomberg_euro_agg_monthly.csv` | Bloomberg Euro Aggregate TR monthly | Bloomberg / iShares Euro Aggregate Bond proxy (Yahoo: `IEAG`) | Licensed / personal use | 60/40 bond leg |
| `msci_europe_monthly.csv` | MSCI Europe Net TR (EUR) monthly | MSCI | MSCI licensed | Pension equity-like proxy |
| `crypto_basket_monthly.csv` | Top-10 crypto market-cap index monthly (USD) | CoinMarketCap / CoinGecko — [CMC200](https://coinmarketcap.com/charts/cmc200/) or similar | Personal use (check current ToS) | Crypto basket (20% of crypto sleeve) |
| `india_ftse_monthly.csv` | FTSE India TR (USD) monthly | [FTSE Russell](https://www.ftserussell.com/index/home) | FTSE licensed | EM satellite India |
| `china_ftse_monthly.csv` | FTSE China TR (USD) monthly | [FTSE Russell](https://www.ftserussell.com/index/home) | FTSE licensed | EM satellite China |
| `sp600_scv_tr_monthly.csv` | S&P 600 Small-Cap Value TR (USD) | [S&P Global](https://www.spglobal.com/spdji/en/indices/equity/sp-smallcap-600-value/) | S&P licensed | **Golden Butterfly benchmark** — SCV leg (optional; benchmark skipped if missing) |
| `msci_emerging_tr_monthly.csv` | MSCI Emerging Markets Net TR (USD) | [MSCI](https://www.msci.com/end-of-day-data-search) | MSCI licensed | **Swensen Lazy benchmark** — EM slice (optional) |
| `ftse_nareit_tr_monthly.csv` | FTSE NAREIT All Equity REITs TR (USD) | [FTSE / NAREIT](https://www.reit.com/data-research) | Licensed | **Swensen Lazy benchmark** — REITs slice (optional) |
| `us_tips_tr_monthly.csv` | Bloomberg US TIPS TR (USD) | Bloomberg, or iShares TIPS ETF (TIP) via Yahoo as proxy | Licensed / personal | **Swensen Lazy benchmark** — TIPS slice (optional) |

## Format conventions

### Monthly CSV

Two columns, header row required:

```
date,value
2005-01-31,1234.56
2005-02-28,1298.34
...
```

- `date`: YYYY-MM-DD, typically last trading day of the month
- `value`: the index/price level, currency as specified in the table above

### Daily CSV

```
date,close
2005-01-03,118.04
2005-01-04,117.07
...
```

- `date`: YYYY-MM-DD trading days only
- `close`: end-of-day close price

### Example — downloading a MSCI index

1. Log in to [www.msci.com/end-of-day-data-search](https://www.msci.com/end-of-day-data-search) (free registration).
2. Search for "MSCI World Momentum".
3. Select: Index Level, Net Total Return, Currency = USD (or EUR if available).
4. Set date range: 2005-01-01 to today, monthly frequency.
5. Download CSV.
6. Rename and save to `data/raw/msci_world_momentum_monthly.csv`.
7. Ensure the header row contains `date,value` (you may need to manually adjust the MSCI export format — see `msci_converter.py` in the root for a helper).

## Running `fetch_data.py`

This script handles the public / freely-licensed sources automatically:

```bash
python fetch_data.py
```

It downloads:
- SPY, QQQ, VIX, BTC-USD, IEAA, IEAG, IBGS, SEGA (from Yahoo Finance)
- EUR/USD FX, 3-month Treasury yield (from FRED)
- Computes SP500 TR monthly from SPY + dividends
- Computes Nasdaq 100 TR monthly from QQQ + dividends

It **prints instructions** for MSCI, CBOE, SG, LBMA, S&P, FTSE — because those require manual download.

## Data validation

After populating `data/raw/`, run:

```bash
python -m src.data_loader --validate
```

This checks that every expected file exists, has the right date range, and no gaps > 5 trading days. If any issue is found, it prints an actionable error.

## Alternative: the "quick test" mini dataset

For development / learning purposes, we include a **synthetic mini dataset** covering 2023–2024 only, generated from random-walk price paths. This is **NOT real data** — it exists only to let you verify that the code runs end-to-end without needing to source the full 20-year dataset first.

```bash
python -m src.data_loader --generate-synthetic
python backtest.py --start 2023-01-01 --end 2024-12-31 --synthetic
```

Do **not** treat synthetic results as meaningful. They exist only for plumbing verification.

## Legal reminder

If you intend to publish or redistribute results from this backtest, be aware that:

- **MSCI, S&P, FTSE, CBOE, SG index data** are licensed. You may publish *analysis and charts derived from them* under fair-use / editorial principles, but you may **not** redistribute the raw time series themselves.
- **Yahoo Finance data** is governed by Yahoo's Terms of Service; systematic bulk redistribution is generally prohibited.
- **FRED data** is public domain (US Federal Reserve).
- **LBMA Gold prices** are usable for personal/research/commentary; check the [LBMA terms](https://www.lbma.org.uk/) for redistribution specifics.

When in doubt, cite the source and let users download the data themselves. That is the model this repository follows.
