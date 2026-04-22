"""
Performance metrics — CAGR, Sharpe, Sortino, Max Drawdown, Calmar, Ulcer,
CVaR, plus underwater-period / time-to-recovery metrics.

All functions operate on a pd.Series of monthly returns (decimals, e.g. 0.01 = +1%)
indexed by end-of-month date.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd


MONTHS_PER_YEAR = 12


def cumulative_wealth(returns: pd.Series, start_nav: float = 1.0) -> pd.Series:
    """Cumulative wealth curve from monthly returns, starting at start_nav."""
    return start_nav * (1.0 + returns.fillna(0.0)).cumprod()


def cagr(returns: pd.Series) -> float:
    """Compound annual growth rate from a monthly return series."""
    wealth = cumulative_wealth(returns)
    if len(wealth) < 2:
        return float("nan")
    years = len(returns) / MONTHS_PER_YEAR
    return float(wealth.iloc[-1] ** (1.0 / years) - 1.0)


def annualized_vol(returns: pd.Series) -> float:
    return float(returns.std() * np.sqrt(MONTHS_PER_YEAR))


def sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    """Annualized Sharpe ratio. risk_free_rate is annual decimal (e.g., 0.03 for 3%)."""
    monthly_rf = (1.0 + risk_free_rate) ** (1.0 / MONTHS_PER_YEAR) - 1.0
    excess = returns - monthly_rf
    mu = excess.mean() * MONTHS_PER_YEAR
    sigma = excess.std() * np.sqrt(MONTHS_PER_YEAR)
    # Guard against floating-point noise: pandas .std() on a constant series
    # can return ~1e-18 instead of 0. Treat < 1e-12 as undefined.
    return float(mu / sigma) if sigma > 1e-12 else float("nan")


def sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    """Annualized Sortino ratio — uses downside deviation (below rf)."""
    monthly_rf = (1.0 + risk_free_rate) ** (1.0 / MONTHS_PER_YEAR) - 1.0
    excess = returns - monthly_rf
    downside = excess[excess < 0]
    mu = excess.mean() * MONTHS_PER_YEAR
    if len(downside) == 0:
        return float("nan")
    sigma_down = downside.std() * np.sqrt(MONTHS_PER_YEAR)
    return float(mu / sigma_down) if sigma_down > 1e-12 else float("nan")


def max_drawdown(returns: pd.Series) -> float:
    """Max drawdown from a monthly return series. Returned as a negative number."""
    wealth = cumulative_wealth(returns)
    running_max = wealth.cummax()
    drawdown = (wealth - running_max) / running_max
    return float(drawdown.min())


def calmar_ratio(returns: pd.Series) -> float:
    """CAGR / |Max Drawdown|."""
    mdd = max_drawdown(returns)
    c = cagr(returns)
    return float(c / abs(mdd)) if mdd != 0 else float("nan")


def ulcer_index(returns: pd.Series) -> float:
    """
    Ulcer Index — RMS of drawdowns. More nuanced than Max DD because it
    accounts for the duration and depth of underwater periods.
    """
    wealth = cumulative_wealth(returns)
    running_max = wealth.cummax()
    drawdown_pct = (wealth - running_max) / running_max * 100.0
    return float(np.sqrt((drawdown_pct ** 2).mean()))


def cvar(returns: pd.Series, alpha: float = 0.05) -> float:
    """
    Conditional Value at Risk (Expected Shortfall) at level alpha.
    Returned as a negative number (the expected loss in the tail).
    """
    if len(returns) == 0:
        return float("nan")
    threshold = returns.quantile(alpha)
    tail = returns[returns <= threshold]
    return float(tail.mean()) if len(tail) > 0 else float("nan")


def average_drawdown(returns: pd.Series) -> float:
    """
    Average drawdown — mean of the drawdown series, considering only the
    underwater values (drawdown < 0). Returned as a negative number.

    This gives a different perspective than Max Drawdown: a portfolio with
    one catastrophic -50% event and many small recoveries has a very negative
    Max DD but a relatively mild Average DD. Conversely, a portfolio that
    spends long periods 10% underwater has a modest Max DD but a larger
    (more negative) Average DD.
    """
    wealth = cumulative_wealth(returns)
    running_max = wealth.cummax()
    drawdown = (wealth - running_max) / running_max
    underwater = drawdown[drawdown < 0]
    if len(underwater) == 0:
        return 0.0
    return float(underwater.mean())


def max_drawdown_duration_months(returns: pd.Series) -> int:
    """
    Max Drawdown duration in months — duration (in months) of the worst
    drawdown *episode*, measured from the peak preceding the trough to the
    trough itself. This is specifically the duration of the single
    worst peak-to-trough event, not the total underwater time.

    Related to `longest_underwater_months`, which measures the longest
    consecutive stretch of time below any prior peak; this function focuses
    on the time-to-bottom of the worst single event.
    """
    wealth = cumulative_wealth(returns)
    if len(wealth) < 2:
        return 0
    running_max = wealth.cummax()
    drawdown = (wealth - running_max) / running_max
    trough_idx = drawdown.idxmin()
    # Find the peak that preceded the trough (last time running_max increased before trough)
    pre_trough = wealth.loc[:trough_idx]
    pre_trough_running_max = pre_trough.cummax()
    # The peak is the first index where running_max == max value seen by trough
    peak_value = pre_trough_running_max.iloc[-1]
    peak_candidates = pre_trough[pre_trough >= peak_value]
    if len(peak_candidates) == 0:
        return 0
    peak_idx = peak_candidates.index[0]
    months = (trough_idx.year - peak_idx.year) * 12 + (trough_idx.month - peak_idx.month)
    return int(months)


def ulcer_performance_index(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
    """
    Ulcer Performance Index (UPI) = (CAGR - risk_free_rate) / Ulcer Index.

    Similar to Sharpe, but uses Ulcer Index (drawdown depth × duration)
    as the risk measure instead of volatility. Popular in the drawdown-aware
    risk management community; more meaningful for investors who experience
    risk primarily as drawdowns rather than volatility.

    Reference: Martin & McCann (1989), "The Investor's Guide to Fidelity Funds".
    """
    c = cagr(returns)
    ui = ulcer_index(returns)
    # Ulcer Index is in percentage-points; convert to decimal for ratio
    ui_decimal = ui / 100.0
    if ui_decimal <= 1e-12:
        return float("nan")
    return float((c - risk_free_rate) / ui_decimal)


def longest_underwater_months(returns: pd.Series) -> int:
    """
    Longest underwater period — the longest stretch of months during which
    the portfolio remained below its prior all-time high.
    """
    wealth = cumulative_wealth(returns)
    running_max = wealth.cummax()
    underwater = wealth < running_max
    longest = 0
    current = 0
    for is_uw in underwater:
        if is_uw:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return int(longest)


def recovery_months_after_max_dd(returns: pd.Series) -> Optional[int]:
    """
    After the worst drawdown bottom, how many months until a new all-time
    high was reached? Returns None if recovery did not happen within the
    sample window.
    """
    wealth = cumulative_wealth(returns)
    running_max = wealth.cummax()
    drawdown = (wealth - running_max) / running_max
    trough_idx = drawdown.idxmin()
    pre_trough_max = running_max.loc[trough_idx]
    after_trough = wealth.loc[trough_idx:]
    recovered = after_trough[after_trough >= pre_trough_max]
    if len(recovered) == 0:
        return None
    recovery_date = recovered.index[0]
    # Count months from trough to recovery
    months = (recovery_date.year - trough_idx.year) * 12 + (recovery_date.month - trough_idx.month)
    return int(months)


@dataclass
class PortfolioStats:
    name: str
    cagr: float                          # a.k.a. Annual Return
    annualized_vol: float                # a.k.a. Volatility
    sharpe: float
    sortino: float
    max_drawdown: float
    average_drawdown: float              # NEW
    max_drawdown_duration_months: int    # NEW
    calmar: float
    ulcer_index: float
    upi: float                           # NEW — Ulcer Performance Index
    cvar_5pct: float
    longest_underwater_months: int
    recovery_months: Optional[int]
    best_month: float
    worst_month: float
    pct_positive_months: float
    total_return: float

    def to_dict(self) -> Dict:
        d = self.__dict__.copy()
        if d["recovery_months"] is None:
            d["recovery_months"] = "not recovered"
        return d


def compute_all(name: str, returns: pd.Series, risk_free_rate: float = 0.02) -> PortfolioStats:
    """Compute the full statistics set for a return series."""
    returns = returns.dropna()
    wealth = cumulative_wealth(returns)
    total_return = float(wealth.iloc[-1] - 1.0) if len(wealth) > 0 else float("nan")
    pct_pos = float((returns > 0).mean()) if len(returns) > 0 else float("nan")
    return PortfolioStats(
        name=name,
        cagr=cagr(returns),
        annualized_vol=annualized_vol(returns),
        sharpe=sharpe_ratio(returns, risk_free_rate),
        sortino=sortino_ratio(returns, risk_free_rate),
        max_drawdown=max_drawdown(returns),
        average_drawdown=average_drawdown(returns),
        max_drawdown_duration_months=max_drawdown_duration_months(returns),
        calmar=calmar_ratio(returns),
        ulcer_index=ulcer_index(returns),
        upi=ulcer_performance_index(returns, risk_free_rate),
        cvar_5pct=cvar(returns, 0.05),
        longest_underwater_months=longest_underwater_months(returns),
        recovery_months=recovery_months_after_max_dd(returns),
        best_month=float(returns.max()),
        worst_month=float(returns.min()),
        pct_positive_months=pct_pos,
        total_return=total_return,
    )


def crisis_drawdown(returns: pd.Series, start: str, end: str) -> float:
    """
    Peak-to-trough drawdown during a specified period (e.g. 2008 crisis).
    """
    sliced = returns.loc[start:end].dropna()
    if len(sliced) == 0:
        return float("nan")
    wealth = cumulative_wealth(sliced)
    running_max = wealth.cummax()
    return float(((wealth - running_max) / running_max).min())


def stats_to_dataframe(stats_list) -> pd.DataFrame:
    """Convert a list of PortfolioStats to a pandas DataFrame for display."""
    return pd.DataFrame([s.to_dict() for s in stats_list]).set_index("name")
