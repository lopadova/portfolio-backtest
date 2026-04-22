"""
Monte Carlo simulation — block bootstrap of historical monthly returns.

Block bootstrap preserves the short-term autocorrelation structure of financial
returns (momentum, volatility clustering) much better than IID resampling.
Default block size is 3 months, a common choice in the financial literature
(Politis & White, 2004 "Automatic block-length selection for the dependent bootstrap").
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd


def block_bootstrap(
    returns: pd.Series,
    n_paths: int,
    n_periods: int,
    block_size: int = 3,
    seed: int = 42,
) -> np.ndarray:
    """
    Generate n_paths simulated return sequences of length n_periods, each by
    sampling contiguous blocks of length `block_size` from the historical
    returns with replacement.

    Returns: np.ndarray of shape (n_paths, n_periods) containing simulated monthly returns.
    """
    rng = np.random.default_rng(seed)
    historical = returns.dropna().values
    n_hist = len(historical)
    if n_hist < block_size:
        raise ValueError(f"Historical returns too short ({n_hist} months) for block size {block_size}")

    n_blocks_per_path = int(np.ceil(n_periods / block_size))
    max_start = n_hist - block_size + 1

    paths = np.empty((n_paths, n_periods))
    for i in range(n_paths):
        block_starts = rng.integers(0, max_start, size=n_blocks_per_path)
        path = np.concatenate([historical[s:s + block_size] for s in block_starts])
        paths[i] = path[:n_periods]
    return paths


def simulate_wealth_paths(simulated_returns: np.ndarray) -> np.ndarray:
    """
    Convert a (n_paths, n_periods) array of monthly returns into a (n_paths, n_periods+1)
    array of cumulative wealth starting at 1.0.
    """
    n_paths, n_periods = simulated_returns.shape
    wealth = np.empty((n_paths, n_periods + 1))
    wealth[:, 0] = 1.0
    wealth[:, 1:] = np.cumprod(1.0 + simulated_returns, axis=1)
    return wealth


@dataclass
class MonteCarloStats:
    n_paths: int
    n_years: float
    median_terminal: float
    pct5_terminal: float
    pct25_terminal: float
    pct75_terminal: float
    pct95_terminal: float
    prob_positive_real_return: float
    prob_drawdown_gt_20pct: float
    prob_drawdown_gt_40pct: float
    median_max_drawdown: float
    pct5_max_drawdown: float   # "worst 5%"


def compute_mc_stats(wealth_paths: np.ndarray, n_months: int) -> MonteCarloStats:
    """
    Compute summary statistics from MC wealth paths.
    wealth_paths: (n_paths, n_months+1) starting at 1.0
    """
    n_paths = wealth_paths.shape[0]
    terminal = wealth_paths[:, -1]

    # Max drawdown per path
    max_dd = np.zeros(n_paths)
    for i in range(n_paths):
        running_max = np.maximum.accumulate(wealth_paths[i])
        dd = (wealth_paths[i] - running_max) / running_max
        max_dd[i] = dd.min()

    return MonteCarloStats(
        n_paths=n_paths,
        n_years=n_months / 12.0,
        median_terminal=float(np.median(terminal)),
        pct5_terminal=float(np.percentile(terminal, 5)),
        pct25_terminal=float(np.percentile(terminal, 25)),
        pct75_terminal=float(np.percentile(terminal, 75)),
        pct95_terminal=float(np.percentile(terminal, 95)),
        prob_positive_real_return=float((terminal > 1.0).mean()),
        prob_drawdown_gt_20pct=float((max_dd < -0.20).mean()),
        prob_drawdown_gt_40pct=float((max_dd < -0.40).mean()),
        median_max_drawdown=float(np.median(max_dd)),
        pct5_max_drawdown=float(np.percentile(max_dd, 5)),  # the worst 5%
    )
