"""
Monte Carlo simulation — block bootstrap of historical monthly returns.

Block bootstrap preserves the short-term autocorrelation structure of financial
returns (momentum, volatility clustering) much better than IID resampling.
Default block size is 3 months, a common choice in the financial literature
(Politis & White, 2004 "Automatic block-length selection for the dependent bootstrap").
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

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
class MonteCarloScenarios:
    """
    User-facing Monte Carlo scenarios (10th / 50th / 90th percentiles).

    Framing:
      - Prudente (10°): "pessimistic but not catastrophic" — only 10% of
        simulated paths end worse than this
      - Mediana (50°): central expectation
      - Ottimista (90°): "favorable but not exceptional" — only 10% end better
    """
    n_years: float
    # Wealth multipliers (multiply by starting NAV to get €)
    prudente_wealth: float
    mediana_wealth: float
    ottimista_wealth: float
    # Implied annualized CAGR over the horizon (from wealth multiplier)
    prudente_cagr: float
    mediana_cagr: float
    ottimista_cagr: float


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
    pct5_max_drawdown: float
    # Phase 2 scenarios — embedded for convenience
    scenarios: Optional[MonteCarloScenarios] = None

    def to_dict(self) -> dict:
        """
        Return a flat, CSV-safe representation of the Monte Carlo stats.

        Nested scenario data is flattened into scalar fields so callers that
        serialize via pandas (e.g. `backtest.py` writes MC stats to CSV) do
        not end up with the `MonteCarloScenarios` object serialized as an
        opaque string.
        """
        data = {
            "n_paths": self.n_paths,
            "n_years": self.n_years,
            "median_terminal": self.median_terminal,
            "pct5_terminal": self.pct5_terminal,
            "pct25_terminal": self.pct25_terminal,
            "pct75_terminal": self.pct75_terminal,
            "pct95_terminal": self.pct95_terminal,
            "prob_positive_real_return": self.prob_positive_real_return,
            "prob_drawdown_gt_20pct": self.prob_drawdown_gt_20pct,
            "prob_drawdown_gt_40pct": self.prob_drawdown_gt_40pct,
            "median_max_drawdown": self.median_max_drawdown,
            "pct5_max_drawdown": self.pct5_max_drawdown,
        }
        if self.scenarios is None:
            data.update({
                "scenario_n_years": None,
                "scenario_prudente_wealth": None,
                "scenario_mediana_wealth": None,
                "scenario_ottimista_wealth": None,
                "scenario_prudente_cagr": None,
                "scenario_mediana_cagr": None,
                "scenario_ottimista_cagr": None,
            })
        else:
            data.update({
                "scenario_n_years": self.scenarios.n_years,
                "scenario_prudente_wealth": self.scenarios.prudente_wealth,
                "scenario_mediana_wealth": self.scenarios.mediana_wealth,
                "scenario_ottimista_wealth": self.scenarios.ottimista_wealth,
                "scenario_prudente_cagr": self.scenarios.prudente_cagr,
                "scenario_mediana_cagr": self.scenarios.mediana_cagr,
                "scenario_ottimista_cagr": self.scenarios.ottimista_cagr,
            })
        return data


def _wealth_to_cagr(wealth_multiplier: float, n_years: float) -> float:
    """Convert a terminal wealth multiplier to an implied annualized CAGR."""
    if wealth_multiplier <= 0 or n_years <= 0:
        return float("nan")
    return wealth_multiplier ** (1.0 / n_years) - 1.0


def compute_scenarios(wealth_paths: np.ndarray, n_months: int) -> MonteCarloScenarios:
    """
    Compute the 3 user-facing scenarios (Prudente 10° / Mediana 50° / Ottimista 90°)
    from Monte Carlo terminal wealth distribution.
    """
    terminal = wealth_paths[:, -1]
    n_years = n_months / 12.0
    prudente = float(np.percentile(terminal, 10))
    mediana = float(np.percentile(terminal, 50))
    ottimista = float(np.percentile(terminal, 90))
    return MonteCarloScenarios(
        n_years=n_years,
        prudente_wealth=prudente,
        mediana_wealth=mediana,
        ottimista_wealth=ottimista,
        prudente_cagr=_wealth_to_cagr(prudente, n_years),
        mediana_cagr=_wealth_to_cagr(mediana, n_years),
        ottimista_cagr=_wealth_to_cagr(ottimista, n_years),
    )


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
        pct5_max_drawdown=float(np.percentile(max_dd, 5)),
        scenarios=compute_scenarios(wealth_paths, n_months),
    )
