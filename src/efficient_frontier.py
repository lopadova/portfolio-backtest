"""
Efficient Frontier — Markowitz mean-variance optimization.

Two complementary visualizations:
1. Random sampling (Dirichlet) — 50k random portfolios to show the full
   feasible region of risk/return
2. Markowitz optimization — for each target volatility, find the portfolio
   with maximum return → traces the frontier as a smooth line

Three key portfolios highlighted:
- Max Sharpe (tangency portfolio)
- Min Volatility
- Max Return (100% of the highest-mean asset)

Plus the "Four Umbrellas" target position for reference.

References:
    Markowitz H. (1952) "Portfolio Selection", Journal of Finance 7(1):77-91.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from scipy.optimize import minimize

from .data_loader import DataBundle


@dataclass
class FrontierPoint:
    weights: Dict[str, float]
    expected_return: float  # annualized
    volatility: float       # annualized
    sharpe: float
    label: str = ""         # e.g. "Max Sharpe", "Min Vol"


def compute_annual_moments(returns_df: pd.DataFrame) -> Tuple[pd.Series, pd.DataFrame]:
    """
    Compute annualized mean returns and covariance matrix from monthly returns.
    """
    monthly_mean = returns_df.mean()
    monthly_cov = returns_df.cov()
    annual_mean = monthly_mean * 12
    annual_cov = monthly_cov * 12
    return annual_mean, annual_cov


def _portfolio_stats(weights: np.ndarray, annual_mean: np.ndarray, annual_cov: np.ndarray, rf: float = 0.02):
    """Return (portfolio_return, portfolio_vol, sharpe)."""
    port_return = float(weights @ annual_mean)
    port_vol = float(np.sqrt(weights @ annual_cov @ weights))
    sharpe = (port_return - rf) / port_vol if port_vol > 1e-12 else float("nan")
    return port_return, port_vol, sharpe


def sample_random_portfolios(
    n_assets: int,
    n_samples: int = 50_000,
    seed: int = 42,
) -> np.ndarray:
    """
    Sample n_samples random portfolios (rows sum to 1.0, all weights >= 0).
    Uses Dirichlet distribution with alpha=1 → uniform over the simplex.
    """
    rng = np.random.default_rng(seed)
    return rng.dirichlet(alpha=np.ones(n_assets), size=n_samples)


def evaluate_portfolios(
    weights_matrix: np.ndarray,
    annual_mean: np.ndarray,
    annual_cov: np.ndarray,
    rf: float = 0.02,
) -> pd.DataFrame:
    """
    Given a (n_samples, n_assets) weights matrix, compute return/vol/sharpe for each.
    """
    returns = weights_matrix @ annual_mean
    # Vectorized vol computation: diag(W @ Cov @ W.T)
    vols = np.sqrt(np.einsum("ij,jk,ik->i", weights_matrix, annual_cov, weights_matrix))
    sharpes = np.where(vols > 1e-12, (returns - rf) / vols, np.nan)
    return pd.DataFrame({
        "expected_return": returns,
        "volatility": vols,
        "sharpe": sharpes,
    })


def find_key_portfolios(
    weights_matrix: np.ndarray,
    eval_df: pd.DataFrame,
    asset_names: List[str],
) -> Dict[str, FrontierPoint]:
    """
    Identify Max Sharpe, Min Volatility, Max Return from a set of random portfolios.
    """
    idx_max_sharpe = eval_df["sharpe"].idxmax()
    idx_min_vol = eval_df["volatility"].idxmin()
    idx_max_return = eval_df["expected_return"].idxmax()

    def _to_point(idx: int, label: str) -> FrontierPoint:
        w = weights_matrix[idx]
        return FrontierPoint(
            weights=dict(zip(asset_names, w)),
            expected_return=float(eval_df.loc[idx, "expected_return"]),
            volatility=float(eval_df.loc[idx, "volatility"]),
            sharpe=float(eval_df.loc[idx, "sharpe"]),
            label=label,
        )

    return {
        "max_sharpe": _to_point(idx_max_sharpe, "Max Sharpe"),
        "min_vol": _to_point(idx_min_vol, "Min Volatility"),
        "max_return": _to_point(idx_max_return, "Max Return"),
    }


def markowitz_frontier(
    annual_mean: np.ndarray,
    annual_cov: np.ndarray,
    n_points: int = 50,
    rf: float = 0.02,
) -> pd.DataFrame:
    """
    Trace the efficient frontier analytically: for each target vol level,
    find the portfolio with maximum expected return subject to:
        - sum(weights) = 1
        - weights >= 0 (long-only)
        - portfolio_vol = target_vol
    """
    n_assets = len(annual_mean)
    min_ret = annual_mean.min()
    max_ret = annual_mean.max()
    target_returns = np.linspace(min_ret, max_ret, n_points)

    frontier_points = []
    for target_return in target_returns:
        # Minimize variance subject to achieving target_return
        constraints = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
            {"type": "eq", "fun": lambda w, tr=target_return: w @ annual_mean - tr},
        ]
        bounds = [(0.0, 1.0)] * n_assets
        x0 = np.ones(n_assets) / n_assets

        def objective(w):
            return w @ annual_cov @ w

        result = minimize(
            objective, x0, method="SLSQP",
            bounds=bounds, constraints=constraints,
            options={"ftol": 1e-9, "maxiter": 200},
        )
        if result.success:
            w = result.x
            port_ret, port_vol, sharpe = _portfolio_stats(w, annual_mean, annual_cov, rf)
            frontier_points.append({
                "expected_return": port_ret,
                "volatility": port_vol,
                "sharpe": sharpe,
            })

    return pd.DataFrame(frontier_points)


def plot_efficient_frontier(
    random_eval_df: pd.DataFrame,
    key_portfolios: Dict[str, FrontierPoint],
    frontier_df: pd.DataFrame,
    four_umbrellas_point: Optional[FrontierPoint],
    path: Path,
):
    """
    Produce the efficient-frontier scatter chart.
    """
    fig, ax = plt.subplots(figsize=(13, 8))

    # Cloud of random portfolios (colored by Sharpe)
    sc = ax.scatter(
        random_eval_df["volatility"] * 100,
        random_eval_df["expected_return"] * 100,
        c=random_eval_df["sharpe"],
        cmap="viridis",
        s=4, alpha=0.35, edgecolors="none",
    )
    cbar = plt.colorbar(sc, ax=ax, fraction=0.035, pad=0.01)
    cbar.set_label("Sharpe ratio", rotation=270, labelpad=15)

    # Markowitz efficient frontier line
    if len(frontier_df) > 0:
        # Filter: only the upper envelope (at each vol, highest return)
        sorted_df = frontier_df.sort_values("volatility")
        ax.plot(
            sorted_df["volatility"] * 100,
            sorted_df["expected_return"] * 100,
            color="#d62728", linewidth=2.2, alpha=0.85,
            label="Efficient Frontier (Markowitz)",
        )

    # Key portfolio markers (matplotlib-native marker codes)
    marker_configs = {
        "max_sharpe": ("*", "#ff7f0e", 400, "Max Sharpe"),
        "min_vol":    ("D", "#2ca02c", 180, "Min Volatility"),
        "max_return": ("^", "#9467bd", 220, "Max Return"),
    }
    for key, point in key_portfolios.items():
        symbol, color, size, label = marker_configs[key]
        ax.scatter(
            point.volatility * 100, point.expected_return * 100,
            marker=symbol, s=size, color=color,
            edgecolors="black", linewidth=1.2, zorder=5,
            label=f"{label} ({point.sharpe:.2f} Sharpe)",
        )
        ax.annotate(
            label, (point.volatility * 100, point.expected_return * 100),
            xytext=(10, 10), textcoords="offset points",
            fontweight="bold", fontsize=9,
        )

    # Four Umbrellas reference
    if four_umbrellas_point is not None:
        ax.scatter(
            four_umbrellas_point.volatility * 100, four_umbrellas_point.expected_return * 100,
            marker="o", s=220, color="#1f77b4",
            edgecolors="black", linewidth=1.5, zorder=5,
            label=f"Four Umbrellas ({four_umbrellas_point.sharpe:.2f} Sharpe)",
        )
        ax.annotate(
            "Four Umbrellas", (four_umbrellas_point.volatility * 100, four_umbrellas_point.expected_return * 100),
            xytext=(10, -15), textcoords="offset points",
            fontweight="bold", fontsize=9,
        )

    ax.set_title(
        f"Efficient Frontier — {len(random_eval_df):,} random portfolios + Markowitz",
        fontsize=13, fontweight="bold",
    )
    ax.set_xlabel("Annualized volatility")
    ax.set_ylabel("Expected return (annualized)")
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=1))
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def run_efficient_frontier(
    bundle: DataBundle,
    four_umbrellas_returns: Optional[pd.Series] = None,
    n_random: int = 50_000,
    risk_free_rate: float = 0.02,
    seed: int = 42,
) -> Tuple[pd.DataFrame, Dict[str, FrontierPoint], pd.DataFrame, Optional[FrontierPoint]]:
    """
    Orchestrate the full efficient-frontier analysis on the sleeve returns in
    the data bundle. Returns: (random_eval_df, key_portfolios, frontier_df, four_umbrellas_point).
    """
    # Use only non-pension sleeves for the frontier — pension is locked,
    # so including it as a free weight would be misleading
    excluded = {"pension_bond", "pension_equity"}
    sleeve_returns = bundle.monthly_returns_eur.drop(columns=[c for c in excluded if c in bundle.monthly_returns_eur.columns])
    sleeve_returns = sleeve_returns.dropna(how="any")
    asset_names = list(sleeve_returns.columns)

    annual_mean, annual_cov = compute_annual_moments(sleeve_returns)

    # Random portfolios
    weights_matrix = sample_random_portfolios(
        n_assets=len(asset_names), n_samples=n_random, seed=seed,
    )
    random_eval_df = evaluate_portfolios(weights_matrix, annual_mean.values, annual_cov.values, rf=risk_free_rate)
    key_portfolios = find_key_portfolios(weights_matrix, random_eval_df, asset_names)

    # Markowitz frontier
    frontier_df = markowitz_frontier(annual_mean.values, annual_cov.values, rf=risk_free_rate)

    # Four Umbrellas reference
    four_umbrellas_point = None
    if four_umbrellas_returns is not None and len(four_umbrellas_returns) > 0:
        fu_returns = four_umbrellas_returns.dropna()
        fu_mean = float(fu_returns.mean() * 12)
        fu_vol = float(fu_returns.std() * np.sqrt(12))
        fu_sharpe = (fu_mean - risk_free_rate) / fu_vol if fu_vol > 1e-12 else float("nan")
        four_umbrellas_point = FrontierPoint(
            weights={},
            expected_return=fu_mean,
            volatility=fu_vol,
            sharpe=fu_sharpe,
            label="Four Umbrellas",
        )

    return random_eval_df, key_portfolios, frontier_df, four_umbrellas_point
