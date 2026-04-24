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

Plus the active portfolio's reference position (Four Umbrellas for the default preset).

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
from .portfolio_model import Portfolio


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
    asset_names: Optional[List[str]] = None,
    n_points: int = 50,
    rf: float = 0.02,
) -> pd.DataFrame:
    """
    Trace the efficient frontier by sweeping target annual return levels and,
    for each target return, finding the long-only fully invested portfolio
    with minimum variance subject to:
        - sum(weights) = 1
        - weights >= 0 (long-only)
        - w @ annual_mean = target_return

    Returns a DataFrame with columns: expected_return, volatility, sharpe,
    and (when asset_names is provided) one `w_<asset>` column per asset so
    the chart can surface the full allocation for each point on hover/click.
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
            row = {
                "expected_return": port_ret,
                "volatility": port_vol,
                "sharpe": sharpe,
            }
            if asset_names is not None:
                for i, name in enumerate(asset_names):
                    row[f"w_{name}"] = float(w[i])
            frontier_points.append(row)

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


def plot_efficient_frontier_interactive(
    random_eval_df: pd.DataFrame,
    weights_matrix: np.ndarray,
    asset_names: List[str],
    key_portfolios: Dict[str, FrontierPoint],
    frontier_df: pd.DataFrame,
    four_umbrellas_point: Optional[FrontierPoint],
    path: Path,
) -> bool:
    """
    Produce an **interactive** HTML efficient-frontier chart using Plotly.
    Hover (and click) each point shows the full portfolio allocation.

    Returns True if plotly is available and the chart was written, False
    if plotly is missing (graceful skip — user gets the static PNG anyway).
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("  (plotly not installed — skipping interactive frontier. "
              "`pip install plotly` to enable.)")
        return False

    def _weights_hover_text(weights_dict_or_array, top_n: int = 6) -> str:
        """Format a weights vector as a compact HTML hover string (top-N by weight)."""
        if isinstance(weights_dict_or_array, dict):
            items = list(weights_dict_or_array.items())
        else:
            items = list(zip(asset_names, weights_dict_or_array))
        items = sorted(items, key=lambda x: -abs(x[1]))[:top_n]
        lines = [f"• {k}: {v * 100:.1f}%" for k, v in items if v > 0.005]
        return "<br>".join(lines) if lines else "(all weights ~0)"

    fig = go.Figure()

    # --- Random cloud (sampled for performance: max 10k points shown) ---
    n_random = len(random_eval_df)
    if n_random > 10_000:
        # Sample 10k indices uniformly for the display (keeps hover snappy)
        sample_idx = np.random.default_rng(42).choice(n_random, size=10_000, replace=False)
    else:
        sample_idx = np.arange(n_random)

    hover_texts = [
        f"Return: {random_eval_df.iloc[i]['expected_return'] * 100:.2f}%<br>"
        f"Vol: {random_eval_df.iloc[i]['volatility'] * 100:.2f}%<br>"
        f"Sharpe: {random_eval_df.iloc[i]['sharpe']:.2f}<br><br>"
        f"<b>Top weights:</b><br>{_weights_hover_text(weights_matrix[i])}"
        for i in sample_idx
    ]
    fig.add_trace(go.Scatter(
        x=random_eval_df.iloc[sample_idx]["volatility"] * 100,
        y=random_eval_df.iloc[sample_idx]["expected_return"] * 100,
        mode="markers",
        marker=dict(
            size=4,
            color=random_eval_df.iloc[sample_idx]["sharpe"],
            colorscale="Viridis",
            colorbar=dict(title="Sharpe", thickness=14),
            opacity=0.45,
            line=dict(width=0),
        ),
        name=f"{len(sample_idx):,} random portfolios",
        text=hover_texts,
        hoverinfo="text",
    ))

    # --- Markowitz frontier line (each point carries full weights) ---
    if len(frontier_df) > 0:
        sorted_df = frontier_df.sort_values("volatility").reset_index(drop=True)
        weight_cols = [c for c in sorted_df.columns if c.startswith("w_")]
        frontier_hover = []
        for _, row in sorted_df.iterrows():
            weights_on_point = {c.removeprefix("w_"): row[c] for c in weight_cols}
            frontier_hover.append(
                f"<b>Frontier point</b><br>"
                f"Return: {row['expected_return'] * 100:.2f}%<br>"
                f"Vol: {row['volatility'] * 100:.2f}%<br>"
                f"Sharpe: {row['sharpe']:.2f}<br><br>"
                f"<b>Allocation:</b><br>{_weights_hover_text(weights_on_point)}"
            )
        fig.add_trace(go.Scatter(
            x=sorted_df["volatility"] * 100,
            y=sorted_df["expected_return"] * 100,
            mode="lines+markers",
            line=dict(color="#d62728", width=2.5),
            marker=dict(size=7, color="#d62728", line=dict(width=1, color="black")),
            name="Efficient Frontier (Markowitz)",
            text=frontier_hover,
            hoverinfo="text",
        ))

    # --- Key portfolios (Max Sharpe, Min Vol, Max Return) ---
    marker_configs = {
        "max_sharpe": ("star", "#ff7f0e", 22, "Max Sharpe"),
        "min_vol":    ("diamond", "#2ca02c", 18, "Min Volatility"),
        "max_return": ("triangle-up", "#9467bd", 20, "Max Return"),
    }
    for key, point in key_portfolios.items():
        symbol, color, size, label = marker_configs[key]
        fig.add_trace(go.Scatter(
            x=[point.volatility * 100],
            y=[point.expected_return * 100],
            mode="markers",
            marker=dict(
                symbol=symbol, color=color, size=size,
                line=dict(width=1.5, color="black"),
            ),
            name=f"{label} (Sharpe {point.sharpe:.2f})",
            text=[
                f"<b>{label}</b><br>"
                f"Return: {point.expected_return * 100:.2f}%<br>"
                f"Vol: {point.volatility * 100:.2f}%<br>"
                f"Sharpe: {point.sharpe:.2f}<br><br>"
                f"<b>Allocation:</b><br>{_weights_hover_text(point.weights)}"
            ],
            hoverinfo="text",
        ))

    # --- Four Umbrellas reference ---
    if four_umbrellas_point is not None:
        fig.add_trace(go.Scatter(
            x=[four_umbrellas_point.volatility * 100],
            y=[four_umbrellas_point.expected_return * 100],
            mode="markers",
            marker=dict(
                symbol="circle", color="#1f77b4", size=18,
                line=dict(width=2, color="black"),
            ),
            name=f"Four Umbrellas (Sharpe {four_umbrellas_point.sharpe:.2f})",
            text=[
                f"<b>Four Umbrellas</b> (liquid sleeves, normalized)<br>"
                f"Return: {four_umbrellas_point.expected_return * 100:.2f}%<br>"
                f"Vol: {four_umbrellas_point.volatility * 100:.2f}%<br>"
                f"Sharpe: {four_umbrellas_point.sharpe:.2f}<br><br>"
                f"<b>Allocation:</b><br>{_weights_hover_text(four_umbrellas_point.weights)}"
            ],
            hoverinfo="text",
        ))

    fig.update_layout(
        title=dict(
            text=f"Efficient Frontier — {n_random:,} random + Markowitz"
                 f" <span style='font-size:12px'>(hover / click any point to see its allocation)</span>",
            x=0.5, xanchor="center",
        ),
        xaxis=dict(title="Annualized Volatility (%)", gridcolor="#eeeeee"),
        yaxis=dict(title="Expected Return (annualized, %)", gridcolor="#eeeeee"),
        hovermode="closest",
        plot_bgcolor="white",
        height=720,
        legend=dict(x=1.08, y=1.0, xanchor="left"),
        margin=dict(l=60, r=240, t=80, b=60),
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(path), include_plotlyjs="cdn", full_html=True)
    return True


def _legacy_reference_weights() -> Dict[str, float]:
    """Read the Four Umbrellas preset weights from the src.portfolio globals.
    Used only when ``run_efficient_frontier`` is called without a Portfolio —
    matches pre-PR3 behavior exactly."""
    from . import portfolio as pf

    raw_weights: Dict[str, float] = {}
    raw_weights["gold"] = pf.WEIGHTS.get("gold", 0.0)
    raw_weights["dbi"] = pf.WEIGHTS.get("dbi", 0.0)
    for src in (pf.EQUITY, pf.CRYPTO, pf.BONDS, pf.EM_SATELLITES):
        raw_weights.update(src)
    return raw_weights


def _compute_reference_point_on_universe(
    sleeve_returns: pd.DataFrame,
    annual_mean: pd.Series,
    annual_cov: pd.DataFrame,
    risk_free_rate: float,
    raw_weights: Optional[Dict[str, float]] = None,
    label: Optional[str] = None,
) -> Optional[FrontierPoint]:
    """
    Compute a reference portfolio's point on the *same* sleeve universe and
    date index used for the frontier (not on the full simulate_portfolio
    series, which would include pension/cash/TER effects and wouldn't be
    directly comparable to the cloud + frontier).

    If ``raw_weights`` is None, falls back to the Four Umbrellas preset read
    from ``src.portfolio`` globals (legacy behavior). Otherwise uses the
    provided weights (typically a user Portfolio's ``to_weights_dict()``).
    In both cases the weights are filtered to sleeves present in
    ``sleeve_returns.columns`` and normalized to sum=1.0.
    """
    if raw_weights is None:
        raw_weights = _legacy_reference_weights()
        label = label or "Four Umbrellas (liquid sleeves, normalized)"
    else:
        label = label or "Reference portfolio (liquid sleeves, normalized)"

    # Filter to sleeves present in the universe, normalize to 1.0
    present = {
        k: v for k, v in raw_weights.items()
        if k in sleeve_returns.columns and v > 0
    }
    total = sum(present.values())
    if total <= 0:
        return None
    normalized = {k: v / total for k, v in present.items()}

    # Vectorized stats on the exact universe the frontier uses
    weight_vec = np.array([normalized.get(c, 0.0) for c in sleeve_returns.columns])
    port_ret = float(weight_vec @ annual_mean.values)
    port_vol = float(np.sqrt(weight_vec @ annual_cov.values @ weight_vec))
    sharpe = (port_ret - risk_free_rate) / port_vol if port_vol > 1e-12 else float("nan")
    return FrontierPoint(
        weights=normalized,
        expected_return=port_ret,
        volatility=port_vol,
        sharpe=sharpe,
        label=label,
    )


def run_efficient_frontier(
    bundle: DataBundle,
    portfolio: Optional[Portfolio] = None,
    n_random: int = 50_000,
    risk_free_rate: float = 0.02,
    seed: int = 42,
    include_reference_point: bool = True,
    # Legacy kw for backward compat with pre-PR3 callers
    include_four_umbrellas_reference: Optional[bool] = None,
) -> Tuple[pd.DataFrame, np.ndarray, Dict[str, FrontierPoint], pd.DataFrame, Optional[FrontierPoint], List[str]]:
    """
    Orchestrate the full efficient-frontier analysis on the sleeve returns in
    the data bundle.

    Behavior depends on ``portfolio``:

    * ``portfolio is None`` (legacy, pre-PR3) → universe = bundle sleeves minus
      pension, reference point = Four Umbrellas preset from ``src.portfolio``
      globals. Matches pre-PR3 output exactly.
    * ``portfolio is not None`` → universe = the portfolio's asset keys minus
      ``cash`` (cash is excluded — the frontier operates on the risky sleeves
      only). Reference point = that portfolio normalized to the universe.

    Returns: (random_eval_df, weights_matrix, key_portfolios, frontier_df,
              reference_point, asset_names).
    """
    # Legacy kw alias: accept `include_four_umbrellas_reference=` for pre-PR3
    # callers. If both are given (shouldn't happen) the new kw wins.
    if include_four_umbrellas_reference is not None:
        include_reference_point = include_four_umbrellas_reference

    if portfolio is None:
        # Legacy path: use all bundle sleeves minus pension
        excluded = {"pension_bond", "pension_equity"}
        sleeve_returns = bundle.monthly_returns_eur.drop(
            columns=[c for c in excluded if c in bundle.monthly_returns_eur.columns]
        )
        reference_raw_weights: Optional[Dict[str, float]] = None  # falls back to globals
        reference_label: Optional[str] = None
    else:
        # Generic path: universe = portfolio assets minus cash (cash excluded
        # from the frontier — it's the residual risk-free sleeve, not an
        # optimization degree of freedom).
        keys = [a.key for a in portfolio.assets if a.key != "cash"]
        available = [k for k in keys if k in bundle.monthly_returns_eur.columns]
        if not available:
            raise ValueError(
                f"None of the portfolio assets {keys!r} are available in the data bundle. "
                f"Check that the CSVs for those keys exist in data/raw/."
            )
        sleeve_returns = bundle.monthly_returns_eur[available].copy()
        reference_raw_weights = {
            a.key: a.weight for a in portfolio.assets if a.key != "cash"
        }
        reference_label = f"{portfolio.name} (liquid sleeves, normalized)"

    sleeve_returns = sleeve_returns.dropna(how="any")
    asset_names = list(sleeve_returns.columns)

    annual_mean, annual_cov = compute_annual_moments(sleeve_returns)

    # Random portfolios
    weights_matrix = sample_random_portfolios(
        n_assets=len(asset_names), n_samples=n_random, seed=seed,
    )
    random_eval_df = evaluate_portfolios(weights_matrix, annual_mean.values, annual_cov.values, rf=risk_free_rate)
    key_portfolios = find_key_portfolios(weights_matrix, random_eval_df, asset_names)

    # Markowitz frontier (with per-point weights for interactive chart)
    frontier_df = markowitz_frontier(
        annual_mean.values, annual_cov.values,
        asset_names=asset_names, rf=risk_free_rate,
    )

    # Reference point on the frontier's coordinate system
    reference_point = None
    if include_reference_point:
        reference_point = _compute_reference_point_on_universe(
            sleeve_returns, annual_mean, annual_cov, risk_free_rate,
            raw_weights=reference_raw_weights,
            label=reference_label,
        )

    return random_eval_df, weights_matrix, key_portfolios, frontier_df, reference_point, asset_names
