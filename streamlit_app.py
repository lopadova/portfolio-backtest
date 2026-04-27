"""
Streamlit dashboard — interactive Portfolio Backtest Engine UI.

Run locally:
    pip install -r requirements-dashboard.txt
    streamlit run streamlit_app.py

Then open http://localhost:8501 in your browser.

Deploy to Streamlit Cloud: push to GitHub, link your repo at share.streamlit.io.
Deploy to HuggingFace Spaces: add a Dockerfile (see README) and push.

PR4 architecture:
  - Sidebar: brand + links only (no widgets).
  - First tab "⚙️ Settings" with 4 sections (Portfolio, Period,
    Engine, Advanced analyses) — assembles a Portfolio in session_state
    without ever mutating module globals.
  - Post-run tabs appear when a run completes: Summary, Charts, Monte
    Carlo, AI Analysis. Frontier / FIRE / Walk-forward are placeholder
    toggles — the integration lands in PR5.
"""

from __future__ import annotations

import dataclasses
import tempfile
from pathlib import Path
from typing import Any, Dict, List

# Load OPENROUTER_API_KEY / OPENAI_API_KEY / ANTHROPIC_API_KEY / LOCAL_API_BASE_URL
# from a `.env` at the project root before anything reads os.environ. Real env
# vars and Streamlit Cloud secrets still take precedence (override=False).
from src.env_check import load_dotenv  # stdlib-only, safe to import first

load_dotenv()

import pandas as pd
import streamlit as st

from src.dashboard_helpers import (
    build_portfolio_from_ui_state,
    common_history_years,
    compute_effective_start,
    fire_config_from_ui_state,
    format_asset_start_date,
)
from src.data_catalog import (
    AssetInfo,
    DEFAULT_CATALOG_PATH,
    DEFAULT_DATA_RAW_DIR,
    augment_with_raw_dates,
    load_catalog,
)
from src.data_loader import load_data
from src.efficient_frontier import (
    plot_efficient_frontier,
    plot_efficient_frontier_interactive,
    run_efficient_frontier,
)
from src.fire import (
    aggregate_fire_results,
    plot_fire_failure_distribution,
    plot_fire_legacy_distribution,
    plot_fire_projection,
    plot_fire_success_probability,
    run_fire_simulation,
)
from src.metrics import compute_all, crisis_drawdown, cumulative_wealth, stats_to_dataframe
from src.monte_carlo import block_bootstrap, compute_mc_stats, simulate_wealth_paths
from src.options_overlay import simulate_options_overlay
from src.rolling_window import (
    plot_rolling_window_results,
    run_rolling_backtest,
    summary_statistics as rolling_summary_statistics,
)
from src.plots import (
    plot_annual_returns,
    plot_crisis_zoom,
    plot_drawdown,
    plot_equity_curve,
    plot_mc_scenarios,
    plot_metrics_comparison,
    plot_monte_carlo_fan,
    plot_return_distribution,
    plot_risk_return_scatter,
    plot_rolling_returns,
    plot_rolling_sharpe,
    plot_underwater,
)
from datetime import datetime

from src.portfolio import BENCHMARKS
from src.portfolio_model import (
    DEFAULT_PORTFOLIOS_DIR,
    Portfolio,
    PortfolioMetricsCache,
    list_available_presets,
    slugify,
)
from src.rebalance import simulate_benchmark, simulate_portfolio
from src.report import generate_markdown_report


# ===========================================================================
# Page config
# ===========================================================================

st.set_page_config(
    page_title="Portfolio Backtest Engine",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ===========================================================================
# Catalog loading (cached)
# ===========================================================================

@st.cache_data(show_spinner=False)
def _load_catalog_cached(synthetic: bool) -> Dict[str, AssetInfo]:
    """Load the data catalog. When ``synthetic=True`` we still load the
    same manifest but skip CSV date augmentation (synthetic bundles don't
    read from ``data/raw/``, so date bounds are meaningless there — we
    simply show "—" for every start date)."""
    catalog = load_catalog(DEFAULT_CATALOG_PATH)
    if not synthetic:
        catalog = augment_with_raw_dates(catalog, DEFAULT_DATA_RAW_DIR)
    return catalog


@st.cache_data(show_spinner=False)
def _load_data_cached(synthetic: bool):
    return load_data(synthetic=synthetic)


# ===========================================================================
# Session-state defaults
# ===========================================================================

def _ensure_session_defaults() -> None:
    """Initialize the session-state keys used by the Settings tab so
    reading them later never KeyErrors."""
    defaults = {
        "data_mode": "Synthetic (demo)",
        "portfolio_name": "My portfolio",
        "portfolio_assets": [],   # list[{key, weight}]
        "options_overlay": False,
        # options_budget is NOT in session_state — the UI widget was removed
        # because the engine reads OPTIONS.budget_nav_per_year from globals
        # and threading a per-run budget requires a per-Portfolio
        # OptionsConfig (planned for a future PR).
        "rebalance_freq": "Semi-annual",
        "transaction_cost_bps": 20.0,
        "start_nav": 100_000.0,
        "mc_enabled": True,
        "mc_paths": 5000,
        "mc_years": 20,
        "mc_block_size": 3,
        # PR5 — advanced analysis toggles + per-analysis config
        "frontier_enabled": False,
        "frontier_n_random": 10_000,
        "fire_enabled": False,
        "fire_current_age": 45,
        "fire_sex": "M",
        "fire_fixed_end_age": 0,               # 0 = unset (handled by the helper)
        "fire_initial_capital": 100_000.0,
        "fire_contribution_amount": 1_000.0,
        "fire_contribution_frequency": "month",
        "fire_fire_age": 60,
        "fire_monthly_spending": 2_500.0,
        "fire_inflation_rate": 0.02,
        "fire_pension_enabled": False,
        "fire_pension_monthly_amount": 1_500.0,
        "fire_pension_start_age": 67,
        "fire_pension_revaluation": 0.75,
        "fire_tax_on_withdrawals": False,
        "fire_tax_rate": 0.26,
        "fire_n_simulations": 1000,
        "fire_block_size": 3,
        "fire_seed": 42,
        "walkforward_enabled": False,
        "walkforward_window_years": 10,
        "walkforward_step_months": 12,
        # Other
        "selected_benchmarks": ["60/40", "100% SWDA (MSCI World EUR)"],
        "save_results": False,
        "last_run": None,
    }
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)


_REBALANCE_FREQ_TO_MONTHS = {
    "Annual":       (1,),
    "Semi-annual":  (1, 7),
    "Quarterly":    (1, 4, 7, 10),
    "Monthly":      tuple(range(1, 13)),
}


# ===========================================================================
# Sidebar — brand + links only
# ===========================================================================

st.sidebar.title("📊 Portfolio Backtest Engine")
st.sidebar.caption("Multi-asset backtest · PR4")
st.sidebar.markdown(
    """
    - [GitHub](https://github.com/padosoft/portfolio-backtest)
    - [Medium article](https://medium.com/@padosoft)
    """
)
st.sidebar.caption("**Disclaimer**: educational use only, not investment advice.")


# ===========================================================================
# Main area — tabs
# ===========================================================================

st.title("📊 Portfolio Backtest Engine")

_ensure_session_defaults()

# Tab labels built dynamically: Settings first, then the post-run tabs
# that are always "visible" (empty info panel if no run yet).
(
    tab_settings, tab_summary, tab_charts, tab_mc,
    tab_frontier, tab_fire, tab_walkforward, tab_ai, tab_saved,
) = st.tabs([
    "⚙️ Settings",
    "📊 Summary",
    "📈 Charts",
    "🎲 Monte Carlo",
    "📐 Efficient Frontier",
    "🔥 FIRE",
    "🌀 Walk-forward",
    "🤖 AI Analysis",
    "📂 Saved portfolios",
])


# ===========================================================================
# Tab: Settings
# ===========================================================================

with tab_settings:
    # ------ Preset + data source + portfolio name ---------------------------
    col_preset, col_name, col_data = st.columns([3, 4, 3])

    with col_preset:
        presets = [p["name"] for p in list_available_presets()]
        preset_options = ["(none — build my own)"] + presets
        # When the user picks a preset, load it into session_state BEFORE the
        # widgets below are rendered (we do it on button click so the choice
        # is deterministic, not magically on dropdown change).
        picked = st.selectbox("Load preset", preset_options, index=0)
        if st.button("📂 Load", disabled=(picked == preset_options[0])):
            portfolio = Portfolio.from_name(picked)
            st.session_state["portfolio_name"] = portfolio.name
            st.session_state["portfolio_assets"] = [
                {"key": a.key, "weight": a.weight} for a in portfolio.assets
            ]
            st.session_state["options_overlay"] = portfolio.options_overlay
            st.session_state["transaction_cost_bps"] = portfolio.transaction_cost_bps
            # Map the preset's rebalance_months back to a user-friendly label
            for label, months in _REBALANCE_FREQ_TO_MONTHS.items():
                if months == portfolio.rebalance_months:
                    st.session_state["rebalance_freq"] = label
                    break
            st.rerun()

    with col_name:
        st.session_state["portfolio_name"] = st.text_input(
            "Portfolio name", value=st.session_state["portfolio_name"],
        )

    with col_data:
        st.session_state["data_mode"] = st.radio(
            "Data source",
            ["Real data (data/raw/)", "Synthetic (demo)"],
            index=1 if st.session_state["data_mode"] == "Synthetic (demo)" else 0,
            help="Use synthetic data if you haven't populated data/raw/ yet.",
        )

    synthetic = st.session_state["data_mode"] == "Synthetic (demo)"

    # Load data + catalog once (cached)
    try:
        bundle = _load_data_cached(synthetic)
        catalog = _load_catalog_cached(synthetic)
    except Exception as e:
        st.error(f"Failed to load data or catalog: {e}")
        st.stop()

    data_start = bundle.monthly_returns_eur.index[0]
    data_end = bundle.monthly_returns_eur.index[-1]

    st.divider()

    # ------ 🎯 Portfolio -----------------------------------------------------
    st.subheader("🎯 Portfolio")

    # Build the picker options list from assets that are actually SIMULATABLE
    # with the currently loaded return bundle. The catalog also contains
    # benchmark-only / metadata-only series (e.g. `msci_world_tr_monthly`)
    # which are not members of ``bundle.monthly_returns_eur.columns`` — if
    # added to the portfolio they would be silently dropped by
    # simulate_portfolio_generic and the remaining weights renormalized,
    # producing a misleading result.
    simulatable_asset_keys = set(bundle.monthly_returns_eur.columns)
    available_asset_keys = sorted(
        k for k in catalog.keys()
        if k == "cash" or k in simulatable_asset_keys
    )
    key_to_display = {
        k: f"{catalog[k].display_name}  ({k})" for k in available_asset_keys
    }

    # If the portfolio already in session_state (e.g. loaded from a preset)
    # contains keys not in the bundle, surface a hard error and block Run.
    invalid_portfolio_keys = sorted({
        row["key"]
        for row in st.session_state["portfolio_assets"]
        if row["key"] != "cash" and row["key"] not in simulatable_asset_keys
    })
    if invalid_portfolio_keys:
        st.error(
            "Some assets in the portfolio cannot be simulated with the current "
            "dataset (missing from `bundle.monthly_returns_eur`). Remove them "
            f"before running: {', '.join(invalid_portfolio_keys)}"
        )

    pick_col, weight_col, btn_col = st.columns([6, 2, 2])
    with pick_col:
        picked_key = st.selectbox(
            "Add asset",
            options=available_asset_keys,
            format_func=lambda k: key_to_display[k],
            key="_asset_picker",
        )
    with weight_col:
        picked_weight_pct = st.number_input(
            "Weight %", min_value=0.0, max_value=100.0, value=10.0, step=0.5,
            key="_asset_picker_weight",
        )
    with btn_col:
        st.write("")  # vertical alignment
        st.write("")
        if st.button("＋ Add", use_container_width=True):
            existing_keys = {a["key"] for a in st.session_state["portfolio_assets"]}
            if picked_key in existing_keys:
                st.warning(f"{picked_key!r} is already in the portfolio; edit the weight on the existing row.")
            else:
                st.session_state["portfolio_assets"].append(
                    {"key": picked_key, "weight": picked_weight_pct / 100.0}
                )
                st.rerun()

    # Render editable list (+ delete column). Using st.data_editor so the
    # user can tweak weights inline and see the badge update on rerun.
    rows: List[Dict[str, Any]] = []
    for a in st.session_state["portfolio_assets"]:
        info = catalog.get(a["key"])
        rows.append({
            "Asset": info.display_name if info else a["key"],
            "Key": a["key"],
            "Data from": format_asset_start_date(info),
            "Weight %": round(a["weight"] * 100, 4),
        })

    if rows:
        # ``num_rows="dynamic"`` exposes Streamlit's native row-delete affordance
        # (left-side checkbox + Delete key, or right-click → Delete row), so we
        # don't render a separate trash-button grid. The "Add row" button it
        # also exposes is harmless: Asset/Key/Dati dal columns are disabled, so
        # any row added inline lands with Key=None and gets filtered out by the
        # sync below — users add real assets via the picker above.
        edited = st.data_editor(
            pd.DataFrame(rows),
            hide_index=True,
            use_container_width=True,
            disabled=["Asset", "Key", "Data from"],
            num_rows="dynamic",
            column_config={
                "Weight %": st.column_config.NumberColumn(
                    "Weight %", min_value=0.0, max_value=100.0, step=0.5, format="%.2f",
                ),
            },
            key="_portfolio_editor",
        )
        # Reflect user edits + deletions back into session_state. Skip rows
        # with a missing/empty Key (placeholder rows added inline by the
        # editor). Use ``pd.isna`` / ``pd.notna`` rather than Python truthiness:
        # ``np.nan`` is truthy and ``pd.NA`` raises in boolean ops, so an
        # ``or 0.0`` / ``if row["Key"]`` check would silently propagate NaN
        # weights or crash with ``TypeError: boolean value of NA is ambiguous``.
        st.session_state["portfolio_assets"] = []
        for _, row in edited.iterrows():
            key = row["Key"]
            if pd.isna(key) or str(key).strip() == "":
                continue
            raw_weight = row["Weight %"]
            weight_pct = 0.0 if pd.isna(raw_weight) else float(raw_weight)
            st.session_state["portfolio_assets"].append(
                {"key": key, "weight": weight_pct / 100.0}
            )
    else:
        st.info("No assets selected. Use the picker above or load a preset.")

    # Sum-of-weights badge
    total_weight = sum(a["weight"] for a in st.session_state["portfolio_assets"])
    total_pct = total_weight * 100
    sum_ok = abs(total_weight - 1.0) <= 0.002
    if sum_ok and st.session_state["portfolio_assets"]:
        st.markdown(f"**Total:** 🟢 {total_pct:.2f}%")
    elif st.session_state["portfolio_assets"]:
        missing = 100.0 - total_pct
        st.markdown(
            f"**Total:** 🔴 {total_pct:.2f}%  "
            f"_(missing {missing:+.2f}% — must equal 100% before you can run)_"
        )
    else:
        st.markdown("**Total:** ⚪ 0.00%")

    st.divider()

    # ------ 📅 Period --------------------------------------------------------
    st.subheader("📅 Period")
    date_range = st.date_input(
        "Analysis interval",
        value=(data_start.date(), data_end.date()),
        min_value=data_start.date(),
        max_value=data_end.date(),
    )
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date = pd.Timestamp(date_range[0])
        end_date = pd.Timestamp(date_range[1])
    else:
        start_date = data_start
        end_date = data_end

    # Date-range warning — only meaningful once the portfolio has assets
    if st.session_state["portfolio_assets"] and sum_ok:
        try:
            _p_peek = build_portfolio_from_ui_state(
                st.session_state["portfolio_name"] or "Peek",
                st.session_state["portfolio_assets"],
            )
            effective_start, warning_text = compute_effective_start(
                start_date, _p_peek, catalog,
            )
            if warning_text:
                st.warning(f"⚠️ {warning_text}")
        except ValueError:
            # Sum or name validation will be reported by the Run path;
            # skip the warning computation here.
            pass

    st.divider()

    # ------ ⚙️ Engine --------------------------------------------------------
    st.subheader("⚙️ Engine")

    nav_col, opt_col, reb_col, tc_col = st.columns([3, 3, 3, 2])
    with nav_col:
        st.session_state["start_nav"] = st.number_input(
            "Initial NAV (€)", min_value=1000.0, step=10_000.0,
            value=float(st.session_state["start_nav"]),
        )
    with opt_col:
        st.session_state["options_overlay"] = st.checkbox(
            "Options overlay (SPY/QQQ put-spread)",
            value=st.session_state["options_overlay"],
            help=(
                "Requires SPY/QQQ/VIX data in the bundle. 6-month put spreads "
                "on 30% of equity notional. Budget is fixed at the global "
                "OPTIONS.budget_nav_per_year default (0.30% NAV/year); a UI "
                "knob for the budget requires a per-Portfolio OptionsConfig, "
                "which is planned for a future PR."
            ),
        )
    with reb_col:
        st.session_state["rebalance_freq"] = st.radio(
            "Rebalancing",
            list(_REBALANCE_FREQ_TO_MONTHS.keys()),
            index=list(_REBALANCE_FREQ_TO_MONTHS.keys()).index(
                st.session_state["rebalance_freq"]
            ),
        )
    with tc_col:
        st.session_state["transaction_cost_bps"] = st.number_input(
            "TC bps", min_value=0.0, max_value=100.0,
            value=float(st.session_state["transaction_cost_bps"]),
            step=1.0,
            help="Round-trip transaction cost on rebalance trades (basis points).",
        )

    st.divider()

    # ------ 🎲 Advanced analyses --------------------------------------------
    st.subheader("🎲 Advanced analyses")

    st.session_state["mc_enabled"] = st.checkbox(
        "Monte Carlo (block bootstrap)", value=st.session_state["mc_enabled"],
    )
    mc_col1, mc_col2, mc_col3 = st.columns(3)
    with mc_col1:
        st.session_state["mc_paths"] = st.number_input(
            "Paths", min_value=100, max_value=50_000,
            value=int(st.session_state["mc_paths"]),
            step=500, disabled=not st.session_state["mc_enabled"],
        )
    with mc_col2:
        st.session_state["mc_years"] = st.number_input(
            "Horizon (years)", min_value=1, max_value=40,
            value=int(st.session_state["mc_years"]),
            disabled=not st.session_state["mc_enabled"],
        )
    with mc_col3:
        st.session_state["mc_block_size"] = st.number_input(
            "Block size (months)", min_value=1, max_value=24,
            value=int(st.session_state["mc_block_size"]),
            disabled=not st.session_state["mc_enabled"],
        )

    # --- Efficient Frontier ------------------------------------------------
    st.session_state["frontier_enabled"] = st.checkbox(
        "Efficient Frontier",
        value=st.session_state["frontier_enabled"],
        help=(
            "Markowitz mean-variance optimization + Dirichlet-sampled random "
            "portfolios restricted to the ASSETS of your portfolio (cash "
            "excluded)."
        ),
    )
    st.session_state["frontier_n_random"] = st.number_input(
        "Random portfolios",
        min_value=500, max_value=50_000,
        value=int(st.session_state["frontier_n_random"]),
        step=1000,
        disabled=not st.session_state["frontier_enabled"],
        help="UI default is 10k for responsiveness; CLI still defaults to 50k.",
    )

    # --- FIRE --------------------------------------------------------------
    st.session_state["fire_enabled"] = st.checkbox(
        "FIRE calculator",
        value=st.session_state["fire_enabled"],
        help=(
            "Monte Carlo FIRE simulator with Italian mortality sampling and "
            "optional pension / CGT on withdrawals. Uses the main portfolio's "
            "return series as the underlying."
        ),
    )
    if st.session_state["fire_enabled"]:
        with st.expander("FIRE parameters", expanded=True):
            # Personal
            c_age, c_sex, c_end = st.columns(3)
            with c_age:
                st.session_state["fire_current_age"] = st.number_input(
                    "Current age", min_value=18, max_value=100,
                    value=int(st.session_state["fire_current_age"]),
                )
            with c_sex:
                st.session_state["fire_sex"] = st.radio(
                    "Sex", ["M", "F"],
                    index=0 if st.session_state["fire_sex"] == "M" else 1,
                    horizontal=True,
                )
            with c_end:
                st.session_state["fire_fixed_end_age"] = st.number_input(
                    "Fixed end age (0 = mortality)", min_value=0, max_value=120,
                    value=int(st.session_state["fire_fixed_end_age"]),
                    help="0 = sample from ISTAT mortality table. Any value >0 forces that as the end age.",
                )

            # Wealth + contributions
            c_cap, c_contr, c_freq = st.columns(3)
            with c_cap:
                st.session_state["fire_initial_capital"] = st.number_input(
                    "Initial capital (€)", min_value=0.0, step=10_000.0,
                    value=float(st.session_state["fire_initial_capital"]),
                )
            with c_contr:
                st.session_state["fire_contribution_amount"] = st.number_input(
                    "Periodic contribution (€)", min_value=0.0, step=100.0,
                    value=float(st.session_state["fire_contribution_amount"]),
                )
            with c_freq:
                st.session_state["fire_contribution_frequency"] = st.radio(
                    "Contribution frequency",
                    ["month", "year"],
                    index=0 if st.session_state["fire_contribution_frequency"] == "month" else 1,
                    horizontal=True,
                )

            # Objective
            c_fire, c_spend, c_infl = st.columns(3)
            with c_fire:
                st.session_state["fire_fire_age"] = st.number_input(
                    "FIRE age", min_value=20, max_value=120,
                    value=int(st.session_state["fire_fire_age"]),
                )
            with c_spend:
                st.session_state["fire_monthly_spending"] = st.number_input(
                    "Monthly spending post-FIRE (€)", min_value=0.0, step=100.0,
                    value=float(st.session_state["fire_monthly_spending"]),
                )
            with c_infl:
                st.session_state["fire_inflation_rate"] = st.number_input(
                    "Annual inflation", min_value=0.0, max_value=0.25, step=0.005,
                    value=float(st.session_state["fire_inflation_rate"]),
                    format="%.3f",
                )

            # Pension
            st.session_state["fire_pension_enabled"] = st.checkbox(
                "Pension active",
                value=st.session_state["fire_pension_enabled"],
            )
            if st.session_state["fire_pension_enabled"]:
                c_pamt, c_page, c_prev = st.columns(3)
                with c_pamt:
                    st.session_state["fire_pension_monthly_amount"] = st.number_input(
                        "Monthly pension (€)", min_value=0.0, step=50.0,
                        value=float(st.session_state["fire_pension_monthly_amount"]),
                    )
                with c_page:
                    st.session_state["fire_pension_start_age"] = st.number_input(
                        "Pension start age", min_value=40, max_value=80,
                        value=int(st.session_state["fire_pension_start_age"]),
                    )
                with c_prev:
                    st.session_state["fire_pension_revaluation"] = st.slider(
                        "INPS revaluation (fraction)", min_value=0.0, max_value=1.0,
                        value=float(st.session_state["fire_pension_revaluation"]),
                        step=0.05,
                        help="1.0 = full inflation match (100% INPS perequazione); 0.75 = 75%.",
                    )

            # Tax
            c_tax_on, c_tax_r = st.columns(2)
            with c_tax_on:
                st.session_state["fire_tax_on_withdrawals"] = st.checkbox(
                    "CGT on withdrawals",
                    value=st.session_state["fire_tax_on_withdrawals"],
                )
            with c_tax_r:
                st.session_state["fire_tax_rate"] = st.number_input(
                    "Tax rate (fraction)", min_value=0.0, max_value=1.0, step=0.01,
                    value=float(st.session_state["fire_tax_rate"]),
                    format="%.2f",
                    disabled=not st.session_state["fire_tax_on_withdrawals"],
                )

            # Simulation
            c_n, c_blk, c_seed = st.columns(3)
            with c_n:
                st.session_state["fire_n_simulations"] = st.number_input(
                    "N simulations", min_value=100, max_value=10_000,
                    value=int(st.session_state["fire_n_simulations"]),
                    step=100,
                )
            with c_blk:
                st.session_state["fire_block_size"] = st.number_input(
                    "Block size (months)", min_value=1, max_value=24,
                    value=int(st.session_state["fire_block_size"]),
                )
            with c_seed:
                st.session_state["fire_seed"] = st.number_input(
                    "Seed", min_value=0, max_value=2_147_483_647,
                    value=int(st.session_state["fire_seed"]),
                )

    # --- Walk-forward (historical rolling-window simulation) ----------------
    st.session_state["walkforward_enabled"] = st.checkbox(
        "Walk-forward (historical rolling window)",
        value=st.session_state["walkforward_enabled"],
        help=(
            "Rolls a fixed-size window across the history, simulating the "
            "portfolio on each window. Requires common history across all "
            "portfolio assets at least as long as the selected window size "
            "(in years) — otherwise the run is skipped with a warning. For "
            "stable percentile tail estimates, at least 20 years of common "
            "history is generally recommended."
        ),
    )
    c_ww, c_ws = st.columns(2)
    with c_ww:
        st.session_state["walkforward_window_years"] = st.number_input(
            "Window (years)", min_value=1, max_value=30,
            value=int(st.session_state["walkforward_window_years"]),
            disabled=not st.session_state["walkforward_enabled"],
        )
    with c_ws:
        st.session_state["walkforward_step_months"] = st.number_input(
            "Step (months)", min_value=1, max_value=60,
            value=int(st.session_state["walkforward_step_months"]),
            disabled=not st.session_state["walkforward_enabled"],
        )

    # Benchmarks
    st.session_state["selected_benchmarks"] = st.multiselect(
        "Comparison benchmarks",
        options=list(BENCHMARKS.keys()),
        default=st.session_state["selected_benchmarks"],
    )

    st.divider()

    # ------ Azioni -----------------------------------------------------------
    run_col, save_col = st.columns([2, 1])
    with run_col:
        run_btn = st.button(
            "▶ Run backtest", type="primary", use_container_width=True,
            disabled=(
                (not sum_ok)
                or (not st.session_state["portfolio_assets"])
                or bool(invalid_portfolio_keys)
            ),
        )
    with save_col:
        st.session_state["save_results"] = st.checkbox(
            "💾 Save to output/", value=st.session_state["save_results"],
            help="Persist charts + REPORT.md under output/dashboard_run/",
        )


# ===========================================================================
# Run the backtest (triggered from the Settings tab)
# ===========================================================================

if run_btn:
    # Build the Portfolio from session_state. If validation fails, surface
    # the error in the Settings tab and don't proceed.
    try:
        user_portfolio = build_portfolio_from_ui_state(
            name=st.session_state["portfolio_name"],
            assets_state=st.session_state["portfolio_assets"],
            options_overlay=st.session_state["options_overlay"],
            rebalance_months=_REBALANCE_FREQ_TO_MONTHS[st.session_state["rebalance_freq"]],
            transaction_cost_bps=st.session_state["transaction_cost_bps"],
        )
    except ValueError as e:
        with tab_settings:
            st.error(f"Portfolio validation failed: {e}")
        st.stop()

    # Clamp the user-selected start to the earliest date at which all
    # portfolio assets have data — otherwise the engine would implicitly
    # treat pre-start months as 0% returns for assets with partial history,
    # contradicting the warning we showed in the Settings tab
    # ("effective start = ..."). compute_effective_start returns the
    # constraining start_date when user_start is too early.
    effective_start, _ = compute_effective_start(start_date, user_portfolio, catalog)
    run_start = effective_start if effective_start > start_date else start_date

    # Slice bundle to the clamped range
    bundle_sliced = bundle.slice(run_start, end_date)

    with st.spinner("Running backtest..."):
        # Core simulation (no options) — exactly what PR2/3 exposed.
        base_returns = simulate_portfolio(
            bundle_sliced.monthly_returns_eur,
            bundle_sliced.btc_activation_date,
            apply_ter=True,
            portfolio=user_portfolio,
        )
        returns_dict: Dict[str, pd.Series] = {
            f"{user_portfolio.name} (no options)": base_returns
        }

        # Options overlay (if portfolio enables it AND bundle has SPY/QQQ/VIX).
        # Pass the user-selected rebalance frequency through so overlay rolls
        # align with the main portfolio's rebalance schedule; otherwise
        # simulate_options_overlay falls back to its hardcoded (1, 7) default
        # and the user's Monthly/Quarterly/Annual choice would be ignored.
        if user_portfolio.options_overlay:
            nav_series = cumulative_wealth(base_returns, st.session_state["start_nav"])
            overlay_returns = simulate_options_overlay(
                spy_daily=bundle_sliced.spy_daily,
                qqq_daily=bundle_sliced.qqq_daily,
                vix_daily=bundle_sliced.vix_daily,
                rf_daily=bundle_sliced.rf_daily,
                nav_series=nav_series,
                rebalance_months=user_portfolio.rebalance_months,
                # PR7: per-Portfolio override. None falls back to the global
                # OPTIONS so loading a TOML preset without [options] keeps
                # the same overlay defaults as before.
                options_config=user_portfolio.options_config,
            )
            overlay_aligned = overlay_returns.reindex(base_returns.index).fillna(0.0)
            returns_dict[user_portfolio.name] = base_returns + overlay_aligned

        # Benchmarks
        for bench_name in st.session_state["selected_benchmarks"]:
            bench_weights = BENCHMARKS[bench_name]
            bench_r = simulate_benchmark(bundle_sliced.benchmark_monthly_eur, bench_weights)
            if len(bench_r) > 0:
                returns_dict[bench_name] = bench_r

        stats_list = [compute_all(n, r) for n, r in returns_dict.items()]
        stats_df = stats_to_dataframe(stats_list)

        mc_stats = None
        wealth_paths = None
        main_returns = None
        if st.session_state["mc_enabled"]:
            main_name = (
                user_portfolio.name if user_portfolio.name in returns_dict
                else f"{user_portfolio.name} (no options)"
            )
            main_returns = returns_dict[main_name]
            sim_returns = block_bootstrap(
                main_returns,
                n_paths=int(st.session_state["mc_paths"]),
                n_periods=int(st.session_state["mc_years"]) * 12,
                block_size=int(st.session_state["mc_block_size"]),
            )
            wealth_paths = simulate_wealth_paths(sim_returns)
            mc_stats = compute_mc_stats(
                wealth_paths, n_months=int(st.session_state["mc_years"]) * 12,
            )

        # --- PR5 advanced analyses (conditional on toggles) ----------------

        frontier_result = None
        if st.session_state["frontier_enabled"]:
            with st.spinner("Computing efficient frontier..."):
                frontier_result = run_efficient_frontier(
                    bundle_sliced,
                    portfolio=user_portfolio,
                    n_random=int(st.session_state["frontier_n_random"]),
                )

        fire_result = None
        fire_error = None
        if st.session_state["fire_enabled"]:
            try:
                fire_config = fire_config_from_ui_state(st.session_state)
                with st.spinner(f"Running {fire_config.n_simulations} FIRE simulations..."):
                    main_name = (
                        user_portfolio.name if user_portfolio.name in returns_dict
                        else f"{user_portfolio.name} (no options)"
                    )
                    fire_portfolio_returns = returns_dict[main_name]
                    fire_sims = run_fire_simulation(fire_config, fire_portfolio_returns)
                    fire_summary = aggregate_fire_results(fire_sims, fire_config)
                    fire_result = (fire_sims, fire_summary, fire_config)
            except ValueError as e:
                fire_error = str(e)

        walkforward_result = None
        walkforward_skipped_reason = None
        if st.session_state["walkforward_enabled"]:
            window_years = int(st.session_state["walkforward_window_years"])
            step_months = int(st.session_state["walkforward_step_months"])
            # Gate: walk-forward needs common history (across the actual
            # bundle, sliced to the user's date range) ≥ window_years.
            # We compute the gate from the SAME bundle that will be fed to
            # run_rolling_backtest, so any gate pass is also a guarantee
            # the run won't crash with 'Data range shorter than window'.
            available_years = common_history_years(bundle_sliced, user_portfolio)
            if available_years < window_years:
                walkforward_skipped_reason = (
                    f"Insufficient history: {available_years:.1f} years of common "
                    f"history across portfolio assets, at least {window_years} are required. "
                    f"Walk-forward simulation was skipped."
                )
            else:
                with st.spinner(
                    f"Running {window_years}-year walk-forward (step {step_months} months)..."
                ):
                    # Defensive try/except: even with the gate above, a
                    # pathological combination (e.g. all-NaN tail rows at
                    # the bundle end) could still trip rolling_window's
                    # own validation. Convert any ValueError into a clean
                    # "skipped" message instead of a Streamlit red traceback.
                    try:
                        wf_df = run_rolling_backtest(
                            bundle_sliced,
                            portfolio=user_portfolio,
                            window_years=window_years,
                            step_months=step_months,
                        )
                        wf_stats = rolling_summary_statistics(wf_df)
                        # Snapshot step_months into the result so the tab
                        # renders the chart with the step size that was
                        # ACTUALLY used for this run — not whatever value
                        # the live session_state holds now (the user may
                        # have tweaked the slider since).
                        walkforward_result = (wf_df, wf_stats, window_years, step_months)
                    except ValueError as e:
                        walkforward_skipped_reason = (
                            f"Walk-forward not runnable on this bundle: {e}"
                        )

    # Output directory: ephemeral tempdir unless user ticked Save
    _tempdir_key = "_dashboard_tempdir"
    existing_tempdir = st.session_state.get(_tempdir_key)
    if st.session_state["save_results"]:
        if existing_tempdir is not None:
            try:
                existing_tempdir.cleanup()
            except Exception:
                pass
            del st.session_state[_tempdir_key]
        output_dir = Path("output/dashboard_run")
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        if existing_tempdir is not None:
            try:
                existing_tempdir.cleanup()
            except Exception:
                pass
        new_tempdir = tempfile.TemporaryDirectory(prefix="pbe_dashboard_")
        st.session_state[_tempdir_key] = new_tempdir
        output_dir = Path(new_tempdir.name)

    # Cache everything the post-run tabs read
    st.session_state["last_run"] = {
        "portfolio": user_portfolio,
        "returns_dict": returns_dict,
        "stats_list": stats_list,
        "stats_df": stats_df,
        "mc_stats": mc_stats,
        "wealth_paths": wealth_paths,
        "main_returns": main_returns,
        "start_date": bundle_sliced.monthly_returns_eur.index[0],
        "end_date": bundle_sliced.monthly_returns_eur.index[-1],
        "output_dir": output_dir,
        "start_nav": st.session_state["start_nav"],
        "mc_years": int(st.session_state["mc_years"]),
        # PR5 — advanced analyses (None when toggle was off)
        "frontier_result": frontier_result,
        "fire_result": fire_result,
        "fire_error": fire_error,
        "walkforward_result": walkforward_result,
        "walkforward_skipped_reason": walkforward_skipped_reason,
    }

    # Save full report if requested
    if st.session_state["save_results"]:
        generate_markdown_report(
            output_dir=output_dir,
            returns_dict=returns_dict,
            stats_list=stats_list,
            start_date=bundle_sliced.monthly_returns_eur.index[0],
            end_date=bundle_sliced.monthly_returns_eur.index[-1],
            start_nav=st.session_state["start_nav"],
            options_enabled=user_portfolio.options_overlay,
            monte_carlo_stats=mc_stats,
            crisis_drawdown_fn=crisis_drawdown,
        )


# ===========================================================================
# Post-run tabs — read from session_state["last_run"]
# ===========================================================================

last_run = st.session_state.get("last_run")

with tab_summary:
    if last_run is None:
        st.info("👈 Configure the portfolio and press **▶ Run backtest** in the Settings tab.")
    else:
        st.subheader("Summary statistics")
        st.dataframe(
            last_run["stats_df"].style.format({
                "cagr": "{:.2%}", "annualized_vol": "{:.2%}",
                "max_drawdown": "{:.2%}", "average_drawdown": "{:.2%}",
                "cvar_5pct": "{:.2%}", "best_month": "{:.2%}", "worst_month": "{:.2%}",
                "pct_positive_months": "{:.1%}", "total_return": "{:.1%}",
                "sharpe": "{:.2f}", "sortino": "{:.2f}",
                "calmar": "{:.2f}", "ulcer_index": "{:.1f}", "upi": "{:.2f}",
            }),
            use_container_width=True,
        )

        st.subheader("Crisis period drawdowns")
        crisis_rows = []
        for name, r in last_run["returns_dict"].items():
            crisis_rows.append({
                "Portfolio": name,
                "GFC 2008-2009": f"{crisis_drawdown(r, '2008-01-01', '2009-06-30'):.2%}",
                "COVID 2020": f"{crisis_drawdown(r, '2020-01-01', '2020-12-31'):.2%}",
                "Stagflation 2022": f"{crisis_drawdown(r, '2022-01-01', '2022-12-31'):.2%}",
            })
        st.dataframe(
            pd.DataFrame(crisis_rows).set_index("Portfolio"),
            use_container_width=True,
        )

        # ---- PR6: save the portfolio that was just simulated --------------
        st.divider()
        st.subheader("💾 Save portfolio")
        with st.form("save_portfolio_form", clear_on_submit=False):
            default_name = last_run["portfolio"].name
            save_name = st.text_input(
                "Name", value=default_name,
                help="Human-readable name; the filename slug is derived automatically.",
            )
            save_notes = st.text_area(
                "Notes (optional)", value=last_run["portfolio"].notes,
                help="Free-form comment stored in the TOML 'notes' field.",
            )
            save_overwrite = st.checkbox(
                "Overwrite if a preset with the same slug already exists",
                value=False,
            )
            submitted = st.form_submit_button("💾 Save", type="primary")
        if submitted:
            try:
                slug = slugify(save_name)
                target = DEFAULT_PORTFOLIOS_DIR / f"{slug}.toml"
                # Build the cache from the main series at the moment of save
                main_name = (
                    last_run["portfolio"].name
                    if last_run["portfolio"].name in last_run["returns_dict"]
                    else f"{last_run['portfolio'].name} (no options)"
                )
                main_returns = last_run["returns_dict"][main_name]
                main_stats = next(
                    (s for s in last_run["stats_list"] if s.name == main_name),
                    last_run["stats_list"][0],
                )
                metrics_cache = PortfolioMetricsCache(
                    cagr=float(main_stats.cagr),
                    annualized_vol=float(main_stats.annualized_vol),
                    max_drawdown=float(main_stats.max_drawdown),
                    period_start=last_run["start_date"],
                    period_end=last_run["end_date"],
                    run_timestamp=datetime.utcnow().replace(microsecond=0),
                )
                to_save = dataclasses.replace(
                    last_run["portfolio"],
                    name=save_name,
                    notes=save_notes,
                    cached_metrics=metrics_cache,
                )
                written = to_save.save_to(target, overwrite=save_overwrite)
                st.success(f"Portfolio saved to `{written}`")
            except FileExistsError as e:
                st.error(
                    f"{e} Tick 'Overwrite' and retry, or pick a different name."
                )
            except ValueError as e:
                st.error(f"Unable to save: {e}")


with tab_charts:
    if last_run is None:
        st.info("No results yet. Run a backtest to see the charts.")
    else:
        output_dir = last_run["output_dir"]
        chart_specs = [
            ("Equity curve", plot_equity_curve, (last_run["returns_dict"], None, last_run["start_nav"])),
            ("Drawdown", plot_drawdown, (last_run["returns_dict"], None)),
            ("Underwater", plot_underwater, (last_run["returns_dict"], None)),
            ("Annual returns", plot_annual_returns, (last_run["returns_dict"], None)),
            ("Return distribution", plot_return_distribution, (last_run["returns_dict"], None)),
            ("Rolling Sharpe (3Y)", plot_rolling_sharpe, (last_run["returns_dict"], None, 36)),
            ("Rolling Returns (3Y)", plot_rolling_returns, (last_run["returns_dict"], None, 36)),
            ("Crisis zoom", plot_crisis_zoom, (last_run["returns_dict"], None)),
            ("Risk/return scatter", plot_risk_return_scatter, (last_run["stats_list"], None)),
            ("Metrics comparison", plot_metrics_comparison, (last_run["stats_list"], None)),
        ]
        for title, fn, args in chart_specs:
            st.subheader(title)
            img_path = output_dir / f"{title.replace(' ', '_').replace('/', '_')}.png"
            args_with_path = tuple(img_path if a is None else a for a in args)
            fn(*args_with_path)
            st.image(str(img_path), use_container_width=True)


with tab_mc:
    if last_run is None or last_run["mc_stats"] is None:
        st.info(
            "Monte Carlo not enabled. Turn it on in **Settings → 🎲 Advanced analyses** "
            "and run the backtest again."
        )
    else:
        st.subheader("Monte Carlo scenarios")
        sc = last_run["mc_stats"].scenarios
        start_nav = last_run["start_nav"]
        col1, col2, col3 = st.columns(3)
        col1.metric(
            "🔴 Conservative (10th)", f"€{sc.prudente_wealth * start_nav:,.0f}",
            f"{sc.prudente_cagr * 100:+.2f}% CAGR",
        )
        col2.metric(
            "🔵 Median (50th)", f"€{sc.mediana_wealth * start_nav:,.0f}",
            f"{sc.mediana_cagr * 100:+.2f}% CAGR",
        )
        col3.metric(
            "🟢 Optimistic (90th)", f"€{sc.ottimista_wealth * start_nav:,.0f}",
            f"{sc.ottimista_cagr * 100:+.2f}% CAGR",
        )

        output_dir = last_run["output_dir"]
        scenarios_img = output_dir / "mc_scenarios.png"
        plot_mc_scenarios(sc, scenarios_img, start_nav=start_nav)
        st.image(str(scenarios_img), use_container_width=True)

        fan_img = output_dir / "mc_fan.png"
        plot_monte_carlo_fan(
            last_run["wealth_paths"],
            pd.date_range(
                last_run["main_returns"].index[-1],
                periods=last_run["mc_years"] * 12 + 1,
                freq="ME",
            ),
            fan_img, start_nav=start_nav,
        )
        st.image(str(fan_img), use_container_width=True)


with tab_frontier:
    if last_run is None or last_run.get("frontier_result") is None:
        st.info(
            "Efficient Frontier not enabled. Turn it on in **Settings → "
            "🎲 Advanced analyses** and run the backtest again."
        )
    else:
        (
            random_eval_df, weights_matrix, key_portfolios,
            frontier_df, ref_point, asset_names,
        ) = last_run["frontier_result"]
        output_dir = last_run["output_dir"]

        st.subheader(f"Efficient Frontier — universe {last_run['portfolio'].name}")
        st.caption(
            f"{len(random_eval_df):,} random portfolios + Markowitz frontier, "
            f"on {len(asset_names)} assets (cash excluded)."
        )

        # Static PNG (matplotlib)
        frontier_png = output_dir / "efficient_frontier.png"
        plot_efficient_frontier(
            random_eval_df, key_portfolios, frontier_df, ref_point, frontier_png,
        )
        st.image(str(frontier_png), use_container_width=True)

        # Interactive HTML (Plotly) — inline if plotly is installed
        frontier_html = output_dir / "efficient_frontier_interactive.html"
        if plot_efficient_frontier_interactive(
            random_eval_df, weights_matrix, asset_names,
            key_portfolios, frontier_df, ref_point, frontier_html,
        ):
            with st.expander("Interactive Plotly chart (hover/click for allocations)"):
                st.components.v1.html(
                    frontier_html.read_text(encoding="utf-8"),
                    height=740, scrolling=True,
                )

        # Key portfolios table
        rows = []
        for key, point in key_portfolios.items():
            rows.append({
                "Portfolio": point.label,
                "Return (annualized)": f"{point.expected_return * 100:.2f}%",
                "Volatility": f"{point.volatility * 100:.2f}%",
                "Sharpe": f"{point.sharpe:.3f}",
            })
        if ref_point is not None:
            rows.append({
                "Portfolio": ref_point.label,
                "Return (annualized)": f"{ref_point.expected_return * 100:.2f}%",
                "Volatility": f"{ref_point.volatility * 100:.2f}%",
                "Sharpe": f"{ref_point.sharpe:.3f}",
            })
        st.subheader("Key portfolios on the frontier")
        st.dataframe(pd.DataFrame(rows).set_index("Portfolio"), use_container_width=True)


with tab_fire:
    if last_run is None or (last_run.get("fire_result") is None and not last_run.get("fire_error")):
        st.info(
            "FIRE calculator not enabled. Turn it on in **Settings → "
            "🎲 Advanced analyses** and configure the parameters."
        )
    elif last_run.get("fire_error"):
        st.error(f"FIRE configuration error: {last_run['fire_error']}")
    else:
        fire_sims, fire_summary, fire_config = last_run["fire_result"]
        output_dir = last_run["output_dir"]

        st.subheader(f"FIRE Monte Carlo — {fire_summary.n_simulations:,} simulations")

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Success probability", f"{fire_summary.probability_success:.1%}")
        m2.metric("Median survival age", f"{fire_summary.median_survival_age}")
        m3.metric("Median legacy (nominal)", f"€{fire_summary.median_legacy_nominal:,.0f}")
        m4.metric("Median legacy (real)", f"€{fire_summary.median_legacy_real:,.0f}")
        m5.metric(
            "Worst failure age",
            f"{fire_summary.worst_case_failure_age}" if fire_summary.worst_case_failure_age else "—",
        )

        # 4 charts — generate once to the output_dir and display
        proj_png = output_dir / "fire_projection.png"
        success_png = output_dir / "fire_success_probability.png"
        failure_png = output_dir / "fire_failure_distribution.png"
        legacy_png = output_dir / "fire_legacy_distribution.png"

        plot_fire_projection(fire_summary, fire_config, proj_png)
        plot_fire_success_probability(fire_sims, fire_config, success_png)
        plot_fire_failure_distribution(fire_sims, failure_png)
        plot_fire_legacy_distribution(fire_sims, legacy_png)

        st.subheader("Portfolio projection (percentile bands)")
        st.image(str(proj_png), use_container_width=True)

        st.subheader("Success probability by age")
        st.image(str(success_png), use_container_width=True)

        c_fail, c_leg = st.columns(2)
        with c_fail:
            st.subheader("Failure age distribution")
            st.image(str(failure_png), use_container_width=True)
        with c_leg:
            st.subheader("Legacy distribution (nominal vs real)")
            st.image(str(legacy_png), use_container_width=True)


with tab_walkforward:
    if last_run is None or (
        last_run.get("walkforward_result") is None
        and not last_run.get("walkforward_skipped_reason")
    ):
        st.info(
            "Walk-forward not enabled. Turn it on in **Settings → "
            "🎲 Advanced analyses** and run the backtest again."
        )
    elif last_run.get("walkforward_skipped_reason"):
        st.warning(f"⚠️ {last_run['walkforward_skipped_reason']}")
        st.caption(
            "The gate requires at least `window_years` of common history across "
            "all assets (cash excluded). Shrink the window or build a portfolio "
            "with longer-lived assets."
        )
    else:
        # Unpack the snapshot saved by the Run path — step_months is the
        # value that was in effect when the simulation ran, NOT the live
        # session_state (which the user could have tweaked since).
        wf_df, wf_stats, window_years, step_months = last_run["walkforward_result"]
        output_dir = last_run["output_dir"]

        st.subheader(f"Walk-forward {window_years}-year backtest")
        st.caption(
            f"{len(wf_df)} windows (step {step_months} months) across the configured period."
        )

        m1, m2, m3 = st.columns(3)
        m1.metric("CAGR median", f"{wf_stats['cagr_median']:.2%}",
                  f"worst {wf_stats['cagr_worst']:.2%}")
        m2.metric("Max DD median", f"{wf_stats['maxdd_median']:.2%}",
                  f"worst {wf_stats['maxdd_worst']:.2%}")
        m3.metric("Sharpe median", f"{wf_stats['sharpe_median']:.2f}",
                  f"worst {wf_stats['sharpe_worst']:.2f}")

        st.caption(
            f"{wf_stats['pct_windows_positive_cagr']:.1%} di windows con CAGR positivo · "
            f"{wf_stats['pct_windows_cagr_above_4pct']:.1%} con CAGR > 4%"
        )

        wf_png = output_dir / f"walkforward_{window_years}y.png"
        plot_rolling_window_results(
            wf_df, window_years, wf_png, step_months=step_months,
        )
        st.image(str(wf_png), use_container_width=True)

        with st.expander("Per-window raw results"):
            st.dataframe(wf_df, use_container_width=True)


with tab_ai:
    if last_run is None:
        st.info("Run a backtest first so the results can be sent to the LLM.")
    else:
        st.subheader("🤖 AI Analysis")
        st.caption("Default provider: **OpenRouter**. Requires OPENROUTER_API_KEY (see README / .env).")
        ai_provider = st.selectbox(
            "Provider", ["openrouter", "openai", "anthropic", "local"], index=0,
        )
        ai_model = st.text_input("Model (empty = provider default)", "")
        if st.button("🤖 Run AI analysis"):
            from src.ai_analyzer import build_analysis_prompt, get_analyzer, save_analysis

            portfolio_name = last_run["portfolio"].name
            prompt_text = f"""
Portfolio: {portfolio_name}
Period: {last_run['start_date'].date()} to {last_run['end_date'].date()}
Starting NAV: €{last_run['start_nav']:,.0f}

Summary statistics:
{last_run['stats_df'].to_markdown()}
"""
            prompt = build_analysis_prompt(prompt_text)
            try:
                with st.spinner(f"Querying {ai_provider}..."):
                    analyzer = get_analyzer(ai_provider, ai_model if ai_model else None)
                    response = analyzer.analyze(prompt)
                st.success(f"Analysis complete — {response.model}")
                st.markdown(response.content)
                if st.session_state["save_results"]:
                    save_analysis(response, last_run["output_dir"] / "AI_ANALYSIS.md")
                    st.caption(f"Saved to {last_run['output_dir'] / 'AI_ANALYSIS.md'}")
            except Exception as e:
                st.error(f"AI analysis failed: {e}")


with tab_saved:
    st.subheader("📂 Saved portfolios")
    st.caption(
        "Portfolios written to `portfolios/*.toml` by the 💾 Save portfolio "
        "button (or by `backtest.py --save-as NAME`). Presets shipped in the "
        "repo (marked 📌) cannot be deleted or overwritten from the UI — they "
        "are part of the codebase and change via PRs."
    )

    _entries = list_available_presets()
    if not _entries:
        st.info("No presets available. Save a portfolio from the Summary tab after a run.")
    else:
        for _e in _entries:
            _name = _e["name"]
            _display = _e["display_name"]
            _is_reserved = _e.get("is_reserved", False)
            _metrics = _e.get("cached_metrics")

            # Row header: name + reserved pin
            _header = f"{'📌 ' if _is_reserved else ''}**{_display}**  `({_name})`"
            with st.container(border=True):
                st.markdown(_header)
                if _e["notes"]:
                    st.caption(_e["notes"])

                # Metrics row
                if _metrics is not None:
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("CAGR", f"{_metrics.cagr * 100:.2f}%")
                    c2.metric("Vol", f"{_metrics.annualized_vol * 100:.2f}%")
                    c3.metric("Max DD", f"{_metrics.max_drawdown * 100:.2f}%")
                    _period_label = (
                        f"{_metrics.period_start.date().isoformat()[:7]} → "
                        f"{_metrics.period_end.date().isoformat()[:7]}"
                    )
                    c4.metric("Period", _period_label)
                else:
                    st.caption(
                        f"{_e['n_assets']} assets · no cached metrics "
                        "(no run has been saved against this preset)"
                    )

                # Actions row
                _act_load, _act_del, _pad = st.columns([1, 1, 4])
                with _act_load:
                    if st.button("▶ Load", key=f"_load_{_name}", use_container_width=True):
                        try:
                            _loaded = Portfolio.from_toml(_e["path"])
                        except Exception as exc:
                            st.error(f"Unable to load {_name!r}: {exc}")
                        else:
                            # Repopulate the Settings state
                            st.session_state["portfolio_name"] = _loaded.name
                            st.session_state["portfolio_assets"] = [
                                {"key": a.key, "weight": a.weight}
                                for a in _loaded.assets
                            ]
                            st.session_state["options_overlay"] = _loaded.options_overlay
                            st.session_state["transaction_cost_bps"] = _loaded.transaction_cost_bps
                            # Map the preset's rebalance_months tuple back to
                            # one of the 4 UI labels. If the preset uses a
                            # custom cadence (valid in the Portfolio model but
                            # not selectable from the radio), fall back to the
                            # first UI option and warn the user — otherwise the
                            # Settings tab would silently keep the PREVIOUS
                            # selection, masking the mismatch.
                            _matched_label = next(
                                (
                                    _label
                                    for _label, _months in _REBALANCE_FREQ_TO_MONTHS.items()
                                    if _months == _loaded.rebalance_months
                                ),
                                None,
                            )
                            if _matched_label is None:
                                _fallback_label = next(iter(_REBALANCE_FREQ_TO_MONTHS))
                                st.session_state["rebalance_freq"] = _fallback_label
                                st.warning(
                                    f"Preset uses rebalance_months={_loaded.rebalance_months} "
                                    "— not in the UI radio options "
                                    f"({list(_REBALANCE_FREQ_TO_MONTHS.keys())}). "
                                    f"Defaulted to {_fallback_label!r}; "
                                    "edit manually if needed."
                                )
                            else:
                                st.session_state["rebalance_freq"] = _matched_label
                            st.success(
                                f"Loaded {_loaded.name!r}. Go to the ⚙️ Settings tab and press "
                                "▶ Run backtest to re-run."
                            )
                            st.rerun()

                with _act_del:
                    if _is_reserved:
                        st.button(
                            "🔒 Shipped", key=f"_del_{_name}",
                            disabled=True, use_container_width=True,
                            help="Preset shipped in the repo: cannot be deleted from the UI.",
                        )
                    else:
                        # Two-click delete: the first click arms a confirm
                        # button in session_state; the second click actually
                        # removes the file. Prevents accidental deletions.
                        _confirm_key = f"_confirm_del_{_name}"
                        if st.session_state.get(_confirm_key):
                            if st.button(
                                "⚠️ Confirm delete", key=f"_del_{_name}",
                                use_container_width=True, type="primary",
                            ):
                                try:
                                    Path(_e["path"]).unlink()
                                    st.success(f"Deleted {_name!r}.")
                                except Exception as exc:
                                    st.error(f"Unable to delete: {exc}")
                                finally:
                                    st.session_state.pop(_confirm_key, None)
                                st.rerun()
                        else:
                            if st.button(
                                "🗑️ Delete", key=f"_del_{_name}",
                                use_container_width=True,
                            ):
                                st.session_state[_confirm_key] = True
                                st.rerun()
