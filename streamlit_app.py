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
  - First tab "⚙️ Impostazioni" with 4 sections (Portafoglio, Periodo,
    Motore, Analisi avanzate) — assembles a Portfolio in session_state
    without ever mutating module globals.
  - Post-run tabs appear when a run completes: Summary, Charts, Monte
    Carlo, AI Analysis. Frontier / FIRE / Walk-forward are placeholder
    toggles — the integration lands in PR5.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Dict, List

# Load OPENROUTER_API_KEY / OPENAI_API_KEY / ANTHROPIC_API_KEY / LOCAL_API_BASE_URL
# from a `.env` at the project root before anything reads os.environ. Real env
# vars and Streamlit Cloud secrets still take precedence (override=False).
from src.env_check import load_dotenv  # stdlib-only, safe to import first

load_dotenv()

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src.dashboard_helpers import (
    build_portfolio_from_ui_state,
    compute_effective_start,
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
from src.metrics import compute_all, crisis_drawdown, cumulative_wealth, stats_to_dataframe
from src.monte_carlo import block_bootstrap, compute_mc_stats, simulate_wealth_paths
from src.options_overlay import simulate_options_overlay
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
from src.portfolio import BENCHMARKS
from src.portfolio_model import Portfolio, list_available_presets
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
    """Initialize the session-state keys used by the Impostazioni tab so
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

# Tab labels built dynamically: Impostazioni first, then the post-run tabs
# that are always "visible" (empty info panel if no run yet).
tab_settings, tab_summary, tab_charts, tab_mc, tab_ai = st.tabs(
    ["⚙️ Impostazioni", "📊 Summary", "📈 Charts", "🎲 Monte Carlo", "🤖 AI Analysis"]
)


# ===========================================================================
# Tab: Impostazioni
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
        picked = st.selectbox("Carica preset", preset_options, index=0)
        if st.button("📂 Carica", disabled=(picked == preset_options[0])):
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
            "Nome portafoglio", value=st.session_state["portfolio_name"],
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

    # ------ 🎯 Portafoglio ---------------------------------------------------
    st.subheader("🎯 Portafoglio")

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
            "Alcuni asset nel portafoglio non sono simulabili con il dataset "
            "corrente (mancano in `bundle.monthly_returns_eur`). Rimuovili "
            f"prima di lanciare: {', '.join(invalid_portfolio_keys)}"
        )

    pick_col, weight_col, btn_col = st.columns([6, 2, 2])
    with pick_col:
        picked_key = st.selectbox(
            "Aggiungi asset",
            options=available_asset_keys,
            format_func=lambda k: key_to_display[k],
            key="_asset_picker",
        )
    with weight_col:
        picked_weight_pct = st.number_input(
            "Peso %", min_value=0.0, max_value=100.0, value=10.0, step=0.5,
            key="_asset_picker_weight",
        )
    with btn_col:
        st.write("")  # vertical alignment
        st.write("")
        if st.button("＋ Aggiungi", use_container_width=True):
            existing_keys = {a["key"] for a in st.session_state["portfolio_assets"]}
            if picked_key in existing_keys:
                st.warning(f"{picked_key!r} è già nel portafoglio; modifica il peso sulla riga esistente.")
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
            "Dati dal": format_asset_start_date(info),
            "Peso %": round(a["weight"] * 100, 4),
        })

    if rows:
        edited = st.data_editor(
            pd.DataFrame(rows),
            hide_index=True,
            use_container_width=True,
            disabled=["Asset", "Key", "Dati dal"],
            column_config={
                "Peso %": st.column_config.NumberColumn(
                    "Peso %", min_value=0.0, max_value=100.0, step=0.5, format="%.2f",
                ),
            },
            key="_portfolio_editor",
        )
        # Reflect user edits back into session_state (before the Delete button
        # handler reads it).
        st.session_state["portfolio_assets"] = [
            {"key": row["Key"], "weight": float(row["Peso %"]) / 100.0}
            for _, row in edited.iterrows()
        ]

        # Per-row delete: one small button per asset. Dense layout.
        del_cols = st.columns(max(len(rows), 1))
        for i, (col, row) in enumerate(zip(del_cols, rows)):
            with col:
                if st.button(f"🗑️ {row['Key']}", key=f"_del_{row['Key']}"):
                    st.session_state["portfolio_assets"].pop(i)
                    st.rerun()
    else:
        st.info("Nessun asset selezionato. Usa il picker qui sopra o carica un preset.")

    # Sum-of-weights badge
    total_weight = sum(a["weight"] for a in st.session_state["portfolio_assets"])
    total_pct = total_weight * 100
    sum_ok = abs(total_weight - 1.0) <= 0.002
    if sum_ok and st.session_state["portfolio_assets"]:
        st.markdown(f"**Totale:** 🟢 {total_pct:.2f}%")
    elif st.session_state["portfolio_assets"]:
        missing = 100.0 - total_pct
        st.markdown(
            f"**Totale:** 🔴 {total_pct:.2f}%  "
            f"_(manca {missing:+.2f}% — deve essere 100% per poter lanciare)_"
        )
    else:
        st.markdown("**Totale:** ⚪ 0.00%")

    st.divider()

    # ------ 📅 Periodo -------------------------------------------------------
    st.subheader("📅 Periodo")
    date_range = st.date_input(
        "Intervallo di analisi",
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

    # ------ ⚙️ Motore --------------------------------------------------------
    st.subheader("⚙️ Motore")

    nav_col, opt_col, reb_col, tc_col = st.columns([3, 3, 3, 2])
    with nav_col:
        st.session_state["start_nav"] = st.number_input(
            "NAV iniziale (€)", min_value=1000.0, step=10_000.0,
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
            "Ribilanciamento",
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

    # ------ 🎲 Analisi avanzate ---------------------------------------------
    st.subheader("🎲 Analisi avanzate")

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
            "Horizon (anni)", min_value=1, max_value=40,
            value=int(st.session_state["mc_years"]),
            disabled=not st.session_state["mc_enabled"],
        )
    with mc_col3:
        st.session_state["mc_block_size"] = st.number_input(
            "Block (mesi)", min_value=1, max_value=24,
            value=int(st.session_state["mc_block_size"]),
            disabled=not st.session_state["mc_enabled"],
        )

    # Placeholder toggles — PR5 wires the corresponding tabs
    st.checkbox("Efficient Frontier 🔒 (PR5)", value=False, disabled=True)
    st.checkbox("FIRE calculator 🔒 (PR5)", value=False, disabled=True)
    st.checkbox("Simulazione storica (walk-forward) 🔒 (PR5)", value=False, disabled=True)

    # Benchmarks
    st.session_state["selected_benchmarks"] = st.multiselect(
        "Benchmarks di confronto",
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
# Run the backtest (triggered from the Impostazioni tab)
# ===========================================================================

if run_btn:
    # Build the Portfolio from session_state. If validation fails, surface
    # the error in the Impostazioni tab and don't proceed.
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
    # contradicting the warning we showed in the Impostazioni tab
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
        st.info("👈 Configura il portafoglio e premi **▶ Run backtest** nel tab Impostazioni.")
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


with tab_charts:
    if last_run is None:
        st.info("Nessun risultato ancora. Lancia un backtest per vedere i grafici.")
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
            "Monte Carlo non attivato. Abilitalo nel tab **Impostazioni → 🎲 Analisi avanzate** "
            "e rilancia il backtest."
        )
    else:
        st.subheader("Monte Carlo scenarios")
        sc = last_run["mc_stats"].scenarios
        start_nav = last_run["start_nav"]
        col1, col2, col3 = st.columns(3)
        col1.metric(
            "🔴 Prudente (10°)", f"€{sc.prudente_wealth * start_nav:,.0f}",
            f"{sc.prudente_cagr * 100:+.2f}% CAGR",
        )
        col2.metric(
            "🔵 Mediana (50°)", f"€{sc.mediana_wealth * start_nav:,.0f}",
            f"{sc.mediana_cagr * 100:+.2f}% CAGR",
        )
        col3.metric(
            "🟢 Ottimista (90°)", f"€{sc.ottimista_wealth * start_nav:,.0f}",
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


with tab_ai:
    if last_run is None:
        st.info("Lancia prima un backtest per poter inviare i risultati all'LLM.")
    else:
        st.subheader("🤖 AI Analysis")
        st.caption("Default provider: **OpenRouter**. Richiede OPENROUTER_API_KEY (vedi README / .env).")
        ai_provider = st.selectbox(
            "Provider", ["openrouter", "openai", "anthropic", "local"], index=0,
        )
        ai_model = st.text_input("Model (vuoto = default per provider)", "")
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
