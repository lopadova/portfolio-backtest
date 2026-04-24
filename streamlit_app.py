"""
Streamlit dashboard — interactive Four Umbrellas Portfolio backtest UI.

Run locally:
    pip install -r requirements-dashboard.txt
    streamlit run streamlit_app.py

Then open http://localhost:8501 in your browser.

Deploy to Streamlit Cloud: push to GitHub, link your repo at share.streamlit.io.
Deploy to HuggingFace Spaces: add a Dockerfile (see README) and push.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

# Load OPENROUTER_API_KEY / OPENAI_API_KEY / ANTHROPIC_API_KEY / LOCAL_API_BASE_URL
# from a `.env` at the project root before anything reads os.environ. Real env
# vars and Streamlit Cloud secrets still take precedence (override=False).
from src.env_check import load_dotenv  # stdlib-only, safe to import first

load_dotenv()

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src import portfolio as portfolio_cfg
from src.data_loader import load_data
from src.rebalance import simulate_portfolio, simulate_benchmark
from src.metrics import compute_all, stats_to_dataframe, crisis_drawdown
from src.monte_carlo import block_bootstrap, simulate_wealth_paths, compute_mc_stats
from src.options_overlay import simulate_options_overlay
from src.plots import (
    plot_equity_curve, plot_drawdown, plot_underwater, plot_rolling_sharpe,
    plot_rolling_returns, plot_crisis_zoom, plot_annual_returns,
    plot_return_distribution, plot_risk_return_scatter, plot_metrics_comparison,
    plot_correlation_heatmap, plot_monte_carlo_fan, plot_mc_distribution,
    plot_mc_scenarios,
)
from src.report import generate_markdown_report


# ===========================================================================
# Page config
# ===========================================================================
st.set_page_config(
    page_title="Four Umbrellas Portfolio",
    page_icon="☂",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ===========================================================================
# Helpers — imported from dashboard_helpers so tests exercise the REAL code.
# (Previously duplicated inside this file; Copilot review pointed out the
# risk of silent drift between prod and test. The shared module is
# Streamlit-free so importing it in pytest doesn't require Streamlit.)
# ===========================================================================
from src.dashboard_helpers import snapshot_config, restore_config, apply_macro_weights


@st.cache_data(show_spinner=False)
def load_data_cached(synthetic: bool):
    return load_data(synthetic=synthetic)


# ===========================================================================
# Sidebar — Configuration
# ===========================================================================
st.sidebar.title("☂ Four Umbrellas Portfolio")
st.sidebar.caption("Interactive backtest dashboard")

data_mode = st.sidebar.radio(
    "Data source",
    ["Real data (data/raw/)", "Synthetic (demo)"],
    index=1,
    help="Use synthetic data if you haven't populated data/raw/ yet.",
)
synthetic = data_mode == "Synthetic (demo)"

# Load bundle once
try:
    bundle = load_data_cached(synthetic)
except Exception as e:
    st.sidebar.error(f"Failed to load data: {e}")
    st.stop()

# Date range
data_start = bundle.monthly_returns_eur.index[0]
data_end = bundle.monthly_returns_eur.index[-1]
date_range = st.sidebar.date_input(
    "Date range",
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

# NAV
nav = st.sidebar.number_input("Starting NAV (€)", min_value=1000, value=100_000, step=10_000)

# Portfolio weights
st.sidebar.subheader("Portfolio weights (% NAV)")
gold_weight = st.sidebar.slider("Gold (%)", 5.0, 30.0, 18.25, step=0.25)
dbi_weight = st.sidebar.slider("DBi Managed Futures (%)", 0.0, 15.0, 5.0, step=0.5)

# Features
st.sidebar.subheader("Features")
options_enabled = st.sidebar.checkbox("Options overlay", value=True)
options_budget = st.sidebar.slider(
    "Options budget (% NAV/year)", 0.0, 1.0, 0.30, step=0.05, disabled=not options_enabled,
) / 100

# Monte Carlo
st.sidebar.subheader("Monte Carlo")
mc_enabled = st.sidebar.checkbox("Run Monte Carlo", value=True)
mc_paths = st.sidebar.number_input("MC paths", 100, 50_000, 5000, step=500, disabled=not mc_enabled)
mc_years = st.sidebar.number_input("MC horizon (years)", 1, 40, 20, disabled=not mc_enabled)

# Benchmarks selection
st.sidebar.subheader("Benchmarks")
selected_benchmarks = st.sidebar.multiselect(
    "Compare against",
    options=list(portfolio_cfg.BENCHMARKS.keys()),
    default=["60/40", "100% SWDA (MSCI World EUR)"],
)

# Run + Save buttons
st.sidebar.subheader("Actions")
col_run, col_save = st.sidebar.columns(2)
run_btn = col_run.button("▶ Run backtest", type="primary", use_container_width=True)
save_results = col_save.checkbox("💾 Save to output/", value=False,
                                  help="If checked, results are written to output/dashboard_run/")

# ===========================================================================
# Main area
# ===========================================================================
st.title("☂ Four Umbrellas Portfolio — Interactive Backtest")
st.markdown(
    """
    > A growth-oriented global portfolio with **four layered defensive protections**:
    > gold, defensive put-write, managed futures, options overlay.
    > [Read the article](https://medium.com/@padosoft) · [GitHub](https://github.com/padosoft/portfolio-backtest) ·
    > **Disclaimer**: educational use only, not investment advice.
    """
)

if run_btn:
    # Apply config overrides. Use the shared helper `apply_macro_weights`
    # which atomically updates gold + dbi and derives cash = 1 - Σ(non-cash)
    # so the WEIGHTS sum stays exactly 1.0 (otherwise the rebalance engine's
    # build_target_weights() assertion fires). Previous code had a double bug:
    # (a) cash computed BEFORE dbi was set, (b) hardcoded 0.1825 gold offset
    # that was wrong after changing the slider.
    snapshot = snapshot_config()
    try:
        apply_macro_weights(gold_pct=gold_weight / 100, dbi_pct=dbi_weight / 100)
        portfolio_cfg.OPTIONS.enabled = options_enabled
        portfolio_cfg.OPTIONS.budget_nav_per_year = options_budget

        bundle_sliced = bundle.slice(start_date, end_date)

        with st.spinner("Running backtest..."):
            # Four Umbrellas (no options)
            base_returns = simulate_portfolio(
                bundle_sliced.monthly_returns_eur,
                bundle_sliced.btc_activation_date,
                apply_ter=True,
            )
            returns_dict = {"Four Umbrellas (no options)": base_returns}

            # With options overlay if enabled
            if options_enabled:
                from src.metrics import cumulative_wealth
                nav_series = cumulative_wealth(base_returns, nav)
                overlay_returns = simulate_options_overlay(
                    spy_daily=bundle_sliced.spy_daily,
                    qqq_daily=bundle_sliced.qqq_daily,
                    vix_daily=bundle_sliced.vix_daily,
                    rf_daily=bundle_sliced.rf_daily,
                    nav_series=nav_series,
                )
                overlay_aligned = overlay_returns.reindex(base_returns.index).fillna(0.0)
                returns_dict["Four Umbrellas"] = base_returns + overlay_aligned

            # Benchmarks
            for bench_name in selected_benchmarks:
                bench_weights = portfolio_cfg.BENCHMARKS[bench_name]
                bench_r = simulate_benchmark(bundle_sliced.benchmark_monthly_eur, bench_weights)
                if len(bench_r) > 0:
                    returns_dict[bench_name] = bench_r

            # Compute stats
            stats_list = [compute_all(n, r) for n, r in returns_dict.items()]
            stats_df = stats_to_dataframe(stats_list)

            # Monte Carlo
            mc_stats = None
            if mc_enabled:
                main_returns = returns_dict.get("Four Umbrellas", base_returns)
                sim_returns = block_bootstrap(main_returns, n_paths=mc_paths, n_periods=mc_years * 12)
                wealth_paths = simulate_wealth_paths(sim_returns)
                mc_stats = compute_mc_stats(wealth_paths, n_months=mc_years * 12)

        # Output directory.
        # When save_results=True: persist to output/dashboard_run/ on disk.
        # When save_results=False: use a TemporaryDirectory stored in
        # st.session_state that is cleaned up on the NEXT run (ensuring
        # only one ephemeral dir exists at a time — prevents disk bloat
        # from tempfile.mkdtemp() accumulating across Streamlit reruns).
        _tempdir_key = "_dashboard_tempdir"
        existing_tempdir = st.session_state.get(_tempdir_key)
        if save_results:
            # Clean up any leftover session-only tempdir (we won't need it now)
            if existing_tempdir is not None:
                try:
                    existing_tempdir.cleanup()
                except Exception:
                    pass  # best-effort; avoid blocking the run
                del st.session_state[_tempdir_key]
            output_dir = Path("output/dashboard_run")
            output_dir.mkdir(parents=True, exist_ok=True)
        else:
            # Clean up the previous session tempdir BEFORE creating a new one
            if existing_tempdir is not None:
                try:
                    existing_tempdir.cleanup()
                except Exception:
                    pass
            new_tempdir = tempfile.TemporaryDirectory(prefix="fourumbrellas_")
            st.session_state[_tempdir_key] = new_tempdir
            output_dir = Path(new_tempdir.name)

        # Tabs
        tab_summary, tab_charts, tab_mc, tab_ai = st.tabs(
            ["📊 Summary", "📈 Charts", "🎲 Monte Carlo", "🤖 AI Analysis"]
        )

        with tab_summary:
            st.subheader("Summary statistics")
            st.dataframe(stats_df.style.format({
                "cagr": "{:.2%}", "annualized_vol": "{:.2%}",
                "max_drawdown": "{:.2%}", "average_drawdown": "{:.2%}",
                "cvar_5pct": "{:.2%}", "best_month": "{:.2%}", "worst_month": "{:.2%}",
                "pct_positive_months": "{:.1%}", "total_return": "{:.1%}",
                "sharpe": "{:.2f}", "sortino": "{:.2f}",
                "calmar": "{:.2f}", "ulcer_index": "{:.1f}", "upi": "{:.2f}",
            }), use_container_width=True)

            st.subheader("Crisis period drawdowns")
            crisis_data = []
            for name, r in returns_dict.items():
                crisis_data.append({
                    "Portfolio": name,
                    "GFC 2008-2009": f"{crisis_drawdown(r, '2008-01-01', '2009-06-30'):.2%}",
                    "COVID 2020": f"{crisis_drawdown(r, '2020-01-01', '2020-12-31'):.2%}",
                    "Stagflation 2022": f"{crisis_drawdown(r, '2022-01-01', '2022-12-31'):.2%}",
                })
            st.dataframe(pd.DataFrame(crisis_data).set_index("Portfolio"), use_container_width=True)

        with tab_charts:
            chart_fns = [
                ("Equity curve", plot_equity_curve, (returns_dict, None, nav)),
                ("Drawdown", plot_drawdown, (returns_dict, None)),
                ("Underwater", plot_underwater, (returns_dict, None)),
                ("Annual returns", plot_annual_returns, (returns_dict, None)),
                ("Return distribution", plot_return_distribution, (returns_dict, None)),
                ("Rolling Sharpe (3Y)", plot_rolling_sharpe, (returns_dict, None, 36)),
                ("Rolling Returns (3Y)", plot_rolling_returns, (returns_dict, None, 36)),
                ("Crisis zoom", plot_crisis_zoom, (returns_dict, None)),
                ("Risk/return scatter", plot_risk_return_scatter, (stats_list, None)),
                ("Metrics comparison", plot_metrics_comparison, (stats_list, None)),
            ]
            for title, fn, args in chart_fns:
                st.subheader(title)
                img_path = output_dir / f"{title.replace(' ', '_').replace('/', '_')}.png"
                # Replace None with img_path
                args_with_path = tuple(img_path if a is None else a for a in args)
                fn(*args_with_path)
                st.image(str(img_path), use_container_width=True)

        with tab_mc:
            if mc_stats is not None:
                st.subheader("Monte Carlo scenarios")
                sc = mc_stats.scenarios
                col1, col2, col3 = st.columns(3)
                col1.metric("🔴 Prudente (10°)", f"€{sc.prudente_wealth * nav:,.0f}",
                            f"{sc.prudente_cagr * 100:+.2f}% CAGR")
                col2.metric("🔵 Mediana (50°)", f"€{sc.mediana_wealth * nav:,.0f}",
                            f"{sc.mediana_cagr * 100:+.2f}% CAGR")
                col3.metric("🟢 Ottimista (90°)", f"€{sc.ottimista_wealth * nav:,.0f}",
                            f"{sc.ottimista_cagr * 100:+.2f}% CAGR")

                scenarios_img = output_dir / "mc_scenarios.png"
                plot_mc_scenarios(sc, scenarios_img, start_nav=nav)
                st.image(str(scenarios_img), use_container_width=True)

                fan_img = output_dir / "mc_fan.png"
                plot_monte_carlo_fan(wealth_paths,
                                     pd.date_range(main_returns.index[-1], periods=mc_years * 12 + 1, freq="ME"),
                                     fan_img, start_nav=nav)
                st.image(str(fan_img), use_container_width=True)
            else:
                st.info("Enable Monte Carlo in the sidebar to see results here.")

        with tab_ai:
            st.subheader("🤖 AI Analysis")
            st.caption("Default provider: **OpenRouter**. Requires OPENROUTER_API_KEY in your environment.")
            ai_provider = st.selectbox("Provider",
                                       ["openrouter", "openai", "anthropic", "local"], index=0)
            ai_model = st.text_input("Model (leave empty for default)", "")
            if st.button("🤖 Run AI analysis"):
                from src.ai_analyzer import get_analyzer, build_analysis_prompt, save_analysis

                # Generate a minimal prompt from the stats
                prompt_text = f"""
Portfolio: Four Umbrellas
Period: {start_date.date()} to {end_date.date()}
Starting NAV: €{nav:,.0f}

Summary statistics:
{stats_df.to_markdown()}
"""
                prompt = build_analysis_prompt(prompt_text)
                try:
                    with st.spinner(f"Querying {ai_provider}..."):
                        analyzer = get_analyzer(ai_provider, ai_model if ai_model else None)
                        response = analyzer.analyze(prompt)
                    st.success(f"Analysis complete — {response.model}")
                    st.markdown(response.content)
                    if save_results:
                        save_analysis(response, output_dir / "AI_ANALYSIS.md")
                        st.caption(f"Saved to {output_dir / 'AI_ANALYSIS.md'}")
                except Exception as e:
                    st.error(f"AI analysis failed: {e}")

        # Save report if requested
        if save_results:
            report_path = generate_markdown_report(
                output_dir=output_dir,
                returns_dict=returns_dict,
                stats_list=stats_list,
                start_date=bundle_sliced.monthly_returns_eur.index[0],
                end_date=bundle_sliced.monthly_returns_eur.index[-1],
                start_nav=nav,
                options_enabled=options_enabled,
                monte_carlo_stats=mc_stats,
                crisis_drawdown_fn=crisis_drawdown,
            )
            st.sidebar.success(f"💾 Saved to {output_dir}")

    finally:
        restore_config(snapshot)

else:
    st.info("👈 Configure parameters in the sidebar and click **▶ Run backtest**.")
