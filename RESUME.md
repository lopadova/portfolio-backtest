# RESUME — Portfolio Backtest v2.0 feature rollout

**Purpose**: crash-safe progress tracking for a multi-phase feature implementation. If the session dies, a new agent reads this file and picks up from the first ⬜ (or 🟡 in-progress) phase.

**Project path**: `<repo-root>` (the directory where this repo is cloned — substitute your own path)

**Remote**: `git@github.com:lopadova/portfolio-backtest.git`

**Branching strategy**: each phase on its own branch, stacked on the previous.
- `main` → base
- `feat/phase-1-metrics` branches from `main`
- `feat/phase-2-mc-scenarios` branches from `feat/phase-1-metrics`
- …and so on.
- Each branch = 1 PR. User reviews and merges sequentially.

**User's decisions on open questions** (from the planning turn):
1. AI Analysis default provider: **OpenRouter**
2. Streamlit deployment: **local default + configs for Streamlit Cloud & HuggingFace Spaces**
3. Efficient Frontier random portfolios: **50k**
4. Tax: pension **exempt during accumulation** (taxed at exit, modeled elsewhere)
5. Rolling window step: **configurable, default monthly**
6. MC scenarios percentiles: **10 / 50 / 90** (Prudente / Mediana / Ottimista)
7. Dashboard save: **optional, "💾 save" button saves to `output/`, otherwise session-only**
8. Execution: **PR per phase, stacked**
9. Extra benchmark data missing: **skip with warning, no crash**
10. **NEW feature: FIRE calculator** (full spec in Phase 10 below)

---

## Phase status

Legend: ⬜ not started · 🟡 in progress · ✅ merged to main · 📤 pushed/PR open · ❌ blocked

| # | Phase | Branch | Status | Commit / Notes |
|---|---|---|---|---|
| 1 | Metrics++ (Avg DD, DD duration, UPI) | `feat/phase-1-metrics` | ✅ | Merged — PR #1 (incl. Copilot fixes, 126 tests) |
| 2 | MC scenarios (Prudente/Mediana/Ottimista, N-years flag) | `feat/phase-2-mc-scenarios` | ✅ | Merged — PR #2 (incl. Copilot fixes, 133 tests) |
| 3 | Additional benchmarks (Golden Butterfly, Permanent, Swensen, Dalio AW official) | `feat/phase-3-benchmarks` | ✅ | Merged — PR #3 (incl. Copilot fixes, 142 tests) |
| 4 | Programmatic sensitivity sweep | `feat/phase-4-sensitivity` | ✅ | Merged — PR #4 (incl. Copilot fixes, 162 tests) |
| 5 | Rolling-window backtest | `feat/phase-5-rolling-window` | ✅ | Merged — PRs #5 + #6 (incl. Copilot fixes, 180 tests) |
| 6 | Efficient Frontier (Markowitz + 50k random + interactive Plotly) | `feat/phase-6-efficient-frontier` | ✅ | Merged — PR #7 (incl. Copilot fixes + interactive chart, 193 tests) |
| 7 | Italian tax modeling (26% CGT + 4y carry) | `feat/phase-7-tax-it` | ✅ | Merged — PR #8 (incl. Copilot fixes, 207 tests) |
| 8 | FIRE calculator (2-phase + Italian mortality tables + pension + Black Swan) | `feat/phase-8-fire` | 📤 | Pushed — rebased on main |
| 9 | AI analysis (OpenAI/Anthropic/OpenRouter/Local, default=OpenRouter) | `feat/phase-9-ai-analysis` | ⬜ | Branch from phase-8 |
| 10 | Streamlit dashboard (form + run + save + deploy config for Cloud/HF) | `feat/phase-10-dashboard` | ⬜ | Branch from phase-9 |
| 11 | README + CI polish + final integration | `feat/phase-11-docs-ci` | ⬜ | Branch from phase-10 |

---

## Resume protocol — instructions for the next agent

If this file is being read because the previous session crashed:

1. `cd <repo-root>` (for example, `cd portfolio-backtest`)
2. Identify the first 🟡 or ⬜ phase in the table above.
3. `git checkout <branch-name>` (or create it from the previous phase's branch if it doesn't exist yet)
4. `git status` to see uncommitted work — resume that first if any
5. Continue implementation following the phase's spec below
6. After finishing a phase: commit, push, update this RESUME.md status to ✅ or 📤, then move to the next phase

---

## Phase specifications (detailed)

### Phase 1 — Metrics++

**Goal**: add 3 metrics to `PortfolioStats` and surface them in reports/CSV.

**Files to modify**:
- `src/metrics.py` — add `average_drawdown()`, `max_drawdown_duration_months()`, `ulcer_performance_index()`. Add fields to `PortfolioStats`. Update `compute_all()`.
- `src/report.py` — update `_stats_table_markdown()` to include new columns.
- `backtest.py` — no change (automatic via PortfolioStats.to_dict()).
- `tests/test_metrics.py` — add 8 tests (known cases for avg DD, DD duration, UPI).

**Acceptance criteria**:
- All 114 existing tests pass + 8 new tests
- `output/REPORT.md` summary table includes Avg DD, DD Duration, UPI columns
- `output/summary_statistics.csv` includes same columns

**Current status**: complete — metrics.py, report.py, and tests all done; acceptance criteria met (124 tests passing including 10 new).

---

### Phase 2 — MC scenarios (Prudente / Mediana / Ottimista)

**Spec**:
- New `MonteCarloScenarios` dataclass in `src/monte_carlo.py`:
  - `prudente_terminal_eur, mediana_terminal_eur, ottimista_terminal_eur`
  - `prudente_cagr, mediana_cagr, ottimista_cagr`
  - Computed at 10 / 50 / 90 percentiles
- New chart `plot_mc_scenarios()` — 3 bars with €/% labels, colored green/blue/red
- CLI flag: `--mc-years N` already exists; default should be `max available years from data` (compute at runtime)
- Report section: expand MC block with the 3 scenario metrics

**Files**: `src/monte_carlo.py`, `src/plots.py`, `src/report.py`, `backtest.py` (CLI), `tests/test_monte_carlo.py` (+4).

---

### Phase 3 — Additional benchmarks

**Golden Butterfly**: 20/20/20/20/20 US large + US SCV + LT Treas + ST Treas + Gold
**Harry Browne Permanent**: 25/25/25/25 US stocks + LT Treas + Cash (≈ST Treas) + Gold
**Dalio All-Weather official**: 30% equity + 40% LT Treas + 15% IT Treas + 7.5% Gold + 7.5% Commodities
**Swensen Lazy**: 30% US eq + 15% intl dev + 5% EM + 15% REITs + 15% LT Treas + 15% TIPS

**Data needed**: SCV proxy, EM index, REITs index, TIPS index. Skip benchmark if data missing (log warning).

**Files**: `src/portfolio.py` (BENCHMARKS dict), `data/README.md` (new csv entries), `src/data_loader.py` (add new bench series to `bench_files`), `tests/test_portfolio.py` (+4 tests).

---

### Phase 4 — Programmatic sensitivity sweep

**CLI**:
```
python backtest.py --sensitivity gold --range 0.10 0.25 --step 0.025
python backtest.py --sensitivity dbi --range 0.03 0.10 --step 0.01
python backtest.py --sensitivity options_budget --range 0.0 0.01 --step 0.001
python backtest.py --sensitivity rebalance_freq --values 1 2 4 12
```

**New module**: `src/sensitivity.py` with `run_sensitivity_sweep()` and chart generator.

Output: `output/sensitivity/<param>.csv` + `.png` (2x2 subplot: CAGR, MaxDD, Sharpe, Calmar vs param).

**Tests** (+3): `tests/test_sensitivity.py`

---

### Phase 5 — Rolling-window backtest

**CLI**:
```
python backtest.py --rolling-window --window-years 10 --step-months 1   # monthly default
python backtest.py --rolling-window --window-years 15 --step-months 12  # annual
```

**New module**: `src/rolling_window.py`.
Output: table + 3-panel chart (CAGR distribution, MaxDD distribution, Sharpe by starting year).

**Tests** (+3): `tests/test_rolling_window.py`

---

### Phase 6 — Efficient Frontier

**Implementation**: 50k random Dirichlet portfolios + Markowitz optimization for frontier line.

**CLI**: `--efficient-frontier`, `--n-random 50000` (default)

**New module**: `src/efficient_frontier.py`. Chart: scatter with cloud + frontier line + 3 markers (Max Sharpe, Min Vol, Max Return) + Four Umbrellas reference position.

**Tests** (+4): `tests/test_efficient_frontier.py`

---

### Phase 7 — Italian tax modeling

**Regole**:
- 26% CGT su plusvalenze realizzate (vendite durante ribilanciamenti)
- Minusvalenze in "zainetto fiscale" con 4-year rollover
- Compensazione FIFO (usa le più vecchie prima)
- Aliquota 12.5% opzionale su bond gov whitelist (configurabile)
- Pensione esente durante accumulo

**New module**: `src/tax.py` with `TaxLedger` class.

**Config additions** (`src/portfolio.py`):
```python
@dataclass
class TaxConfig:
    enabled: bool = False
    capital_gains_rate: float = 0.26
    gov_bond_rate: float = 0.125
    loss_carryforward_years: int = 4
    pension_exempt: bool = True
```

**Modifications**:
- `src/rebalance.py`: integrate `TaxLedger` when `TaxConfig.enabled`
- `src/report.py`: add tax section if enabled

**Tests** (+8): `tests/test_tax.py`

---

### Phase 8 — FIRE calculator (BIG — user's killer feature)

**Features requested**:

**Dati Personali**:
- Età attuale
- Sesso (needed for mortality table)
- Età fine simulazione (fissa, opzionale — se non specificata usa mortality sampling)

**Patrimonio e contributi**:
- Capitale iniziale (€)
- Contributi periodici (€) + frequenza (mese / anno)
- In fase accumulo: versamenti; in fase FIRE: sostituiti da prelievi

**Pensione (opzionale)**:
- Importo mensile (€) — 12 mensilità, no 13esima
- Età inizio pensione
- Rivalutazione % annua (100/75/0 — perequazione INPS)

**Obbiettivi**:
- Anno FIRE target
- Spese post-FIRE (€/mese)
- Inflazione (% annua)

**Tasse plusvalenze**: SI/NO (collega a Phase 7)

**Motore**:
- Monte Carlo con block bootstrap (3-mo blocks) sui rendimenti mensili del portfolio
- Per ogni simulazione: sampling mortality age dalle tabelle ISTAT (per sesso)
- Modello 2-fasi: accumulo → FIRE age → decumulo fino a morte
- Spese prelevate mensilmente con aggiustamento inflazione
- Pensione se abilitata
- Tasse CGT sui prelievi se abilitate
- Successo = portfolio sopravvive fino a mortality sampled age
- Record: wealth finale (eredità) se successo, età fallimento se insuccesso

**Output chart**:
1. Portfolio projection (area chart 25°-75° percentile + mediana + linee FIRE/pensione/spese)
2. Success probability (linea % per età)
3. Failure age distribution (histogram se fallimenti presenti)
4. Legacy distribution (histogram nominale + reale)
5. Stats panel (% success, median survival age, median legacy €, worst-case)

**New module**: `src/fire.py` + `src/mortality.py` (tabelle ISTAT 2023 bundled CSV).

**New CSVs nel repo (pubblici ISTAT)**:
- `data/mortality/istat_mortality_2023.csv` — age / sex / qx

**CLI**:
```
python fire.py --age 45 --sex M --capital 100000 --contributions 1000 --frequency month \
  --fire-age 60 --spending 2500 --inflation 0.02 --pension 1500 --pension-age 67 \
  --pension-revaluation 0.75 --tax-on-withdrawals
```

**Tests** (+8): `tests/test_fire.py`, `tests/test_mortality.py`

---

### Phase 9 — AI Analysis

**Provider abstraction**: `src/ai_analyzer.py`
- Base class `AiAnalyzer` with 4 implementations (OpenAI, Anthropic, OpenRouter, Local)
- Default: **OpenRouter** (user's choice)
- Env vars: `OPENROUTER_API_KEY` (primary), `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `LOCAL_API_BASE_URL`
- Default model per provider; override with `--ai-model`

**Prompt**:
- System: expert quant analyst role, explicit Italian/English output (config)
- User: JSON/Markdown dump of results (stats + MC + sensitivity + rolling)
- Output: `output/AI_ANALYSIS.md`

**CLI**:
```
python backtest.py --ai-analysis                        # default OpenRouter
python backtest.py --ai-analysis --ai-provider openai
python backtest.py --ai-analysis --ai-provider local --ai-model kimi-k2
```

**Also**: standalone `python analyze.py --results output/ --provider openrouter`

**Tests** (+4, mocked HTTP): `tests/test_ai_analyzer.py`

---

### Phase 10 — Streamlit dashboard

**Single-page app** with sidebar form + tabbed results.

**Sidebar sections**:
- Portfolio selector (dropdown from `data/raw/` folders)
- Date range pickers, NAV slider
- Checkbox: options overlay, tax modeling, TER
- Weight sliders (gold, dbi, options budget)
- Benchmark selector (multi-select)
- MC config: enabled, n-paths, horizon
- Tax config: rate, carry-forward years
- **FIRE config** (new tab): personal data, contributions, pension, goals
- "▶ Run backtest" button
- "💾 Save results" button (saves to `output/`)

**Main area** — tabs:
- "Summary" (stats table)
- "Charts" (all 13+ PNG inline)
- "Monte Carlo" (fan + distribution + scenarios)
- "Sensitivity" (if configured)
- "Rolling Window" (if configured)
- "Efficient Frontier" (if configured)
- "FIRE" (if configured — 4 charts + stats panel)
- "AI Analysis" (button + output display)

**Deploy configs**:
- `streamlit_app.py` (root, main app)
- `.streamlit/config.toml` (theme, settings)
- `Dockerfile` for HF Spaces
- `requirements-dashboard.txt`
- `README.md` section with deploy instructions

**Tests** (+3): `tests/test_dashboard_smoke.py`

---

### Phase 11 — README + CI polish

- Update `README.md` with all new features, expanded TOC, new screenshots
- **Add dedicated "Running the interactive dashboard locally" section** with:
  - `pip install -r requirements-dashboard.txt`
  - `streamlit run streamlit_app.py`
  - Default port (8501), how to open in browser
  - Screenshot of the UI
  - Deploy-to-Cloud and Deploy-to-HF-Spaces sub-sections
- Update `data/README.md` with new benchmark data
- Update `.github/workflows/tests.yml` to upload `output/` artifacts on PR for manual inspection
- Final integration test: run full `backtest.py --synthetic --monte-carlo --efficient-frontier --rolling-window --ai-analysis` and verify REPORT.md contains all sections

---

## Decisions embedded in code

- CLI flag naming: `--feature-name` with hyphens (Python argparse will convert to underscores internally)
- Config modifications: via `src/portfolio.py` dataclasses, editable → re-run model
- Backward compat: every new feature gated behind a flag (default OFF where it changes behavior)
- Tax backward compat: `TaxConfig.enabled = False` by default → existing tests pass unchanged

## Known blockers / gotchas

- **Python 3.14** on Windows: working; user already has venv set up
- **pandas 2.2+**: use `freq="ME"` not `"M"`, `.bfill()` not `fillna(method="bfill")`
- **Line endings**: Windows CRLF warnings in git — cosmetic only
- **Data licensing**: never commit `data/raw/` CSVs (already in `.gitignore`)
- **Streamlit on HF Spaces**: Docker image must expose port 7860 (HF convention)
- **ISTAT mortality data**: public domain; ship bundled in `data/mortality/`
