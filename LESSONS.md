# LESSONS — Copilot PR review distillation

Running log of every Copilot review finding across the refactor PRs, grouped by theme.
At end-of-project this is the raw material for distilling CLAUDE.md guidance, Claude Code skills, and project-level rules so the next round of PRs lands cleaner out of the gate.

**How to use this file**

- New finding after a Copilot review → add it under the matching theme. Create a new theme if none fits. One bullet per finding with: `PR#<n> (<shortname>): <one-line fact>. **Fix**: <what we did>.`
- If a finding spans multiple themes (e.g. "unused import + misleading docstring"), split into multiple bullets so each theme stays pure.
- If the same mistake reappears in a later PR, keep BOTH bullets — repeat counts are signal for promoting the rule into CLAUDE.md.
- Reviews that are purely stylistic (typo, whitespace) are fair game if Copilot flagged them — they still teach us a pre-commit gap.

**End-of-project distillation task (PR-final)**

When the refactor is done:

1. Re-read every row of this file, plus as a safety net re-fetch every Copilot review via `gh api repos/lopadova/portfolio-backtest/pulls/<N>/{reviews,comments}` for PR15…PR#N. Compare against this file; if anything's missing, backfill it.
2. Group themes by frequency (the "keeps coming back" themes are the priority).
3. For each high-frequency theme, produce one of:
   - a paragraph in `CLAUDE.md` under a "Standing rules" section (short, imperative),
   - a Claude Code *skill* with a tight trigger description + instructions,
   - a project-level pre-commit hook / CI check if the rule is lintable (e.g. "no unused imports", "TOML round-trip for every preset").
4. Cross-link from `CLAUDE.md` → the relevant skill / rule so future sessions pick it up automatically.

---

## Theme 1 — Marketing copy vs. delivered capability

Claims (README, docstrings, UI intros) must not advertise features the PR doesn't actually deliver. Copilot is sharp at catching this.

- PR #15 (PR1 — product rename). README TL;DR said "backtest any multi-asset portfolio" / "swap the preset for any custom composition", but at that point the engine was still hardwired to Four Umbrellas globals and the CLI had no `--portfolio` flag. Streamlit intro claimed "you can build any other portfolio from the available data" while the sidebar only had gold/DBi sliders. `src/portfolio.py` docstring said "Portfolio presets" (plural) with only one preset defined. **Fix**: reverted the overreaching phrasing to match current reality, added explicit "comes in PR2+" forward pointers.
- PR #16 (PR2 — Portfolio dataclass). `src/efficient_frontier.py` docstring referred to the "active preset's weights" as if preset selection existed. **Fix**: reworded to "current module-level weights ... i.e. the current global Four Umbrellas configuration" with a pointer to the PR3 generalization.
- PR #18 (PR4 — UI rewrite). `src/dashboard_helpers.compute_effective_start` docstring claimed the engine "drops per period" assets without a start_date, but the rebalance engine only drops assets missing from the returns *columns* — missing dates get `fillna(0.0)`. **Fix**: reworded to "ignored for this calculation: they do not constrain the computed effective start date".

**Candidate rule**: before committing a PR, grep the diff for new claims in user-visible text (README, page_title, help=, `st.title`, `st.markdown`, docstrings) against what the PR actually wires. If a claim references a capability, confirm there's a code path that delivers it — otherwise rephrase to "planned for PR<N>".

---

## Theme 2 — Unused imports / dead symbols

- PR #16 (PR2). `src/data_catalog.py` imported `field`, `Iterable` but used neither. `src/portfolio_model.py` imported `replace`, `Iterable`, `Optional` unused. `src/rebalance.py` kept importing `SYMBOL_MAP` after the refactor dropped all call sites. **Fix**: removed.
- PR #18 (PR4). `streamlit_app.py` imported `plot_correlation_heatmap` but never added it to `chart_specs`. **Fix**: removed the import.

**Candidate rule**: run `ruff --select F401` (or equivalent) as a pre-commit hook on every touched file. Alternatively: before `git commit`, grep the diff's import lines against the file body to assert each imported symbol appears at least once in the non-import region.

---

## Theme 3 — Robust input parsing: `ValueError` over raw `KeyError`/`TypeError`

All external-input loaders (TOML, JSON, YAML, CLI spec) must validate shape proactively and raise `ValueError` with actionable messages. An uncaught `KeyError` on `data["assets"][0]["key"]` bypasses the caller's exception handler and shows a raw traceback to the user.

- PR #16 (PR2). `Portfolio.from_dict` could raise `KeyError`/`TypeError` on malformed TOML/JSON (non-list assets, non-dict entries, missing `key`/`weight`, non-numeric `weight`, non-int `rebalance_months`, non-numeric `transaction_cost_bps`). The CLI's `except (FileNotFoundError, ValueError)` wouldn't catch those → raw traceback to the user. **Fix**: per-field validation with targeted `ValueError("Portfolio asset at index 2 missing required field: key")`-style messages.
- PR #16 (PR2). `_resolve_portfolio_or_exit` caught `FileNotFoundError` and `ValueError` but not `tomllib.TOMLDecodeError`, so a corrupted preset file produced a traceback instead of exit 2. **Fix**: added `tomllib.TOMLDecodeError` to the `except` tuple.

**Candidate rule**: for every `@classmethod from_*` loader, write a test per possible malformed-input branch (missing field, wrong type, malformed structure). At the CLI entry point, catch `(FileNotFoundError, ValueError, <format>DecodeError)` — mentally enumerate every parser exception the loader chain can raise.

---

## Theme 4 — Don't route logic decisions through user-controlled fields

User-controlled display names / IDs can be spoofed. Control flow must branch on the CLI/config spec (what the user typed), not on fields that the user could set to match a sentinel.

- PR #16 (PR2). `_warn_if_custom_portfolio_with_advanced_analysis` detected "custom portfolio" by checking `portfolio.name != "Four Umbrellas"`. A user-supplied preset with `name = "Four Umbrellas"` silently bypassed the warning. **Fix**: a new `_is_default_portfolio_spec(spec: str | None)` that matches on the CLI argparse value (None, bare name, or path resolving to the shipped preset file). Added a regression test `test_custom_name_spoofing_still_warns`.

**Candidate rule**: when a function takes a user-supplied object and needs to branch on "is this the default", use the selector (CLI flag value, preset path, env var) — not a string field inside the object.

---

## Theme 5 — Type hints must match actual defaults

- PR #16 (PR2). `run_backtest(..., portfolio: Portfolio = None)` annotated non-Optional but defaulted to `None`. **Fix**: `Portfolio | None = None` (PEP 604, py3.10+).

**Candidate rule**: default of `None` ⇔ `Optional` / `| None` in the annotation. Catchable by mypy/pyright with `--strict-optional`; consider enabling at least on `src/`.

---

## Theme 6 — Help text drift from actual defaults

- PR #16 (PR2). `--portfolio` argparse help said `Default: 'four_umbrellas'` but argparse's default was `None` (the defaulting happens downstream in `_resolve_portfolio_or_exit`). Anyone calling `parse_args()` programmatically got `None` back and was confused. **Fix**: help text now says "argparse default is None; the CLI defaults to 'four_umbrellas' when the value is omitted, via `_resolve_portfolio_or_exit`".

**Candidate rule**: help text must describe the argparse-level default truthfully. If defaulting is deferred, that fact must be in the help string so programmatic callers know what they get from `parse_args()`.

---

## Theme 7 — Compute → display, but also compute → apply

If you compute a value, show it to the user, and then don't use it downstream, the number in the UI lies.

- PR #18 (PR4). `effective_start` was computed and shown in a yellow warning ("effective start = 2014-09-17"), but the actual bundle was still sliced with the raw user-picked `start_date`. Pre-start months silently became `fillna(0.0)` returns, biasing the backtest. **Fix**: clamp `run_start = max(start_date, effective_start)` and slice the bundle with `run_start`.

**Candidate rule**: any value computed for display that represents a "will actually be used" number must flow into the downstream call. In reviews / self-review, trace every computed variable that's shown in UI text to the function call it's supposed to parameterize. If no call reads it, it's a lying display.

---

## Theme 8 — UI widgets must be wired end-to-end or removed

A widget in the UI that does not change behavior is a bug, even if it looks fine.

- PR #18 (PR4). UI collected `options_budget_bps` into `st.session_state`, but `simulate_options_overlay()` reads `OPTIONS.budget_nav_per_year` from module globals — the slider had zero effect. **Fix**: removed the widget; left a doc note in the options-overlay checkbox's `help=` explaining the budget is fixed until a per-Portfolio OptionsConfig lands.
- PR #18 (PR4). UI offered a rebalance-frequency radio (Annual / Semi-annual / Quarterly / Monthly) and threaded it into `Portfolio.rebalance_months`, but the overlay call `simulate_options_overlay(...)` omitted `rebalance_months=`, so the overlay always used its hardcoded `(1, 7)` default. **Fix**: pass `rebalance_months=user_portfolio.rebalance_months`.

**Candidate rule**: when a PR introduces a new UI widget, grep the codebase for every function that should consume the new value and add a failing-before-fix test that reads back what the function actually used. A widget without a test proving its effect is DOA.

---

## Theme 9 — Filter UI options to what the backend can actually do

Showing options the engine can't honor produces silent wrong-answer results.

- PR #18 (PR4). The asset picker used `sorted(catalog.keys())`, which includes benchmark-only series like `msci_world_tr_monthly` / `sp500_tr_monthly` / `msci_emerging_tr_monthly`. Those aren't columns in `bundle.monthly_returns_eur`, so `simulate_portfolio_generic` silently drops them and renormalizes the remaining weights → the user gets a result that doesn't match what they configured. **Fix**: restrict `available_asset_keys` to `{k for k in catalog if k == "cash" or k in bundle.monthly_returns_eur.columns}` and hard-error on Run if the current portfolio contains any non-cash key missing from the bundle.

**Candidate rule**: options lists that map to backend capabilities must be intersected with the actual backend surface. Symmetrically, on Run / Submit: re-validate the submitted values against the backend surface and hard-error on mismatch instead of letting the engine silently drop.

---

## Theme 10 — Test comments/docstrings must reflect actual behavior

A test that says "the engine raises X" when the engine actually silently drops is worse than no test — it encodes the wrong mental model in the code review record.

- PR #18 (PR4). `test_unknown_asset_in_catalog_ignored`'s docstring said "The engine will surface a clear error" for unknown asset keys, but the actual engine silently drops them (then renormalizes). The test only verified no crash in the helper. **Fix**: reword docstring to "This matches current downstream behavior, where unknown asset keys may be silently dropped if they are missing from `monthly_returns`" — preserves the intent (helper doesn't crash) but stops lying about downstream behavior.

**Candidate rule**: when writing a test, the docstring and assertion must describe **what the test actually checks**, never an idealized version. If the actual engine behavior differs from what you want, open a follow-up issue — don't document the wanted behavior in a test that doesn't enforce it.

---

## Theme 11 — Cleanups not to forget

- PR #16 (PR2). `src/rebalance.py` kept `SYMBOL_MAP` imported after all consumers were removed. **Fix**: removed the import.
- PR #18 (PR4). After removing `options_budget_bps` from the UI, the session_state default and the doc needed the corresponding cleanup. **Fix**: removed the entry from `_ensure_session_defaults` and replaced with a comment.

**Candidate rule**: when removing a widget / call site, run the project's dead-symbol detector (or `ruff F401` + grep) over the touched file to confirm no orphan references remain.

---

## Theme 12 — Metadata vs. ground truth (prefer ground truth)

When the same fact is encoded in both a **metadata catalog** (e.g. `AssetInfo.start_date` shipped in `data/catalog.toml`) and in the **live data** (e.g. `bundle.monthly_returns_eur[key].first_valid_index()`), derived computations must read from the live data. Metadata is easy to read, easy to cache, easy to be out-of-date or missing.

- PR #19 (PR5 — walk-forward gating). `common_history_years(bundle, portfolio, catalog)` used `AssetInfo.start_date` to compute the gate. Three concrete bugs this introduced:
  - **Synthetic mode**: catalog wasn't `augment_with_raw_dates`-ed so every `start_date` was `None` → gate returned 0.0 → walk-forward was ALWAYS skipped even when the synthetic bundle had 20 years.
  - **Catalog predates bundle**: if `AssetInfo.start_date=2003-01` but the bundle was sliced to `[2015-01, 2024-12]`, the gate overestimated available years and the UI called `run_rolling_backtest`, which then raised `ValueError: Data range shorter than the window` — a red Streamlit traceback.
  - **Sliced bundle ignored**: the user could narrow the date range; the gate didn't notice. **Fix**: rewrote the helper to use `bundle.monthly_returns_eur.index[-1]` and per-column `first_valid_index()`. Dropped the `catalog` arg from the signature. The gate now agrees with the engine exactly.

**Candidate rule**: for any predicate / metric used to gate a downstream engine call, compute the predicate from the SAME data the engine will read, not from a separate metadata source that can drift. Rule of thumb: "would this still be correct if the catalog were empty? if yes, good; if not, rewrite."

---

## Theme 13 — UI help text must match runtime behavior

Help / tooltip strings written "aspirationally" instead of descriptively drift from the actual code and either mislead users or become untrue when the code changes.

- PR #19 (PR5 — walk-forward tooltip). The checkbox help said "Requires ≥ 20 years of common history", but the gating code compared `available_years >= window_years` with `window_years` defaulting to 10. Users with a 10-year window would never see the "insufficient history" branch even when they had only 9 years of common data for that window size. **Fix**: reworded the help to "Requires common history across all portfolio assets at least as long as the selected window size (in years)", with an editorial note that ≥ 20 years is generally recommended for stable percentile tail estimates.

**Candidate rule**: help strings describe what the code does, not what you'd like the code to do. When reviewing a PR, grep for hardcoded numbers in help=/tooltip= strings and verify each one is enforced somewhere downstream.

---

## Theme 14 — Post-run renders must use snapshotted inputs, not live session_state

Streamlit reruns the whole script on every widget change. A post-run render that reads `st.session_state[...]` for a parameter that was used during the run itself will show mislabeled results when the user tweaks the slider after the run completed.

- PR #19 (PR5 — walk-forward tab). `plot_rolling_window_results(... step_months=st.session_state["walkforward_step_months"])` used the live session value. If the user ran with step=12 then changed the slider to step=1, the chart would re-render with the wrong title ("step 1 month(s)") even though the underlying `wf_df` was computed with step=12. **Fix**: snapshot `step_months` into the `walkforward_result` tuple at run-time and read from the snapshot in the tab.

**Candidate rule**: anything you cache in `st.session_state["last_run"]` (or equivalent) must include the parameters that were used to produce it. Rendering code reads from the snapshot, never from live widget state.

---

## Theme 15 — Leave no user-facing crash as a bare traceback

Even with a gate that "should" prevent bad inputs, the handful of pathological combinations that slip through must produce a clean message, not a red Streamlit exception box.

- PR #19 (PR5 — walk-forward). Even with the bundle-driven gate from Theme 12, an edge case (pure NaN tail on the bundle + a boundary window size) could still trip `run_rolling_backtest`'s internal validation. **Fix**: wrapped the call in a belt-and-suspenders `try/except ValueError` that converts the exception into a friendly `walkforward_skipped_reason` — same message shape as the primary gate.

**Candidate rule**: external-facing engine calls inside a UI path must be guarded by the narrowest reasonable exception handler that converts known error shapes into user-friendly status messages. Reserve uncaught exceptions for true bugs.

---

## Theme 17 — Docstring drift: docstrings are promises, not wishful thinking

A docstring that advertises schema fields, format-char choices, or "stdlib-only" guarantees becomes a lie the moment the code drifts. Reviewers and future-me trust docstrings; they shouldn't.

- PR #20 (PR6). `list_available_presets` docstring listed a 6-key return schema but the implementation also added `is_reserved` (used by the CLI `*` prefix and the UI Shipped flag). Callers reading the docstring would miss a field they need. **Fix**: added `is_reserved` to the documented schema with a one-line semantic note.
- PR #20 (PR6). `Portfolio.to_toml` docstring said "Hand-rolled to keep the module stdlib-only" — but by PR6 the module had started importing pandas for `PortfolioMetricsCache` timestamps, making the stdlib-only claim false. **Fix**: reworded to "to avoid pulling in a third-party TOML writer" with an explicit note that the module itself is not stdlib-only.
- PR #20 (PR6). `_print_preset_listing` docstring said outputs used "—" and "📌", but the implementation used ASCII `-` and `*` (to avoid Windows cp1252 crashes — see Theme 18). **Fix**: docstring now describes the ASCII actuals and points at LESSONS Theme 18.

**Candidate rule**: every code change that modifies a return schema, a dependency set, or a format character bumps against the docstring of the same function. In PR review, grep the diff for function-level docstring changes that should have accompanied the logic change but didn't.

---

## Theme 18 — ASCII-only CLI output on Windows

Emoji, em-dashes, and Unicode arrows (U+2192 `→`) crash Windows' default `cp1252` stdout codec with `UnicodeEncodeError`, aborting the CLI mid-print. Inside a Streamlit app the browser renders UTF-8 natively so the same characters work; the contamination happens when Copilot-review-style "just make it pretty" suggestions leak Unicode into `print()` calls shared across CLI and UI.

- PR #20 (PR6). `_save_portfolio_or_exit` printed `💾 Portfolio saved: ...` and the preset-listing header used `→` for the period separator + `📌` for reserved presets. Every Windows-default terminal crashed with `UnicodeEncodeError` on those characters. **Fix**: ASCII replacements everywhere — `[SAVED] Portfolio written to`, `->` in dates, `* ` prefix for reserved.

**Candidate rule**: `print()` / CLI stdout strings must stay 7-bit ASCII, full stop. If a string is meant for the browser (Streamlit), keep the emoji; if it could ever flow through `print()` or a logger, strip it. Consider a `ruff`-style check that scans `backtest.py` and `src/*.py` for non-ASCII characters inside `print(...)` arguments.

---

## Theme 19 — Cleanup must be total: orphan references are worse than no cleanup

When a module retires a public symbol, every *importer* must be updated in the same commit. Leaving a `from x import Foo` where `Foo` was deleted produces the worst kind of regression: the file still imports cleanly, tests still pass, but the unused import is a landmine for a future reader / linter.

- PR #20 (PR6). `streamlit_app.py` imported `RESERVED_PRESET_SLUGS` from `src.portfolio_model` but never referenced it in the dashboard code (I had planned to use it for the Shipped check but ended up using `_e['is_reserved']` from the listing dict). The import survived the UI rewrite as dead weight. **Fix**: removed.

**Candidate rule**: when finalizing a PR, run `ruff --select F401` (or `pyflakes`) over every touched file. If F401 reports unused imports, either delete them or add a `# noqa: F401` with a rationale comment — never leave them silent.

---

## Theme 20 — UI state reloads must handle the "no mapping" case explicitly

When a UI offers a fixed set of choices (radio buttons, dropdowns) but the underlying data model allows richer values, loading an out-of-UI-set value must produce a visible warning. Silently leaving the widget on its PREVIOUS value is a trap: the user thinks they loaded the preset faithfully, but the simulation runs with a different cadence.

- PR #20 (PR6). The Portafogli-salvati Load button mapped the loaded Portfolio's `rebalance_months` tuple back to one of the 4 UI radio labels (Annual / Semi-annual / Quarterly / Monthly). For a preset with a non-standard cadence (e.g. `(1, 3, 6, 9)`, valid in the Portfolio dataclass), the `for/break` loop silently fell through — `st.session_state["rebalance_freq"]` kept its previous value. The next Run would use the wrong cadence. **Fix**: replaced the `for/break` with a `next((... if match ...), None)` expression; when `None`, set the radio to the first UI option and emit `st.warning` with the preset's raw tuple + the chosen fallback.

**Candidate rule**: any code that maps a rich model value back to a widget's constrained choice set must have an explicit "no match" branch that is either (a) a hard failure, (b) a sensible default + user warning, or (c) an extension of the choice set with a "Custom" option. Silent carry-over of the previous widget value is never correct.

---

## Theme 16 — Hardcoded product/preset labels leak through when names become data

When a preset's name used to be a hardcoded product name but is now user-configurable, every chart legend / annotation / hover card that built a string around the old hardcoded name needs to switch to the runtime `label` field.

- PR #19 (PR5 — Efficient Frontier). `plot_efficient_frontier` and `plot_efficient_frontier_interactive` hardcoded `"Four Umbrellas"` in the legend label, annotation, and hover card — even though PR3 made `run_efficient_frontier(..., portfolio=...)` return a `FrontierPoint` whose `.label` already carries the correct runtime name ("Toy", "My strategy", etc.). The Streamlit integration (PR5) therefore mislabeled every custom-portfolio frontier as "Four Umbrellas". **Fix**: both plot functions now derive the short legend label and the annotation text from `ref_point.label` (stripping any parenthetical suffix for compactness); the hover card uses the full label.

**Candidate rule**: when a function accepts a dataclass whose fields (name, label, description) are surfaced in user-facing output, search the function body for string literals matching historic/default values and replace with references to the dataclass fields. This is a "names become data" refactor that must happen simultaneously with the runtime plumbing.

---

## Distillation targets (build during PR-final)

- `CLAUDE.md` addendum "Before every PR" with themes 1, 2, 3, 5, 7, 8, 9, 10 as imperative bullets.
- Skill (Claude Code) tentatively named `review-self-before-copilot` — activated by the `create-pr` / `pr-review` flow to self-audit against themes 1/7/8/9.
- Pre-commit hooks: `ruff --select F401` (theme 2), `mypy --strict-optional` on `src/` (theme 5), a custom check that every `--portfolio`/preset-file-change is round-trip-tested (theme 3).
