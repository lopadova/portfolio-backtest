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

## Distillation targets (build during PR-final)

- `CLAUDE.md` addendum "Before every PR" with themes 1, 2, 3, 5, 7, 8, 9, 10 as imperative bullets.
- Skill (Claude Code) tentatively named `review-self-before-copilot` — activated by the `create-pr` / `pr-review` flow to self-audit against themes 1/7/8/9.
- Pre-commit hooks: `ruff --select F401` (theme 2), `mypy --strict-optional` on `src/` (theme 5), a custom check that every `--portfolio`/preset-file-change is round-trip-tested (theme 3).
