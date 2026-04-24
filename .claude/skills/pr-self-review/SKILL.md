---
name: pr-self-review
description: Self-audit a PR against patterns Copilot has historically flagged in this repo. Run AFTER staging the diff but BEFORE pushing. Covers the 23 LESSONS themes accumulated across PR1-PR7 of the Portfolio Backtest Engine refactor. Use when you're about to push a branch, open a PR, or address Copilot review comments.
---

# PR self-review

You are about to push a branch / open a PR. Run through this checklist
against the staged diff before pushing. Each item references a LESSONS.md
theme; if you fail any, fix it now — Copilot would have flagged it
post-push and the round-trip is wasteful.

The full distilled rules live in `CLAUDE.md` at the project root. This
skill is the **operational checklist** — short, scannable, with a per-item
diagnostic command where applicable.

## Step 1 — Survey the diff

```bash
git diff --stat                           # files touched
git diff main...HEAD                      # full content diff vs main
git log main..HEAD --oneline              # commit list
```

Note any new file > 200 lines, any cross-cutting change (signature
changes touching 3+ files), or any new public symbol. Those are where
the patterns below tend to bite.

## Step 2 — Run the lint gate locally

```bash
.venv/Scripts/python.exe -m ruff check --select F401 src/ tests/ \
    backtest.py fire.py fetch_data.py analyze.py streamlit_app.py
.venv/Scripts/python.exe scripts/check_ascii_print.py \
    backtest.py fire.py fetch_data.py analyze.py src/
```

Both must exit 0. The CI `lint` job runs the same commands on every
push — fixing locally avoids the round-trip.

## Step 3 — Run the suite

```bash
.venv/Scripts/python.exe -m pytest -q
```

Must end with `N passed, 0 warnings`. A new RuntimeWarning means the
code path emits expected-but-noisy output that should be silenced
(`warnings.catch_warnings()` + targeted filter, like `src/fire.py`
already does for the All-NaN slice case).

## Step 4 — Walk the 23-theme checklist

Mark each ✅ if you've actively verified, ⏭️ if not applicable to this
PR. Don't ⏭️-tag without thinking — the cost of false-skipping is the
exact pattern that bit us in PRs 1–7.

### Marketing copy vs. capability

- [ ] **README claims** match the code's actual delivered surface in
  this PR. No "users can build any portfolio" if `--portfolio` only
  accepts the default preset. (Theme 1, PR15/PR16/PR18.)
- [ ] **Streamlit `st.title` / `st.markdown` / `help=`** strings
  describe behavior the user can actually invoke. (Theme 1.)
- [ ] **Module / class / method docstrings** match the symbol's
  current signature, return shape, and exception list. (Theme 17,
  PR20.)

### Imports / cleanup

- [ ] `ruff check --select F401` exits clean. (Themes 2, 19.)
- [ ] When deleting a public helper, every `from x import Foo`
  callsite is also deleted or `# noqa: F401`-marked with reason.
  (Theme 11.)
- [ ] When deleting a feature, the corresponding test file is
  deleted too — don't leave tests asserting against stubs.
  (Theme 11.)

### Function signature changes

- [ ] **Transitive callers audited.** Grep for the symbol in every
  module under `src/`, `tests/`, root entry points. Any helper
  reaching the symbol via a None-default fallback to a global has
  been threaded with the override. (Theme 21, PR21.)
- [ ] **Branch decisions read from CLI/config spec, not display
  names.** A custom Portfolio with `name="Four Umbrellas"` doesn't
  bypass any "is this default?" check. (Theme 4, PR16.)
- [ ] **Type hints match defaults.** `field: Foo = None` is wrong;
  `field: Foo | None = None` is right. (Theme 5.)

### External-input parsing (TOML / JSON / argparse / env)

- [ ] **Per-field type coercion** at the parser. String for `float`
  rejected. `bool` for `int` rejected (bool is an int subclass).
  List of wrong length rejected. (Theme 22, PR21.)
- [ ] **Allowlist of permitted fields** maintained as a constant.
  Unknown fields produce `ValueError` naming the offending key.
  Don't rely on `dataclasses.fields()` as the schema — that includes
  fields with no runtime effect that shouldn't be persistable.
  (Themes 3, 23, PR16/PR21.)
- [ ] **All expected exceptions caught at the CLI entry point.**
  `FileNotFoundError`, `ValueError`, `tomllib.TOMLDecodeError`,
  `json.JSONDecodeError`, etc. Map to `sys.exit(2)` + clean stderr
  message. (Themes 3, 11, 15.)
- [ ] **Help text matches argparse default.** If `argparse` default
  is `None` and downstream resolves it, say that in `help=`.
  (Theme 6, PR16.)
- [ ] **Source of truth = live data, not metadata cache.** When the
  same fact is in both a manifest and the bundle, derived
  computations read from the bundle. (Theme 12, PR19.)

### Streamlit changes

- [ ] **Computed values are applied, not just displayed.** A yellow
  warning showing `effective_start` must mean the engine call uses
  it. (Theme 7, PR18.)
- [ ] **UI options filtered to backend surface.** Asset pickers
  intersect with `bundle.monthly_returns_eur.columns`. Submit-time
  re-validation present. (Theme 9, PR18.)
- [ ] **`st.session_state["last_run"]` snapshots run-time params.**
  Post-run renders read from the snapshot, never from live widget
  state. (Theme 14, PR19.)
- [ ] **New widgets are end-to-end-wired.** Grep for the consuming
  function. There's a test reading back what the engine actually
  used. (Theme 8, PR18.)
- [ ] **No-mapping fallback handled.** Loading a Portfolio whose
  `rebalance_months` doesn't match a UI radio choice falls back to
  a default + emits `st.warning`. (Theme 20, PR20.)
- [ ] **No bare tracebacks.** Engine calls wrapped in
  narrow-scoped `try/except` mapping to `st.error` /
  `st.warning`. (Theme 15, PR19.)

### CLI changes

- [ ] `python scripts/check_ascii_print.py ...` exits clean for
  every CLI module touched. (Theme 18, PR20/PR21.)
- [ ] Exit codes: 0 for success, 2 for input errors, 1 reserved
  for runtime failures. (Theme 11.)

### Test changes

- [ ] **Test docstrings describe what the assertion enforces** —
  not what you wished the engine did. (Themes 10, 16.)
- [ ] **Wrong-type-per-allowed-field matrix** present for new
  parsers (string for float, bool for int, list of wrong length,
  etc.). (Theme 22.)
- [ ] **No hardcoded display names** in assertions where the
  runtime label is dynamic. (Theme 16, PR19.)

## Step 5 — Self-summary

After running the checklist:

1. State which themes you verified and which you skipped (with
   reason).
2. State whether the diff preserves the byte-identical guarantee on
   `python backtest.py --synthetic`. If you touched `src/rebalance.py`,
   `src/options_overlay.py`, or `src/portfolio.py`, run:
   ```bash
   .venv/Scripts/python.exe backtest.py --synthetic --output-dir /tmp/check
   cmp /tmp/check/summary_statistics.csv \
       <(git show main:output/summary_statistics.csv)
   ```
3. If anything failed, fix it, re-run the relevant lint commands,
   then re-run this skill.

## When NOT to use this skill

- For documentation-only PRs (README typos, no Python touched). The
  lint gate still applies but the 23-theme checklist is overkill.
- For dependency bumps (e.g. `pandas` minor version). Run pytest only.

For everything else — engine refactors, UI changes, new features,
Copilot review fixes — run this skill.
