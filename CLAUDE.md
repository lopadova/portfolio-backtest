# CLAUDE.md — Standing rules for this repo

Distilled from 23 themes accumulated across 7 PRs of Copilot reviews on the
Portfolio Backtest Engine refactor (PR1–PR7). Read top-to-bottom before
touching code; the patterns here repeated enough times to deserve permanent
guardrails. Per-incident history: see `LESSONS.md`.

The format is **deliberately imperative and grouped by moment of code
change** (not by theme number) — that's how you actually decide what to
check while you work.

---

## Project conventions

- **Python 3.11+** (`tomllib` is stdlib; type hints use `X | None`).
- **Engine entry points**: `backtest.py`, `fire.py`, `fetch_data.py`,
  `analyze.py`. UI: `streamlit_app.py`. Engine library: `src/*.py`.
- **The byte-identical guarantee**: `python backtest.py --synthetic`
  on a fresh `main` checkout always produces a `summary_statistics.csv`
  byte-identical to the previous commit. Every refactor preserves this
  on the default Four Umbrellas preset. When you change anything in
  `src/rebalance.py`, `src/options_overlay.py`, or `src/portfolio.py`,
  prove the guarantee with a `cmp` against `git show main:...`.
- **Workflow per PR**: `EnterPlanMode` → write a plan to
  `~/.claude/plans/<plan-file>.md` → user approves → execute via small
  atomic commits → push → open PR via `gh pr create`. After Copilot
  review: append findings to `LESSONS.md`, fix, push.
- **Parallel-PR stacking**: while a PR is in review, the next PR's branch
  is created **off the current PR's branch**, not main. After merge, rebase
  on main. Avoids waiting on review for unrelated downstream work.

---

## Before you commit any PR

These five run on every PR. If any fires, fix before pushing.

1. **No marketing copy that overshoots delivered capability.** Grep the
   diff for new claims in user-visible text (README, `st.title`,
   `st.markdown`, `--help` strings, docstrings). For each claim, prove
   there's a code path that delivers it. If not, rephrase to "planned
   for PR<N>". (Themes 1, 6.)

2. **No unused imports.** Run `ruff check --select F401`. The CI
   `lint` job blocks merge on F401 — see if you can fix locally first
   via `pre-commit install` (`.pre-commit-config.yaml`). (Themes 2, 19.)

3. **Docstrings match behavior.** Any function whose return shape, raised
   exceptions, or fallback dependencies changed in this PR — update its
   docstring. Pretend you're a reviewer reading only the docstring: would
   it lie about anything? (Theme 17.)

4. **Cleanup is total.** When removing a public symbol, grep the codebase
   for every importer/caller. Delete or `noqa`-mark each one. Dead
   imports are worse than no cleanup. (Themes 11, 19.)

5. **CLI output is ASCII-only.** No emoji, em-dash, U+2192 arrow, or any
   char > 127 inside `print()` literals in `backtest.py`, `fire.py`,
   `fetch_data.py`, `analyze.py`, or `src/*.py`. The check covers BOTH
   plain string constants (`print("Done -- ok")`) AND the literal
   segments of f-strings (`print(f"NAV: EUR {n}")` is fine; `print(f"NAV: €{n}")`
   gets flagged on the leading `"NAV: €"` segment). Computed expression
   slots inside f-strings (`{...}`) are NOT inspected — their runtime
   value is unknowable at AST time; if you compute a Unicode string
   externally and pass it to `print()`, the linter won't catch it but
   you should still avoid it. Run
   `python scripts/check_ascii_print.py backtest.py fire.py fetch_data.py analyze.py src/`.
   The CI `lint` job blocks merge if it fires.
   Streamlit (`streamlit_app.py`) is exempt — the browser is UTF-8 native.
   (Theme 18.)

---

## When you change a function signature

6. **Audit every transitive caller.** A new kwarg or removed kwarg
   propagates only as far as the immediate callsite. Helpers that the
   immediate callsite invokes may still read the OLD source (often a
   global). Grep the full chain. (Theme 21.)

7. **Don't route logic through user-controlled fields.** When checking
   "is this the default preset?", branch on the CLI spec / config selector
   the user typed (`args.portfolio`), not on a string field of an object
   the user could spoof (`portfolio.name`). (Theme 4.)

---

## When you parse external input (TOML / JSON / CLI args / env vars)

8. **Per-field type coercion at the parser.** `dataclasses.replace` does
   NOT enforce annotations: a wrong type from TOML lives on the dataclass
   until the engine crashes deep with a less-actionable error. Coerce + reject
   bad types at parse time with `ValueError` naming the offending field.
   `bool` is an `int` subclass — explicitly reject it where you mean `int`.
   (Theme 22.)

9. **Reject unknown fields explicitly.** A TOML hand-edit typo
   (`budgetnav_per_year` instead of `budget_nav_per_year`) silently
   drops on the floor unless the parser checks against an allowlist.
   Maintain `_<TYPE>_ALLOWED_FIELDS` and validate. (Theme 3.)

10. **Reject fields with no runtime effect.** When a dataclass has both
    runtime-honored fields AND fields that are shadowed by another switch
    elsewhere (e.g. `OptionsConfig.enabled` shadowed by
    `Portfolio.options_overlay`), exclude the shadowed fields from the
    persisted schema. Maintain an allowlist; the dataclass's `fields()` is
    the wrong source of truth for what users can persist. (Theme 23.)

11. **Catch every expected exception at the CLI entry point.** Loaders
    can raise `FileNotFoundError`, `ValueError`, `tomllib.TOMLDecodeError`,
    `json.JSONDecodeError`. Map all of them to `sys.exit(2)` + a clean
    stderr message. A raw traceback is a UX regression. (Themes 3, 15.)

12. **Read from ground truth, not metadata cache.** When the same fact is
    available in both a metadata catalog (e.g. `AssetInfo.start_date`
    pre-computed from a TOML) and the live data (e.g.
    `bundle.monthly_returns_eur[k].first_valid_index()`), derived
    computations must read from the live data. Catalogs drift; live data
    is what the engine sees. (Theme 12.)

---

## When you write Streamlit code

13. **Apply computed values, don't just display them.** If you compute
    an `effective_start` and show it to the user in a yellow warning,
    the bundle slice / engine call must use that `effective_start`,
    not the raw user input. Otherwise the warning lies. (Theme 7.)

14. **Filter UI options to the simulatable backend surface.** Asset
    pickers, dropdowns, etc. that map to backend capabilities must be
    intersected with the actual backend surface. Don't show options
    that get silently dropped downstream. Symmetrically: on Submit /
    Run, re-validate and hard-error on mismatches. (Theme 9.)

15. **Snapshot run-time inputs into `st.session_state["last_run"]`.**
    Streamlit reruns the script on every widget change. Post-run renders
    that read live `st.session_state` for parameters used during the
    run will mislabel results when the user tweaks the slider after
    the run completed. Cache the params used at run-time and read from
    that cache in the render. (Theme 14.)

16. **Wire UI widgets end-to-end or remove them.** A widget whose value
    never reaches the engine is a bug, even if it looks fine. When
    introducing a new widget, grep for the function that should consume
    it; add a failing-before-fix test that reads back what the engine
    actually used. (Theme 8.)

17. **Handle the "no mapping" case explicitly.** If a UI radio/dropdown
    offers a fixed set of choices but the underlying model allows richer
    values, loading an out-of-set value must show a visible warning +
    pick a sensible fallback. Silently leaving the widget on its
    PREVIOUS value is a correctness trap. (Theme 20.)

18. **Don't surface bare tracebacks.** Wrap engine calls inside UI
    paths with the narrowest reasonable `except` that converts known
    error shapes into friendly status messages. Reserve uncaught
    exceptions for true bugs. (Theme 15.)

---

## When you write CLI code

19. **Help text matches actual default.** If `argparse` default is
    `None` but defaulting happens downstream, say so in `help=`. A help
    string that lies about the default is worse than no help string —
    programmatic callers of `parse_args()` get bitten silently.
    (Theme 6.)

20. **Exit code 2 for input errors.** Bad CLI args / malformed config
    files / unknown flags → `sys.exit(2)`. Reserve exit 1 for genuine
    runtime failures and 0 for success. (Theme 11.)

---

## When you write tests

21. **Test docstring describes what the test actually checks.** Don't
    write "the engine raises ValueError on X" if your test only verifies
    "the helper doesn't crash on X". Encoding wrong mental models in
    test docstrings is worse than no docstring at all. (Theme 10.)

22. **Reflect runtime config, not hardcoded names.** When a value is
    derived from a Portfolio's display name or a runtime label, don't
    hardcode the historical default in the assertion. Let the test
    accept any non-empty label, or assert a known runtime value via the
    same path the production code uses. (Theme 16.)

23. **Cover the wrong-type-per-allowed-field matrix for parsers.** When
    parsing a typed dataclass from TOML/JSON, write one test per
    (field, wrong-type) pair: string for float, bool for int, list for
    tuple-of-2, etc. The matrix is small and catches Theme 22 regressions.

---

## When you delete a feature

24. **Remove the imports too.** Run `ruff check --select F401` after
    every cleanup commit. Orphan `from x import Foo` survives until a
    linter catches it; the file imports cleanly, tests pass, but it
    confuses every future reader. (Theme 19.)

25. **Remove the tests too.** Delete the test file or the test class
    that exercised the deleted helper. Don't leave it asserting against
    a stub. (Theme 11.)

26. **Update the docs.** If README mentioned the feature, edit the
    section. If a docstring referred to the helper as "deprecated, will
    be removed in PR<N>", remove the doc reference now that PR<N> is
    here. (Theme 17.)

---

## Known tech debt

(none currently)

---

## Tooling reference

- **CI lint job**: `.github/workflows/tests.yml` job `lint`. Single
  Ubuntu/3.12 runner. Runs `ruff check --select F401` + `python
  scripts/check_ascii_print.py`. Blocks merge on failure.

- **Pre-commit (optional, opt-in)**: `.pre-commit-config.yaml`. Same
  two checks for fast local feedback. Setup:
  `pip install pre-commit && pre-commit install`.

- **PR self-review skill**: `.claude/skills/pr-self-review/SKILL.md`.
  Auto-invoked checklist for the patterns above. Trigger with the
  skill's name; it reads the staged diff and runs through the rules.

- **Per-incident archive**: `LESSONS.md` at project root. 23 themes,
  each with the concrete PR# / file:line where Copilot caught it and
  the fix that landed. CLAUDE.md is the operational digest; LESSONS.md
  is the canonical history.
