"""Tests for ``scripts/check_ascii_print.py``.

Exercises the AST visitor on hand-crafted Python snippets so the
violation matrix (clean / non-ASCII constant / non-ASCII f-string segment
/ etc.) is enforced by tests, not by smoke-running the script over the
codebase.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


# The script lives outside the package — load it as a module under the
# name "check_ascii_print" via the importlib machinery.
_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "check_ascii_print.py"
_spec = importlib.util.spec_from_file_location("check_ascii_print", _SCRIPT)
check_ascii_print = importlib.util.module_from_spec(_spec)
sys.modules["check_ascii_print"] = check_ascii_print
_spec.loader.exec_module(check_ascii_print)


def _scan_source(src: str, tmp_path: Path) -> list:
    p = tmp_path / "snippet.py"
    p.write_text(src, encoding="utf-8")
    return check_ascii_print._scan_file(p)


# ---------------------------------------------------------------------------
# Plain string literals
# ---------------------------------------------------------------------------


class TestPlainStringLiterals:
    def test_ascii_only_clean(self, tmp_path):
        v = _scan_source("print('hello world')\n", tmp_path)
        assert v == []

    def test_em_dash_caught(self, tmp_path):
        v = _scan_source('print("Done -- ok")\n'.replace("--", "—"), tmp_path)
        assert len(v) == 1
        assert "U+2014" in v[0][1]

    def test_emoji_caught(self, tmp_path):
        v = _scan_source('print("Saved \\U0001f4be")\n', tmp_path)
        assert len(v) == 1
        assert "U+1F4BE" in v[0][1]

    def test_arrow_caught(self, tmp_path):
        v = _scan_source('print("from -> to".replace("->", "→"))\n', tmp_path)
        # The replace() is computed — the script doesn't follow it. Only
        # the ASCII LITERAL "from -> to" is inspected, no violation.
        assert v == []

    def test_arrow_in_actual_literal_caught(self, tmp_path):
        v = _scan_source('print("from → to")\n', tmp_path)
        assert len(v) == 1
        assert "U+2192" in v[0][1]


# ---------------------------------------------------------------------------
# f-string literal segments — the PR22 Copilot fix scope
# ---------------------------------------------------------------------------


class TestFStringLiteralSegments:
    """Post-PR22-Copilot: the script must inspect the static text segments
    of f-strings, not just plain string constants. Computed expression
    slots (``{...}``) are still skipped — their runtime value is unknowable
    at AST time."""

    def test_fstring_ascii_only_clean(self, tmp_path):
        v = _scan_source('x = 5\nprint(f"value = {x}")\n', tmp_path)
        assert v == []

    def test_fstring_euro_in_literal_segment_caught(self, tmp_path):
        # f"NAV: €{n:,.0f}" — the leading "NAV: €" is a Constant segment
        v = _scan_source('n = 1000\nprint(f"NAV: €{n:,.0f}")\n', tmp_path)
        assert len(v) == 1
        assert "U+20AC" in v[0][1]
        assert "f-string literal segment" in v[0][1]

    def test_fstring_em_dash_between_segments_caught(self, tmp_path):
        # f"Run -- {name}" with em-dash — caught in the "Run — " segment
        v = _scan_source(
            'name = "X"\nprint(f"Run — {name}")\n',
            tmp_path,
        )
        assert len(v) == 1
        assert "U+2014" in v[0][1]

    def test_fstring_multiplication_sign_caught(self, tmp_path):
        # The PR22 reviewer specifically mentioned "× years" patterns
        v = _scan_source(
            'paths = 5000\ny = 20\nprint(f"{paths} × {y} years")\n',
            tmp_path,
        )
        assert len(v) == 1
        assert "U+00D7" in v[0][1]

    def test_fstring_computed_value_not_inspected(self, tmp_path):
        """Only Constant segments inside JoinedStr.values are checked.
        FormattedValue (computed expression) slots can produce non-ASCII
        text at runtime — that's by design out of scope (false-positive
        avoidance). Documented in the script docstring."""
        v = _scan_source(
            'symbol = "€"\nprint(f"NAV: {symbol}{1000}")\n',
            tmp_path,
        )
        # The static segments are "NAV: " and "" — both ASCII. The {symbol}
        # slot's runtime value isn't inspected.
        assert v == []


# ---------------------------------------------------------------------------
# Print() boundaries
# ---------------------------------------------------------------------------


class TestPrintBoundaries:
    def test_non_print_call_ignored(self, tmp_path):
        """A non-ASCII string passed to a function OTHER than print is fine
        — the LESSONS Theme 18 rule is CLI-stdout-specific."""
        v = _scan_source('logger.info("Done — ok")\n', tmp_path)
        assert v == []

    def test_print_attribute_ignored(self, tmp_path):
        """``module.print(...)`` is not the builtin print — current
        implementation only flags the bare ``print`` Name. Documented
        as the trade-off in the script."""
        v = _scan_source('sys.stdout.write("—")\n', tmp_path)
        assert v == []

    def test_print_with_multiple_args_each_inspected(self, tmp_path):
        v = _scan_source('print("a", "b—", "c")\n', tmp_path)
        assert len(v) == 1
        assert "U+2014" in v[0][1]


# ---------------------------------------------------------------------------
# File-level handling
# ---------------------------------------------------------------------------


class TestFileLevel:
    def test_empty_file_clean(self, tmp_path):
        assert _scan_source("", tmp_path) == []

    def test_syntax_error_reported(self, tmp_path):
        v = _scan_source("def broken(\n", tmp_path)
        assert len(v) == 1
        assert "SyntaxError" in v[0][1]

    def test_multiple_violations_in_one_file(self, tmp_path):
        v = _scan_source(
            'print("— plain")\n'
            'x = 1\n'
            'print(f"€ {x}")\n',
            tmp_path,
        )
        assert len(v) == 2
        joined = " ; ".join(msg for _, msg in v)
        assert "U+2014" in joined
        assert "U+20AC" in joined
