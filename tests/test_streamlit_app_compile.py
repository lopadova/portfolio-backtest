"""Lightweight compile-time smoke test for ``streamlit_app.py``.

This replaces the dashboard-globals tests that PR6 retired with
``tests/test_dashboard_smoke.py``. It does NOT import streamlit_app at
module level (that would require a running Streamlit context) — it
just runs ``py_compile`` on the file so CI catches syntax errors,
undefined names at the AST level, and missing imports that only show
up when the dashboard is hit.

If the file compiles and every name referenced has a plausible binding
source (via ``ast.parse`` + a top-level name-check), broken dashboards
cannot land on main.
"""

from __future__ import annotations

import ast
import py_compile
from pathlib import Path

import pytest

STREAMLIT_APP = Path(__file__).resolve().parent.parent / "streamlit_app.py"


def test_streamlit_app_exists():
    """Guard rail: if someone renames/moves the dashboard, the CI must
    notice and this file must be updated (not silently skipped)."""
    assert STREAMLIT_APP.is_file(), (
        f"Expected dashboard at {STREAMLIT_APP}; "
        "if you moved it, update tests/test_streamlit_app_compile.py too."
    )


def test_streamlit_app_compiles_cleanly(tmp_path):
    """py_compile -> .pyc. Fails loud on any SyntaxError, indentation
    error, or bytecode-level surprise in the dashboard module."""
    try:
        py_compile.compile(
            str(STREAMLIT_APP),
            cfile=str(tmp_path / "streamlit_app.pyc"),
            doraise=True,
        )
    except py_compile.PyCompileError as e:  # pragma: no cover — only on regressions
        pytest.fail(f"streamlit_app.py failed to compile:\n{e}")


def test_streamlit_app_parses_to_ast():
    """ast.parse -> catches any syntax oddity that py_compile smooths over
    (e.g. a future PEP change, tokenization regression). Also provides
    the AST for the name-check below."""
    source = STREAMLIT_APP.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(STREAMLIT_APP))
    assert tree.body, "streamlit_app.py parsed as empty AST"


def test_streamlit_app_uses_no_deleted_helpers():
    """Regression guard: the post-PR6 dashboard must not reference the
    legacy globals-mutation helpers that PR6 retired. This catches a
    partial cleanup in any future refactor that reintroduces them."""
    source = STREAMLIT_APP.read_text(encoding="utf-8")
    forbidden = {"snapshot_config", "restore_config", "apply_macro_weights"}
    tree = ast.parse(source)
    referenced = {
        node.id for node in ast.walk(tree) if isinstance(node, ast.Name)
    } | {
        node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
    }
    leaked = forbidden & referenced
    assert not leaked, (
        f"streamlit_app.py references retired helpers: {leaked}. "
        f"These were deleted in PR6 with tests/test_dashboard_smoke.py."
    )


def test_streamlit_app_imports_portfolio_model():
    """The dashboard must go through the Portfolio dataclass + loaders.
    If a future refactor drops the import, the UI has silently lost the
    Save/Load capability — fail fast."""
    source = STREAMLIT_APP.read_text(encoding="utf-8")
    tree = ast.parse(source)
    portfolio_model_imported = False
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "src.portfolio_model":
            portfolio_model_imported = True
            break
    assert portfolio_model_imported, (
        "streamlit_app.py must import from src.portfolio_model — the dashboard "
        "relies on Portfolio + PortfolioMetricsCache + list_available_presets + "
        "slugify for the Save/Load UX."
    )
