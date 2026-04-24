"""
Regression tests for CLI parser construction.

Python 3.14 tightened argparse: `ArgumentParser.add_argument()` now calls
`_check_help()`, which runs `help_string % params`. A literal `%` that is not
escaped as `%%` (or not part of a valid named format like `%(default)s`) will
raise `ValueError: badly formed help string` at parser-construction time —
before the user ever runs `--help`.

This caught a real bug on Python 3.14:

    fire.py:48: help="Annual inflation rate (default 2%)"  # %) → crash
    fire.py:55: help="... 0.75 = 75%, 0 = no adjustment)"  # %, → crash

Fix: escape the literal `%` as `%%` (e.g. `(default 2%%)`).

These tests import each top-level CLI script and call its `parse_args` — if any
help string is malformed, argparse raises `ValueError` during `add_argument`
and the test fails.

`SystemExit` from argparse (e.g. missing required args) is unrelated to what
we are testing and is absorbed. Any `ValueError` from `_check_help` is not
absorbed and fails the test.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parent.parent


def _load_cli_module(name: str, filename: str):
    """
    Load a top-level CLI script by absolute path.
    Using importlib rather than `import fire` avoids any ambiguity with
    `src/fire.py` (the engine module) and keeps this test self-contained.
    """
    spec = importlib.util.spec_from_file_location(name, ROOT / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestCliHelpStringsValid:
    """Every top-level CLI entrypoint must build its argparse parser cleanly
    on Python 3.14+. This was added after a literal `%` in `fire.py` help
    strings caused `ValueError: badly formed help string` at runtime."""

    def test_fire_parser_builds(self):
        fire_mod = _load_cli_module("fire_cli", "fire.py")
        try:
            fire_mod.parse_args([])
        except SystemExit:
            pass  # required-arg missing is unrelated; we only care about ValueError

    def test_backtest_parser_builds(self):
        backtest_mod = _load_cli_module("backtest_cli", "backtest.py")
        try:
            backtest_mod.parse_args([])
        except SystemExit:
            pass

    def test_analyze_parser_builds(self):
        analyze_mod = _load_cli_module("analyze_cli", "analyze.py")
        try:
            analyze_mod.parse_args([])
        except SystemExit:
            pass

    def test_fire_parser_accepts_realistic_args(self):
        """Positive path: with all required args supplied, parse_args returns
        a namespace — confirming the parser is fully functional, not just
        buildable."""
        fire_mod = _load_cli_module("fire_cli2", "fire.py")
        args = fire_mod.parse_args([
            "--age", "45",
            "--sex", "M",
            "--capital", "100000",
            "--contributions", "1000",
            "--fire-age", "60",
            "--spending", "2500",
            "--simulations", "10",
            "--synthetic",
        ])
        assert args.age == 45
        assert args.fire_age == 60
        assert args.simulations == 10
        assert args.inflation == pytest.approx(0.02)  # default preserved


class TestTopLevelScriptImportsCleanly:
    """Top-level CLI scripts must import successfully without side-effects that
    require network or filesystem state. This caught a real bug where
    `fetch_data.py` imported `pandas_datareader`, which in turn imports
    `distutils.version.LooseVersion` — `distutils` was removed from the Python
    stdlib in 3.12, so the import crashed before any user interaction.

    These are pure import-time tests: success means the module's top-level
    code runs without raising. They cannot catch runtime bugs inside `main()`
    — those are covered by dedicated tests in other files."""

    def test_fetch_data_imports_cleanly(self):
        """fetch_data.py must import on Python 3.12+ without stdlib-removed
        dependencies (distutils) or abandoned packages."""
        _load_cli_module("fetch_data_mod", "fetch_data.py")

    def test_backtest_imports_cleanly(self):
        _load_cli_module("backtest_import_check", "backtest.py")

    def test_fire_imports_cleanly(self):
        _load_cli_module("fire_import_check", "fire.py")

    def test_analyze_imports_cleanly(self):
        _load_cli_module("analyze_import_check", "analyze.py")
