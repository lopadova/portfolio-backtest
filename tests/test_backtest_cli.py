"""CLI-level tests for backtest.py --portfolio / --list-portfolios flags.

These run the ``backtest`` main function in-process (no subprocess) so the
assertions can inspect the shape and behavior directly, without paying the
cost of a fresh Python interpreter per test.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

import backtest as backtest_mod
from src.portfolio_model import Portfolio


PROJECT_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def monkeypatched_cwd(monkeypatch, tmp_path):
    """Run with cwd = tmp_path so the default output dir doesn't litter the repo.
    Falls back to explicit --output-dir anyway in most tests."""
    monkeypatch.chdir(tmp_path)
    return tmp_path


class TestResolve:
    def test_default_resolves_to_four_umbrellas(self):
        p = backtest_mod._resolve_portfolio_or_exit(None)
        assert p.name == "Four Umbrellas"
        assert len(p.assets) == 19  # current preset size

    def test_by_name(self):
        p = backtest_mod._resolve_portfolio_or_exit("four_umbrellas")
        assert p.name == "Four Umbrellas"

    def test_by_path(self):
        p = backtest_mod._resolve_portfolio_or_exit(
            str(PROJECT_ROOT / "portfolios" / "four_umbrellas.toml")
        )
        assert p.name == "Four Umbrellas"

    def test_inline_json(self):
        raw = '{"name":"X","assets":[{"key":"gold","weight":0.5},{"key":"cash","weight":0.5}]}'
        p = backtest_mod._resolve_portfolio_or_exit(raw)
        assert p.name == "X"
        assert p.cash_weight() == 0.5

    def test_invalid_exits_2(self, capsys):
        with pytest.raises(SystemExit) as ei:
            backtest_mod._resolve_portfolio_or_exit(
                '{"name":"Bad","assets":[{"key":"gold","weight":0.5}]}'
            )
        assert ei.value.code == 2
        captured = capsys.readouterr()
        # Message goes to stderr
        assert "ERROR loading portfolio spec" in captured.err
        assert "sum to" in captured.err.lower()

    def test_missing_preset_exits_2(self, capsys):
        with pytest.raises(SystemExit) as ei:
            backtest_mod._resolve_portfolio_or_exit("nonexistent_preset")
        assert ei.value.code == 2


class TestListPresets:
    def test_lists_four_umbrellas(self, capsys):
        backtest_mod._print_preset_listing()
        out = capsys.readouterr().out
        assert "four_umbrellas" in out
        assert "assets" in out  # the header line
        # Shows asset count; four_umbrellas ships with 19 assets
        assert "19" in out

    def test_empty_dir(self, capsys, tmp_path, monkeypatch):
        """If portfolios/ is empty (or doesn't exist), listing prints a
        user-friendly notice instead of crashing."""
        # Make list_available_presets point at an empty dir
        monkeypatch.setattr(
            backtest_mod, "list_available_presets", lambda: []
        )
        backtest_mod._print_preset_listing()
        out = capsys.readouterr().out
        assert "No presets found" in out


class TestAdvancedAnalysisWarning:
    """The warning guard about custom-portfolio + --sensitivity/etc."""

    def _args(self, **overrides):
        """Build a minimal argparse Namespace with the flags the warning checks."""
        import argparse
        ns = argparse.Namespace(
            sensitivity=None,
            rolling_window=False,
            efficient_frontier=False,
        )
        for k, v in overrides.items():
            setattr(ns, k, v)
        return ns

    def test_silent_on_default_preset(self, capsys):
        preset = backtest_mod._resolve_portfolio_or_exit(None)
        backtest_mod._warn_if_custom_portfolio_with_advanced_analysis(
            self._args(sensitivity="gold"), preset
        )
        assert capsys.readouterr().err == ""

    def test_silent_on_default_preset_without_advanced(self, capsys):
        preset = backtest_mod._resolve_portfolio_or_exit(None)
        backtest_mod._warn_if_custom_portfolio_with_advanced_analysis(
            self._args(), preset
        )
        assert capsys.readouterr().err == ""

    def test_silent_on_custom_without_advanced(self, capsys):
        raw = '{"name":"Toy","assets":[{"key":"gold","weight":0.5},{"key":"cash","weight":0.5}]}'
        p = Portfolio.resolve(raw)
        backtest_mod._warn_if_custom_portfolio_with_advanced_analysis(
            self._args(), p
        )
        assert capsys.readouterr().err == ""

    def test_warns_on_custom_with_sensitivity(self, capsys):
        raw = '{"name":"Toy","assets":[{"key":"gold","weight":0.5},{"key":"cash","weight":0.5}]}'
        p = Portfolio.resolve(raw)
        backtest_mod._warn_if_custom_portfolio_with_advanced_analysis(
            self._args(sensitivity="gold"), p
        )
        err = capsys.readouterr().err
        assert "WARNING" in err
        assert "'Toy'" in err
        assert "PR3" in err

    def test_warns_on_custom_with_rolling_window(self, capsys):
        raw = '{"name":"Toy","assets":[{"key":"gold","weight":0.5},{"key":"cash","weight":0.5}]}'
        p = Portfolio.resolve(raw)
        backtest_mod._warn_if_custom_portfolio_with_advanced_analysis(
            self._args(rolling_window=True), p
        )
        err = capsys.readouterr().err
        assert "WARNING" in err

    def test_warns_on_custom_with_efficient_frontier(self, capsys):
        raw = '{"name":"Toy","assets":[{"key":"gold","weight":0.5},{"key":"cash","weight":0.5}]}'
        p = Portfolio.resolve(raw)
        backtest_mod._warn_if_custom_portfolio_with_advanced_analysis(
            self._args(efficient_frontier=True), p
        )
        err = capsys.readouterr().err
        assert "WARNING" in err


class TestParseArgsNewFlags:
    def test_default_portfolio_is_none(self):
        args = backtest_mod.parse_args(["--synthetic"])
        assert args.portfolio is None
        assert args.list_portfolios is False

    def test_explicit_portfolio(self):
        args = backtest_mod.parse_args(["--portfolio", "four_umbrellas"])
        assert args.portfolio == "four_umbrellas"

    def test_list_portfolios_flag(self):
        args = backtest_mod.parse_args(["--list-portfolios"])
        assert args.list_portfolios is True
