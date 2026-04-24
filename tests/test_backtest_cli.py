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
    """The warning guard about custom-portfolio + --sensitivity/etc.

    Post-Copilot-review: detection is based on the CLI spec
    (``args.portfolio``), not on the loaded Portfolio's display name,
    because display names are user-controlled and could spoof 'Four
    Umbrellas' on an otherwise-custom preset."""

    def _args(self, portfolio=None, **overrides):
        """Build a minimal argparse Namespace with the flags the warning checks."""
        import argparse
        ns = argparse.Namespace(
            portfolio=portfolio,
            sensitivity=None,
            rolling_window=False,
            efficient_frontier=False,
        )
        for k, v in overrides.items():
            setattr(ns, k, v)
        return ns

    def test_silent_on_default_preset(self, capsys):
        backtest_mod._warn_if_custom_portfolio_with_advanced_analysis(
            self._args(portfolio=None, sensitivity="gold")
        )
        assert capsys.readouterr().err == ""

    def test_silent_on_explicit_default_by_name(self, capsys):
        backtest_mod._warn_if_custom_portfolio_with_advanced_analysis(
            self._args(portfolio="four_umbrellas", sensitivity="gold")
        )
        assert capsys.readouterr().err == ""

    def test_silent_on_explicit_default_by_path(self, capsys):
        path = str(PROJECT_ROOT / "portfolios" / "four_umbrellas.toml")
        backtest_mod._warn_if_custom_portfolio_with_advanced_analysis(
            self._args(portfolio=path, sensitivity="gold")
        )
        assert capsys.readouterr().err == ""

    def test_silent_on_default_preset_without_advanced(self, capsys):
        backtest_mod._warn_if_custom_portfolio_with_advanced_analysis(
            self._args(portfolio=None)
        )
        assert capsys.readouterr().err == ""

    def test_silent_on_custom_without_advanced(self, capsys):
        raw = '{"name":"Toy","assets":[{"key":"gold","weight":0.5},{"key":"cash","weight":0.5}]}'
        backtest_mod._warn_if_custom_portfolio_with_advanced_analysis(
            self._args(portfolio=raw)
        )
        assert capsys.readouterr().err == ""

    def test_warns_on_custom_with_sensitivity(self, capsys):
        raw = '{"name":"Toy","assets":[{"key":"gold","weight":0.5},{"key":"cash","weight":0.5}]}'
        backtest_mod._warn_if_custom_portfolio_with_advanced_analysis(
            self._args(portfolio=raw, sensitivity="gold")
        )
        err = capsys.readouterr().err
        assert "WARNING" in err
        assert "PR3" in err

    def test_warns_on_custom_with_rolling_window(self, capsys):
        raw = '{"name":"Toy","assets":[{"key":"gold","weight":0.5},{"key":"cash","weight":0.5}]}'
        backtest_mod._warn_if_custom_portfolio_with_advanced_analysis(
            self._args(portfolio=raw, rolling_window=True)
        )
        err = capsys.readouterr().err
        assert "WARNING" in err

    def test_warns_on_custom_with_efficient_frontier(self, capsys):
        raw = '{"name":"Toy","assets":[{"key":"gold","weight":0.5},{"key":"cash","weight":0.5}]}'
        backtest_mod._warn_if_custom_portfolio_with_advanced_analysis(
            self._args(portfolio=raw, efficient_frontier=True)
        )
        err = capsys.readouterr().err
        assert "WARNING" in err

    def test_custom_name_spoofing_still_warns(self, capsys):
        """Regression: a custom preset that sets name='Four Umbrellas' but is
        passed via inline JSON must STILL trigger the warning — detection is
        from the CLI spec, not the display name."""
        spoofed = '{"name":"Four Umbrellas","assets":[{"key":"gold","weight":1.0}]}'
        backtest_mod._warn_if_custom_portfolio_with_advanced_analysis(
            self._args(portfolio=spoofed, sensitivity="gold")
        )
        assert "WARNING" in capsys.readouterr().err


class TestIsDefaultPortfolioSpec:
    def test_none_is_default(self):
        assert backtest_mod._is_default_portfolio_spec(None) is True

    def test_name_is_default(self):
        assert backtest_mod._is_default_portfolio_spec("four_umbrellas") is True

    def test_explicit_path_to_default(self):
        path = str(PROJECT_ROOT / "portfolios" / "four_umbrellas.toml")
        assert backtest_mod._is_default_portfolio_spec(path) is True

    def test_other_name_is_custom(self):
        assert backtest_mod._is_default_portfolio_spec("my_strategy") is False

    def test_other_toml_is_custom(self):
        assert backtest_mod._is_default_portfolio_spec("portfolios/other.toml") is False

    def test_inline_json_is_custom(self):
        assert backtest_mod._is_default_portfolio_spec('{"name":"X","assets":[]}') is False


class TestResolveBadToml:
    def test_malformed_toml_exits_2(self, tmp_path, capsys):
        """Regression for Copilot review: a corrupted preset file must exit 2
        with a clean error message, not a TOMLDecodeError traceback."""
        bad = tmp_path / "bad.toml"
        bad.write_text("this is [not = valid[ toml", encoding="utf-8")
        with pytest.raises(SystemExit) as ei:
            backtest_mod._resolve_portfolio_or_exit(str(bad))
        assert ei.value.code == 2
        err = capsys.readouterr().err
        assert "ERROR loading portfolio spec" in err


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
