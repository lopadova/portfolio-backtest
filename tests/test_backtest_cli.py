"""CLI-level tests for backtest.py --portfolio / --list-portfolios flags.

These run the ``backtest`` main function in-process (no subprocess) so the
assertions can inspect the shape and behavior directly, without paying the
cost of a fresh Python interpreter per test.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import backtest as backtest_mod


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
        # The header contains stable column labels — match on ones that
        # don't get truncated by the fixed-width formatter (PR6 shortened
        # "assets" → "as" to fit the metrics columns into 110-char width).
        assert "CAGR" in out
        assert "Period" in out
        # Shows asset count for four_umbrellas (19)
        assert "19" in out
        # Shipped preset is flagged as reserved (ASCII '*' marker — emoji
        # crashes Windows cp1252, see CLI print ASCII rule in LESSONS.md).
        assert "* four_umbrellas" in out

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


class TestEnginePortfolioDispatch:
    """PR3: ``_engine_portfolio`` returns the user's Portfolio when the CLI
    spec is not the default preset, and ``None`` otherwise. The None case
    sends the advanced-analysis engines down the legacy globals path,
    preserving byte-identical output for the default preset."""

    def _args(self, portfolio=None):
        import argparse
        return argparse.Namespace(portfolio=portfolio)

    def test_none_for_default_none_spec(self):
        p = backtest_mod._resolve_portfolio_or_exit(None)
        assert backtest_mod._engine_portfolio(self._args(portfolio=None), p) is None

    def test_none_for_bare_name_default(self):
        p = backtest_mod._resolve_portfolio_or_exit("four_umbrellas")
        assert backtest_mod._engine_portfolio(self._args(portfolio="four_umbrellas"), p) is None

    def test_none_for_explicit_path_to_default(self):
        path = str(PROJECT_ROOT / "portfolios" / "four_umbrellas.toml")
        p = backtest_mod._resolve_portfolio_or_exit(path)
        assert backtest_mod._engine_portfolio(self._args(portfolio=path), p) is None

    def test_returns_portfolio_for_custom_inline_json(self):
        raw = '{"name":"Toy","assets":[{"key":"gold","weight":0.5},{"key":"cash","weight":0.5}]}'
        p = backtest_mod._resolve_portfolio_or_exit(raw)
        result = backtest_mod._engine_portfolio(self._args(portfolio=raw), p)
        assert result is p
        assert result.name == "Toy"

    def test_returns_portfolio_for_custom_named_preset(self, tmp_path, monkeypatch):
        # Create an ad-hoc preset in a tmp root and redirect the default root
        custom_toml = tmp_path / "custom.toml"
        custom_toml.write_text(
            'name = "Custom"\n'
            'assets = [\n'
            '  { key = "gold", weight = 0.5 },\n'
            '  { key = "cash", weight = 0.5 },\n'
            ']\n',
            encoding="utf-8",
        )
        spec = str(custom_toml)
        p = backtest_mod._resolve_portfolio_or_exit(spec)
        result = backtest_mod._engine_portfolio(self._args(portfolio=spec), p)
        assert result is p
        assert result.name == "Custom"

    def test_no_longer_exposes_removed_warning_function(self):
        """PR3 removed _warn_if_custom_portfolio_with_advanced_analysis.
        Make sure the symbol is gone so nobody depends on it accidentally."""
        assert not hasattr(
            backtest_mod, "_warn_if_custom_portfolio_with_advanced_analysis"
        )


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
