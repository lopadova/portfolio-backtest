"""CLI-level tests for the PR6 save/load flags on ``backtest.py``.

Exercises ``_build_metrics_cache`` / ``_save_portfolio_or_exit`` /
``_print_preset_listing`` in isolation, without spawning subprocesses.
Full round-trip CLI smoke is covered by the battery of manual checks in
the PR verification section — this file focuses on the fast, in-process
validation paths (slug traversal, collision, overwrite, reserved names).
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import backtest as backtest_mod
from src.portfolio_model import AssetAllocation, Portfolio, PortfolioMetricsCache


def _toy_portfolio() -> Portfolio:
    return Portfolio(
        name="Toy",
        assets=[
            AssetAllocation("gold", 0.5),
            AssetAllocation("cash", 0.5),
        ],
    )


def _returns_dict(name: str) -> dict:
    idx = pd.date_range("2020-01-31", periods=24, freq="ME")
    values = np.linspace(0.001, 0.01, 24)
    return {name: pd.Series(values, index=idx)}


class TestBuildMetricsCache:
    def test_populates_all_fields(self):
        cache = backtest_mod._build_metrics_cache(
            _returns_dict("Toy"),
            _toy_portfolio(),
            pd.Timestamp("2020-01-31"),
            pd.Timestamp("2021-12-31"),
        )
        assert isinstance(cache, PortfolioMetricsCache)
        assert cache.period_start == pd.Timestamp("2020-01-31")
        assert cache.period_end == pd.Timestamp("2021-12-31")
        assert isinstance(cache.run_timestamp, datetime)
        # Ballpark sanity — positive-trending synthetic series has positive CAGR
        assert cache.cagr > 0

    def test_prefers_with_options_series(self):
        """If returns_dict has both '<name>' and '<name> (no options)',
        _build_metrics_cache picks the '<name>' series as representative."""
        idx = pd.date_range("2020-01-31", periods=12, freq="ME")
        rd = {
            "Toy":                pd.Series(np.ones(12) * 0.01, index=idx),
            "Toy (no options)":   pd.Series(np.ones(12) * 0.005, index=idx),
        }
        cache = backtest_mod._build_metrics_cache(
            rd, _toy_portfolio(),
            pd.Timestamp("2020-01-31"), pd.Timestamp("2020-12-31"),
        )
        # Should reflect the 1% series, not the 0.5% one
        assert cache.cagr == pytest.approx(
            (1.01 ** 12) - 1, rel=0.01,
        )


class TestSavePortfolioOrExit:
    def _make_cache(self) -> PortfolioMetricsCache:
        return PortfolioMetricsCache(
            cagr=0.05, annualized_vol=0.08, max_drawdown=-0.1,
            period_start=pd.Timestamp("2020-01-31"),
            period_end=pd.Timestamp("2024-12-31"),
            run_timestamp=datetime(2026, 4, 24, 20, 0, 0),
        )

    def test_happy_path(self, tmp_path, capsys):
        p = _toy_portfolio()
        result = backtest_mod._save_portfolio_or_exit(
            p, "My Strategy", overwrite=False,
            metrics_cache=self._make_cache(), root=tmp_path,
        )
        assert (tmp_path / "my_strategy.toml").is_file()
        assert result == (tmp_path / "my_strategy.toml").resolve()
        assert "[SAVED] Portfolio written to" in capsys.readouterr().out

    def test_collision_exits_2(self, tmp_path, capsys):
        p = _toy_portfolio()
        backtest_mod._save_portfolio_or_exit(
            p, "Existing", overwrite=False,
            metrics_cache=self._make_cache(), root=tmp_path,
        )
        with pytest.raises(SystemExit) as ei:
            backtest_mod._save_portfolio_or_exit(
                p, "Existing", overwrite=False,
                metrics_cache=self._make_cache(), root=tmp_path,
            )
        assert ei.value.code == 2
        err = capsys.readouterr().err
        assert "already" in err.lower()
        assert "--overwrite" in err

    def test_overwrite_succeeds(self, tmp_path):
        p = _toy_portfolio()
        backtest_mod._save_portfolio_or_exit(
            p, "Existing", overwrite=False,
            metrics_cache=self._make_cache(), root=tmp_path,
        )
        backtest_mod._save_portfolio_or_exit(
            p, "Existing", overwrite=True,
            metrics_cache=self._make_cache(), root=tmp_path,
        )
        assert (tmp_path / "existing.toml").is_file()

    def test_traversal_blocked(self, tmp_path, capsys):
        """Slugify strips path separators, so '../../evil' → 'evil' and
        the file is always written inside root. Verify: the file does NOT
        appear outside root."""
        p = _toy_portfolio()
        backtest_mod._save_portfolio_or_exit(
            p, "../../evil", overwrite=False,
            metrics_cache=self._make_cache(), root=tmp_path,
        )
        # The save went into the root as 'evil.toml', not outside
        assert (tmp_path / "evil.toml").is_file()
        # Parent / grandparent dirs did NOT receive a file
        assert not (tmp_path.parent / "evil.toml").exists()
        assert not (tmp_path.parent.parent / "evil.toml").exists()

    def test_empty_name_exits_2(self, tmp_path, capsys):
        p = _toy_portfolio()
        with pytest.raises(SystemExit) as ei:
            backtest_mod._save_portfolio_or_exit(
                p, "   ", overwrite=False,
                metrics_cache=self._make_cache(), root=tmp_path,
            )
        assert ei.value.code == 2
        assert "no alphanumeric" in capsys.readouterr().err

    def test_reserved_name_exits_2(self, tmp_path, capsys):
        p = _toy_portfolio()
        with pytest.raises(SystemExit) as ei:
            backtest_mod._save_portfolio_or_exit(
                p, "Four Umbrellas", overwrite=False,
                metrics_cache=self._make_cache(), root=tmp_path,
            )
        assert ei.value.code == 2
        err = capsys.readouterr().err
        assert "reserved" in err

    def test_does_not_mutate_input_portfolio(self, tmp_path):
        """The input portfolio must be unchanged — we operate on a replace() copy."""
        p = _toy_portfolio()
        original_name = p.name
        original_metrics = p.cached_metrics
        backtest_mod._save_portfolio_or_exit(
            p, "Renamed For Save", overwrite=False,
            metrics_cache=self._make_cache(), root=tmp_path,
        )
        assert p.name == original_name
        assert p.cached_metrics is original_metrics


class TestArgparseFlags:
    def test_save_as_default_none(self):
        args = backtest_mod.parse_args(["--synthetic"])
        assert args.save_as is None
        assert args.overwrite is False

    def test_save_as_explicit(self):
        args = backtest_mod.parse_args(["--synthetic", "--save-as", "my_test"])
        assert args.save_as == "my_test"

    def test_overwrite_flag(self):
        args = backtest_mod.parse_args(
            ["--synthetic", "--save-as", "x", "--overwrite"]
        )
        assert args.overwrite is True


class TestListPresetListingWithMetrics:
    def test_lists_metrics_when_present(self, capsys, tmp_path, monkeypatch):
        """The --list-portfolios output should render CAGR/Vol/MaxDD/Period
        when the preset has a [metrics] section."""
        metrics = PortfolioMetricsCache(
            cagr=0.0612, annualized_vol=0.0843, max_drawdown=-0.1274,
            period_start=pd.Timestamp("2005-01-31"),
            period_end=pd.Timestamp("2024-12-31"),
            run_timestamp=datetime(2026, 4, 24, 20, 0, 0),
        )
        p = Portfolio(
            name="Saved strategy",
            assets=[
                AssetAllocation("gold", 0.5),
                AssetAllocation("cash", 0.5),
            ],
            cached_metrics=metrics,
        )
        p.save_to(tmp_path / "saved_strategy.toml")
        # Redirect list_available_presets to the tmp dir
        from src.portfolio_model import list_available_presets as real
        monkeypatch.setattr(
            backtest_mod, "list_available_presets",
            lambda: real(tmp_path),
        )
        backtest_mod._print_preset_listing()
        out = capsys.readouterr().out
        assert "saved_strategy" in out
        assert "6.12%" in out   # CAGR
        assert "8.43%" in out   # Vol
        assert "2005-01" in out  # Period start (truncated to YYYY-MM)

    def test_lists_dashes_when_no_metrics(self, capsys, tmp_path, monkeypatch):
        p = Portfolio(
            name="Bare",
            assets=[AssetAllocation("gold", 1.0)],
        )
        p.save_to(tmp_path / "bare.toml")
        from src.portfolio_model import list_available_presets as real
        monkeypatch.setattr(
            backtest_mod, "list_available_presets",
            lambda: real(tmp_path),
        )
        backtest_mod._print_preset_listing()
        out = capsys.readouterr().out
        assert "bare" in out
        # ASCII dash for missing metrics (em-dash crashes Windows cp1252).
        assert "-" in out
