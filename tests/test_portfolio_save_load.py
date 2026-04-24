"""Tests for the PR6 save/load additions on :mod:`src.portfolio_model`.

Covered surface:

- :func:`slugify` — reserved-names guard, edge cases (accents, punctuation,
  multiple spaces, unicode, all-whitespace input).
- :class:`PortfolioMetricsCache` — round-trip through ``to_toml`` / ``from_toml``
  with exact numeric equality.
- :meth:`Portfolio.save_to` — filesystem write, overwrite guard, reserved-preset
  refusal.
- :func:`list_available_presets` — surfaces ``cached_metrics`` and ``is_reserved``.
- ``from_dict`` on partial/malformed ``[metrics]`` sections — reports the
  offending field rather than raising ``KeyError``.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

from src.portfolio_model import (
    AssetAllocation,
    Portfolio,
    PortfolioMetricsCache,
    RESERVED_PRESET_SLUGS,
    list_available_presets,
    slugify,
)


# ----------------------------- fixtures -------------------------------------


def _sample_portfolio_with_metrics() -> Portfolio:
    return Portfolio(
        name="My strategy",
        assets=[
            AssetAllocation("gold", 0.5),
            AssetAllocation("cash", 0.5),
        ],
        notes="Example for tests",
        cached_metrics=PortfolioMetricsCache(
            cagr=0.0612,
            annualized_vol=0.0843,
            max_drawdown=-0.1274,
            period_start=pd.Timestamp("2005-01-31"),
            period_end=pd.Timestamp("2024-12-31"),
            run_timestamp=datetime(2026, 4, 24, 20, 30, 15),
        ),
    )


# ----------------------------- slugify --------------------------------------


class TestSlugify:
    def test_simple(self):
        assert slugify("my strategy") == "my_strategy"

    def test_collapses_whitespace_and_punctuation(self):
        assert slugify("  My    Defensive Strategy  ") == "my_defensive_strategy"
        assert slugify("My-Strategy, v2!") == "my_strategy_v2"

    def test_unicode_letters_dropped_conservatively(self):
        # ASCII-only slugs — accented characters are stripped. This keeps
        # the filesystem stable across OSes / shells that don't agree on
        # Unicode normalisation.
        assert slugify("stratégia cautélosa") == "strat_gia_caut_losa"

    def test_preserves_digits(self):
        assert slugify("60/40 classic") == "60_40_classic"

    def test_non_string_input(self):
        with pytest.raises(ValueError, match="must be a string"):
            slugify(42)  # type: ignore[arg-type]

    def test_empty_result_raises(self):
        with pytest.raises(ValueError, match="no alphanumeric characters"):
            slugify("   ")
        with pytest.raises(ValueError, match="no alphanumeric characters"):
            slugify("!!!")

    def test_reserved_name(self):
        with pytest.raises(ValueError, match="reserved"):
            slugify("four_umbrellas")
        with pytest.raises(ValueError, match="reserved"):
            slugify("Four Umbrellas")   # slug would be 'four_umbrellas'

    def test_reserved_set_frozen(self):
        assert "four_umbrellas" in RESERVED_PRESET_SLUGS


# ----------------------------- TOML round-trip -----------------------------


class TestMetricsRoundTrip:
    def test_round_trip(self, tmp_path):
        p = _sample_portfolio_with_metrics()
        path = tmp_path / "my.toml"
        p.save_to(path)

        loaded = Portfolio.from_toml(path)
        assert loaded.name == p.name
        assert loaded.cached_metrics is not None
        assert loaded.cached_metrics.cagr == pytest.approx(p.cached_metrics.cagr)
        assert loaded.cached_metrics.annualized_vol == pytest.approx(
            p.cached_metrics.annualized_vol
        )
        assert loaded.cached_metrics.max_drawdown == pytest.approx(
            p.cached_metrics.max_drawdown
        )
        assert loaded.cached_metrics.period_start == p.cached_metrics.period_start
        assert loaded.cached_metrics.period_end == p.cached_metrics.period_end
        assert loaded.cached_metrics.run_timestamp == p.cached_metrics.run_timestamp

    def test_missing_metrics_section_is_ok(self, tmp_path):
        """A portfolio WITHOUT metrics must round-trip with cached_metrics=None."""
        p = Portfolio(
            name="No metrics",
            assets=[AssetAllocation("gold", 1.0)],
        )
        path = tmp_path / "no_metrics.toml"
        p.save_to(path)

        loaded = Portfolio.from_toml(path)
        assert loaded.cached_metrics is None
        # The dumped file should not contain a [metrics] header either.
        assert "[metrics]" not in path.read_text(encoding="utf-8")

    def test_partial_metrics_rejected(self, tmp_path):
        """If the user hand-edits the TOML and leaves a partial [metrics]
        block, we refuse it rather than silently dropping fields."""
        path = tmp_path / "bad.toml"
        path.write_text(
            'name = "Partial"\n'
            'options_overlay = false\n'
            'rebalance_months = [1, 7]\n'
            'transaction_cost_bps = 20.0\n'
            '\n'
            '[[assets]]\n'
            'key = "gold"\n'
            'weight = 1.0\n'
            '\n'
            '[metrics]\n'
            'cagr = 0.05\n'  # missing the other 5 required fields
            ,
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="missing required field"):
            Portfolio.from_toml(path)

    def test_badly_typed_metric_field_rejected(self, tmp_path):
        path = tmp_path / "bad.toml"
        path.write_text(
            'name = "X"\n'
            'options_overlay = false\n'
            'rebalance_months = [1, 7]\n'
            'transaction_cost_bps = 20.0\n'
            '\n'
            '[[assets]]\n'
            'key = "gold"\n'
            'weight = 1.0\n'
            '\n'
            '[metrics]\n'
            'cagr = "not a number"\n'
            'annualized_vol = 0.1\n'
            'max_drawdown = -0.1\n'
            'period_start = "2020-01-01"\n'
            'period_end = "2024-12-31"\n'
            'run_timestamp = "2026-04-24T20:30:00"\n',
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="badly-typed field"):
            Portfolio.from_toml(path)


# ----------------------------- save_to / overwrite -------------------------


class TestSaveTo:
    def test_write_new_file(self, tmp_path):
        p = _sample_portfolio_with_metrics()
        out = tmp_path / "new.toml"
        result = p.save_to(out)
        assert out.is_file()
        assert result == out.resolve()

    def test_refuses_existing_without_overwrite(self, tmp_path):
        p = _sample_portfolio_with_metrics()
        out = tmp_path / "existing.toml"
        p.save_to(out)
        with pytest.raises(FileExistsError, match="already saved"):
            p.save_to(out)

    def test_overwrite_allowed(self, tmp_path):
        p = _sample_portfolio_with_metrics()
        out = tmp_path / "existing.toml"
        p.save_to(out)
        # Modify notes to prove the second write actually replaced the file
        p2 = Portfolio(name=p.name, assets=p.assets, notes="updated")
        p2.save_to(out, overwrite=True)
        assert "updated" in out.read_text(encoding="utf-8")

    def test_refuses_shipped_preset_even_with_overwrite(self, tmp_path):
        p = _sample_portfolio_with_metrics()
        out = tmp_path / "four_umbrellas.toml"
        with pytest.raises(ValueError, match="shipped preset"):
            p.save_to(out, overwrite=True)

    def test_creates_parent_directory(self, tmp_path):
        """If the target directory doesn't exist, save_to creates it."""
        p = _sample_portfolio_with_metrics()
        out = tmp_path / "deep" / "nested" / "portfolio.toml"
        p.save_to(out)
        assert out.is_file()


# ----------------------------- list_available_presets ----------------------


class TestListAvailablePresetsMetrics:
    def test_surfaces_cached_metrics(self, tmp_path):
        p = _sample_portfolio_with_metrics()
        p.save_to(tmp_path / "has_metrics.toml")
        # Portfolio without metrics
        Portfolio(
            name="Bare", assets=[AssetAllocation("gold", 1.0)],
        ).save_to(tmp_path / "no_metrics.toml")

        entries = list_available_presets(tmp_path)
        by_name = {e["name"]: e for e in entries}
        assert by_name["has_metrics"]["cached_metrics"] is not None
        assert by_name["has_metrics"]["cached_metrics"].cagr == pytest.approx(0.0612)
        assert by_name["no_metrics"]["cached_metrics"] is None

    def test_is_reserved_flag(self, tmp_path):
        """list_available_presets labels Four Umbrellas as reserved so the
        UI can disable delete/overwrite on that row. Note: save_to refuses
        reserved names — we bypass by writing the TOML directly, which is
        exactly how the shipped preset lands in the repo (committed to git,
        not saved via the UI)."""
        (tmp_path / "four_umbrellas.toml").write_text(
            Portfolio(
                name="Four Umbrellas",
                assets=[AssetAllocation("gold", 1.0)],
            ).to_toml(),
            encoding="utf-8",
        )
        (tmp_path / "my_custom.toml").write_text(
            Portfolio(
                name="Custom",
                assets=[AssetAllocation("gold", 1.0)],
            ).to_toml(),
            encoding="utf-8",
        )

        entries = list_available_presets(tmp_path)
        by_name = {e["name"]: e for e in entries}
        assert by_name["four_umbrellas"]["is_reserved"] is True
        assert by_name["my_custom"]["is_reserved"] is False
