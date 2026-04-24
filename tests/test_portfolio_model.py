"""Tests for src/portfolio_model.py — Portfolio + AssetAllocation + loaders."""

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from src.portfolio_model import (
    AssetAllocation,
    Portfolio,
    _dump_toml,
    list_available_presets,
)


def _sample_portfolio() -> Portfolio:
    return Portfolio(
        name="Toy",
        assets=[
            AssetAllocation("gold", 0.5),
            AssetAllocation("cash", 0.5),
        ],
    )


class TestAssetAllocation:
    def test_frozen(self):
        a = AssetAllocation("gold", 0.5)
        with pytest.raises(FrozenInstanceError):
            a.key = "silver"  # type: ignore[misc]


class TestValidate:
    def test_ok(self):
        _sample_portfolio().validate()

    def test_weights_sum_below(self):
        p = Portfolio(name="x", assets=[AssetAllocation("gold", 0.97)])
        with pytest.raises(ValueError, match="sum to 0.97"):
            p.validate()

    def test_weights_sum_above(self):
        p = Portfolio(
            name="x",
            assets=[AssetAllocation("gold", 0.6), AssetAllocation("cash", 0.5)],
        )
        with pytest.raises(ValueError, match="sum to 1.10"):
            p.validate()

    def test_weights_within_tolerance(self):
        # 0.999 is inside the ±0.002 window
        p = Portfolio(name="x", assets=[AssetAllocation("gold", 0.999)])
        p.validate()

    def test_duplicate_keys(self):
        p = Portfolio(
            name="x",
            assets=[AssetAllocation("gold", 0.5), AssetAllocation("gold", 0.5)],
        )
        with pytest.raises(ValueError, match="Duplicate asset key"):
            p.validate()

    def test_negative_weight(self):
        p = Portfolio(
            name="x",
            assets=[AssetAllocation("gold", 1.2), AssetAllocation("short", -0.2)],
        )
        with pytest.raises(ValueError, match="negative weight"):
            p.validate()

    def test_empty_name(self):
        p = Portfolio(name="", assets=[AssetAllocation("gold", 1.0)])
        with pytest.raises(ValueError, match="non-empty string"):
            p.validate()

    def test_empty_assets(self):
        p = Portfolio(name="x", assets=[])
        with pytest.raises(ValueError, match="at least one allocation"):
            p.validate()

    def test_empty_key(self):
        p = Portfolio(name="x", assets=[AssetAllocation("", 1.0)])
        with pytest.raises(ValueError, match="non-empty string"):
            p.validate()

    def test_bad_rebalance_month(self):
        p = Portfolio(
            name="x",
            assets=[AssetAllocation("gold", 1.0)],
            rebalance_months=(13,),
        )
        with pytest.raises(ValueError, match="ints in 1..12"):
            p.validate()

    def test_rebalance_months_must_be_tuple(self):
        # Someone passing a list instead of a tuple (common mistake)
        p = Portfolio(
            name="x",
            assets=[AssetAllocation("gold", 1.0)],
            rebalance_months=[1, 7],  # type: ignore[arg-type]
        )
        with pytest.raises(ValueError, match="tuple"):
            p.validate()

    def test_negative_transaction_cost(self):
        p = Portfolio(
            name="x",
            assets=[AssetAllocation("gold", 1.0)],
            transaction_cost_bps=-5.0,
        )
        with pytest.raises(ValueError, match="transaction_cost_bps"):
            p.validate()


class TestAccessors:
    def test_to_weights_dict_includes_cash(self):
        p = _sample_portfolio()
        assert p.to_weights_dict() == {"gold": 0.5, "cash": 0.5}

    def test_to_legacy_target_weights_excludes_cash(self):
        p = _sample_portfolio()
        assert p.to_legacy_target_weights() == {"gold": 0.5}

    def test_cash_weight(self):
        assert _sample_portfolio().cash_weight() == 0.5
        p = Portfolio(name="x", assets=[AssetAllocation("gold", 1.0)])
        assert p.cash_weight() == 0.0

    def test_keys(self):
        assert _sample_portfolio().keys() == ["gold", "cash"]


class TestFromDict:
    def test_valid(self):
        p = Portfolio.from_dict(
            {
                "name": "T",
                "assets": [{"key": "gold", "weight": 1.0}],
            }
        )
        assert p.name == "T"
        assert p.assets[0].key == "gold"

    def test_missing_name(self):
        with pytest.raises(ValueError, match="missing required field: name"):
            Portfolio.from_dict({"assets": [{"key": "gold", "weight": 1.0}]})

    def test_missing_assets(self):
        with pytest.raises(ValueError, match="missing required field: assets"):
            Portfolio.from_dict({"name": "x"})

    def test_invalid_weights_sum_rejected_in_from_dict(self):
        with pytest.raises(ValueError, match="sum to"):
            Portfolio.from_dict(
                {"name": "x", "assets": [{"key": "gold", "weight": 0.5}]}
            )

    def test_assets_not_a_list(self):
        """Copilot review: malformed inputs should raise ValueError, not
        KeyError/TypeError, so the CLI can translate them to a clean exit 2."""
        with pytest.raises(ValueError, match="must be a list"):
            Portfolio.from_dict({"name": "x", "assets": "gold"})

    def test_asset_not_a_dict(self):
        with pytest.raises(ValueError, match="index 0 must be a mapping"):
            Portfolio.from_dict({"name": "x", "assets": [123]})

    def test_asset_missing_key(self):
        with pytest.raises(ValueError, match="missing required field: key"):
            Portfolio.from_dict({"name": "x", "assets": [{"weight": 1.0}]})

    def test_asset_missing_weight(self):
        with pytest.raises(ValueError, match="missing required field: weight"):
            Portfolio.from_dict({"name": "x", "assets": [{"key": "gold"}]})

    def test_asset_weight_not_numeric(self):
        with pytest.raises(ValueError, match="invalid weight"):
            Portfolio.from_dict(
                {"name": "x", "assets": [{"key": "gold", "weight": "half"}]}
            )

    def test_rebalance_months_bad_value(self):
        with pytest.raises(ValueError, match="rebalance_months"):
            Portfolio.from_dict(
                {
                    "name": "x",
                    "assets": [{"key": "gold", "weight": 1.0}],
                    "rebalance_months": ["jan"],
                }
            )

    def test_transaction_cost_bad_value(self):
        with pytest.raises(ValueError, match="transaction_cost_bps"):
            Portfolio.from_dict(
                {
                    "name": "x",
                    "assets": [{"key": "gold", "weight": 1.0}],
                    "transaction_cost_bps": "free",
                }
            )


class TestFromToml:
    def test_round_trip(self, tmp_path):
        p = _sample_portfolio()
        toml = _dump_toml(p)
        path = tmp_path / "toy.toml"
        path.write_text(toml, encoding="utf-8")

        loaded = Portfolio.from_toml(path)
        assert loaded.name == p.name
        assert loaded.to_weights_dict() == p.to_weights_dict()
        assert loaded.options_overlay == p.options_overlay
        assert loaded.rebalance_months == p.rebalance_months

    def test_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="Portfolio file not found"):
            Portfolio.from_toml(tmp_path / "missing.toml")

    def test_notes_with_quotes(self, tmp_path):
        p = Portfolio(
            name="x",
            assets=[AssetAllocation("gold", 1.0)],
            notes='Has "quotes" and \\ backslash',
        )
        path = tmp_path / "p.toml"
        path.write_text(_dump_toml(p), encoding="utf-8")
        loaded = Portfolio.from_toml(path)
        assert loaded.notes == p.notes


class TestFromJsonInline:
    def test_valid(self):
        raw = json.dumps(
            {"name": "J", "assets": [{"key": "gold", "weight": 1.0}]}
        )
        p = Portfolio.from_json_inline(raw)
        assert p.name == "J"

    def test_bad_syntax(self):
        with pytest.raises(ValueError, match="Invalid inline-JSON"):
            Portfolio.from_json_inline("{ not valid json")

    def test_not_object(self):
        with pytest.raises(ValueError, match="top-level object"):
            Portfolio.from_json_inline("[1, 2, 3]")

    def test_missing_required_field(self):
        with pytest.raises(ValueError, match="missing required field"):
            Portfolio.from_json_inline('{"assets":[{"key":"gold","weight":1.0}]}')


class TestFromName:
    def test_resolves(self, tmp_path):
        toml = _dump_toml(_sample_portfolio())
        (tmp_path / "toy.toml").write_text(toml, encoding="utf-8")
        p = Portfolio.from_name("toy", root=tmp_path)
        assert p.name == "Toy"

    def test_not_found_lists_available(self, tmp_path):
        (tmp_path / "a.toml").write_text(_dump_toml(_sample_portfolio()), encoding="utf-8")
        (tmp_path / "b.toml").write_text(_dump_toml(_sample_portfolio()), encoding="utf-8")
        with pytest.raises(FileNotFoundError) as ei:
            Portfolio.from_name("missing", root=tmp_path)
        assert "a, b" in str(ei.value) or "a" in str(ei.value)

    def test_missing_dir_raises_clear(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            Portfolio.from_name("anything", root=tmp_path / "nope")


class TestResolve:
    def test_inline_json(self):
        raw = '{"name":"I","assets":[{"key":"gold","weight":1.0}]}'
        p = Portfolio.resolve(raw)
        assert p.name == "I"

    def test_path_toml(self, tmp_path):
        path = tmp_path / "p.toml"
        path.write_text(_dump_toml(_sample_portfolio()), encoding="utf-8")
        p = Portfolio.resolve(str(path))
        assert p.name == "Toy"

    def test_bare_name(self, tmp_path):
        (tmp_path / "toy.toml").write_text(_dump_toml(_sample_portfolio()), encoding="utf-8")
        p = Portfolio.resolve("toy", root=tmp_path)
        assert p.name == "Toy"


class TestListAvailablePresets:
    def test_lists_valid_presets(self, tmp_path):
        (tmp_path / "a.toml").write_text(_dump_toml(_sample_portfolio()), encoding="utf-8")
        (tmp_path / "b.toml").write_text(_dump_toml(_sample_portfolio()), encoding="utf-8")
        entries = list_available_presets(tmp_path)
        names = [e["name"] for e in entries]
        assert "a" in names and "b" in names
        for e in entries:
            assert e["n_assets"] == 2

    def test_missing_dir_returns_empty(self, tmp_path):
        assert list_available_presets(tmp_path / "nope") == []
