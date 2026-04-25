"""Tests for src/data_catalog.py — catalog manifest + runtime date augmentation."""

from __future__ import annotations

import tomllib

import pandas as pd
import pytest

from src.data_catalog import (
    AssetInfo,
    DEFAULT_CATALOG_PATH,
    augment_with_raw_dates,
    find_by_alias,
    load_catalog,
)


class TestLoadCatalog:
    def test_ships_with_repo(self):
        """data/catalog.toml must exist at the project root."""
        assert DEFAULT_CATALOG_PATH.is_file(), (
            f"expected catalog at {DEFAULT_CATALOG_PATH}"
        )

    def test_default_catalog_loads(self):
        cat = load_catalog()
        assert len(cat) >= 15
        # All entries are typed
        for key, info in cat.items():
            assert isinstance(info, AssetInfo)
            assert info.key == key
            assert info.display_name
            assert info.filename
            assert info.category in {"equity", "bond", "commodity", "crypto", "fx", "cash", "alt"}
            assert info.native_ccy in {"USD", "EUR", "GBP", "JPY", "CHF"}
            assert 0.0 <= info.ter <= 0.05

    def test_default_catalog_covers_symbol_map(self):
        """Every sleeve key in the legacy SYMBOL_MAP globals must have a
        catalog entry — otherwise the generic engine (PR2 shim path) would
        produce a Portfolio whose assets point to keys we know nothing about."""
        from src.portfolio import SYMBOL_MAP, HEDGED, TER_ANNUAL

        cat = load_catalog()
        for sleeve_key in SYMBOL_MAP.keys():
            assert sleeve_key in cat, f"SYMBOL_MAP key {sleeve_key!r} missing from catalog"

        # Catalog ter matches legacy TER_ANNUAL for every sleeve
        for sleeve_key, legacy_ter in TER_ANNUAL.items():
            assert cat[sleeve_key].ter == pytest.approx(legacy_ter), (
                f"TER mismatch for {sleeve_key}: catalog {cat[sleeve_key].ter} vs legacy {legacy_ter}"
            )

        # Catalog is_hedged matches legacy HEDGED for every sleeve
        for sleeve_key, legacy_hedged in HEDGED.items():
            assert cat[sleeve_key].is_hedged == legacy_hedged, (
                f"HEDGED mismatch for {sleeve_key}"
            )

    def test_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="Catalog file not found"):
            load_catalog(tmp_path / "nope.toml")

    def test_malformed_toml(self, tmp_path):
        path = tmp_path / "bad.toml"
        path.write_text("this = [is not = valid", encoding="utf-8")
        with pytest.raises(tomllib.TOMLDecodeError):
            load_catalog(path)

    def test_missing_required_field(self, tmp_path):
        path = tmp_path / "partial.toml"
        path.write_text(
            '[assets.gold]\n'
            'display_name = "Gold"\n'
            'filename = "gold"\n'
            # missing category / native_ccy / is_hedged / ter
            ,
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="missing required field"):
            load_catalog(path)

    def test_invalid_category(self, tmp_path):
        path = tmp_path / "bad_cat.toml"
        path.write_text(
            '[assets.x]\n'
            'display_name = "x"\n'
            'filename = "x"\n'
            'category = "stonks"\n'
            'native_ccy = "USD"\n'
            'is_hedged = false\n'
            'ter = 0.001\n',
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="invalid category"):
            load_catalog(path)

    def test_aliases_parsed_as_tuple(self, tmp_path):
        path = tmp_path / "aliased.toml"
        path.write_text(
            '[assets.btc]\n'
            'display_name = "BTC"\n'
            'filename = "btc"\n'
            'category = "crypto"\n'
            'native_ccy = "USD"\n'
            'is_hedged = false\n'
            'ter = 0.009\n'
            'aliases = ["bitcoin", "XBT"]\n',
            encoding="utf-8",
        )
        cat = load_catalog(path)
        assert cat["btc"].aliases == ("bitcoin", "XBT")


class TestAugmentWithRawDates:
    def test_populates_start_end(self, tmp_path):
        # Build a toy catalog pointing to a CSV in this tmp data/raw/
        catalog_path = tmp_path / "cat.toml"
        catalog_path.write_text(
            '[assets.toy]\n'
            'display_name = "Toy"\n'
            'filename = "toy_daily"\n'
            'category = "equity"\n'
            'native_ccy = "USD"\n'
            'is_hedged = false\n'
            'ter = 0.0\n',
            encoding="utf-8",
        )
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        csv_path = raw_dir / "toy_daily.csv"
        csv_path.write_text(
            "date,close\n2010-01-04,100\n2020-12-31,200\n", encoding="utf-8"
        )

        cat = load_catalog(catalog_path)
        augmented = augment_with_raw_dates(cat, raw_dir)
        assert augmented["toy"].start_date == pd.Timestamp("2010-01-04")
        assert augmented["toy"].end_date == pd.Timestamp("2020-12-31")

    def test_missing_csv_keeps_none(self, tmp_path):
        catalog_path = tmp_path / "cat.toml"
        catalog_path.write_text(
            '[assets.absent]\n'
            'display_name = "A"\n'
            'filename = "absent"\n'
            'category = "equity"\n'
            'native_ccy = "USD"\n'
            'is_hedged = false\n'
            'ter = 0.0\n',
            encoding="utf-8",
        )
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()  # empty

        cat = load_catalog(catalog_path)
        augmented = augment_with_raw_dates(cat, raw_dir)
        assert augmented["absent"].start_date is None
        assert augmented["absent"].end_date is None

    def test_does_not_mutate_input(self, tmp_path):
        catalog_path = tmp_path / "cat.toml"
        catalog_path.write_text(
            '[assets.x]\n'
            'display_name = "X"\n'
            'filename = "x"\n'
            'category = "equity"\n'
            'native_ccy = "USD"\n'
            'is_hedged = false\n'
            'ter = 0.0\n',
            encoding="utf-8",
        )
        cat = load_catalog(catalog_path)
        before = cat["x"]
        augment_with_raw_dates(cat, tmp_path / "no_raw_dir")
        # The original AssetInfo instance is unchanged (frozen dataclass)
        assert cat["x"] is before
        assert cat["x"].start_date is None


class TestFindByAlias:
    def test_primary_key(self):
        cat = load_catalog()
        assert find_by_alias(cat, "btc") is cat["btc"]

    def test_alias(self):
        cat = load_catalog()
        assert find_by_alias(cat, "bitcoin") is cat["btc"]

    def test_not_found(self):
        cat = load_catalog()
        assert find_by_alias(cat, "wut_is_this") is None
