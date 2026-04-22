"""
Unit tests for src/data_loader.py — synthetic bundle generation,
date slicing, bundle structure.
"""

import pandas as pd
import pytest

from src.data_loader import (
    load_data,
    DataBundle,
    _generate_synthetic_bundle,
)


class TestSyntheticBundle:
    def test_synthetic_returns_databundle(self):
        bundle = _generate_synthetic_bundle()
        assert isinstance(bundle, DataBundle)

    def test_synthetic_has_all_required_series(self):
        bundle = _generate_synthetic_bundle()
        # Core time series must exist
        assert not bundle.monthly_returns_eur.empty
        assert not bundle.spy_daily.empty
        assert not bundle.qqq_daily.empty
        assert not bundle.vix_daily.empty
        assert not bundle.eurusd_daily.empty
        assert not bundle.rf_daily.empty

    def test_synthetic_has_sleeve_columns(self):
        bundle = _generate_synthetic_bundle()
        expected_sleeves = {
            "put_write", "nasdaq_top30", "hc_us_hedged", "hc_world",
            "quality", "momentum", "ex_usa", "gold", "btc",
            "eur_gov_1_3y", "eur_gov_7_10y", "dbi",
        }
        assert expected_sleeves.issubset(set(bundle.monthly_returns_eur.columns))

    def test_synthetic_has_benchmark_series(self):
        bundle = _generate_synthetic_bundle()
        expected = {"msci_world_tr_monthly", "sp500_tr_monthly", "bloomberg_euro_agg_monthly", "gold_lbma_monthly"}
        assert expected.issubset(set(bundle.benchmark_monthly_eur.columns))


class TestDateSlicing:
    def test_slice_reduces_length(self):
        bundle = _generate_synthetic_bundle()
        n_full = len(bundle.monthly_returns_eur)
        sliced = bundle.slice(pd.Timestamp("2024-01-01"), pd.Timestamp("2024-06-30"))
        assert len(sliced.monthly_returns_eur) < n_full

    def test_slice_preserves_btc_activation_date(self):
        bundle = _generate_synthetic_bundle()
        sliced = bundle.slice(pd.Timestamp("2024-01-01"), pd.Timestamp("2024-06-30"))
        assert sliced.btc_activation_date == bundle.btc_activation_date

    def test_slice_respects_bounds(self):
        bundle = _generate_synthetic_bundle()
        start = pd.Timestamp("2024-01-01")
        end = pd.Timestamp("2024-06-30")
        sliced = bundle.slice(start, end)
        if len(sliced.monthly_returns_eur) > 0:
            assert sliced.monthly_returns_eur.index[0] >= start
            assert sliced.monthly_returns_eur.index[-1] <= end + pd.Timedelta(days=1)


class TestLoadDataFallback:
    def test_load_data_synthetic_mode(self):
        """load_data(synthetic=True) should succeed without external CSVs."""
        bundle = load_data(synthetic=True)
        assert isinstance(bundle, DataBundle)
        assert len(bundle.monthly_returns_eur) > 0
