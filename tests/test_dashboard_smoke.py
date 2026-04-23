"""
Smoke + unit tests for Streamlit dashboard (streamlit_app.py + src/dashboard_helpers.py).

Smoke level: verify `streamlit_app.py` parses and its companion configs exist.
Unit level: exercise the REAL `src/dashboard_helpers` functions (snapshot /
restore / apply_macro_weights) so any drift between prod and test is caught.

Full UI automation (Selenium / Streamlit AppTest) is intentionally excluded
to keep CI fast; structural coverage is enough to catch the common breakages.
"""

from pathlib import Path

import pytest

from src import portfolio as portfolio_cfg
from src.dashboard_helpers import (
    snapshot_config,
    restore_config,
    apply_macro_weights,
)


class TestDashboardImport:
    def test_streamlit_app_module_parseable(self):
        """The streamlit_app.py file should parse without SyntaxError."""
        project_root = Path(__file__).resolve().parent.parent
        app_path = project_root / "streamlit_app.py"
        assert app_path.exists(), "streamlit_app.py must exist"
        code = app_path.read_text(encoding="utf-8")
        compile(code, str(app_path), "exec")

    def test_requirements_dashboard_exists(self):
        project_root = Path(__file__).resolve().parent.parent
        req = project_root / "requirements-dashboard.txt"
        assert req.exists()
        assert "streamlit" in req.read_text(encoding="utf-8").lower()

    def test_streamlit_config_exists(self):
        project_root = Path(__file__).resolve().parent.parent
        cfg = project_root / ".streamlit" / "config.toml"
        assert cfg.exists()

    def test_dockerfile_exists(self):
        project_root = Path(__file__).resolve().parent.parent
        dockerfile = project_root / "Dockerfile"
        assert dockerfile.exists()
        content = dockerfile.read_text(encoding="utf-8")
        assert "streamlit" in content.lower()
        assert "7860" in content  # HF Spaces port


class TestSnapshotRestore:
    """Exercise the REAL snapshot_config / restore_config from
    src/dashboard_helpers.py (no reimplementation in-test)."""

    def test_snapshot_returns_expected_keys(self):
        snap = snapshot_config()
        assert set(snap.keys()) == {"WEIGHTS", "EQUITY", "OPTIONS", "REBALANCE"}

    def test_restore_reverts_mutation(self):
        snap = snapshot_config()
        try:
            original_gold = portfolio_cfg.WEIGHTS["gold"]
            portfolio_cfg.WEIGHTS["gold"] = 0.99
            restore_config(snap)
            assert portfolio_cfg.WEIGHTS["gold"] == original_gold
        finally:
            restore_config(snap)

    def test_snapshot_is_deep_copy(self):
        """Mutating the snapshot's dict must NOT affect the live module state."""
        snap = snapshot_config()
        snap["WEIGHTS"]["gold"] = 0.99
        assert portfolio_cfg.WEIGHTS["gold"] != 0.99


class TestApplyMacroWeights:
    """Regression (Copilot PR #11): dashboard's cash rebalance bug.
    Previous code set gold, then computed cash using a hardcoded 0.1825
    offset, THEN set dbi — leaving cash out of sync and breaking the
    WEIGHTS sum=1.0 invariant. New shared helper `apply_macro_weights`
    derives cash = 1 - Σ(non-cash) atomically."""

    def test_sum_equals_one_after_gold_change(self):
        snap = snapshot_config()
        try:
            apply_macro_weights(gold_pct=0.25, dbi_pct=portfolio_cfg.WEIGHTS["dbi"])
            assert abs(sum(portfolio_cfg.WEIGHTS.values()) - 1.0) < 1e-9
        finally:
            restore_config(snap)

    def test_sum_equals_one_after_dbi_change(self):
        snap = snapshot_config()
        try:
            apply_macro_weights(gold_pct=portfolio_cfg.WEIGHTS["gold"], dbi_pct=0.10)
            assert abs(sum(portfolio_cfg.WEIGHTS.values()) - 1.0) < 1e-9
        finally:
            restore_config(snap)

    def test_sum_equals_one_after_both_change(self):
        """gold=0.20 + dbi=0.08 is a realistic slider combo that fits the
        ~0.88 non-cash budget (other sleeves sum to ~0.66)."""
        snap = snapshot_config()
        try:
            apply_macro_weights(gold_pct=0.20, dbi_pct=0.08)
            assert abs(sum(portfolio_cfg.WEIGHTS.values()) - 1.0) < 1e-9
            assert portfolio_cfg.WEIGHTS["gold"] == pytest.approx(0.20)
            assert portfolio_cfg.WEIGHTS["dbi"] == pytest.approx(0.08)
            # Cash should be positive: 1 - (0.66 fixed + 0.20 gold + 0.08 dbi) ≈ 0.06
            assert portfolio_cfg.WEIGHTS["cash"] > 0.0
        finally:
            restore_config(snap)

    def test_rejects_out_of_range(self):
        snap = snapshot_config()
        try:
            with pytest.raises(ValueError, match="gold_pct must be in"):
                apply_macro_weights(gold_pct=-0.1, dbi_pct=0.05)
            with pytest.raises(ValueError, match="gold_pct must be in"):
                apply_macro_weights(gold_pct=1.5, dbi_pct=0.05)
            with pytest.raises(ValueError, match="dbi_pct must be in"):
                apply_macro_weights(gold_pct=0.18, dbi_pct=1.5)
        finally:
            restore_config(snap)

    def test_rejects_negative_cash(self):
        """gold + dbi too large → cash would be negative → ValueError."""
        snap = snapshot_config()
        try:
            # Other non-cash weights sum to ~0.8 (equity 0.47 + crypto/bonds/em/pension).
            # Setting gold=0.5 and dbi=0.5 would push non-cash to ~1.8, leaving cash=-0.8.
            with pytest.raises(ValueError, match="negative cash"):
                apply_macro_weights(gold_pct=0.5, dbi_pct=0.5)
        finally:
            restore_config(snap)
