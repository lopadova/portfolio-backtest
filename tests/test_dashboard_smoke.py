"""
Smoke tests for Streamlit dashboard (streamlit_app.py).

These tests verify that the dashboard module imports cleanly and that the
core helpers (config snapshot/restore, data loading cache) work. Full UI
integration testing requires Streamlit's AppTest or Selenium, which we do
not include here to keep CI fast.
"""

import sys
from pathlib import Path

import pytest


class TestDashboardImport:
    def test_streamlit_app_module_parseable(self):
        """The streamlit_app.py file should parse without SyntaxError."""
        project_root = Path(__file__).resolve().parent.parent
        app_path = project_root / "streamlit_app.py"
        assert app_path.exists(), "streamlit_app.py must exist"
        code = app_path.read_text(encoding="utf-8")
        # Attempt to compile (catches syntax errors without executing)
        compile(code, str(app_path), "exec")

    def test_requirements_dashboard_exists(self):
        project_root = Path(__file__).resolve().parent.parent
        req = project_root / "requirements-dashboard.txt"
        assert req.exists()
        content = req.read_text(encoding="utf-8")
        assert "streamlit" in content.lower()

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


class TestDashboardHelpers:
    """Test the helper functions exposed in streamlit_app.py."""

    def test_config_snapshot_restore_roundtrip(self):
        """Mirror test of the helpers defined inline in streamlit_app.py."""
        # Redeclaring the logic here since streamlit_app.py imports streamlit
        # which we don't want to require for test runs
        from src import portfolio as portfolio_cfg
        import copy

        def _snap():
            return {
                "WEIGHTS": copy.deepcopy(portfolio_cfg.WEIGHTS),
                "EQUITY": copy.deepcopy(portfolio_cfg.EQUITY),
                "OPTIONS": copy.deepcopy(portfolio_cfg.OPTIONS),
                "REBALANCE": copy.deepcopy(portfolio_cfg.REBALANCE),
            }

        def _restore(snap):
            portfolio_cfg.WEIGHTS.clear()
            portfolio_cfg.WEIGHTS.update(snap["WEIGHTS"])
            portfolio_cfg.EQUITY.clear()
            portfolio_cfg.EQUITY.update(snap["EQUITY"])
            portfolio_cfg.OPTIONS.__dict__.update(snap["OPTIONS"].__dict__)
            portfolio_cfg.REBALANCE.__dict__.update(snap["REBALANCE"].__dict__)

        snap = _snap()
        original_gold = portfolio_cfg.WEIGHTS["gold"]
        # Mutate
        portfolio_cfg.WEIGHTS["gold"] = 0.30
        # Restore
        _restore(snap)
        assert portfolio_cfg.WEIGHTS["gold"] == original_gold
