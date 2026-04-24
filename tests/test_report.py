"""
Unit tests for src/report.py — Markdown report generation.
"""

import pandas as pd
import pytest

from src.metrics import compute_all
from src.report import generate_markdown_report


class TestReportGeneration:
    def test_report_creates_file(self, tmp_output_dir, sample_returns_dict):
        stats_list = [compute_all(name, r) for name, r in sample_returns_dict.items()]
        report_path = generate_markdown_report(
            output_dir=tmp_output_dir,
            returns_dict=sample_returns_dict,
            stats_list=stats_list,
            start_date=list(sample_returns_dict.values())[0].index[0],
            end_date=list(sample_returns_dict.values())[0].index[-1],
            start_nav=100_000.0,
            options_enabled=True,
        )
        assert report_path.exists()
        assert report_path.name == "REPORT.md"

    def test_report_contains_expected_sections(self, tmp_output_dir, sample_returns_dict):
        stats_list = [compute_all(name, r) for name, r in sample_returns_dict.items()]
        generate_markdown_report(
            output_dir=tmp_output_dir,
            returns_dict=sample_returns_dict,
            stats_list=stats_list,
            start_date=list(sample_returns_dict.values())[0].index[0],
            end_date=list(sample_returns_dict.values())[0].index[-1],
            start_nav=100_000.0,
            options_enabled=True,
        )
        content = (tmp_output_dir / "REPORT.md").read_text(encoding="utf-8")

        for section in [
            "# Portfolio Backtest Report",
            "## Run configuration",
            "## Summary statistics",
            "## Equity curve",
            "## Drawdown analysis",
            "## Crisis period zoom",
            "## Rolling metrics",
            "## Annual returns",
            "## Metrics comparison",
            "Disclaimer",
        ]:
            assert section in content, f"Missing section: {section}"

    def test_report_includes_portfolio_names(self, tmp_output_dir, sample_returns_dict):
        stats_list = [compute_all(name, r) for name, r in sample_returns_dict.items()]
        generate_markdown_report(
            output_dir=tmp_output_dir,
            returns_dict=sample_returns_dict,
            stats_list=stats_list,
            start_date=list(sample_returns_dict.values())[0].index[0],
            end_date=list(sample_returns_dict.values())[0].index[-1],
            start_nav=100_000.0,
            options_enabled=False,
        )
        content = (tmp_output_dir / "REPORT.md").read_text(encoding="utf-8")
        for portfolio_name in sample_returns_dict.keys():
            assert portfolio_name in content

    def test_report_indicates_disabled_options(self, tmp_output_dir, sample_returns_dict):
        stats_list = [compute_all(name, r) for name, r in sample_returns_dict.items()]
        generate_markdown_report(
            output_dir=tmp_output_dir,
            returns_dict=sample_returns_dict,
            stats_list=stats_list,
            start_date=list(sample_returns_dict.values())[0].index[0],
            end_date=list(sample_returns_dict.values())[0].index[-1],
            start_nav=100_000.0,
            options_enabled=False,
        )
        content = (tmp_output_dir / "REPORT.md").read_text(encoding="utf-8")
        assert "DISABLED" in content
