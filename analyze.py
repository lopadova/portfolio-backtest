"""
Standalone AI analyzer — takes an existing output/REPORT.md and sends it
to a configured LLM provider for analysis.

Usage:
    # Default provider: OpenRouter (requires OPENROUTER_API_KEY in env)
    python analyze.py --results output/

    # Specific provider + model
    python analyze.py --results output/ --provider anthropic --model claude-opus-4-7
    python analyze.py --results output/ --provider openai --model gpt-4o
    python analyze.py --results output/ --provider local --model kimi-k2

    # Extra context from sensitivity analysis
    python analyze.py --results output/ --extra output/sensitivity/gold.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.ai_analyzer import (
    get_analyzer,
    build_analysis_prompt,
    save_analysis,
    DEFAULT_MODELS,
)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="AI analysis of backtest results")
    p.add_argument("--results", type=Path, default=Path("output"),
                   help="Path to the output directory containing REPORT.md")
    p.add_argument("--provider",
                   choices=["openrouter", "openai", "anthropic", "local"],
                   default="openrouter",
                   help="AI provider (default: openrouter)")
    p.add_argument("--model", type=str, default=None,
                   help=f"Model name (defaults per provider: {DEFAULT_MODELS})")
    p.add_argument("--extra", type=Path, nargs="+",
                   help="Additional context files (CSVs, Markdown)")
    p.add_argument("--output", type=Path, default=None,
                   help="Output path (default: <results>/AI_ANALYSIS.md)")
    return p.parse_args(argv)


def main():
    args = parse_args()

    report_path = args.results / "REPORT.md"
    if not report_path.exists():
        raise SystemExit(
            f"REPORT.md not found at {report_path}. "
            f"Run `python backtest.py` first to generate the report."
        )

    report_md = report_path.read_text(encoding="utf-8")

    extra_context = None
    if args.extra:
        parts = []
        for p in args.extra:
            if p.exists():
                parts.append(f"\n### {p.name}\n\n```\n{p.read_text(encoding='utf-8')}\n```")
        extra_context = "\n".join(parts) if parts else None

    prompt = build_analysis_prompt(report_md, extra_context=extra_context)

    print(f"Sending analysis request to {args.provider} ({args.model or DEFAULT_MODELS[args.provider]})...")
    analyzer = get_analyzer(args.provider, args.model)
    response = analyzer.analyze(prompt)

    output_path = args.output or (args.results / "AI_ANALYSIS.md")
    save_analysis(response, output_path)

    print(f"\n{'=' * 60}")
    print("AI Analysis complete")
    print(f"{'=' * 60}")
    print(f"Provider:      {response.provider}")
    print(f"Model:         {response.model}")
    # Use `is not None` rather than truthy: 0 tokens is a legitimate value
    # worth surfacing (e.g. some providers report 0 for streamed responses).
    if response.prompt_tokens is not None:
        print(f"Tokens in:     {response.prompt_tokens:,}")
    if response.completion_tokens is not None:
        print(f"Tokens out:    {response.completion_tokens:,}")
    print(f"\nAnalysis saved: {output_path.resolve()}")


if __name__ == "__main__":
    main()
