"""
Stdlib-only environment diagnostics for CLI entry points.

If the user runs `python fire.py` (or any other entry point) without first
activating the project virtual environment, they hit a raw
`ModuleNotFoundError: No module named 'numpy'` traceback with no guidance
on how to fix it. This module intercepts that failure mode and prints a
friendly, actionable message that covers:

  * activating the venv (Windows + POSIX),
  * invoking the venv Python directly (no activation needed),
  * creating the venv from scratch if it doesn't exist yet.

Because this module runs *before* third-party deps are imported, it MUST
use stdlib only. `importlib.util.find_spec` has been stable since 3.4 and
does not import the target package — it only checks whether the loader
can locate it.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from textwrap import dedent
from typing import Iterable, List, Optional, Sequence


def find_missing(deps: Sequence[str]) -> List[str]:
    """Return the subset of `deps` that cannot be imported in the current
    Python environment. Pure helper: no side-effects, no stdout."""
    return [d for d in deps if importlib.util.find_spec(d) is None]


def format_missing_deps_message(
    missing: Iterable[str],
    script_name: str = "this script",
) -> str:
    """Build the user-facing error message for missing deps. Kept separate
    from `require_runtime_deps` so tests can assert on its contents without
    triggering `sys.exit`."""
    missing_list = ", ".join(missing)
    # Use single-width box chars; no Unicode heavy lifting to keep this
    # readable even on Windows terminals with legacy code pages.
    return dedent(
        f"""\
        ============================================================
        ERROR: missing required Python package(s): {missing_list}
        ============================================================

        This almost always means the project's virtual environment is
        not active for this shell — the system Python doesn't have the
        project dependencies installed.

          Windows (PowerShell):
            .venv\\Scripts\\Activate.ps1
            python {script_name}

          Windows (cmd.exe):
            .venv\\Scripts\\activate.bat
            python {script_name}

          macOS / Linux:
            source .venv/bin/activate
            python {script_name}

          Or invoke the venv Python directly (no activation needed):
            .venv\\Scripts\\python.exe {script_name}      # Windows
            .venv/bin/python {script_name}               # macOS / Linux

        If the venv does not exist yet:
            python -m venv .venv
            .venv\\Scripts\\Activate.ps1     # or: source .venv/bin/activate
            pip install -r requirements.txt
        ============================================================
        """
    )


def require_runtime_deps(
    deps: Sequence[str],
    script_name: str = "this script",
    _exit: "callable" = sys.exit,
    _stderr=None,
) -> None:
    """If any of `deps` is not importable, print a friendly message to stderr
    and exit with status 1. No-op when every dep is already available.

    Arguments `_exit` and `_stderr` are injection points for tests; real
    callers should pass only `deps` and `script_name`."""
    missing = find_missing(deps)
    if not missing:
        return
    stream = _stderr if _stderr is not None else sys.stderr
    print(format_missing_deps_message(missing, script_name), file=stream)
    _exit(1)


def infer_script_name() -> str:
    """Best-effort script name for the error message. Falls back to the
    Python process argv[0] if called outside a file context."""
    try:
        return os.path.basename(sys.argv[0]) or "this script"
    except Exception:
        return "this script"


# ---------------------------------------------------------------------------
# Minimal stdlib .env loader
# ---------------------------------------------------------------------------
#
# AI-analysis entry points (`analyze.py`, `backtest.py --ai-analysis`, the
# Streamlit dashboard) read API keys from environment variables. Non-technical
# users on Windows find `setx` / `export` awkward, so we support a plain
# `.env` file at the project root as the officially-documented way to set
# keys. We parse it here with stdlib only (no `python-dotenv` dependency):
# keeping this module third-party-free means it can run BEFORE the venv check
# above — i.e. we can load the `.env` whether or not the venv is active.


def _parse_env_line(line: str) -> Optional[tuple]:
    """Parse one line of a .env file. Returns (key, value) or None.

    Supports `KEY=value`, `KEY="quoted value"`, leading `export KEY=...`,
    inline `#` comments outside quotes, and strips surrounding whitespace.
    Ignores blank lines and full-line comments."""
    s = line.strip()
    if not s or s.startswith("#"):
        return None
    if s.startswith("export "):
        s = s[len("export "):].lstrip()
    if "=" not in s:
        return None
    key, _, raw = s.partition("=")
    key = key.strip()
    if not key or not key.replace("_", "").isalnum():
        return None
    value = raw.strip()
    # Strip matching single- or double-quote wrappers (no escape handling —
    # fine for API keys, which are alphanumeric). If unquoted, strip inline
    # comments (` # comment` after the value).
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
        value = value[1:-1]
    else:
        comment_pos = value.find(" #")
        if comment_pos >= 0:
            value = value[:comment_pos].rstrip()
    return key, value


def load_dotenv(path: Optional[Path] = None, override: bool = False) -> dict:
    """Load variables from a `.env` file into `os.environ`.

    Args:
        path: Path to the `.env` file. Defaults to `<project_root>/.env`
            (project root = parent of `src/`).
        override: If True, values in `.env` replace pre-existing env vars.
            Default False — real environment wins, matches `python-dotenv`.

    Returns:
        Dict of keys that were actually applied to `os.environ` (useful for
        logging / debugging). Returns `{}` if the file is absent — loading is
        a best-effort no-op so users who prefer shell exports aren't affected.
    """
    if path is None:
        path = Path(__file__).resolve().parent.parent / ".env"
    else:
        path = Path(path)
    if not path.is_file():
        return {}
    applied = {}
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return {}
    for line in text.splitlines():
        parsed = _parse_env_line(line)
        if parsed is None:
            continue
        key, value = parsed
        if not override and key in os.environ:
            continue
        os.environ[key] = value
        applied[key] = value
    return applied
