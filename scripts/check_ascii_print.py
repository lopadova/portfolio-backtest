#!/usr/bin/env python3
"""ASCII-only check for ``print()`` arguments in CLI-facing modules.

LESSONS.md Theme 18 (Windows ``cp1252`` stdout codec): emoji, em-dashes,
arrows, and other non-ASCII characters inside ``print(...)`` arguments
crash with ``UnicodeEncodeError`` on Windows' default terminal codec.
This check enforces ASCII-only literals inside ``print(...)`` calls so
those crashes can't reach a release.

Usage::

    python scripts/check_ascii_print.py backtest.py src/

Each argument can be a file or a directory; directories are walked
recursively for ``.py`` files. Excluded by default:
- ``streamlit_app.py``: Streamlit renders via the browser (UTF-8 native);
  the CLI-codec constraint doesn't apply there.
- ``.venv/`` and any directory starting with ``.``: virtualenv + dotfile dirs.
- ``__pycache__/``: caches.

Stdlib only (``ast``, ``pathlib``, ``sys``) so the script doubles as a
pre-commit hook with zero install footprint beyond Python itself.

Exits with status 0 when clean, 1 when at least one violation is found.
Each violation is reported on stderr in ``file:line: <msg>`` format so
editors and CI logs can jump to it.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path
from typing import Iterable, List, Tuple

EXCLUDED_FILES = {"streamlit_app.py"}
EXCLUDED_DIR_PARTS = {"__pycache__"}


def _iter_python_files(targets: Iterable[Path]) -> Iterable[Path]:
    for t in targets:
        if t.is_file() and t.suffix == ".py":
            if t.name in EXCLUDED_FILES:
                continue
            yield t
        elif t.is_dir():
            for p in t.rglob("*.py"):
                if p.name in EXCLUDED_FILES:
                    continue
                if any(part in EXCLUDED_DIR_PARTS for part in p.parts):
                    continue
                # Skip dotfile dirs (e.g. .venv, .git, .pytest_cache)
                if any(part.startswith(".") for part in p.parts):
                    continue
                yield p


def _check_string_literal(
    text: str, lineno: int, kind: str,
) -> List[Tuple[int, str]]:
    """Helper: emit a violation tuple if ``text`` has non-ASCII chars."""
    non_ascii = [c for c in text if ord(c) > 127]
    if not non_ascii:
        return []
    chars_repr = ", ".join(
        f"U+{ord(c):04X} ({c!r})" for c in dict.fromkeys(non_ascii)
    )
    return [(
        lineno,
        f"non-ASCII char(s) inside print() {kind}: {chars_repr}",
    )]


def _scan_print_arg(arg: ast.AST) -> List[Tuple[int, str]]:
    """Walk a single ``print(...)`` argument expression and return any
    non-ASCII violations.

    Covers two kinds of literals:

    1. Plain string constants: ``print("Done -- ok")`` — caught via
       ``ast.Constant(value=str)``.
    2. f-string literal segments: ``print(f"Done -- {n} items")`` —
       caught by walking ``ast.JoinedStr.values`` and checking each
       ``ast.Constant(value=str)`` (the static text between ``{...}``
       expression slots). The ``FormattedValue`` slots themselves
       (computed expressions) are NOT inspected — their runtime value
       is unknowable at AST time.

    Computed exprs (Call/Name/BinOp/...) get a pass: false-positive
    avoidance trumps theoretical completeness. The point of the rule
    is to catch obvious authored Unicode in CLI ``print()`` calls,
    which is what LESSONS Theme 18 documents.
    """
    if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
        return _check_string_literal(arg.value, arg.lineno, "string literal")

    if isinstance(arg, ast.JoinedStr):
        out: List[Tuple[int, str]] = []
        for piece in arg.values:
            if isinstance(piece, ast.Constant) and isinstance(piece.value, str):
                out.extend(_check_string_literal(
                    piece.value, piece.lineno, "f-string literal segment",
                ))
        return out

    return []


def _scan_file(path: Path) -> List[Tuple[int, str]]:
    """Return ``[(lineno, message), ...]`` for every print() arg whose
    literal text contains non-ASCII characters.

    Coverage: plain string constants AND the literal segments of f-strings.
    Computed expressions (``Name``, ``Call``, ``BinOp``, ...) are skipped.
    """
    try:
        source = path.read_text(encoding="utf-8")
    except OSError as e:  # pragma: no cover — defensive
        return [(0, f"read error: {e}")]
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError as e:
        return [(getattr(e, "lineno", 0) or 0, f"SyntaxError: {e.msg}")]

    violations: List[Tuple[int, str]] = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)):
            continue
        if node.func.id != "print":
            continue
        for arg in node.args:
            violations.extend(_scan_print_arg(arg))
    return violations


def main(argv: List[str]) -> int:
    if len(argv) < 2:
        print(
            "usage: check_ascii_print.py <file_or_dir> [<file_or_dir> ...]",
            file=sys.stderr,
        )
        return 2

    targets = [Path(a) for a in argv[1:]]
    missing = [t for t in targets if not t.exists()]
    if missing:
        print(
            f"error: target(s) not found: {', '.join(str(m) for m in missing)}",
            file=sys.stderr,
        )
        return 2

    total_violations = 0
    for py_file in sorted(_iter_python_files(targets)):
        for lineno, msg in _scan_file(py_file):
            print(f"{py_file}:{lineno}: {msg}", file=sys.stderr)
            total_violations += 1

    if total_violations:
        print(
            f"\ncheck_ascii_print: {total_violations} violation(s). "
            "CLI stdout must be ASCII-only (LESSONS.md Theme 18).",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
