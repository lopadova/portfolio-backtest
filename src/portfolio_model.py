"""
Generic portfolio model — typed container for arbitrary multi-asset allocations.

This module is the SSOT for what a "portfolio" means in the engine, independent
of any specific preset. A Portfolio is a name + a list of (asset_key, weight)
allocations + a handful of simulation settings. Weights MUST sum to ~1.0.

The engine's ``simulate_portfolio_generic`` (in ``src.rebalance``) consumes a
Portfolio directly — no module-level ``WEIGHTS``/``EQUITY``/... globals needed.
The legacy ``simulate_portfolio`` still works via a shim that constructs a
Portfolio from those globals, so pre-PR2 callers keep working unchanged.

Serialization format: **TOML** (``tomllib`` stdlib for reading). A portfolio
file looks like::

    name = "My portfolio"
    options_overlay = false
    rebalance_months = [1, 7]
    transaction_cost_bps = 20.0
    notes = "optional free-text"

    [[assets]]
    key    = "gold_lbma_monthly"
    weight = 0.5

    [[assets]]
    key    = "cash"
    weight = 0.5

The TOML writer used by tests and by the future "save portfolio" UI is a small
hand-rolled emitter in ``_dump_toml`` — keeps this module stdlib-only.
"""

from __future__ import annotations

import json
import tomllib
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

_WEIGHTS_SUM_TOLERANCE = 0.002
DEFAULT_PORTFOLIOS_DIR = Path(__file__).resolve().parent.parent / "portfolios"


@dataclass(frozen=True)
class AssetAllocation:
    """One row of a portfolio: which asset and how much of the NAV."""

    key: str
    weight: float


@dataclass
class Portfolio:
    """Generic, serializable portfolio definition."""

    name: str
    assets: List[AssetAllocation] = field(default_factory=list)
    options_overlay: bool = False
    rebalance_months: Tuple[int, ...] = (1, 7)
    transaction_cost_bps: float = 20.0
    notes: str = ""

    # ---------------------------------------------------------------- validate
    def validate(self) -> None:
        """Raise ``ValueError`` with an actionable message if the portfolio is
        malformed. The engine calls this before simulating — callers that load
        from user input should call it explicitly too."""
        if not self.name or not self.name.strip():
            raise ValueError("Portfolio.name must be a non-empty string")
        if not self.assets:
            raise ValueError("Portfolio.assets must contain at least one allocation")
        seen: set[str] = set()
        for a in self.assets:
            if not a.key or not a.key.strip():
                raise ValueError("AssetAllocation.key must be a non-empty string")
            if a.key in seen:
                raise ValueError(f"Duplicate asset key: {a.key!r}")
            seen.add(a.key)
            if a.weight < 0:
                raise ValueError(
                    f"Asset {a.key!r} has negative weight {a.weight}; long-only portfolios only"
                )
        total = sum(a.weight for a in self.assets)
        if abs(total - 1.0) > _WEIGHTS_SUM_TOLERANCE:
            raise ValueError(
                f"Weights sum to {total:.4f}, expected 1.0 ± {_WEIGHTS_SUM_TOLERANCE}. "
                f"Adjust the allocations so they total 100%."
            )
        if not isinstance(self.rebalance_months, tuple):
            raise ValueError(
                f"rebalance_months must be a tuple of ints (1–12), got {type(self.rebalance_months).__name__}"
            )
        for m in self.rebalance_months:
            if not isinstance(m, int) or not (1 <= m <= 12):
                raise ValueError(
                    f"rebalance_months entries must be ints in 1..12, got {m!r}"
                )
        if self.transaction_cost_bps < 0:
            raise ValueError(
                f"transaction_cost_bps must be >= 0, got {self.transaction_cost_bps}"
            )

    # --------------------------------------------------------------- accessors
    def to_weights_dict(self) -> Dict[str, float]:
        """Return a flat ``{key: weight}`` dict including cash. Used by the
        generic engine."""
        return {a.key: a.weight for a in self.assets}

    def to_legacy_target_weights(self) -> Dict[str, float]:
        """Return weights EXCLUDING the ``cash`` sleeve — shape matching the
        pre-PR2 ``src.rebalance.build_target_weights()`` output so the legacy
        shim can hand this straight to the engine."""
        return {a.key: a.weight for a in self.assets if a.key != "cash"}

    def cash_weight(self) -> float:
        for a in self.assets:
            if a.key == "cash":
                return a.weight
        return 0.0

    def keys(self) -> List[str]:
        return [a.key for a in self.assets]

    # ----------------------------------------------------------------- loaders
    @classmethod
    def from_dict(cls, data: dict) -> "Portfolio":
        """Build a Portfolio from a plain dict (used by all loaders)."""
        if "name" not in data:
            raise ValueError("Portfolio definition missing required field: name")
        if "assets" not in data or not data["assets"]:
            raise ValueError("Portfolio definition missing required field: assets")
        assets = [
            AssetAllocation(key=str(a["key"]), weight=float(a["weight"]))
            for a in data["assets"]
        ]
        rebalance_months = tuple(int(m) for m in data.get("rebalance_months", [1, 7]))
        p = cls(
            name=str(data["name"]),
            assets=assets,
            options_overlay=bool(data.get("options_overlay", False)),
            rebalance_months=rebalance_months,
            transaction_cost_bps=float(data.get("transaction_cost_bps", 20.0)),
            notes=str(data.get("notes", "")),
        )
        p.validate()
        return p

    @classmethod
    def from_toml(cls, path: str | Path) -> "Portfolio":
        """Load a Portfolio from a TOML file."""
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"Portfolio file not found: {path}")
        with path.open("rb") as fh:
            data = tomllib.load(fh)
        return cls.from_dict(data)

    @classmethod
    def from_json_inline(cls, raw: str) -> "Portfolio":
        """Parse a portfolio from an inline JSON string."""
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid inline-JSON portfolio: {e}") from e
        if not isinstance(data, dict):
            raise ValueError("Inline JSON must be a top-level object")
        return cls.from_dict(data)

    @classmethod
    def from_name(
        cls, name: str, root: Path = DEFAULT_PORTFOLIOS_DIR
    ) -> "Portfolio":
        """Resolve ``name`` to ``<root>/<name>.toml``. Raises a helpful error
        listing available names if not found."""
        root = Path(root)
        candidate = root / f"{name}.toml"
        if candidate.is_file():
            return cls.from_toml(candidate)
        available = sorted(p.stem for p in root.glob("*.toml")) if root.is_dir() else []
        hint = f"  Available: {', '.join(available)}" if available else "  (no presets found)"
        raise FileNotFoundError(
            f"No preset named {name!r} in {root}\n{hint}"
        )

    @classmethod
    def resolve(cls, spec: str, root: Path = DEFAULT_PORTFOLIOS_DIR) -> "Portfolio":
        """Single entry point for the CLI. Accepts:

        - inline JSON starting with ``{``
        - a filesystem path (ends in ``.toml`` or contains ``/`` / ``\\``)
        - a bare preset name resolved under ``root``
        """
        s = spec.strip()
        if s.startswith("{"):
            return cls.from_json_inline(s)
        if s.endswith(".toml") or "/" in s or "\\" in s:
            return cls.from_toml(s)
        return cls.from_name(s, root=root)

    # --------------------------------------------------------------- emit TOML
    def to_toml(self) -> str:
        """Emit a TOML representation of this Portfolio. Used by tests and the
        future "save portfolio" UI (PR5). Hand-rolled to keep the module
        stdlib-only; covers the exact schema we read in ``from_dict``."""
        return _dump_toml(self)


def _escape_toml_basic_string(s: str) -> str:
    """Escape a string for a TOML basic string literal (``"..."``)."""
    return (
        s.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", "\\n")
        .replace("\r", "\\r")
        .replace("\t", "\\t")
    )


def _dump_toml(p: Portfolio) -> str:
    """Minimal TOML emitter for the Portfolio schema."""
    lines: list[str] = []
    lines.append(f'name = "{_escape_toml_basic_string(p.name)}"')
    if p.notes:
        lines.append(f'notes = "{_escape_toml_basic_string(p.notes)}"')
    lines.append(f"options_overlay = {'true' if p.options_overlay else 'false'}")
    lines.append(
        "rebalance_months = [" + ", ".join(str(m) for m in p.rebalance_months) + "]"
    )
    lines.append(f"transaction_cost_bps = {p.transaction_cost_bps}")
    for a in p.assets:
        lines.append("")
        lines.append("[[assets]]")
        lines.append(f'key    = "{_escape_toml_basic_string(a.key)}"')
        lines.append(f"weight = {a.weight}")
    return "\n".join(lines) + "\n"


def list_available_presets(root: Path = DEFAULT_PORTFOLIOS_DIR) -> List[dict]:
    """Return one dict per preset in ``root``: ``{name, path, n_assets, notes}``.
    Used by the CLI ``--list-portfolios`` flag. Errors on individual files are
    caught so one malformed preset doesn't break listing."""
    root = Path(root)
    if not root.is_dir():
        return []
    entries: list[dict] = []
    for p in sorted(root.glob("*.toml")):
        try:
            portfolio = Portfolio.from_toml(p)
            entries.append(
                {
                    "name": p.stem,
                    "display_name": portfolio.name,
                    "path": str(p),
                    "n_assets": len(portfolio.assets),
                    "notes": portfolio.notes.splitlines()[0] if portfolio.notes else "",
                }
            )
        except Exception as e:  # pragma: no cover — defensive listing
            entries.append(
                {
                    "name": p.stem,
                    "display_name": "(invalid)",
                    "path": str(p),
                    "n_assets": 0,
                    "notes": f"ERROR: {e}",
                }
            )
    return entries
