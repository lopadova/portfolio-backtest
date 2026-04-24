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
import re
import tomllib
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

_WEIGHTS_SUM_TOLERANCE = 0.002
DEFAULT_PORTFOLIOS_DIR = Path(__file__).resolve().parent.parent / "portfolios"

# Slug names that point at repo-shipped presets — these must never be
# overwritten or deleted by user operations, only updated intentionally
# via a PR to the repo itself.
RESERVED_PRESET_SLUGS = frozenset({"four_umbrellas"})

_SLUG_VALID_CHARS = re.compile(r"[a-z0-9]+")


def slugify(name: str) -> str:
    """Turn a human-readable portfolio name into a stable filesystem slug.

    Rules: lowercase, non-alphanumeric → ``_``, collapse consecutive ``_``,
    strip leading/trailing ``_``. Raises ``ValueError`` if the result is
    empty (the input was all whitespace / punctuation) or if it collides
    with a :data:`RESERVED_PRESET_SLUGS` entry.
    """
    if not isinstance(name, str):
        raise ValueError(f"Portfolio name must be a string, got {type(name).__name__}")
    tokens = _SLUG_VALID_CHARS.findall(name.lower())
    slug = "_".join(tokens)
    if not slug:
        raise ValueError(
            f"Cannot derive a slug from {name!r}: the name contains no "
            f"alphanumeric characters."
        )
    if slug in RESERVED_PRESET_SLUGS:
        raise ValueError(
            f"Slug {slug!r} is reserved for a shipped preset and cannot be "
            f"used as a user-portfolio filename. Pick a different name."
        )
    return slug


@dataclass
class PortfolioMetricsCache:
    """Cached summary statistics of the last successful simulation.

    Stored as the ``[metrics]`` section of a saved portfolio TOML so the
    "Portafogli salvati" UI page can show a preview (CAGR / Vol / MaxDD /
    Period) without re-running the simulation.

    ``run_timestamp`` is a naive UTC datetime (no tz info) — TOML
    round-trips best with RFC 3339 strings, and naive UTC keeps the
    comparison between saves deterministic regardless of the user's
    local timezone.
    """

    cagr: float
    annualized_vol: float
    max_drawdown: float
    period_start: pd.Timestamp
    period_end: pd.Timestamp
    run_timestamp: datetime


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
    # PR6 — optional cached metrics from the last successful simulation.
    # Never enters `validate()` (it's auxiliary metadata, not engine config).
    cached_metrics: Optional[PortfolioMetricsCache] = None

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
        """Build a Portfolio from a plain dict (used by all loaders).

        Malformed input always surfaces as ``ValueError`` with a clear message
        — never an uncaught ``KeyError`` / ``TypeError`` — so the CLI's
        ``except ValueError`` path in ``_resolve_portfolio_or_exit`` can
        translate it to a friendly exit-2 message instead of a traceback.
        """
        if "name" not in data:
            raise ValueError("Portfolio definition missing required field: name")
        if "assets" not in data or not data["assets"]:
            raise ValueError("Portfolio definition missing required field: assets")
        if not isinstance(data["assets"], list):
            raise ValueError(
                f"Portfolio field 'assets' must be a list of allocations, "
                f"got {type(data['assets']).__name__}"
            )
        assets: List[AssetAllocation] = []
        for i, a in enumerate(data["assets"]):
            if not isinstance(a, dict):
                raise ValueError(
                    f"Portfolio asset at index {i} must be a mapping with "
                    f"'key' and 'weight' fields, got {type(a).__name__}"
                )
            if "key" not in a:
                raise ValueError(
                    f"Portfolio asset at index {i} missing required field: key"
                )
            if "weight" not in a:
                raise ValueError(
                    f"Portfolio asset at index {i} missing required field: weight"
                )
            try:
                asset = AssetAllocation(key=str(a["key"]), weight=float(a["weight"]))
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Portfolio asset at index {i} (key={a.get('key')!r}) has "
                    f"invalid weight {a.get('weight')!r}: {exc}"
                ) from exc
            assets.append(asset)
        try:
            rebalance_months = tuple(int(m) for m in data.get("rebalance_months", [1, 7]))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Portfolio 'rebalance_months' must be a list of ints 1..12, "
                f"got {data.get('rebalance_months')!r}: {exc}"
            ) from exc
        try:
            transaction_cost_bps = float(data.get("transaction_cost_bps", 20.0))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Portfolio 'transaction_cost_bps' must be a number, "
                f"got {data.get('transaction_cost_bps')!r}: {exc}"
            ) from exc
        cached_metrics = _parse_metrics_section(data.get("metrics"))
        p = cls(
            name=str(data["name"]),
            assets=assets,
            options_overlay=bool(data.get("options_overlay", False)),
            rebalance_months=rebalance_months,
            transaction_cost_bps=transaction_cost_bps,
            notes=str(data.get("notes", "")),
            cached_metrics=cached_metrics,
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
        """Emit a TOML representation of this Portfolio, including the
        ``[metrics]`` section when ``cached_metrics`` is set. Hand-rolled
        to avoid pulling in a third-party TOML writer; covers the exact
        schema read by :meth:`from_dict`. Note that this module itself is
        not stdlib-only — it imports pandas for timestamp handling — so
        the rationale is "no extra TOML-writer dep", not "zero 3rd-party
        imports"."""
        return _dump_toml(self)

    def save_to(self, path: str | Path, overwrite: bool = False) -> Path:
        """Serialize to TOML and write to ``path``.

        Returns the resolved output path for chaining / logging.

        Raises:
            FileExistsError: target exists and ``overwrite`` is False.
            ValueError: ``path`` targets the shipped Four Umbrellas preset
                — shipped presets live in the repo and must only be updated
                via a code change, never from a user-initiated save.
        """
        path = Path(path)
        stem = path.stem
        if stem in RESERVED_PRESET_SLUGS:
            raise ValueError(
                f"Refusing to overwrite shipped preset {stem!r}. Shipped "
                f"presets are part of the repo and must be updated via a "
                f"code change, not a user save."
            )
        if path.exists() and not overwrite:
            raise FileExistsError(
                f"Portfolio already saved at {path}; pass overwrite=True to replace."
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_toml(), encoding="utf-8")
        return path.resolve()


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
    """Minimal TOML emitter for the Portfolio schema (+ optional metrics)."""
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
    if p.cached_metrics is not None:
        m = p.cached_metrics
        lines.append("")
        lines.append("[metrics]")
        lines.append(f"cagr = {m.cagr}")
        lines.append(f"annualized_vol = {m.annualized_vol}")
        lines.append(f"max_drawdown = {m.max_drawdown}")
        # Dates as ISO YYYY-MM-DD strings (TOML's native local-date would
        # also work but tomllib returns it as datetime.date — one less
        # conversion to reason about if we just use strings both ways).
        lines.append(f'period_start = "{m.period_start.date().isoformat()}"')
        lines.append(f'period_end = "{m.period_end.date().isoformat()}"')
        lines.append(f'run_timestamp = "{m.run_timestamp.isoformat(timespec="seconds")}"')
    return "\n".join(lines) + "\n"


def _parse_metrics_section(section: Any) -> Optional[PortfolioMetricsCache]:
    """Parse the optional ``[metrics]`` section of a Portfolio TOML.

    Returns ``None`` when the section is absent. Raises ``ValueError`` on
    a partial or badly-typed section — presence of ``[metrics]`` means the
    caller is committing to all six required fields.
    """
    if section is None:
        return None
    if not isinstance(section, dict):
        raise ValueError("Portfolio [metrics] section must be a TOML table")
    required = (
        "cagr", "annualized_vol", "max_drawdown",
        "period_start", "period_end", "run_timestamp",
    )
    missing = [f for f in required if f not in section]
    if missing:
        raise ValueError(
            f"Portfolio [metrics] section missing required field(s): "
            f"{', '.join(missing)}"
        )
    try:
        return PortfolioMetricsCache(
            cagr=float(section["cagr"]),
            annualized_vol=float(section["annualized_vol"]),
            max_drawdown=float(section["max_drawdown"]),
            period_start=pd.Timestamp(section["period_start"]),
            period_end=pd.Timestamp(section["period_end"]),
            run_timestamp=_parse_iso_datetime(section["run_timestamp"]),
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Portfolio [metrics] section has a badly-typed field: {exc}"
        ) from exc


def _parse_iso_datetime(raw: Any) -> datetime:
    """Accept a string or an already-parsed datetime (tomllib returns
    datetime objects for native TOML datetimes)."""
    if isinstance(raw, datetime):
        return raw
    if isinstance(raw, str):
        return datetime.fromisoformat(raw)
    raise ValueError(
        f"run_timestamp must be an ISO string or datetime, got {type(raw).__name__}"
    )


def list_available_presets(root: Path = DEFAULT_PORTFOLIOS_DIR) -> List[dict]:
    """Return one dict per preset in ``root``. Schema:
    ``{name, display_name, path, n_assets, notes, cached_metrics, is_reserved}``.

    - ``cached_metrics``: parsed :class:`PortfolioMetricsCache` when the
      TOML has a ``[metrics]`` section, else ``None``.
    - ``is_reserved``: ``True`` when the preset slug is in
      :data:`RESERVED_PRESET_SLUGS` (i.e. shipped with the repo and
      protected against save/overwrite/delete via the CLI or UI).

    Errors on individual files are caught so one malformed preset
    doesn't break listing the others.
    """
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
                    "cached_metrics": portfolio.cached_metrics,
                    "is_reserved": p.stem in RESERVED_PRESET_SLUGS,
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
                    "cached_metrics": None,
                    "is_reserved": p.stem in RESERVED_PRESET_SLUGS,
                }
            )
    return entries
