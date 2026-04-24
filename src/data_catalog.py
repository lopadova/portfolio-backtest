"""
Data catalog — typed metadata index of all datasets this engine can consume.

The catalog is the SSOT for "what datasets exist, what are their properties".
It decouples the engine from the hardcoded ``SYMBOL_MAP`` / ``HEDGED`` /
``TER_ANNUAL`` dicts in ``src/portfolio.py`` so user-facing UIs (PR3+) can
enumerate available assets with display names, categories, and start dates.

The catalog manifest lives at ``data/catalog.toml`` and is read stdlib-only
(``tomllib``). Start/end dates are inferred at runtime from the first/last
row of the corresponding CSV in ``data/raw/``; missing CSVs leave those
fields as ``None`` (not an error — the asset is simply "not downloaded yet").
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd

DEFAULT_CATALOG_PATH = Path(__file__).resolve().parent.parent / "data" / "catalog.toml"
DEFAULT_DATA_RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

_REQUIRED_FIELDS = (
    "display_name",
    "filename",
    "category",
    "native_ccy",
    "is_hedged",
    "ter",
)
_VALID_CATEGORIES = frozenset(
    {"equity", "bond", "commodity", "crypto", "fx", "cash", "alt"}
)


@dataclass(frozen=True)
class AssetInfo:
    """One entry of the catalog."""

    key: str
    display_name: str
    filename: str
    category: str
    native_ccy: str
    is_hedged: bool
    ter: float
    description: str = ""
    aliases: Tuple[str, ...] = ()
    # Inferred from the CSV when present:
    start_date: Optional[pd.Timestamp] = None
    end_date: Optional[pd.Timestamp] = None


def load_catalog(path: Path = DEFAULT_CATALOG_PATH) -> Dict[str, AssetInfo]:
    """Load the catalog manifest from a TOML file.

    Raises:
        FileNotFoundError: path does not exist.
        ValueError: a catalog entry is missing a required field or has a
            category outside the allowed set.
        tomllib.TOMLDecodeError: malformed TOML.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Catalog file not found: {path}")
    with path.open("rb") as fh:
        data = tomllib.load(fh)
    assets_table = data.get("assets", {})
    if not isinstance(assets_table, dict):
        raise ValueError(
            "Catalog 'assets' must be a table of [assets.<key>] entries"
        )
    out: Dict[str, AssetInfo] = {}
    for key, entry in assets_table.items():
        if not isinstance(entry, dict):
            raise ValueError(f"Catalog entry {key!r} must be a table")
        for required in _REQUIRED_FIELDS:
            if required not in entry:
                raise ValueError(
                    f"Catalog entry {key!r} missing required field: {required}"
                )
        category = str(entry["category"])
        if category not in _VALID_CATEGORIES:
            raise ValueError(
                f"Catalog entry {key!r}: invalid category {category!r}. "
                f"Allowed: {sorted(_VALID_CATEGORIES)}"
            )
        aliases_raw = entry.get("aliases", [])
        if not isinstance(aliases_raw, list):
            raise ValueError(f"Catalog entry {key!r}: 'aliases' must be a list")
        out[key] = AssetInfo(
            key=str(key),
            display_name=str(entry["display_name"]),
            filename=str(entry["filename"]),
            category=category,
            native_ccy=str(entry["native_ccy"]),
            is_hedged=bool(entry["is_hedged"]),
            ter=float(entry["ter"]),
            description=str(entry.get("description", "")),
            aliases=tuple(str(a) for a in aliases_raw),
        )
    return out


def _csv_date_bounds(csv_path: Path) -> Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]:
    """Return (start, end) dates from a catalog CSV.

    Accepts the project's two CSV conventions: ``date,value`` or
    ``date,close``. Returns ``(None, None)`` if the file can't be parsed —
    callers treat missing CSVs as "data not fetched yet", not as errors.
    """
    try:
        df = pd.read_csv(csv_path, parse_dates=["date"])
    except Exception:
        return None, None
    if df.empty or "date" not in df.columns:
        return None, None
    dates = df["date"].dropna()
    if dates.empty:
        return None, None
    return pd.Timestamp(dates.min()), pd.Timestamp(dates.max())


def augment_with_raw_dates(
    catalog: Dict[str, AssetInfo],
    data_raw_dir: Path = DEFAULT_DATA_RAW_DIR,
) -> Dict[str, AssetInfo]:
    """Return a NEW catalog dict with ``start_date`` / ``end_date`` populated
    from the CSV files in ``data_raw_dir`` where they exist.

    Does not mutate the input. Non-existent CSVs leave the fields as ``None``.
    """
    data_raw_dir = Path(data_raw_dir)
    out: Dict[str, AssetInfo] = {}
    for key, info in catalog.items():
        csv_path = data_raw_dir / f"{info.filename}.csv"
        if csv_path.is_file():
            start, end = _csv_date_bounds(csv_path)
            out[key] = replace(info, start_date=start, end_date=end)
        else:
            out[key] = info
    return out


def find_by_alias(
    catalog: Dict[str, AssetInfo], alias_or_key: str
) -> Optional[AssetInfo]:
    """Look up by primary key first, then by alias. Returns ``None`` if no
    match. Matching is case-sensitive to keep catalog keys unambiguous."""
    if alias_or_key in catalog:
        return catalog[alias_or_key]
    for info in catalog.values():
        if alias_or_key in info.aliases:
            return info
    return None
