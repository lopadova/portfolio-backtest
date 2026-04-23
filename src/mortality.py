"""
Italian mortality tables — ISTAT-based lifetime mortality probabilities.

The bundled CSV `data/mortality/istat_mortality_2023.csv` contains annual
mortality probabilities (qx) by age (0-110) and sex (male/female),
calibrated to ISTAT's published 2023 life tables.

Used by the FIRE calculator to stochastically sample an age-at-death for
each Monte Carlo simulation path, rather than using a fixed horizon.

References:
    - ISTAT Tavole di mortalità: https://www.istat.it/
    - Italian life expectancy 2023: ~81.5y (males), ~85.6y (females)
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

import numpy as np
import pandas as pd


MORTALITY_CSV = Path(__file__).resolve().parent.parent / "data" / "mortality" / "istat_mortality_2023.csv"


def load_mortality_table(path: Optional[Path] = None) -> pd.DataFrame:
    """
    Load the ISTAT mortality table from CSV. Columns: age, qx_male, qx_female.
    """
    csv_path = path if path is not None else MORTALITY_CSV
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Mortality table not found at {csv_path}. "
            f"Ensure data/mortality/istat_mortality_2023.csv is present."
        )
    df = pd.read_csv(csv_path)
    required = {"age", "qx_male", "qx_female"}
    if not required.issubset(df.columns):
        raise ValueError(f"Mortality table missing columns. Required: {required}, found: {set(df.columns)}")
    return df


def sample_death_age(
    current_age: int,
    sex: Literal["M", "F", "male", "female"],
    n_samples: int,
    seed: int = 42,
    table: Optional[pd.DataFrame] = None,
) -> np.ndarray:
    """
    Sample death ages for n_samples simulated individuals using year-by-year
    Bernoulli draws against the qx table.

    Algorithm:
        For each simulation, start at current_age and advance one year at a time.
        At each age, Bernoulli(qx) → if dies, record age; else continue.

    Returns: np.ndarray of shape (n_samples,) with integer ages of death.
    """
    if table is None:
        table = load_mortality_table()
    col = _qx_column_for_sex(sex)
    qx_by_age = dict(zip(table["age"], table[col]))
    max_age = int(table["age"].max())

    rng = np.random.default_rng(seed)
    death_ages = np.empty(n_samples, dtype=np.int32)

    # Vectorize across all simulations: at each age, draw Bernoulli for all alive
    alive = np.ones(n_samples, dtype=bool)
    death_ages.fill(max_age)  # default to max age if never triggered
    for age in range(current_age, max_age + 1):
        qx = qx_by_age.get(age, 1.0)
        if not alive.any():
            break
        draws = rng.random(n_samples)
        dies_this_year = alive & (draws < qx)
        death_ages[dies_this_year] = age
        alive &= ~dies_this_year

    return death_ages


def _qx_column_for_sex(sex: str) -> str:
    """
    Validate `sex` and return the corresponding qx column name.
    Accepts: "M", "F", "male", "female" (case-insensitive).
    Raises ValueError for any other value to prevent silent misclassification.
    """
    if not isinstance(sex, str):
        raise ValueError(f"sex must be a string, got {sex!r}")
    normalized = sex.strip().lower()
    if normalized in ("m", "male"):
        return "qx_male"
    if normalized in ("f", "female"):
        return "qx_female"
    raise ValueError(
        f"sex must be one of 'M', 'F', 'male', 'female' (case-insensitive), got {sex!r}"
    )


def life_expectancy(current_age: int, sex: str, table: Optional[pd.DataFrame] = None) -> float:
    """
    Compute remaining life expectancy from the mortality table analytically
    (no sampling needed). Returns expected years of life remaining.
    """
    if table is None:
        table = load_mortality_table()
    col = _qx_column_for_sex(sex)
    qx_by_age = dict(zip(table["age"], table[col]))
    max_age = int(table["age"].max())

    # P(alive at each future age) and expected years
    survival = 1.0
    expected_years = 0.0
    for age in range(current_age, max_age + 1):
        qx = qx_by_age.get(age, 1.0)
        px = 1.0 - qx  # probability of surviving this year
        # Expected years in this age = survival × (probability of dying this year × 0.5 + probability of surviving this year × 1.0)
        # Approximation: each year alive contributes 1.0 to life expectancy; we accrue survival × 1
        expected_years += survival * (1.0 - 0.5 * qx)  # half-year adjustment for those who die this year
        survival *= px
    return expected_years
