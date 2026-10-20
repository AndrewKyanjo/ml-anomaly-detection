"""Convert the raw loan predictors into numeric features."""

import numpy as np
import pandas as pd

from .schema import FEATURE_COLUMNS, NUMERIC_COLUMNS, TARGET


def build_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Encode tenure and term, add income ratios, and exclude row identifiers."""
    if frame.empty or not frame.columns.is_unique:
        raise ValueError("Provide nonempty data with unique column names.")
    if TARGET in frame:
        raise ValueError(f"Remove {TARGET!r} before building features.")
    missing = set(FEATURE_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing predictors: {sorted(missing)}")
    result = frame.loc[:, list(FEATURE_COLUMNS)].copy()
    result[list(NUMERIC_COLUMNS)] = result[list(NUMERIC_COLUMNS)].apply(
        pd.to_numeric, errors="raise"
    ).astype(float)

    term = result.pop("term").astype("string").str.strip().str.lower()
    if term.isna().any() or not term.isin(["36 months", "60 months"]).all():
        raise ValueError("term must be '36 months' or '60 months'.")
    result["term_months"] = term.map({"36 months": 36, "60 months": 60}).astype(float)

    employment = result.pop("emp_length").astype("string").str.strip().str.lower()
    employment = employment.replace("", pd.NA)
    years = {"< 1 year": 0, "1 year": 1, "10+ years": 10}
    years.update({f"{year} years": year for year in range(2, 10)})
    if (employment.notna() & ~employment.isin(years)).any():
        raise ValueError("emp_length contains an unsupported category.")
    result["emp_length_years"] = employment.map(years).astype(float)
    result["emp_length_missing"] = employment.isna().astype(float)

    income = result["annual_inc"]
    denominator = income.where((income > 0) & np.isfinite(income))
    result["annual_inc_zero"] = income.eq(0).astype(float)
    result["loan_to_income"] = result["loan_amnt"] / denominator
    result["payment_to_income"] = result["installment"] * 12 / denominator
    return result
