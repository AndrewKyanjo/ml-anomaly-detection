"""Loan data checks built on ml-gearbox."""

import numpy as np
import pandas as pd
from ml_gearbox import audit as ga

from .schema import (
    CATEGORICAL_COLUMNS,
    COUNT_COLUMNS,
    FEATURE_COLUMNS,
    NUMERIC_COLUMNS,
    TARGET,
)


def run_loan_audit(frame: pd.DataFrame) -> dict:
    """Report schema, missingness, duplicates, ranges, and invalid counts."""
    if frame.empty or not frame.columns.is_unique:
        raise ValueError("Provide nonempty data with unique column names.")
    missing = set((*FEATURE_COLUMNS, TARGET)) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    invalid = [
        c for c in NUMERIC_COLUMNS if not pd.api.types.is_numeric_dtype(frame[c])
    ]
    if invalid:
        raise TypeError(f"Expected numeric dtypes for: {invalid}")

    audited = frame.copy()
    # Gearbox discovers categorical columns through object/category dtypes.
    for column in CATEGORICAL_COLUMNS:
        audited[column] = audited[column].astype(object)
    audited["term"] = (
        audited["term"]
        .map(lambda value: value.strip() if isinstance(value, str) else value)
        .astype(object)
    )
    
    rules = {column: {"min": 0} for column in NUMERIC_COLUMNS}
    rules.update(
        {
            "fico_range_low": {"min": 300, "max": 850},
            "term": {"allowed": ["36 months", "60 months"]},
            TARGET: {"allowed": [0, 1]},
        }
    )

    reports = ga.run_data_audit(
        audited,
        target_col=TARGET,
        range_rules=rules,
        cross_field_rules=[("pub_rec_bankruptcies", "<=", "pub_rec")],
    )

    counts = audited[list(COUNT_COLUMNS)].replace([np.inf, -np.inf], np.nan)
    non_integer = ((counts % 1 != 0) & counts.notna()).sum()
    reports["count_integrality"] = (
        non_integer.rename("non_integer_count").rename_axis("feature").reset_index()
    )
    reports["negative_dti_count"] = int(audited["dti"].lt(0).sum())
    return reports
