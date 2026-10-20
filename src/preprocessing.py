"""Audit numeric features and construct fitted sklearn preprocessing."""

import numpy as np
import pandas as pd
from ml_gearbox import audit as ga
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from .schema import ROW_ID, TARGET


def clean_numeric(frame: pd.DataFrame) -> pd.DataFrame:
    """Replace infinities and negative DTI with missing values; keep row order."""
    if frame.empty or not frame.columns.is_unique:
        raise ValueError("Provide nonempty data with unique column names.")
    if {TARGET, ROW_ID} & set(frame.columns):
        raise ValueError("Targets and row identifiers cannot be numeric predictors.")
    if not all(pd.api.types.is_numeric_dtype(dtype) for dtype in frame.dtypes):
        raise ValueError("Build numeric features before preprocessing.")
    
    cleaned = frame.astype(float).replace([np.inf, -np.inf], np.nan)
    if "dti" in cleaned:
        cleaned["dti"] = cleaned["dti"].mask(cleaned["dti"] < 0)
    return cleaned


def audit_preprocessing(frame: pd.DataFrame) -> dict:
    """Inspect missingness and invalid values with ml-gearbox before fitting."""
    cleaned = clean_numeric(frame)
    reports = {
        "missingness_before": ga.missingness_report(frame),
        "infinite_values_before": ga.infinite_value_report(frame),
        "missingness_after_cleaning": ga.missingness_report(cleaned),
    }
    
    if "dti" in frame:
        reports["dti_validation"] = ga.range_validation(frame, {"dti": {"min": 0}})
    return reports


def make_preprocessor(*, scale: bool = True) -> Pipeline:
    """Return cleaning, median imputation, and optional scaling; fit on train only.

    ml-gearbox provides the audits above. Its current release has no fitted
    imputer or scaler, so sklearn supplies those operations. Empty training
    columns receive zero. Large finite values remain unchanged before scaling.
    """
    imputer = SimpleImputer(
        strategy="median", add_indicator=True, keep_empty_features=True,
    ).set_output(transform="pandas")
    steps = [("clean", FunctionTransformer(clean_numeric)), ("impute", imputer)]
    if scale:
        steps.append(("scale", StandardScaler().set_output(transform="pandas")))
    return Pipeline(steps)
