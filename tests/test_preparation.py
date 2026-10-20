"""Check feature rules and preprocessing learned from training data only."""

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

from src.features import build_features
from src.preprocessing import audit_preprocessing, clean_numeric, make_preprocessor
from src.schema import ROW_ID, TARGET


def test_feature_encoding_ratios_and_metadata_exclusion(loan_frame):
    raw = loan_frame.drop(columns=TARGET)
    raw[ROW_ID] = np.arange(len(raw))
    raw.loc[raw.index[0], "annual_inc"] = 0
    original = raw.copy(deep=True)

    features = build_features(raw)

    pd.testing.assert_frame_equal(raw, original)
    assert features.index.equals(raw.index)
    assert ROW_ID not in features and TARGET not in features
    assert features["term_months"].tolist()[:2] == [36, 60]
    assert features["emp_length_years"].tolist()[:2] == [0, 10]
    assert features.iloc[2]["emp_length_missing"] == 1
    assert features.iloc[0]["annual_inc_zero"] == 1
    assert np.isnan(features.iloc[0]["loan_to_income"])
    assert np.isnan(features.iloc[0]["payment_to_income"])
    assert not np.isinf(features.to_numpy()).any()


@pytest.mark.parametrize(
    "column,value",
    [("term", "72 months"), ("term", None), ("emp_length", "20 years")],
)
def test_unknown_categories_are_rejected(loan_frame, column, value):
    raw = loan_frame.drop(columns=TARGET)
    raw.loc[raw.index[0], column] = value
    with pytest.raises(ValueError):
        build_features(raw)


def test_features_reject_target_and_missing_predictors(loan_frame):
    with pytest.raises(ValueError, match=TARGET):
        build_features(loan_frame)
    with pytest.raises(ValueError):
        build_features(loan_frame.drop(columns=[TARGET, "annual_inc"]))


def test_preprocessing_learns_training_statistics_only():
    train = pd.DataFrame(
        {
            "signal": [0.0, 2.0, np.nan],
            "dti": [10.0, -1.0, 30.0],
            "empty": [np.nan, np.nan, np.nan],
            "revol_util": [120.0, 40.0, 50.0],
        },
        index=[10, 20, 30],
    )
    original = train.copy(deep=True)
    preprocessor = make_preprocessor(scale=False)
    prepared = preprocessor.fit_transform(train)

    pd.testing.assert_frame_equal(train, original)
    assert prepared.index.equals(train.index)
    assert prepared.loc[30, "signal"] == 1.0
    assert prepared.loc[20, "dti"] == 20.0
    assert prepared.loc[10, "revol_util"] == 120.0
    assert prepared["empty"].eq(0).all()
    statistics = preprocessor.named_steps["impute"].statistics_.copy()

    holdout = pd.DataFrame(
        {
            "signal": [1e9, np.nan],
            "dti": [999.0, -1.0],
            "empty": [100.0, np.nan],
            "revol_util": [172.0, np.inf],
        },
        index=[90, 80],
    )
    result = preprocessor.transform(holdout)
    assert result.index.equals(holdout.index)
    assert result.loc[80, "signal"] == 1.0
    assert result.loc[80, "dti"] == 20.0
    assert result.loc[90, "dti"] == 999.0
    assert result.loc[90, "revol_util"] == 172.0
    assert np.isfinite(result.to_numpy()).all()
    np.testing.assert_equal(
        preprocessor.named_steps["impute"].statistics_, statistics
    )
    reports = audit_preprocessing(train)
    assert reports["dti_validation"]["violation_count"].sum() == 1


def test_preprocessing_scaler_uses_training_only():
    preprocessor = make_preprocessor()
    train = pd.DataFrame({"signal": [1.0, 2.0, 3.0]})
    prepared = preprocessor.fit_transform(train)
    mean = preprocessor.named_steps["scale"].mean_.copy()

    preprocessor.transform(pd.DataFrame({"signal": [1e10]}))

    np.testing.assert_equal(preprocessor.named_steps["scale"].mean_, mean)
    np.testing.assert_allclose(prepared.mean(), 0, atol=1e-12)


def test_preprocessing_requires_fitted_state_and_matching_columns():
    preprocessor = make_preprocessor()
    with pytest.raises(NotFittedError):
        preprocessor.transform(pd.DataFrame({"x": [1.0]}))
    preprocessor.fit(pd.DataFrame({"x": [1.0]}))
    with pytest.raises(ValueError):
        preprocessor.transform(pd.DataFrame({"other": [1.0]}))


@pytest.mark.parametrize(
    "frame",
    [
        pd.DataFrame({TARGET: [0.0]}),
        pd.DataFrame({ROW_ID: [1.0]}),
        pd.DataFrame({"signal": ["unknown"]}),
        pd.DataFrame([[1.0, 2.0]], columns=["signal", "signal"]),
    ],
)
def test_cleaning_rejects_metadata_and_invalid_feature_schema(frame):
    with pytest.raises((ValueError, TypeError)):
        clean_numeric(frame)


def test_saved_preprocessing_reproduces_transformations(loan_frame, tmp_path):
    features = build_features(loan_frame.drop(columns=TARGET))
    preprocessor = make_preprocessor()
    expected = preprocessor.fit_transform(features)
    path = tmp_path / "preparation.joblib"

    joblib.dump(preprocessor, path)
    actual = joblib.load(path).transform(features)

    pd.testing.assert_frame_equal(actual, expected)
