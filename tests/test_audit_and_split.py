"""Audit diagnostics and persistent holdout membership."""

import pandas as pd
import pytest

from src.audit import run_loan_audit
from src.data import get_or_create_split
from src.schema import ROW_ID, TARGET


def test_audit_handles_pandas_strings_and_preserves_input(loan_frame):
    loan_frame["emp_length"] = loan_frame["emp_length"].astype("string")
    loan_frame.loc[loan_frame.index[0], "emp_length"] = "   "
    loan_frame.loc[loan_frame.index[1], "dti"] = -1
    loan_frame.loc[loan_frame.index[2], "pub_rec"] = 0.5
    original = loan_frame.copy(deep=True)
    report = run_loan_audit(loan_frame)
    pd.testing.assert_frame_equal(loan_frame, original)
    assert report["negative_dti_count"] == 1
    assert report["hidden_missingness"]["hidden_missing_count"].sum() == 1
    assert report["count_integrality"]["non_integer_count"].sum() == 1
    term_rules = report["range_validation"].query("feature == 'term'")
    assert term_rules["violation_count"].sum() == 0


def test_saved_split_survives_reordering_and_rejects_changed_data(loan_frame, tmp_path):
    path = tmp_path / "splits" / "membership.parquet"
    original = loan_frame.copy(deep=True)
    split = get_or_create_split(loan_frame, path)
    restored = get_or_create_split(loan_frame.sample(frac=1, random_state=7), path)
    pd.testing.assert_frame_equal(loan_frame, original)
    assert len(split["train"]) == 32 and len(split["test"]) == 8
    assert set(split["train"][ROW_ID]).isdisjoint(split["test"][ROW_ID])
    assert split["test"][TARGET].value_counts().to_dict() == {0: 4, 1: 4}
    for partition in ["train", "test"]:
        assert set(split[partition][ROW_ID]) == set(restored[partition][ROW_ID])
    changed = loan_frame.copy()
    changed.loc[changed.index[0], "loan_amnt"] += 1
    with pytest.raises(ValueError, match="unknown row keys"):
        get_or_create_split(changed, path)


def test_duplicate_content_is_rejected(loan_frame, tmp_path):
    duplicates = pd.concat([loan_frame, loan_frame.iloc[[0]]])
    with pytest.raises(ValueError, match="fingerprints"):
        get_or_create_split(duplicates, tmp_path / "split.parquet")


def test_audit_rejects_missing_columns_and_nonnumeric_data(loan_frame):
    with pytest.raises(ValueError, match="Missing required"):
        run_loan_audit(loan_frame.drop(columns="annual_inc"))
    malformed = loan_frame.copy()
    malformed["annual_inc"] = "unknown"
    with pytest.raises(TypeError, match="numeric dtypes"):
        run_loan_audit(malformed)
