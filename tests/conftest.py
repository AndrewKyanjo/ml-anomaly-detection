"""Small loan records independent of the local CSV."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def loan_frame() -> pd.DataFrame:
    size = 40
    return pd.DataFrame({
        "loan_amnt": np.arange(size, dtype=float) * 100 + 1000,
        "term": [" 36 months", " 60 months"] * (size // 2),
        "int_rate": np.full(size, 12.0),
        "installment": np.full(size, 100.0),
        "emp_length": ["< 1 year", "10+ years", None, "1 year"] * (size // 4),
        "annual_inc": np.full(size, 60000.0),
        "loan_status": [0, 1] * (size // 2),
        "dti": np.full(size, 20.0),
        "delinq_2yrs": np.zeros(size),
        "fico_range_low": np.full(size, 700.0),
        "inq_last_6mths": np.zeros(size),
        "pub_rec": np.zeros(size),
        "revol_util": np.full(size, 40.0),
        "total_acc": np.full(size, 10.0),
        "mort_acc": np.ones(size),
        "pub_rec_bankruptcies": np.zeros(size),
    }, index=pd.Index(np.arange(size) + 100, name="source_row"))
