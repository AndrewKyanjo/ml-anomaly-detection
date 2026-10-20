"""Predictor contract for the extracted Lending Club dataset."""

TARGET = "loan_status"
ROW_ID = "_row_id"
CATEGORICAL_COLUMNS = ("term", "emp_length")
COUNT_COLUMNS = (
    "delinq_2yrs",
    "inq_last_6mths",
    "pub_rec",
    "total_acc",
    "mort_acc",
    "pub_rec_bankruptcies",
)
FEATURE_COLUMNS = (
    "loan_amnt",
    "term",
    "int_rate",
    "installment",
    "emp_length",
    "annual_inc",
    "dti",
    "delinq_2yrs",
    "fico_range_low",
    "inq_last_6mths",
    "pub_rec",
    "revol_util",
    "total_acc",
    "mort_acc",
    "pub_rec_bankruptcies",
)
NUMERIC_COLUMNS = tuple(c for c in FEATURE_COLUMNS if c not in CATEGORICAL_COLUMNS)
