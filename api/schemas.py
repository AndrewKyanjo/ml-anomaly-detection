"""Request and response validation for loan predictions."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

ModelName = Literal["random_forest", "logistic_regression", "xgboost", "prior_baseline"]
EmploymentLength = Literal[
    "< 1 year",
    "1 year",
    "2 years",
    "3 years",
    "4 years",
    "5 years",
    "6 years",
    "7 years",
    "8 years",
    "9 years",
    "10+ years",
]


class LoanRecord(BaseModel):
    """Raw predictors accepted by the saved preprocessing pipeline."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    loan_amnt: float = Field(gt=0)
    term: str = Field(pattern=r"^\s*(36|60) months\s*$")
    int_rate: float = Field(ge=0, le=100)
    installment: float = Field(ge=0)
    emp_length: EmploymentLength | None = None
    annual_inc: float = Field(ge=0)
    dti: float | None = Field(default=None, ge=-1)
    delinq_2yrs: int = Field(ge=0)
    fico_range_low: int = Field(ge=300, le=850)
    inq_last_6mths: int | None = Field(default=None, ge=0)
    pub_rec: int = Field(ge=0)
    revol_util: float | None = Field(default=None, ge=0)
    total_acc: int = Field(ge=0)
    mort_acc: int = Field(ge=0)
    pub_rec_bankruptcies: int = Field(ge=0)


class PredictRequest(BaseModel):
    """A bounded batch, with an optional model override."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    records: list[LoanRecord] = Field(min_length=1, max_length=1000)
    model: ModelName | None = None


class Prediction(BaseModel):
    """Probability and thresholded label for one loan."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    probability: float = Field(ge=0, le=1)
    label: Literal[0, 1]
    id: int = Field(ge=1)


class PredictResponse(BaseModel):
    """Predictions in the same order as the submitted records."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    model: str
    model_version: str
    threshold: float = Field(ge=0, le=1)
    predictions: list[Prediction]


class HealthResponse(BaseModel):
    """Availability of the loaded model collection."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    status: Literal["ok", "unavailable"]
    models: list[str]
    default_model: str | None = None


class StoredPredictRequest(BaseModel):
    """Predict existing database records without resubmitting their features."""

    model_config = ConfigDict(extra="forbid")

    record_ids: list[Annotated[int, Field(ge=1)]] = Field(min_length=1, max_length=1000)
    model: ModelName | None = None


class OutcomeRequest(BaseModel):
    """An observed outcome supplied after prediction."""

    model_config = ConfigDict(extra="forbid")

    observed_label: Literal[0, 1]


class RecordPage(BaseModel):
    total: int
    items: list[dict]


class SavedPrediction(Prediction):
    record_id: int | None
    model: str
    model_version: str
    threshold: float
    inputs: dict
    created_at: str
    observed_label: Literal[0, 1] | None
    observed_at: str | None


class PredictionPage(BaseModel):
    total: int
    items: list[SavedPrediction]
