"""Persistent, content-keyed train/test membership through ml-gearbox."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from ml_gearbox import create_split, load_split
from ml_gearbox import audit as gearbox_audit

from .schema import ROW_ID, TARGET


def get_or_create_split(frame: pd.DataFrame, path: Path) -> dict[str, pd.DataFrame]:
    """Return a persistent 80/20 target-stratified split with seed 42.

    ``_row_id`` is a deterministic content fingerprint, not a borrower ID.
    Reordering rows preserves membership. Changed content, duplicate rows,
    or hash collisions are rejected rather than regenerating assignments.
    The random split cannot measure future-loan or unseen-borrower
    performance because this extract contains no dates or borrower IDs.

    Only the small Parquet membership mapping is persisted. Existing
    ``train.csv`` and ``test.csv`` files are never modified.
    """
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame.")
    if not frame.columns.is_unique:
        raise ValueError("Column names must be unique.")
    if TARGET not in frame.columns:
        raise ValueError(f"Missing target column: {TARGET}")
    prepared = frame.drop(columns=ROW_ID, errors="ignore").copy()
    prepared[ROW_ID] = pd.util.hash_pandas_object(prepared, index=False).astype(
        "uint64"
    )
    keys = gearbox_audit.id_report(prepared, ROW_ID)
    if keys["missing_ids"] or keys["duplicate_ids"]:
        raise ValueError(
            "Content fingerprints must be unique and non-null; "
            "duplicate rows or hash collisions prevent a safe split."
        )

    destination = Path(path)
    if not destination.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        create_split(
            prepared,
            id_col=ROW_ID,
            stratify_col=TARGET,
            test_size=0.2,
            random_state=42,
            path=destination,
            overwrite=False,
        )
    return load_split(prepared, destination)
