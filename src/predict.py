"""Input validation and scoring, shared by the Streamlit pages and the tests.

Validation rules come from reports/feature_schema.json, which is learned from the training data:
 - every required column must exist
 - numbers must be numeric and inside the range seen in training
 - categories must be one of the values seen in training (matching ignores upper/lower case)
"""
import json

import joblib
import pandas as pd

from src.calibration import SigmoidCalibrator
from src.config import CATEGORICAL_COLUMNS, MODELS_DIR, NUMERIC_COLUMNS, REPORTS_DIR
from src.preprocessing import FEATURE_COLUMNS, add_features

REQUIRED_COLUMNS = NUMERIC_COLUMNS + CATEGORICAL_COLUMNS
SCHEMA_PATH = REPORTS_DIR / "feature_schema.json"
MAX_ISSUES_LISTED = 500          # keep the on-screen problem list readable for very bad files


class ValidationError(ValueError):
    """Raised when the input cannot be scored (missing columns, or a single record is invalid)."""


def load_schema() -> dict:
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


def load_bundle() -> dict:
    """Final pipeline + calibrator + review threshold, all created by earlier steps."""
    meta = json.loads((REPORTS_DIR / "metrics.json").read_text(encoding="utf-8"))
    return {
        "model_name": meta["final_model"],
        "threshold": float(meta["final_threshold"]),
        "pipeline": joblib.load(MODELS_DIR / meta["final_model_file"]),
        "calibrator": SigmoidCalibrator.from_json(REPORTS_DIR / "calibration_params.json"),
    }


def validate_batch(df: pd.DataFrame, schema: dict | None = None):
    """Check every row. Returns (valid_rows, issues_table, number_of_invalid_rows).
    'Row' in the issues table is the line number as seen in a spreadsheet (header = row 1)."""
    schema = schema or load_schema()
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValidationError("Missing required column(s): " + ", ".join(missing_cols))

    work = df.reset_index(drop=True).copy()
    bad = pd.Series(False, index=work.index)
    issues = []

    def record(col, mask, problem):
        for idx in work.index[mask]:
            if len(issues) >= MAX_ISSUES_LISTED:
                return
            issues.append({"Row": int(idx) + 2, "Column": col, "Problem": problem, "Value": df.iloc[idx][col]})

    for col in NUMERIC_COLUMNS:
        values = pd.to_numeric(work[col], errors="coerce")
        lo, hi = schema["numeric"][col]["min"], schema["numeric"][col]["max"]
        not_number = values.isna()
        out_of_range = ~not_number & ((values < lo) | (values > hi))
        record(col, not_number, "missing or not a number")
        record(col, out_of_range, f"outside the accepted range {lo:g} to {hi:g}")
        bad |= not_number | out_of_range
        work[col] = values

    for col in CATEGORICAL_COLUMNS:
        text = work[col].astype(str).str.strip()
        lookup = {v.lower(): v for v in schema["categorical"][col]}
        mapped = text.str.lower().map(lookup)
        empty = work[col].isna() | text.eq("") | text.str.lower().eq("nan")
        unknown = ~empty & mapped.isna()
        record(col, empty, "missing")
        record(col, unknown, "not one of: " + ", ".join(schema["categorical"][col]))
        bad |= empty | unknown
        work[col] = mapped

    issues_df = pd.DataFrame(issues, columns=["Row", "Column", "Problem", "Value"])
    return work.loc[~bad], issues_df, int(bad.sum())


def validate_single(values: dict) -> pd.DataFrame:
    """Validate one borrower (a dict of feature values). Raises ValidationError with readable messages."""
    valid, issues, n_bad = validate_batch(pd.DataFrame([values]))
    if n_bad:
        raise ValidationError("; ".join(f"{r.Column}: {r.Problem}" for r in issues.itertuples()))
    return valid


def score(bundle: dict, df: pd.DataFrame) -> pd.DataFrame:
    """Score validated rows. Adds:
       ModelScore            raw model output (used for ranking and the review threshold)
       EstimatedDefaultRate  calibrated: how often similar borrowers defaulted in validation data
       ReviewFlag            True when ModelScore >= the chosen threshold"""
    features = add_features(df)[FEATURE_COLUMNS]
    raw = bundle["pipeline"].predict_proba(features)[:, 1]
    out = df.copy()
    out["ModelScore"] = raw.round(4)
    out["EstimatedDefaultRate"] = bundle["calibrator"].predict(raw).round(4)
    out["ReviewFlag"] = raw >= bundle["threshold"]
    return out
