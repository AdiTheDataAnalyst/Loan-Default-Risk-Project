"""Automated checks. Run from the project root:   pytest -v

Needs the artifacts from `python -m src.run_all` (and the raw CSV for the split test).
Tests that need a missing file are skipped, not failed.
"""
import pandas as pd
import pytest

from src.config import MODELS_DIR, RAW_DIR, REPORTS_DIR
from src.predict import REQUIRED_COLUMNS, ValidationError, load_bundle, score, validate_batch, validate_single
from src.preprocessing import add_features, build_preprocessor, split_data, load_raw

needs_artifacts = pytest.mark.skipif(
    not (REPORTS_DIR / "metrics.json").exists() or not (REPORTS_DIR / "calibration_params.json").exists()
    or not (REPORTS_DIR / "feature_schema.json").exists(),
    reason="run `python -m src.run_all` first")
needs_raw = pytest.mark.skipif(not list(RAW_DIR.glob("*.csv")), reason="raw CSV not in data/raw")

GOOD = dict(Age=40, Income=70000, LoanAmount=100000, CreditScore=650, MonthsEmployed=48, NumCreditLines=2, InterestRate=12.0,
            LoanTerm=36, DTIRatio=0.4, Education="Bachelor's", EmploymentType="Full-time", MaritalStatus="Married",
            HasMortgage="Yes", HasDependents="No", LoanPurpose="Home", HasCoSigner="No")


# ---------- feature engineering ----------
def test_add_features_ratios():
    out = add_features(pd.DataFrame([GOOD]))
    assert out["LoanToIncome"].iloc[0] == pytest.approx(100000 / 70000)
    assert out["MonthlyPaymentToIncome"].iloc[0] == pytest.approx((100000 / 36) / (70000 / 12))


def test_zero_income_does_not_crash_or_divide_by_zero():
    out = add_features(pd.DataFrame([{**GOOD, "Income": 0}]))
    assert pd.isna(out["LoanToIncome"].iloc[0])


# ---------- data split / leakage ----------
@needs_raw
def test_split_is_stratified_and_has_no_overlap():
    X_tr, X_va, X_te, y_tr, y_va, y_te = split_data(load_raw())
    assert len(X_tr) + len(X_va) + len(X_te) == len(load_raw())
    assert not (set(X_tr.index) & set(X_te.index)) and not (set(X_tr.index) & set(X_va.index)) and not (set(X_va.index) & set(X_te.index))
    rates = [y_tr.mean(), y_va.mean(), y_te.mean()]
    assert max(rates) - min(rates) < 0.002           # same default rate in every split
    assert "LoanID" not in X_tr.columns and "Default" not in X_tr.columns


@needs_raw
def test_preprocessor_learns_from_training_data_only():
    X_tr, _, X_te, _, _, _ = split_data(load_raw())
    prep = build_preprocessor().fit(X_tr)
    scaler = prep.named_transformers_["num"].named_steps["scale"]
    age_pos = list(prep.transformers_[0][2]).index("Age")
    assert scaler.mean_[age_pos] == pytest.approx(X_tr["Age"].mean())


# ---------- input validation ----------
@needs_artifacts
def test_valid_single_record_passes():
    assert len(validate_single(GOOD)) == 1


@needs_artifacts
@pytest.mark.parametrize("field,value", [("Age", 5), ("Age", 150), ("CreditScore", 1000), ("InterestRate", -3),
                                         ("Age", "abc"), ("Income", None), ("Education", "Wizard"), ("LoanPurpose", "")])
def test_invalid_single_record_is_rejected(field, value):
    with pytest.raises(ValidationError):
        validate_single({**GOOD, field: value})


@needs_artifacts
def test_category_matching_ignores_case_and_spaces():
    valid = validate_single({**GOOD, "EmploymentType": "  full-time "})
    assert valid["EmploymentType"].iloc[0] == "Full-time"


@needs_artifacts
def test_missing_column_is_reported():
    df = pd.DataFrame([GOOD]).drop(columns=["CreditScore"])
    with pytest.raises(ValidationError, match="CreditScore"):
        validate_batch(df)


@needs_artifacts
def test_batch_keeps_good_rows_and_reports_bad_ones():
    df = pd.DataFrame([GOOD, {**GOOD, "Age": 5}, GOOD, {**GOOD, "HasCoSigner": "Maybe"}])
    valid, issues, n_bad = validate_batch(df)
    assert n_bad == 2 and len(valid) == 2
    assert sorted(issues["Row"]) == [3, 5]            # spreadsheet row numbers (header is row 1)


# ---------- model and predictions ----------
@needs_artifacts
def test_bundle_loads_with_expected_parts():
    bundle = load_bundle()
    assert 0 < bundle["threshold"] < 1 and bundle["model_name"]
    assert (MODELS_DIR / "logistic_regression.joblib").exists() or any(MODELS_DIR.glob("*.joblib"))


@needs_artifacts
def test_single_prediction_is_a_valid_probability():
    out = score(load_bundle(), validate_single(GOOD)).iloc[0]
    assert 0 <= out["ModelScore"] <= 1 and 0 <= out["EstimatedDefaultRate"] <= 1
    assert out["ReviewFlag"] == (out["ModelScore"] >= load_bundle()["threshold"])


@needs_artifacts
def test_batch_prediction_keeps_row_order_and_extra_columns():
    df = pd.DataFrame([{**GOOD, "LoanID": "A"}, {**GOOD, "Age": 22, "InterestRate": 20.0, "LoanID": "B"}])
    valid, _, _ = validate_batch(df)
    out = score(load_bundle(), valid)
    assert list(out["LoanID"]) == ["A", "B"] and len(out) == 2
    assert {"ModelScore", "EstimatedDefaultRate", "ReviewFlag"} <= set(out.columns)


@needs_artifacts
def test_calibration_preserves_ranking():
    bundle = load_bundle()
    raw = [0.05, 0.2, 0.4, 0.6, 0.9]
    cal = list(bundle["calibrator"].predict(raw))
    assert cal == sorted(cal)


@needs_artifacts
def test_required_columns_match_schema():
    import json
    schema = json.loads((REPORTS_DIR / "feature_schema.json").read_text())
    assert set(REQUIRED_COLUMNS) == set(schema["numeric"]) | set(schema["categorical"])
