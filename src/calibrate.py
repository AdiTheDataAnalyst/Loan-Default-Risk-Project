"""Calibration: turn the model's raw score into an honest estimated default rate.

Why this exists: we trained with class weights so the model pays attention to defaults.
A side effect is that raw scores run HIGH (average score ~0.43 while only ~12% of loans default).
A 2-parameter sigmoid (Platt scaling) is fitted on the VALIDATION set so that a score of s maps to
the share of similar borrowers who really defaulted. It never reorders borrowers, so ranking and the
review threshold stay the same.

Run after training:   python -m src.calibrate
"""
import json

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score

from src.calibration import SigmoidCalibrator
from src.config import REPORTS_DIR
from src.preprocessing import load_raw, split_data


def reliability_table(y_true, predicted, bins=10) -> list:
    """Group borrowers by predicted value; compare the average prediction with what actually happened."""
    frame = pd.DataFrame({"p": np.asarray(predicted), "y": np.asarray(y_true)})
    frame["bin"] = pd.qcut(frame["p"].rank(method="first"), bins, labels=False)
    grouped = frame.groupby("bin").agg(predicted=("p", "mean"), observed=("y", "mean"), n=("y", "size"))
    return [{k: round(float(v), 4) for k, v in row.items()} for row in grouped.to_dict("records")]


def main():
    meta = json.loads((REPORTS_DIR / "metrics.json").read_text())
    import joblib
    from src.config import MODELS_DIR
    pipe = joblib.load(MODELS_DIR / meta["final_model_file"])
    _, X_val, X_test, _, y_val, y_test = split_data(load_raw())

    val_score = pipe.predict_proba(X_val)[:, 1]
    test_score = pipe.predict_proba(X_test)[:, 1]

    calibrator = SigmoidCalibrator.fit(val_score, y_val)
    test_cal = calibrator.predict(test_score)
    (REPORTS_DIR / "calibration_params.json").write_text(calibrator.to_json())

    report = {
        "model": meta["final_model"],
        "actual_default_rate_test": round(float(y_test.mean()), 4),
        "mean_raw_score_test": round(float(test_score.mean()), 4),
        "mean_calibrated_test": round(float(test_cal.mean()), 4),
        "brier_raw": round(brier_score_loss(y_test, test_score), 4),
        "brier_calibrated": round(brier_score_loss(y_test, test_cal), 4),
        "roc_auc_raw": round(roc_auc_score(y_test, test_score), 4),
        "roc_auc_calibrated": round(roc_auc_score(y_test, test_cal), 4),
        "reliability_raw": reliability_table(y_test, test_score),
        "reliability_calibrated": reliability_table(y_test, test_cal),
    }
    (REPORTS_DIR / "calibration.json").write_text(json.dumps(report, indent=2))

    # test-set scores, used by the threshold explorer on the Model Performance page
    pd.DataFrame({"y_true": y_test.values, "score": test_score.round(5), "calibrated": test_cal.round(5)}
                 ).to_csv(REPORTS_DIR / "test_scores.csv.gz", index=False)

    print(f"Actual default rate (test):      {report['actual_default_rate_test']:.4f}")
    print(f"Average RAW score (test):        {report['mean_raw_score_test']:.4f}   <- too high")
    print(f"Average CALIBRATED (test):       {report['mean_calibrated_test']:.4f}")
    print(f"Brier score  raw -> calibrated:  {report['brier_raw']} -> {report['brier_calibrated']} (lower is better)")
    print(f"ROC-AUC      raw -> calibrated:  {report['roc_auc_raw']} -> {report['roc_auc_calibrated']}")
    print("Saved reports/calibration_params.json, reports/calibration.json, reports/test_scores.csv.gz")


if __name__ == "__main__":
    main()
