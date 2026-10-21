"""Group-level check: does the model flag, and catch defaults, equally across groups?

Run after training:   python -m src.fairness
This is a descriptive check on one prototype dataset. It does not prove or disprove fair treatment.
"""
import json

import joblib

from src.app_data import add_bands
from src.config import MODELS_DIR, REPORTS_DIR
from src.preprocessing import load_raw, split_data

GROUP_COLUMNS = ["AgeGroup", "MaritalStatus", "Education", "EmploymentType", "HasDependents"]


def main():
    meta = json.loads((REPORTS_DIR / "metrics.json").read_text())
    pipe = joblib.load(MODELS_DIR / meta["final_model_file"])
    threshold = meta["final_threshold"]
    _, _, X_test, _, _, y_test = split_data(load_raw())

    frame = add_bands(X_test)
    frame["y"] = y_test.values
    frame["flag"] = pipe.predict_proba(X_test)[:, 1] >= threshold

    result = {"threshold": threshold, "groups": {}}
    for col in GROUP_COLUMNS:
        rows = []
        for group, sub in frame.groupby(col, observed=True):
            defaulted, flagged = sub[sub["y"] == 1], sub[sub["flag"]]
            rows.append({
                "group": str(group), "n": int(len(sub)),
                "default_rate": round(float(sub["y"].mean()), 4),
                "flag_rate": round(float(sub["flag"].mean()), 4),
                "recall": round(float(defaulted["flag"].mean()), 4) if len(defaulted) else None,
                "precision": round(float(flagged["y"].mean()), 4) if len(flagged) else None,
            })
        result["groups"][col] = rows
    (REPORTS_DIR / "fairness.json").write_text(json.dumps(result, indent=2))
    for col, rows in result["groups"].items():
        print(f"\n{col}")
        for r in rows:
            print(f"  {r['group']:<14} n={r['n']:<6} default {r['default_rate']:.3f} | flagged {r['flag_rate']:.3f} | recall {r['recall']}")
    print("\nSaved reports/fairness.json")


if __name__ == "__main__":
    main()
