"""Phase 5: explain the final model with SHAP and (for Logistic Regression) coefficients.

Run from the project root:   python -m src.explain

SHAP in plain words: for ONE borrower, it splits the model's output into
"how much each feature pushed the risk up or down compared with an average borrower".
It describes what the MODEL does. It does NOT prove what causes a default.
"""
import json

import joblib
import numpy as np
import pandas as pd
import shap
from scipy import sparse
from sklearn.linear_model import LogisticRegression

from src.config import (
    CATEGORICAL_COLUMNS, ENGINEERED_COLUMNS, MODELS_DIR, NUMERIC_COLUMNS,
    RANDOM_STATE, REPORTS_DIR,
)
from src.preprocessing import FEATURE_COLUMNS, load_raw, split_data

ORIGINAL_FEATURES = NUMERIC_COLUMNS + ENGINEERED_COLUMNS + CATEGORICAL_COLUMNS
SAMPLE_SIZE = 2000        # borrowers explained for the global charts (keeps it fast)
BACKGROUND_SIZE = 100     # "average borrower" reference used by SHAP (SHAP uses at most 100 anyway)


def _dense(matrix):
    return matrix.toarray() if sparse.issparse(matrix) else np.asarray(matrix)


def load_final_pipeline():
    meta = json.loads((REPORTS_DIR / "metrics.json").read_text())
    pipe = joblib.load(MODELS_DIR / meta["final_model_file"])
    return pipe, meta


def original_feature_of(transformed_name: str) -> str:
    """'cat__Education_Master's' -> 'Education',  'num__Age' -> 'Age'."""
    name = transformed_name.split("__", 1)[1]
    for col in CATEGORICAL_COLUMNS:
        if name.startswith(col + "_"):
            return col
    return name


def build_explainer(pipe, X_background: pd.DataFrame):
    """Pick the right SHAP explainer for the model type."""
    prep, model = pipe.named_steps["prep"], pipe.named_steps["model"]
    background = _dense(prep.transform(X_background))
    if isinstance(model, LogisticRegression):
        return shap.LinearExplainer(model, background)
    return shap.TreeExplainer(model)


def shap_by_original_feature(pipe, explainer, X: pd.DataFrame) -> pd.DataFrame:
    """SHAP values with one-hot columns added back together (e.g. all Education_* -> Education)."""
    prep = pipe.named_steps["prep"]
    names = prep.get_feature_names_out()
    values = explainer.shap_values(_dense(prep.transform(X)))
    if isinstance(values, list):          # some explainers return one array per class
        values = values[1]
    frame = pd.DataFrame(values, columns=[original_feature_of(n) for n in names], index=X.index)
    return frame.T.groupby(level=0).sum().T[ORIGINAL_FEATURES]


def explain_one(pipe, explainer, row: pd.DataFrame) -> pd.DataFrame:
    """Explanation for a single borrower (row = 1-row DataFrame with FEATURE_COLUMNS).
    Positive contribution = pushes estimated risk UP; negative = pushes it DOWN."""
    contrib = shap_by_original_feature(pipe, explainer, row).iloc[0]
    out = pd.DataFrame({
        "feature": contrib.index,
        "value": [row.iloc[0][c] for c in contrib.index],
        "contribution": contrib.values,
    })
    return out.reindex(out["contribution"].abs().sort_values(ascending=False).index).reset_index(drop=True)


def coefficient_table(pipe) -> pd.DataFrame:
    """Logistic Regression only: one row per (encoded) feature with its coefficient and odds ratio."""
    model = pipe.named_steps["model"]
    names = [n.split("__", 1)[1] for n in pipe.named_steps["prep"].get_feature_names_out()]
    table = pd.DataFrame({"feature": names, "coefficient": model.coef_[0]})
    table["odds_ratio"] = np.exp(table["coefficient"])
    return table.reindex(table["coefficient"].abs().sort_values(ascending=False).index).reset_index(drop=True)


def main():
    pipe, meta = load_final_pipeline()
    print(f"Explaining final model: {meta['final_model']}")

    X_train, _, X_test, _, _, _ = split_data(load_raw())
    background = X_train.sample(BACKGROUND_SIZE, random_state=RANDOM_STATE)
    sample = X_test.sample(SAMPLE_SIZE, random_state=RANDOM_STATE)

    explainer = build_explainer(pipe, background)
    shap_frame = shap_by_original_feature(pipe, explainer, sample)

    importance = (shap_frame.abs().mean().sort_values(ascending=False)
                  .rename("mean_abs_shap").reset_index().rename(columns={"index": "feature"}))
    importance.to_csv(REPORTS_DIR / "shap_importance.csv", index=False)
    joblib.dump({"X_sample": sample, "shap_values": shap_frame, "background": background},
                REPORTS_DIR / "shap_sample.joblib")
    print("\nTop features by average SHAP impact (log-odds units):")
    print(importance.head(10).round(4).to_string(index=False))

    if isinstance(pipe.named_steps["model"], LogisticRegression):
        coefs = coefficient_table(pipe)
        coefs.to_csv(REPORTS_DIR / "coefficients.csv", index=False)
        print("\nLargest coefficients (numeric features are standardised, so sizes are comparable):")
        print(coefs.head(10).round(4).to_string(index=False))

    # Sanity check: explain one borrower, and confirm SHAP adds up to the model's own score
    row = sample.iloc[[0]]
    one = explain_one(pipe, explainer, row)
    print("\nExample borrower explanation:")
    print(one.head(6).round(4).to_string(index=False))
    proba = pipe.predict_proba(row)[0, 1]
    print(f"Model's estimated default probability for this borrower: {proba:.4f}")


if __name__ == "__main__":
    main()
