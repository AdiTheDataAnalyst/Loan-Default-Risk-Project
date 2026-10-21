"""Train and compare: Dummy baseline + Logistic Regression, Random Forest, XGBoost.

Run from the project root:   python -m src.train
"""
import json
import time

import joblib
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.config import MODELS_DIR, RANDOM_STATE, REPORTS_DIR
from src.evaluate import best_threshold, compute_metrics, curve_points
from src.preprocessing import build_preprocessor, load_raw, split_data

TUNING_SAMPLE = 40_000   # tune on a sample for speed, then refit the winner on ALL training data
N_ITER = 4               # random hyperparameter combinations tried per model
CV_FOLDS = 3


def get_models(scale_pos_weight: float) -> dict:
    """name -> (model, hyperparameter options to search)."""
    return {
        "Logistic Regression": (
            LogisticRegression(max_iter=1000, class_weight="balanced"),
            {"model__C": [0.01, 0.1, 1, 10]},
        ),
        "Random Forest": (
            RandomForestClassifier(class_weight="balanced", n_jobs=-1, random_state=RANDOM_STATE),
            {"model__n_estimators": [100, 200],
             "model__max_depth": [8, 12, 16],
             "model__min_samples_leaf": [20, 50, 100]},
        ),
        "XGBoost": (
            XGBClassifier(scale_pos_weight=scale_pos_weight, tree_method="hist",
                          eval_metric="logloss", n_jobs=-1, random_state=RANDOM_STATE),
            {"model__n_estimators": [200, 400],
             "model__max_depth": [3, 4, 5],
             "model__learning_rate": [0.05, 0.1],
             "model__subsample": [0.8, 1.0],
             "model__colsample_bytree": [0.8, 1.0]},
        ),
    }


def main():
    MODELS_DIR.mkdir(exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(load_raw())
    print(f"Train {len(X_train):,} | Validation {len(X_val):,} | Test {len(X_test):,}")
    print(f"Default rate  train {y_train.mean():.4f} | val {y_val.mean():.4f} | test {y_test.mean():.4f}\n")

    # Small stratified sample used ONLY for hyperparameter search
    sample_idx = y_train.groupby(y_train).sample(
        frac=TUNING_SAMPLE / len(y_train), random_state=RANDOM_STATE).index
    X_tune, y_tune = X_train.loc[sample_idx], y_train.loc[sample_idx]

    # Class imbalance: XGBoost needs "negatives per positive"; the other two use class_weight.
    spw = float((y_train == 0).sum() / (y_train == 1).sum())

    all_models = {"Dummy Baseline": (DummyClassifier(strategy="prior"), None)}
    all_models.update(get_models(spw))

    results, curves = {}, {}
    for name, (model, grid) in all_models.items():
        print(f"=== {name} ===")
        pipe = Pipeline([("prep", build_preprocessor()), ("model", model)])
        best_params, cv_score = {}, None

        if grid:
            search = RandomizedSearchCV(
                pipe, grid, n_iter=N_ITER, scoring="average_precision",
                cv=StratifiedKFold(CV_FOLDS, shuffle=True, random_state=RANDOM_STATE),
                random_state=RANDOM_STATE, n_jobs=1, refit=False)
            search.fit(X_tune, y_tune)
            best_params = {k.replace("model__", ""): v for k, v in search.best_params_.items()}
            cv_score = round(float(search.best_score_), 4)
            pipe = clone(pipe).set_params(**search.best_params_)
            print(f"  best params: {best_params} | CV PR-AUC: {cv_score}")

        start = time.perf_counter()
        pipe.fit(X_train, y_train)                       # final fit on ALL training data
        train_seconds = time.perf_counter() - start

        start = time.perf_counter()
        val_proba = pipe.predict_proba(X_val)[:, 1]
        predict_seconds = time.perf_counter() - start
        test_proba = pipe.predict_proba(X_test)[:, 1]

        threshold = best_threshold(y_val, val_proba)     # chosen on validation only
        results[name] = {
            "best_params": best_params,
            "cv_pr_auc": cv_score,
            "train_seconds": round(train_seconds, 2),
            "predict_seconds_validation": round(predict_seconds, 3),
            "validation": compute_metrics(y_val, val_proba, threshold),
            "test": compute_metrics(y_test, test_proba, threshold),
        }
        curves[name] = curve_points(y_test, test_proba)
        joblib.dump(pipe, MODELS_DIR / f"{name.lower().replace(' ', '_')}.joblib")
        v = results[name]["validation"]
        print(f"  validation: PR-AUC {v['pr_auc']} | ROC-AUC {v['roc_auc']} | "
              f"recall {v['recall']} | precision {v['precision']} | threshold {v['threshold']}")
        print(f"  train {train_seconds:.1f}s | predict {predict_seconds:.2f}s\n")

    # Final model = best validation PR-AUC among the three real models (not the baseline)
    candidates = {k: v for k, v in results.items() if k != "Dummy Baseline"}
    final = max(candidates, key=lambda k: candidates[k]["validation"]["pr_auc"])
    print(f"Selected model: {final}")

    summary = {
        "final_model": final,
        "final_model_file": f"{final.lower().replace(' ', '_')}.joblib",
        "final_threshold": results[final]["validation"]["threshold"],
        "split_sizes": {"train": len(X_train), "validation": len(X_val), "test": len(X_test)},
        "default_rate": {"train": round(float(y_train.mean()), 4),
                         "validation": round(float(y_val.mean()), 4),
                         "test": round(float(y_test.mean()), 4)},
        "models": results,
    }
    (REPORTS_DIR / "metrics.json").write_text(json.dumps(summary, indent=2))
    (REPORTS_DIR / "curves.json").write_text(json.dumps(curves))
    print("Saved reports/metrics.json and reports/curves.json")


if __name__ == "__main__":
    main()
