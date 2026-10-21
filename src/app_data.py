"""Data helpers for the Streamlit app.

Build the small files the app needs (run once, after training):
    python -m src.app_data

Why: raw data stays out of GitHub, but the deployed app still needs numbers to show.
 - reports/dataset_summary.json = statistics computed from the FULL dataset
 - data/sample/loan_sample.csv  = 20,000-row stratified sample used for charts when the raw file is absent
"""
import json

import pandas as pd

from src.config import (CATEGORICAL_COLUMNS, MODELS_DIR, NUMERIC_COLUMNS, RANDOM_STATE, RAW_DIR,
                        REPORTS_DIR, ROOT, TARGET)

SAMPLE_PATH = ROOT / "data" / "sample" / "loan_sample.csv"
SUMMARY_PATH = REPORTS_DIR / "dataset_summary.json"
SCHEMA_PATH = REPORTS_DIR / "feature_schema.json"
AGE_BINS, AGE_LABELS = [17, 25, 35, 45, 55, 69], ["18-25", "26-35", "36-45", "46-55", "56-69"]
CREDIT_BINS = [299, 579, 669, 739, 799, 850]
CREDIT_LABELS = ["Poor (<580)", "Fair (580-669)", "Good (670-739)", "Very good (740-799)", "Excellent (800+)"]
MEAN_COLUMNS = ["Age", "Income", "LoanAmount", "CreditScore", "MonthsEmployed", "InterestRate", "DTIRatio"]


def add_bands(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["AgeGroup"] = pd.cut(df["Age"], AGE_BINS, labels=AGE_LABELS)
    df["CreditBand"] = pd.cut(df["CreditScore"], CREDIT_BINS, labels=CREDIT_LABELS)
    return df


def build_summary(df: pd.DataFrame) -> dict:
    raw_columns = int(df.shape[1])      # count BEFORE helper columns are added
    df = add_bands(df)
    rate = lambda col: {str(k): round(float(v) * 100, 2)
                        for k, v in df.groupby(col, observed=True)[TARGET].mean().items()}
    return {
        "rows": int(len(df)), "columns": raw_columns,
        "defaults": int(df[TARGET].sum()), "non_defaults": int((df[TARGET] == 0).sum()),
        "default_rate_pct": round(float(df[TARGET].mean() * 100), 2),
        "means": {c: round(float(df[c].mean()), 2) for c in MEAN_COLUMNS},
        "default_rate_by": {c: rate(c) for c in CATEGORICAL_COLUMNS + ["AgeGroup", "CreditBand"]},
    }


def build_feature_schema(df: pd.DataFrame) -> dict:
    """What a valid input looks like, learned from the training data (used by input validation)."""
    numeric = {}
    for col in NUMERIC_COLUMNS:
        entry = {"min": float(df[col].min()), "max": float(df[col].max()), "median": float(df[col].median())}
        if df[col].nunique() <= 10:                       # discrete features such as LoanTerm
            entry["values"] = sorted(float(v) for v in df[col].unique())
        numeric[col] = entry
    categorical = {c: {"values": sorted(df[c].dropna().unique().tolist()),
                       "most_common": str(df[c].mode().iloc[0])} for c in CATEGORICAL_COLUMNS}
    return {"numeric": numeric, "categorical": {c: v["values"] for c, v in categorical.items()},
            "most_common": {c: v["most_common"] for c, v in categorical.items()}}


def main():
    path = next(RAW_DIR.glob("*.csv"))
    df = pd.read_csv(path)
    REPORTS_DIR.mkdir(exist_ok=True)
    SAMPLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps(build_summary(df), indent=2))
    SCHEMA_PATH.write_text(json.dumps(build_feature_schema(df), indent=2))
    sample = (df.drop(columns=["LoanID"], errors="ignore")
                .groupby(TARGET, group_keys=False)
                .sample(frac=20_000 / len(df), random_state=RANDOM_STATE))
    sample.to_csv(SAMPLE_PATH, index=False)
    print(f"Saved {SUMMARY_PATH.name}, {SCHEMA_PATH.name} (full data: {len(df):,} rows) and {SAMPLE_PATH.name} ({len(sample):,} rows)")


if __name__ == "__main__":
    main()


# ---------- loaders used by the Streamlit pages ----------
def _cache():
    import streamlit as st
    return st.cache_data(show_spinner=False)


def load_summary() -> dict | None:
    return _cached_json(str(SUMMARY_PATH))


def load_metrics() -> dict | None:
    return _cached_json(str(REPORTS_DIR / "metrics.json"))


def load_shap_importance() -> pd.DataFrame | None:
    p = REPORTS_DIR / "shap_importance.csv"
    return pd.read_csv(p) if p.exists() else None


def _cached_json(path: str):
    import streamlit as st

    @st.cache_data(show_spinner=False)
    def _read(p: str, mtime: float):
        return json.loads(open(p, encoding="utf-8").read())

    from pathlib import Path
    f = Path(path)
    return _read(path, f.stat().st_mtime) if f.exists() else None


def load_dataframe():
    """Full raw data if present on this machine, otherwise the bundled sample. Returns (df, source)."""
    import streamlit as st
    from src.preprocessing import add_features

    @st.cache_data(show_spinner="Loading data...")
    def _load(use_raw: bool):
        df = pd.read_csv(next(RAW_DIR.glob("*.csv"))) if use_raw else pd.read_csv(SAMPLE_PATH)
        df = add_features(add_bands(df)).drop(columns=["LoanID"], errors="ignore")
        df["Status"] = df[TARGET].map({0: "No default", 1: "Default"})
        return df

    use_raw = RAW_DIR.exists() and any(RAW_DIR.glob("*.csv"))
    if not use_raw and not SAMPLE_PATH.exists():
        return None, None
    return _load(use_raw), ("full dataset" if use_raw else "20,000-row stratified sample")


def load_json(name: str):
    """Read reports/<name> (cached, refreshed when the file changes)."""
    return _cached_json(str(REPORTS_DIR / name))


# files each step creates, so pages can say exactly which command is missing
_NEEDS = {
    "train": ("python -m src.train", [REPORTS_DIR / "metrics.json", REPORTS_DIR / "curves.json"]),
    "calibrate": ("python -m src.calibrate", [REPORTS_DIR / "calibration_params.json", REPORTS_DIR / "calibration.json",
                                              REPORTS_DIR / "test_scores.csv.gz"]),
    "explain": ("python -m src.explain", [REPORTS_DIR / "shap_sample.joblib", REPORTS_DIR / "shap_importance.csv"]),
    "app_data": ("python -m src.app_data", [SUMMARY_PATH, SCHEMA_PATH]),
    "fairness": ("python -m src.fairness", [REPORTS_DIR / "fairness.json"]),
}


def missing_steps(*steps: str) -> list:
    return [(_NEEDS[k][0], [p.name for p in _NEEDS[k][1] if not p.exists()])
            for k in steps if any(not p.exists() for p in _NEEDS[k][1])]


def stop_if_missing(*steps: str):
    """Show a friendly message (instead of a crash) when a page needs files that were not generated yet."""
    import streamlit as st
    missing = missing_steps(*steps)
    if missing:
        lines = "\n".join(f"- run `{cmd}` (missing: {', '.join(files)})" for cmd, files in missing)
        st.error("Some files this page needs have not been created yet:\n" + lines
                 + "\n\nOr run everything at once with `python -m src.run_all`.")
        st.stop()


def get_bundle():
    """Final model + calibrator + threshold, loaded once and reused."""
    import streamlit as st

    @st.cache_resource(show_spinner="Loading model...")
    def _load(token: float):
        from src.predict import load_bundle
        return load_bundle()

    return _load((REPORTS_DIR / "metrics.json").stat().st_mtime)


def get_explainer():
    """SHAP explainer for the final model, built once."""
    import streamlit as st

    @st.cache_resource(show_spinner="Preparing explanations...")
    def _build(token: float):
        import joblib
        from src.explain import build_explainer
        background = joblib.load(REPORTS_DIR / "shap_sample.joblib")["background"]
        return build_explainer(get_bundle()["pipeline"], background)

    return _build((REPORTS_DIR / "shap_sample.joblib").stat().st_mtime)
