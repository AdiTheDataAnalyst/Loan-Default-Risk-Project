"""Page 7: dataset, method, limitations, responsible use, and how to reproduce everything."""
import streamlit as st

from src import app_data as ad
from src.theme import AMBER, BLUE, GREEN, ROSE, TEAL, VIOLET, disclaimer, flow, hero, insight, kpi_row, section

hero("About the Project", "Where the data came from, how the models were built and tested, and what this prototype should and should not be used for.",
     badge="Methodology")
ad.stop_if_missing("train", "app_data")
S, M = ad.load_summary(), ad.load_metrics()
sizes = M["split_sizes"]

section("Dataset")
kpi_row([("Source", "Kaggle", "Loan Default dataset by nikhil1e9", TEAL), ("Rows", f"{S['rows']:,}", f"{S['columns']} columns", BLUE),
         ("Target", "Default", f"1 = defaulted ({S['default_rate_pct']:.2f}%)", ROSE), ("Missing values / duplicates", "0 / 0", "verified on the file", GREEN)], mini=True)
st.markdown("""
- **Link:** <https://www.kaggle.com/datasets/nikhil1e9/loan-default>
- **Identifier removed:** `LoanID` is unique per row and carries no meaning, so it is not used as a feature.
- **Predictors:** 16 application-time columns (age, income, loan amount, credit score, employment, interest rate, term, DTI ratio, education, marital status, purpose, and similar). None describe what happens after a loan defaults.
- **License:** check the license shown on the Kaggle dataset page before further reusing or redistributing the data. The raw file is included in this repository.
""")

section("Method", "Each stage is a separate module in `src/`, so it can be explained and tested on its own.")
flow([("Verify data", "Schema, row count, duplicates, missing values, target balance.", TEAL),
      ("Explore", "Distributions, defaults by group, correlations, outliers.", BLUE),
      ("Split", f"Stratified {sizes['train']:,} train / {sizes['validation']:,} validation / {sizes['test']:,} test.", VIOLET),
      ("Pipeline", "Impute, scale, one-hot encode. Fitted on training data only.", AMBER),
      ("Train and tune", "Three algorithms plus a baseline, random search, 3-fold cross-validation.", TEAL),
      ("Threshold and calibrate", "F2-based review threshold and a sigmoid calibration, both from validation data.", ROSE),
      ("Explain", "SHAP, coefficients and a group fairness check.", GREEN)])

section("Algorithms compared")
st.dataframe([
    {"Model": "Dummy baseline", "Idea in one line": "Ignores the features; the floor every real model must beat.", "Imbalance handling": "none"},
    {"Model": "Logistic Regression", "Idea in one line": "A weighted sum of features turned into a probability. Simple and easy to explain.", "Imbalance handling": "class weights"},
    {"Model": "Random Forest", "Idea in one line": "Many decision trees trained on random slices of the data, voting together.", "Imbalance handling": "class weights"},
    {"Model": "XGBoost", "Idea in one line": "Trees built one after another, each correcting the previous trees' mistakes.", "Imbalance handling": "scale_pos_weight"},
], hide_index=True, width="stretch")

section("Key decisions")
for t, b, c in [
    ("Why not accuracy?", f"About {100 - S['default_rate_pct']:.0f}% of loans do not default, so predicting 'no default' for everyone looks accurate and is useless. PR-AUC, recall and precision show what matters.", ROSE),
    ("Why PR-AUC to pick the model?", "It measures how well the model ranks the rare defaults, without depending on a threshold.", VIOLET),
    ("Why an F2-based threshold?", "A missed default usually costs more than a false alarm, so recall counts twice as much as precision.", AMBER),
    ("Why calibration?", "Class weights make raw scores run high. The calibrated estimate matches the real default rate and is what the app displays as a rate.", TEAL),
    ("How was leakage prevented?", "Preprocessing sits inside each model pipeline and is fitted on training data only. The test set is used once, for final reporting.", GREEN)]:
    insight(f"<b>{t}</b> {b}", c)

section("Limitations")
st.markdown("""
- The data looks synthetic, so results may not carry over to real borrowers.
- The models find associations, not causes. Explanations are not causal proof.
- Several features (age, marital status, education, dependents) are sensitive. See the fairness tab on the Explainable AI page.
- Calibration and thresholds come from one validation set and have not been tested over time or on other data.
- Performance is moderate: the models separate risk better than chance but are far from perfect.
""")

section("Responsible use")
st.markdown(f'<div class="disclaimer"><b>Not a lending decision system.</b> This is an educational prototype. Estimated default rates are statistical '
            'estimates, not guaranteed outcomes. The app never approves or rejects a loan, and it must not be used to do so. '
            'Any real use would need validation on real data, fairness and legal review, monitoring, and human oversight.</div>', unsafe_allow_html=True)

section("Reproduce it")
st.code("""python -m venv venv
.\\venv\\Scripts\\Activate.ps1
pip install -r requirements.txt
kaggle datasets download -d nikhil1e9/loan-default -p data\\raw --unzip
python -m src.run_all          # train, calibrate, explain, app files, fairness
streamlit run app.py""", language="powershell")
st.code("""app.py                  entry point and navigation
pages/                  the 7 Streamlit pages
src/                    config, preprocessing, train, evaluate, calibrate, explain, fairness, predict
models/                 saved model pipelines (.joblib)
reports/                metrics, curves, SHAP and summary files used by the app
notebooks/              EDA notebook
tests/                  automated tests""", language="text")
st.caption("Built with Python, pandas, scikit-learn, XGBoost, SHAP, Plotly and Streamlit.")
disclaimer()
