"""Page 4: estimate risk for one borrower, with input validation."""
import plotly.graph_objects as go
import streamlit as st

from src import app_data as ad
from src.config import CATEGORICAL_COLUMNS
from src.predict import ValidationError, load_schema, score, validate_single
from src.preprocessing import FEATURE_COLUMNS, add_features
from src.theme import AMBER, BLUE, GREEN, MUTED, ROSE, TEAL, VIOLET, disclaimer, hero, insight, kpi_row, pill, section, style_fig

hero("Loan Risk Prediction", "Enter one borrower's details to see the model's risk estimate.",
     badge="Single borrower")
ad.stop_if_missing("train", "calibrate", "app_data")

bundle, schema = ad.get_bundle(), load_schema()
overall = ad.load_summary()["default_rate_pct"]
NUM = schema["numeric"]

# ---------- input definitions: (label, kind, step) ----------
INT_FIELDS = {"Age": 1, "Income": 1000, "LoanAmount": 1000, "CreditScore": 1, "MonthsEmployed": 1}
FLOAT_FIELDS = {"InterestRate": (0.1, "%.2f"), "DTIRatio": (0.01, "%.2f")}
CHOICE_FIELDS = ["LoanTerm", "NumCreditLines"]

# first visit: start from typical values (dataset medians / most common categories)
for col, spec in NUM.items():
    st.session_state.setdefault(f"f_{col}", int(spec["median"]) if (col in INT_FIELDS or col in CHOICE_FIELDS) else float(spec["median"]))
for col in CATEGORICAL_COLUMNS:
    st.session_state.setdefault(f"f_{col}", schema["most_common"][col])

PRESETS = {
    "Typical borrower": {**{c: int(NUM[c]["median"]) if (c in INT_FIELDS or c in CHOICE_FIELDS) else float(NUM[c]["median"]) for c in NUM},
                         **{c: schema["most_common"][c] for c in CATEGORICAL_COLUMNS}},
    "Example: higher-risk inputs": dict(Age=29, Income=50000, LoanAmount=140000, CreditScore=540, MonthsEmployed=18, NumCreditLines=3,
                                        InterestRate=16.0, LoanTerm=48, DTIRatio=0.6, Education="High School", EmploymentType="Part-time",
                                        MaritalStatus="Single", HasMortgage="No", HasDependents="Yes", LoanPurpose="Auto", HasCoSigner="No"),
    "Example: lower-risk inputs": dict(Age=47, Income=95000, LoanAmount=90000, CreditScore=680, MonthsEmployed=70, NumCreditLines=2,
                                       InterestRate=11.0, LoanTerm=36, DTIRatio=0.4, Education="Master's", EmploymentType="Full-time",
                                       MaritalStatus="Married", HasMortgage="Yes", HasDependents="No", LoanPurpose="Home", HasCoSigner="Yes"),
}


def apply_preset(values):
    for key, val in values.items():
        st.session_state[f"f_{key}"] = val


section("Borrower details", "Start from a preset, then change any value. Ranges match what the model saw in training; values outside them are rejected.")
pc = st.columns(len(PRESETS))
for col, (name, vals) in zip(pc, PRESETS.items()):
    col.button(name, on_click=apply_preset, args=(vals,), width="stretch")

with st.form("borrower"):
    a, b, c = st.columns(3)
    with a:
        st.markdown("**Borrower**")
        st.number_input("Age", int(NUM["Age"]["min"]), int(NUM["Age"]["max"]), step=1, key="f_Age")
        st.number_input("Income (annual)", int(NUM["Income"]["min"]), int(NUM["Income"]["max"]), step=1000, key="f_Income")
        st.number_input("Credit score", int(NUM["CreditScore"]["min"]), int(NUM["CreditScore"]["max"]), step=1, key="f_CreditScore")
        st.number_input("Months employed", int(NUM["MonthsEmployed"]["min"]), int(NUM["MonthsEmployed"]["max"]), step=1, key="f_MonthsEmployed")
        st.selectbox("Employment type", schema["categorical"]["EmploymentType"], key="f_EmploymentType")
        st.selectbox("Education", schema["categorical"]["Education"], key="f_Education")
    with b:
        st.markdown("**Loan**")
        st.number_input("Loan amount", int(NUM["LoanAmount"]["min"]), int(NUM["LoanAmount"]["max"]), step=1000, key="f_LoanAmount")
        st.number_input("Interest rate (%)", float(NUM["InterestRate"]["min"]), float(NUM["InterestRate"]["max"]), step=0.1, format="%.2f", key="f_InterestRate")
        st.selectbox("Loan term (months)", [int(v) for v in NUM["LoanTerm"]["values"]], key="f_LoanTerm")
        st.number_input("Debt-to-income ratio", float(NUM["DTIRatio"]["min"]), float(NUM["DTIRatio"]["max"]), step=0.01, format="%.2f", key="f_DTIRatio")
        st.selectbox("Loan purpose", schema["categorical"]["LoanPurpose"], key="f_LoanPurpose")
        st.selectbox("Has co-signer", schema["categorical"]["HasCoSigner"], key="f_HasCoSigner")
    with c:
        st.markdown("**Household and credit**")
        st.selectbox("Number of credit lines", [int(v) for v in NUM["NumCreditLines"]["values"]], key="f_NumCreditLines")
        st.selectbox("Marital status", schema["categorical"]["MaritalStatus"], key="f_MaritalStatus")
        st.selectbox("Has mortgage", schema["categorical"]["HasMortgage"], key="f_HasMortgage")
        st.selectbox("Has dependents", schema["categorical"]["HasDependents"], key="f_HasDependents")
    submitted = st.form_submit_button("Estimate risk", type="primary", width="stretch")

if not submitted:
    st.info("Fill in the form and press **Estimate risk**. Nothing is stored or sent anywhere.")
    disclaimer()
    st.stop()

# ---------- validate and score ----------
values = {col: st.session_state[f"f_{col}"] for col in schema["numeric"].keys() | set(CATEGORICAL_COLUMNS)}
try:
    row = validate_single(values)
except ValidationError as err:
    st.error(f"Please fix the input: {err}")
    st.stop()

res = score(bundle, row).iloc[0]
raw, calibrated, flagged, thr = float(res["ModelScore"]), float(res["EstimatedDefaultRate"]), bool(res["ReviewFlag"]), bundle["threshold"]
features = add_features(row)[FEATURE_COLUMNS]

section("Result", "These are model estimates for a learning project, not a lending decision.")
left, right = st.columns([1.1, 1])
gauge = go.Figure(go.Indicator(
    mode="gauge+number", value=raw * 100, number=dict(suffix="%", font=dict(size=44, color="#F4F7FF")),
    title=dict(text="Model risk score", font=dict(size=15, color=MUTED)),
    gauge=dict(axis=dict(range=[0, 100], tickcolor=MUTED), bar=dict(color=ROSE if flagged else TEAL, thickness=0.28),
               bgcolor="rgba(0,0,0,0)", borderwidth=0,
               steps=[dict(range=[0, thr * 100], color="#0F2A33"), dict(range=[thr * 100, 100], color="#32172B")],
               threshold=dict(line=dict(color=AMBER, width=4), thickness=0.85, value=thr * 100))))
left.plotly_chart(style_fig(gauge, 330, legend=False), width="stretch")
left.caption(f"The amber line is the review threshold ({thr * 100:.1f}). Scores at or above it are flagged for human review.")
with right:
    kpi_row([("Estimated default rate", f"{calibrated * 100:.1f}%", f"vs {overall:.1f}% across all loans", TEAL if not flagged else ROSE)])
    st.write("")
    kpi_row([("Model score", f"{raw * 100:.1f}", "used for ranking", VIOLET),
             ("Loan / income", f"{features['LoanToIncome'].iloc[0]:.2f}", "loan amount ÷ annual income", BLUE)], mini=True)
    st.write("")
    st.markdown(pill("Flagged for review" if flagged else "Not flagged", ROSE if flagged else GREEN), unsafe_allow_html=True)
    st.caption(f"Model: {bundle['model_name']}.")
insight(f"<b>How to read this:</b> among validation borrowers with a similar score, about <b>{calibrated * 100:.1f}%</b> actually defaulted. "
        "That is a statistical estimate, not a prediction that this person will or will not default, and the app never approves or rejects a loan.", TEAL)

if not bundle["calibrator"].is_typical(raw):
    st.warning("This profile scores far from most borrowers in the data, so few similar borrowers exist and "
               "the estimate is less reliable. Treat it as a rough indication only.")

disclaimer()
