"""Page 1: the big picture - dataset size, default rate, borrower profile, model snapshot, key findings."""
import plotly.express as px
import streamlit as st

from src import app_data as ad
from src.theme import (AMBER, BLUE, GREEN, MUTED, ROSE, TEAL, VIOLET, STATUS_COLORS,
                       disclaimer, hero, insight, kpi_row, section, style_fig)

hero("Loan Risk Intelligence Platform",
     "Analyse borrower data, estimate the chance of loan default, compare machine-learning models "
     "and see why a borrower gets a higher or lower risk estimate.")

S, M = ad.load_summary(), ad.load_metrics()
if S is None:
    st.error("Dataset summary not found. In your terminal run:  `python -m src.app_data`")
    st.stop()

# ---------- dataset KPIs (full dataset) ----------
kpi_row([
    ("Loans analysed", f"{S['rows']:,}", f"{S['columns']} columns in the raw file", TEAL),
    ("Observed default rate", f"{S['default_rate_pct']:.2f}%", f"{S['defaults']:,} defaulted loans", ROSE),
    ("Non-defaulted loans", f"{S['non_defaults']:,}", "Majority class", GREEN),
    ("Class balance", f"1 : {S['non_defaults'] / S['defaults']:.1f}", "default : non-default", VIOLET),
])

section("Average borrower profile", "Means across all loans. The dataset does not state a currency, so amounts are shown as plain numbers.")
m = S["means"]
kpi_row([
    ("Age", f"{m['Age']:.1f} yrs", "", BLUE), ("Income", f"{m['Income']:,.0f}", "", BLUE),
    ("Loan amount", f"{m['LoanAmount']:,.0f}", "", BLUE), ("Credit score", f"{m['CreditScore']:.0f}", "", BLUE),
    ("Interest rate", f"{m['InterestRate']:.2f}%", "", BLUE), ("Employed", f"{m['MonthsEmployed']:.0f} mo", "", BLUE),
], mini=True)

# ---------- model snapshot (from the real training run) ----------
if M:
    final = M["final_model"]
    t = M["models"][final]["test"]
    section("Selected model snapshot", "Measured on a held-out test set the model never saw during training or tuning.")
    kpi_row([
        ("Selected model", final, "chosen on validation PR-AUC", TEAL),
        ("ROC-AUC", f"{t['roc_auc']:.3f}", "0.5 = random guessing", VIOLET),
        ("PR-AUC", f"{t['pr_auc']:.3f}", f"baseline = {S['default_rate_pct'] / 100:.3f}", AMBER),
        ("Recall", f"{t['recall'] * 100:.1f}%", f"defaults caught at threshold {t['threshold']:.2f}", GREEN),
        ("Precision", f"{t['precision'] * 100:.1f}%", "of flagged loans truly default", ROSE),
    ], weights=[1.8, 1, 1, 1, 1])      # first card is wider so the model name fits on one line
else:
    st.info("Model metrics not found yet. Run `python -m src.train` to create them.")

# ---------- charts ----------
section("Where default risk concentrates", "Observed default rate (%) by group, computed from the full dataset.")
avg = S["default_rate_pct"]


def rate_bar(key, title, order=None):
    data = S["default_rate_by"][key]
    keys = order or sorted(data, key=data.get, reverse=True)
    fig = px.bar(x=keys, y=[data[k] for k in keys], text=[f"{data[k]:.1f}%" for k in keys],
                 color=[data[k] for k in keys], color_continuous_scale=[TEAL, AMBER, ROSE],
                 labels={"x": "", "y": "Default rate (%)"}, title=title)
    fig.add_hline(y=avg, line_dash="dot", line_color=MUTED, annotation_text=f"overall {avg:.1f}%",
                  annotation_font_color=MUTED)
    fig.update_traces(textposition="outside", marker_line_width=0, cliponaxis=False)
    fig.update_coloraxes(showscale=False)
    return style_fig(fig, 340, legend=False)


c1, c2 = st.columns(2)
donut = px.pie(names=["No default", "Default"], values=[S["non_defaults"], S["defaults"]], hole=0.62,
               color=["No default", "Default"], color_discrete_map=STATUS_COLORS, title="Default vs non-default loans")
donut.update_traces(textinfo="percent", marker=dict(line=dict(color="#0B1020", width=3)))
c1.plotly_chart(style_fig(donut, 340), width="stretch")
c2.plotly_chart(rate_bar("EmploymentType", "Default rate by employment type"), width="stretch")

c3, c4 = st.columns(2)
c3.plotly_chart(rate_bar("AgeGroup", "Default rate by age group", order=ad.AGE_LABELS), width="stretch")
c4.plotly_chart(rate_bar("LoanPurpose", "Default rate by loan purpose"), width="stretch")

# ---------- key findings: generated from the data, never hand-typed numbers ----------
section("Key findings", "Generated automatically from the data. These are associations, not proof of cause.")
R = S["default_rate_by"]
hi_e, lo_e = max(R["EmploymentType"], key=R["EmploymentType"].get), min(R["EmploymentType"], key=R["EmploymentType"].get)
hi_a, lo_a = max(R["AgeGroup"], key=R["AgeGroup"].get), min(R["AgeGroup"], key=R["AgeGroup"].get)
insight(f"<b>Age shows the widest gap.</b> Borrowers aged {hi_a} default at <b>{R['AgeGroup'][hi_a]:.1f}%</b>, "
        f"versus <b>{R['AgeGroup'][lo_a]:.1f}%</b> for those aged {lo_a}.", ROSE)
insight(f"<b>Employment matters.</b> {hi_e} borrowers default at <b>{R['EmploymentType'][hi_e]:.1f}%</b> "
        f"compared with <b>{R['EmploymentType'][lo_e]:.1f}%</b> for {lo_e}.", AMBER)
cs = R["HasCoSigner"]
insight(f"<b>Co-signers.</b> Loans without a co-signer default at <b>{cs.get('No', 0):.1f}%</b>; "
        f"with a co-signer, <b>{cs.get('Yes', 0):.1f}%</b>.", TEAL)
cb = R["CreditBand"]
insight(f"<b>Credit score effect is modest here.</b> Default rate runs from <b>{cb[ad.CREDIT_LABELS[0]]:.1f}%</b> "
        f"({ad.CREDIT_LABELS[0]}) to <b>{cb[ad.CREDIT_LABELS[-1]]:.1f}%</b> ({ad.CREDIT_LABELS[-1]}). "
        "Real lending data usually shows a stronger link, one reason to treat this dataset with caution.", VIOLET)

imp = ad.load_shap_importance()
if imp is not None:
    top = ", ".join(imp["feature"].head(3))
    insight(f"<b>What the model relies on most:</b> {top} (average SHAP impact on the selected model).", BLUE)

disclaimer()
