"""Page 1: the big picture - dataset size, default rate, model performance, and key findings."""
import plotly.express as px
import streamlit as st

from src import app_data as ad
from src.theme import (AMBER, MUTED, ROSE, TEAL, VIOLET, STATUS_COLORS,
                       disclaimer, hero, kpi_row, section, style_fig)

hero("Loan Risk Intelligence Platform",
     "Analyse borrower data, estimate the chance of loan default, compare machine-learning models "
     "and see why a borrower gets a higher or lower risk estimate.")

S, M = ad.load_summary(), ad.load_metrics()
if S is None:
    st.error("Dataset summary not found. In your terminal run:  `python -m src.app_data`")
    st.stop()

# ---------- executive KPIs ----------
if M:
    final = M["final_model"]
    test_metrics = M["models"][final]["test"]
    model_pr_auc = f"{test_metrics['pr_auc']:.3f}"
    model_note = f"{final} · baseline = {S['default_rate_pct'] / 100:.3f}"
else:
    model_pr_auc = "Not available"
    model_note = "Run `python -m src.train` to create metrics"

kpi_row([
    ("Loans analysed", f"{S['rows']:,}", f"{S['columns']} columns in the raw file", TEAL),
    ("Observed default rate", f"{S['default_rate_pct']:.2f}%", f"{S['defaults']:,} defaulted loans", ROSE),
    ("Selected model PR-AUC", model_pr_auc, model_note, VIOLET),
])

if not M:
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

disclaimer()
