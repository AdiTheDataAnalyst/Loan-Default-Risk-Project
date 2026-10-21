"""Page 2: interactive exploration with filters, distributions, categories, relationships and risk bands."""
import pandas as pd
import plotly.express as px
import streamlit as st

from src import app_data as ad
from src.config import CATEGORICAL_COLUMNS, ENGINEERED_COLUMNS, NUMERIC_COLUMNS, TARGET
from src.theme import (AMBER, BLUE, GREEN, MUTED, ROSE, TEAL, VIOLET, STATUS_COLORS,
                       disclaimer, hero, insight, kpi_row, section, style_fig)

hero("Exploratory Analysis", "Filter the borrowers, then explore how each feature relates to default.",
     badge="Interactive")

df, source = ad.load_dataframe()
if df is None:
    st.error("No data found. Put the CSV in `data/raw/` or run `python -m src.app_data` to create the sample.")
    st.stop()
st.caption(f"Charts below use the **{source}** ({len(df):,} loans).")

# ---------- filters ----------
with st.expander("Filters", expanded=True):
    f1, f2, f3, f4 = st.columns(4)
    emp = f1.multiselect("Employment type", sorted(df["EmploymentType"].unique()), default=sorted(df["EmploymentType"].unique()))
    purpose = f2.multiselect("Loan purpose", sorted(df["LoanPurpose"].unique()), default=sorted(df["LoanPurpose"].unique()))
    age = f3.slider("Age", int(df["Age"].min()), int(df["Age"].max()), (int(df["Age"].min()), int(df["Age"].max())))
    cs = f4.slider("Credit score", int(df["CreditScore"].min()), int(df["CreditScore"].max()),
                   (int(df["CreditScore"].min()), int(df["CreditScore"].max())))

view = df[df["EmploymentType"].isin(emp) & df["LoanPurpose"].isin(purpose)
          & df["Age"].between(*age) & df["CreditScore"].between(*cs)]
if view.empty:
    st.warning("No borrowers match these filters. Widen them to continue.")
    st.stop()

overall = df[TARGET].mean() * 100
kpi_row([
    ("Borrowers in view", f"{len(view):,}", f"{len(view) / len(df) * 100:.1f}% of loaded data", TEAL),
    ("Default rate in view", f"{view[TARGET].mean() * 100:.2f}%", f"overall {overall:.2f}%", ROSE),
    ("Avg loan amount", f"{view['LoanAmount'].mean():,.0f}", "", BLUE),
    ("Avg interest rate", f"{view['InterestRate'].mean():.2f}%", "", AMBER),
])
if len(view) < 300:
    st.warning("Fewer than 300 borrowers in view, so rates can swing a lot. Treat them cautiously.")

NUM_ALL = NUMERIC_COLUMNS + ENGINEERED_COLUMNS
tab1, tab2, tab3, tab4 = st.tabs(["Distributions", "Categories", "Relationships", "Risk bands"])

# ---------- tab 1: distributions ----------
with tab1:
    feat = st.selectbox("Numeric feature", NUM_ALL, index=NUM_ALL.index("InterestRate"))
    fig = px.histogram(view, x=feat, color="Status", barmode="overlay", histnorm="percent", nbins=40, opacity=0.7,
                       color_discrete_map=STATUS_COLORS, title=f"{feat}: defaulters vs non-defaulters (% within each group)")
    fig.update_yaxes(title="% of group")
    st.plotly_chart(style_fig(fig, 380), width="stretch")
    st.markdown("**Summary statistics by outcome**")
    st.dataframe(view.groupby("Status")[feat].describe().round(2), width="stretch")

# ---------- tab 2: categories ----------
with tab2:
    cat = st.selectbox("Categorical feature", CATEGORICAL_COLUMNS, index=CATEGORICAL_COLUMNS.index("EmploymentType"))
    grp = view.groupby(cat)[TARGET].agg(rate="mean", loans="size").reset_index()
    grp["rate"] = (grp["rate"] * 100).round(2)
    grp = grp.sort_values("rate", ascending=False)
    c1, c2 = st.columns(2)
    f = px.bar(grp, x=cat, y="rate", text="rate", color="rate", color_continuous_scale=[TEAL, AMBER, ROSE],
               title=f"Default rate (%) by {cat}")
    f.update_traces(textposition="outside", cliponaxis=False, marker_line_width=0)
    f.update_coloraxes(showscale=False)
    f.add_hline(y=view[TARGET].mean() * 100, line_dash="dot", line_color=MUTED)
    c1.plotly_chart(style_fig(f, 360, legend=False), width="stretch")
    g = px.bar(grp, x=cat, y="loans", title=f"Number of loans by {cat}", color_discrete_sequence=[VIOLET])
    g.update_traces(marker_line_width=0)
    c2.plotly_chart(style_fig(g, 360, legend=False), width="stretch")
    st.caption("The dotted line is the default rate for the filtered borrowers. Always check loan counts before reading too much into a bar.")

# ---------- tab 3: relationships ----------
with tab3:
    corr = view[NUM_ALL + [TARGET]].corr()[TARGET].drop(TARGET).sort_values()
    c1, c2 = st.columns([1, 1])
    f = px.bar(x=corr.values, y=corr.index, orientation="h", title="Correlation with default",
               color=corr.values, color_continuous_scale=[TEAL, "#1E2A47", ROSE], range_color=[-0.2, 0.2])
    f.update_coloraxes(showscale=False)
    f.update_traces(marker_line_width=0)
    c1.plotly_chart(style_fig(f, 400, legend=False), width="stretch")
    heat = px.imshow(view[NUM_ALL].corr().round(2), text_auto=True, aspect="auto", zmin=-1, zmax=1,
                     color_continuous_scale=[TEAL, "#101830", ROSE], title="Correlation between features")
    heat.update_coloraxes(showscale=False)
    c2.plotly_chart(style_fig(heat, 400, legend=False), width="stretch")
    insight("Correlation measures how two numbers move together in <b>this</b> data. It is not proof that one causes the other, "
            "and it only captures straight-line relationships.", VIOLET)

    s1, s2 = st.columns(2)
    x = s1.selectbox("X axis", NUM_ALL, index=NUM_ALL.index("Age"))
    y = s2.selectbox("Y axis", NUM_ALL, index=NUM_ALL.index("InterestRate"))
    pts = view.sample(min(4000, len(view)), random_state=1)
    sc = px.scatter(pts, x=x, y=y, color="Status", opacity=0.55, color_discrete_map=STATUS_COLORS,
                    title=f"{x} vs {y} (random sample of up to 4,000 loans)")
    st.plotly_chart(style_fig(sc, 420), width="stretch")

# ---------- tab 4: risk bands ----------
with tab4:
    band_feat = st.selectbox("Group borrowers by", ["Age", "CreditScore", "Income", "InterestRate", "LoanAmount",
                                                    "DTIRatio", "LoanToIncome", "MonthsEmployed"], index=3)
    bins = st.slider("Number of groups", 3, 10, 6)
    banded = pd.qcut(view[band_feat], q=bins, duplicates="drop")
    out = (view.groupby(banded, observed=True)[TARGET].agg(rate="mean", loans="size").reset_index())
    out[band_feat] = out[band_feat].astype(str)
    out["rate"] = (out["rate"] * 100).round(2)
    f = px.line(out, x=band_feat, y="rate", markers=True, title=f"Default rate (%) across {band_feat} groups (equal-sized groups)",
                color_discrete_sequence=[TEAL])
    f.update_traces(line=dict(width=3), marker=dict(size=10))
    f.update_xaxes(title=f"{band_feat} range")
    st.plotly_chart(style_fig(f, 380, legend=False), width="stretch")
    st.dataframe(out.rename(columns={"rate": "Default rate (%)", "loans": "Loans"}), width="stretch", hide_index=True)

disclaimer()
