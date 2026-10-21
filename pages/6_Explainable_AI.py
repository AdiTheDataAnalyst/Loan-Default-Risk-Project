"""Page 6: what drives the model, one borrower at a time, plus fairness and limitations."""
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src import app_data as ad
from src.predict import score
from src.theme import AMBER, BLUE, GREEN, MUTED, ROSE, TEAL, VIOLET, disclaimer, hero, insight, kpi_row, section, style_fig

hero("Explainable AI", "See which features move the model's scores, why one borrower scores higher than another, and where the model needs caution.",
     badge="Transparency")
ad.stop_if_missing("train", "calibrate", "explain", "fairness")


@st.cache_data(show_spinner=False)
def load_sample(token: float):
    d = joblib.load(ad.REPORTS_DIR / "shap_sample.joblib")
    return d["X_sample"].reset_index(drop=True), d["shap_values"].reset_index(drop=True)


X, SHAP = load_sample((ad.REPORTS_DIR / "shap_sample.joblib").stat().st_mtime)
IMP = ad.load_shap_importance()
M, FAIR, bundle = ad.load_metrics(), ad.load_json("fairness.json"), ad.get_bundle()
NUMERIC = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]

tab1, tab2, tab3, tab4, tab5 = st.tabs(["What drives the model", "Feature effects", "One borrower", "Coefficients", "Fairness and limits"])

# ---------------- tab 1 ----------------
with tab1:
    section("Average impact of each feature", f"Mean absolute SHAP value on {len(X):,} test borrowers, for the {M['final_model']} model. Bigger = moves scores more.")
    top = IMP.head(12).iloc[::-1]
    fig = px.bar(top, x="mean_abs_shap", y="feature", orientation="h", color="mean_abs_shap",
                 color_continuous_scale=[VIOLET, TEAL], title="Top 12 features by average impact")
    fig.update_coloraxes(showscale=False)
    fig.update_traces(marker_line_width=0)
    fig.update_xaxes(title="Average impact on score (log-odds)")
    st.plotly_chart(style_fig(fig, 460, legend=False), width="stretch")

    rows = []
    for feat in IMP["feature"].head(12):
        if feat in NUMERIC:
            corr = float(np.corrcoef(X[feat], SHAP[feat])[0, 1])
            direction = ("Higher values raise risk" if corr > 0.3 else "Higher values lower risk" if corr < -0.3 else "Mixed or weak pattern")
        else:
            by = SHAP.groupby(X[feat])[feat].mean().sort_values()
            direction = f"Raises: {by.index[-1]}; lowers: {by.index[0]}"
        rows.append({"Feature": feat, "Average impact": round(float(IMP.set_index('feature').loc[feat, 'mean_abs_shap']), 3), "Pattern in this model": direction})
    st.markdown("**Direction of each effect** (worked out from the data, not typed by hand)")
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    insight("<b>How to read SHAP:</b> it splits one prediction into per-feature pieces that add up to the model's output. "
            "Positive pieces push the risk score up, negative pieces push it down. It explains the <b>model</b>, not the real world.", VIOLET)

# ---------------- tab 2 ----------------
with tab2:
    section("How one feature changes the score", "Each dot is a borrower. Height shows the feature's push on that borrower's score.")
    feat = st.selectbox("Feature", list(IMP["feature"]), index=0)
    if feat in NUMERIC:
        df = pd.DataFrame({feat: X[feat], "Impact": SHAP[feat]})
        fig = px.scatter(df, x=feat, y="Impact", color="Impact", color_continuous_scale=[TEAL, "#3A4A6B", ROSE], opacity=0.6,
                         title=f"Impact of {feat} on the risk score")
        fig.add_hline(y=0, line_dash="dot", line_color=MUTED)
        fig.update_coloraxes(showscale=False)
    else:
        by = SHAP.groupby(X[feat])[feat].agg(["mean", "size"]).reset_index().rename(columns={feat: "Category", "mean": "Average impact", "size": "Borrowers"})
        by = by.sort_values("Average impact")
        fig = px.bar(by, x="Category", y="Average impact", color="Average impact", color_continuous_scale=[TEAL, "#3A4A6B", ROSE],
                     text=by["Average impact"].round(3), title=f"Average impact of {feat} categories")
        fig.update_coloraxes(showscale=False)
        fig.update_traces(marker_line_width=0)
    st.plotly_chart(style_fig(fig, 420, legend=False), width="stretch")
    st.caption("Above zero = raises the risk score. Below zero = lowers it. Compare this with what you saw in the Exploratory Analysis page.")

# ---------------- tab 3 ----------------
with tab3:
    section("Explain a single borrower", "Pick one of the test borrowers the model was never trained on.")
    st.session_state.setdefault("xai_idx", 0)

    def pick_random(n):            # runs BEFORE the widgets are drawn, which Streamlit requires
        st.session_state["xai_idx"] = int(np.random.randint(n))

    c1, c2 = st.columns([3, 1])
    c1.number_input("Borrower number", 0, len(X) - 1, step=1, key="xai_idx")
    c2.write("")
    c2.write("")
    c2.button("Random borrower", on_click=pick_random, args=(len(X),), width="stretch")
    i = int(st.session_state["xai_idx"])
    row = X.iloc[[i]]
    res = score(bundle, row).iloc[0]
    kpi_row([("Model score", f"{res['ModelScore'] * 100:.1f}", "used for ranking", VIOLET),
             ("Estimated default rate", f"{res['EstimatedDefaultRate'] * 100:.1f}%", "calibrated", TEAL),
             ("Review flag", "Flagged" if res["ReviewFlag"] else "Not flagged", f"threshold {bundle['threshold'] * 100:.1f}", ROSE if res["ReviewFlag"] else GREEN)], mini=True)
    contrib = pd.DataFrame({"feature": SHAP.columns, "contribution": SHAP.iloc[i].values, "value": [row.iloc[0][c] for c in SHAP.columns]})
    contrib = contrib.reindex(contrib["contribution"].abs().sort_values(ascending=False).index).head(10).iloc[::-1]
    label = [f"{f} = {v:,.2f}" if isinstance(v, (float, np.floating)) else f"{f} = {v}" for f, v in zip(contrib["feature"], contrib["value"])]
    fig = go.Figure(go.Bar(x=contrib["contribution"], y=label, orientation="h",
                           marker_color=[ROSE if c > 0 else TEAL for c in contrib["contribution"]]))
    fig.update_layout(title="Top 10 contributions for this borrower", xaxis_title="Pushes risk down  ←   impact (log-odds)   →  pushes risk up")
    st.plotly_chart(style_fig(fig, 430, legend=False), width="stretch")
    with st.expander("This borrower's input values"):
        st.dataframe(row.T.rename(columns={row.index[0]: "Value"}).astype(str), width="stretch")

# ---------------- tab 4 ----------------
with tab4:
    section("Logistic Regression coefficients", "A simpler second view. Only available when the selected model is Logistic Regression.")
    coef_path = ad.REPORTS_DIR / "coefficients.csv"
    if M["final_model"] != "Logistic Regression" or not coef_path.exists():
        st.info("The selected model is not Logistic Regression, so there is no coefficient table. SHAP above explains the selected model.")
    else:
        coefs = pd.read_csv(coef_path).head(14).iloc[::-1]
        fig = px.bar(coefs, x="coefficient", y="feature", orientation="h", color=coefs["coefficient"] > 0,
                     color_discrete_map={True: ROSE, False: TEAL}, title="Largest coefficients (positive raises risk)")
        fig.update_traces(marker_line_width=0)
        st.plotly_chart(style_fig(fig, 460, legend=False), width="stretch")
        st.dataframe(coefs.iloc[::-1].round(3).rename(columns={"odds_ratio": "odds ratio"}), hide_index=True, width="stretch")
        insight("<b>Odds ratio:</b> above 1 means higher odds of default, below 1 lower odds. For numeric features the numbers were standardised, "
                "so an odds ratio describes a one-standard-deviation increase. Category rows are relative to the other categories of that feature.", BLUE)

# ---------------- tab 5 ----------------
with tab5:
    section("Fairness check", "Does the model treat groups differently? Computed on the test set at the model's review threshold.")
    attr = st.selectbox("Group by", list(FAIR["groups"]))
    g = pd.DataFrame(FAIR["groups"][attr])
    fig = go.Figure()
    fig.add_bar(x=g["group"], y=g["default_rate"] * 100, name="Actual default rate", marker_color=VIOLET)
    fig.add_bar(x=g["group"], y=g["flag_rate"] * 100, name="Share flagged for review", marker_color=AMBER)
    fig.update_layout(barmode="group", title=f"Actual default rate vs share flagged, by {attr}", yaxis_title="%")
    st.plotly_chart(style_fig(fig, 380), width="stretch")
    table = g.rename(columns={"group": "Group", "n": "Borrowers", "default_rate": "Default rate", "flag_rate": "Flag rate",
                              "recall": "Recall", "precision": "Precision"})
    st.dataframe(table, hide_index=True, width="stretch",
                 column_config={c: st.column_config.NumberColumn(format="%.3f") for c in ["Default rate", "Flag rate", "Recall", "Precision"]})

    gaps = {a: max(r["flag_rate"] for r in rows) - min(r["flag_rate"] for r in rows) for a, rows in FAIR["groups"].items()}
    worst = max(gaps, key=gaps.get)
    hi = max(FAIR["groups"][worst], key=lambda r: r["flag_rate"])
    lo = min(FAIR["groups"][worst], key=lambda r: r["flag_rate"])
    SENSITIVE = ["Age", "MaritalStatus", "Education", "HasDependents"]
    share = IMP[IMP["feature"].isin(SENSITIVE)]["mean_abs_shap"].sum() / IMP["mean_abs_shap"].sum() * 100
    insight(f"<b>Largest gap:</b> by {worst}, the flag rate runs from <b>{lo['flag_rate'] * 100:.0f}%</b> ({lo['group']}) to "
            f"<b>{hi['flag_rate'] * 100:.0f}%</b> ({hi['group']}). Part of this mirrors real differences in default rates in the data, "
            "but a gap this size would need serious review before any real use.", ROSE)
    insight(f"<b>Sensitive features carry weight:</b> age, marital status, education and dependents together account for about "
            f"<b>{share:.0f}%</b> of the model's average impact. Real lending rules often restrict or forbid several of these.", AMBER)

    section("Limits you should state out loud")
    for title, body, colr in [
        ("Association is not causation", "SHAP and coefficients describe patterns the model learned. They do not show that a feature causes default.", VIOLET),
        ("The dataset looks synthetic", "Category sizes are almost perfectly even and credit score has a weak link to default, which real lending data rarely shows. Results may not transfer to real borrowers.", AMBER),
        ("Interest rate may carry the lender's own view of risk", "In real lending the rate is partly set because of perceived risk, so using it can make a model look better than it would at the point of application.", AMBER),
        ("Scores were calibrated on one validation set", "The estimated default rate is only as reliable as that set. It has not been tested over time or on another lender's data.", TEAL),
        ("Threshold choice is a business decision", "We weighted recall twice as much as precision. A real lender would set the trade-off from actual costs.", BLUE),
        ("Not a validated lending system", "No legal, fairness or regulatory review has been done. It must never approve or reject real applications.", ROSE)]:
        insight(f"<b>{title}.</b> {body}", colr)
disclaimer()
