"""Page 3: compare the models using real results saved by training."""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src import app_data as ad
from src.theme import (AMBER, BLUE, GREEN, MUTED, ROSE, TEAL, VIOLET, disclaimer, hero, insight, kpi_row,
                       section, style_fig)

hero("Model Performance", "How the three algorithms compare against a do-nothing baseline, measured on data they never saw.",
     badge="Evaluation")
ad.stop_if_missing("train", "calibrate")

M, CURVES, CAL = ad.load_metrics(), ad.load_json("curves.json"), ad.load_json("calibration.json")
models, final = M["models"], M["final_model"]
COLORS = {"Dummy Baseline": "#64748B", "Logistic Regression": TEAL, "Random Forest": VIOLET, "XGBoost": AMBER}
sizes = M["split_sizes"]

kpi_row([
    ("Selected model", final, "highest validation PR-AUC", TEAL),
    ("Review threshold", f"{M['final_threshold']:.3f}", "chosen on validation data (F2)", AMBER),
    ("Test loans", f"{sizes['test']:,}", "never used for tuning", BLUE),
    ("Test default rate", f"{M['default_rate']['test'] * 100:.2f}%", "same in every split", ROSE),
], weights=[1.6, 1, 1, 1])

# ---------------- comparison table ----------------
section("Model comparison (test set)", "All models use the same split. Each model's threshold was picked on validation data, not on this test set.")
rows = []
for name, r in models.items():
    t = r["test"]
    rows.append({"Model": ("★ " if name == final else "") + name, "PR-AUC": t["pr_auc"], "ROC-AUC": t["roc_auc"],
                 "Precision": t["precision"], "Recall": t["recall"], "F1": t["f1"], "F2": t["f2"],
                 "Train time (s)": r["train_seconds"], "Predict time (s)": r["predict_seconds_validation"]})
table = pd.DataFrame(rows)
st.dataframe(table, hide_index=True, width="stretch",
             column_config={c: st.column_config.NumberColumn(format="%.3f") for c in ["PR-AUC", "ROC-AUC", "Precision", "Recall", "F1", "F2"]})
st.caption("★ = selected model. Predict time is for the 51,069 validation loans. The Dummy Baseline flags every loan, "
           "so its recall is 100% but its precision only equals the default rate.")

metric_names = ["PR-AUC", "ROC-AUC", "Precision", "Recall", "F2"]
long = table.melt(id_vars="Model", value_vars=metric_names, var_name="Metric", value_name="Score")
long["Model"] = long["Model"].str.replace("★ ", "", regex=False)
fig = px.bar(long, x="Metric", y="Score", color="Model", barmode="group", color_discrete_map=COLORS,
             title="Key metrics by model (test set)", text_auto=".2f")
fig.update_traces(textposition="outside", marker_line_width=0, cliponaxis=False)
st.plotly_chart(style_fig(fig, 400), width="stretch")

best_three = {k: v["test"]["pr_auc"] for k, v in models.items() if k != "Dummy Baseline"}
spread = max(best_three.values()) - min(best_three.values())
insight(f"<b>The three real models are close:</b> test PR-AUC ranges over only <b>{spread:.3f}</b>. "
        f"{final} was selected on validation PR-AUC, but the differences are small enough that other reasonable choices exist. "
        "Simpler models are easier to explain and audit.", AMBER)

# ---------------- curves ----------------
section("ROC and precision-recall curves", "Curves are drawn from the saved test-set predictions of each model.")
c1, c2 = st.columns(2)
roc, pr = go.Figure(), go.Figure()
for name, c in CURVES.items():
    if name == "Dummy Baseline":
        continue
    roc.add_trace(go.Scatter(x=c["fpr"], y=c["tpr"], mode="lines", name=f"{name} (AUC {models[name]['test']['roc_auc']:.3f})",
                             line=dict(color=COLORS[name], width=3)))
    pr.add_trace(go.Scatter(x=c["recall"], y=c["precision"], mode="lines", name=f"{name} (AP {models[name]['test']['pr_auc']:.3f})",
                            line=dict(color=COLORS[name], width=3)))
roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random guessing", line=dict(color="#64748B", dash="dot")))
base = M["default_rate"]["test"]
pr.add_trace(go.Scatter(x=[0, 1], y=[base, base], mode="lines", name=f"Baseline ({base:.3f})", line=dict(color="#64748B", dash="dot")))
roc.update_layout(title="ROC curve", xaxis_title="False positive rate", yaxis_title="True positive rate (recall)")
pr.update_layout(title="Precision-recall curve", xaxis_title="Recall", yaxis_title="Precision")
c1.plotly_chart(style_fig(roc, 420), width="stretch")
c2.plotly_chart(style_fig(pr, 420), width="stretch")
st.caption("ROC: the further up and left, the better. Precision-recall is more informative when defaults are rare, "
           "because it ignores the large pile of easy non-defaults.")

# ---------------- confusion matrix ----------------
section("Confusion matrix and report", "Pick a model to see exactly where it is right and wrong at its own threshold.")
choice = st.selectbox("Model", list(models), index=list(models).index(final))
t, v = models[choice]["test"], models[choice]["validation"]
cm = t["confusion_matrix"]
counts = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])
pct = counts / counts.sum(axis=1, keepdims=True) * 100
labels = np.array([[f"{counts[i, j]:,}<br>{pct[i, j]:.1f}% of row" for j in range(2)] for i in range(2)])
heat = go.Figure(go.Heatmap(z=pct, x=["Predicted: no default", "Predicted: default"], y=["Actual: no default", "Actual: default"],
                            text=labels, texttemplate="%{text}", colorscale=[[0, "#101830"], [1, TEAL]], showscale=False,
                            hovertemplate="%{text}<extra></extra>"))
heat.update_yaxes(autorange="reversed")
heat.update_layout(title=f"{choice}: confusion matrix (threshold {t['threshold']:.3f})")
c1, c2 = st.columns([1.1, 1])
c1.plotly_chart(style_fig(heat, 380, legend=False), width="stretch")
with c2:
    st.markdown(f"""
- **True negatives {cm['tn']:,}:** correctly left alone
- **False positives {cm['fp']:,}:** flagged but did not default (false alarms)
- **False negatives {cm['fn']:,}:** defaults the model missed
- **True positives {cm['tp']:,}:** defaults correctly flagged
""")
    report = pd.DataFrame(t["classification_report"]).T.loc[["No default", "Default", "macro avg", "weighted avg"],
                                                            ["precision", "recall", "f1-score", "support"]]
    st.dataframe(report.round(3), width="stretch")

with st.expander("Validation vs test, tuning details and cross-validation"):
    cmp = pd.DataFrame({"Validation": {k: v[k] for k in ["pr_auc", "roc_auc", "precision", "recall", "f1", "f2"]},
                        "Test": {k: t[k] for k in ["pr_auc", "roc_auc", "precision", "recall", "f1", "f2"]}})
    st.markdown("**Validation and test side by side.** Similar numbers suggest the model is not just memorising training data.")
    st.dataframe(cmp.round(4), width="stretch")
    r = models[choice]
    if r["best_params"]:
        st.markdown(f"**Best hyperparameters found** (random search, 3-fold cross-validation, scored on PR-AUC): `{r['best_params']}`")
        st.markdown(f"**Cross-validated PR-AUC on the tuning sample:** {r['cv_pr_auc']}")
    else:
        st.markdown("The baseline has nothing to tune.")

# ---------------- time ----------------
section("Training and prediction time")
c1, c2 = st.columns(2)
tm = table.assign(Model=table["Model"].str.replace("★ ", "", regex=False))
f = px.bar(tm, x="Model", y="Train time (s)", color="Model", color_discrete_map=COLORS, title="Training time (seconds)", text_auto=".1f")
f.update_traces(marker_line_width=0)
g = px.bar(tm, x="Model", y="Predict time (s)", color="Model", color_discrete_map=COLORS, title="Prediction time on 51,069 loans (seconds)", text_auto=".2f")
g.update_traces(marker_line_width=0)
c1.plotly_chart(style_fig(f, 330, legend=False), width="stretch")
c2.plotly_chart(style_fig(g, 330, legend=False), width="stretch")

# ---------------- threshold explorer ----------------
section("Threshold explorer", f"Drag the slider to see the trade-off for the selected model ({final}) on the test set. "
        "A lower threshold catches more defaults but raises more false alarms.")
scores = pd.read_csv(ad.REPORTS_DIR / "test_scores.csv.gz")
thr = st.slider("Review threshold (model score)", 0.05, 0.95, float(round(M["final_threshold"], 2)), 0.01)
flag = scores["score"] >= thr
y = scores["y_true"] == 1
tp, fp, fn = int((flag & y).sum()), int((flag & ~y).sum()), int((~flag & y).sum())
prec, rec = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
f2 = 5 * prec * rec / max(4 * prec + rec, 1e-9)
kpi_row([("Share of loans flagged", f"{flag.mean() * 100:.1f}%", f"{int(flag.sum()):,} loans", BLUE),
         ("Recall", f"{rec * 100:.1f}%", f"{tp:,} defaults caught, {fn:,} missed", GREEN),
         ("Precision", f"{prec * 100:.1f}%", f"{fp:,} false alarms", ROSE), ("F2 score", f"{f2:.3f}", "recall weighted 2x", AMBER)], mini=True)
grid = np.round(np.arange(0.05, 0.96, 0.01), 2)
rows = []
for g_ in grid:
    fl = scores["score"] >= g_
    tp_ = (fl & y).sum()
    p_, r_ = tp_ / max(fl.sum(), 1), tp_ / y.sum()
    rows.append((g_, p_, r_, 5 * p_ * r_ / max(4 * p_ + r_, 1e-9)))
curve = pd.DataFrame(rows, columns=["Threshold", "Precision", "Recall", "F2"]).melt("Threshold", var_name="Metric", value_name="Value")
fig = px.line(curve, x="Threshold", y="Value", color="Metric", color_discrete_map={"Precision": ROSE, "Recall": GREEN, "F2": AMBER},
              title="Precision, recall and F2 across thresholds")
fig.add_vline(x=thr, line_dash="dot", line_color=MUTED)
st.plotly_chart(style_fig(fig, 360), width="stretch")
insight("<b>Why F2?</b> A missed default usually costs a lender more than a false alarm, so we weight recall twice as much as precision "
        "when choosing the threshold. A real lender would set it from actual costs.", VIOLET)

# ---------------- calibration ----------------
section("Are the scores trustworthy as probabilities?", "Because we trained with class weights, raw scores run high. Calibration corrects that.")
c1, c2 = st.columns([1, 1])
with c1:
    kpi_row([("Actual default rate", f"{CAL['actual_default_rate_test'] * 100:.1f}%", "test set", TEAL),
             ("Average raw score", f"{CAL['mean_raw_score_test'] * 100:.1f}%", "too high", ROSE),
             ("Average calibrated", f"{CAL['mean_calibrated_test'] * 100:.1f}%", "matches reality", GREEN)], mini=True)
    st.markdown(f"Brier score (lower is better): **{CAL['brier_raw']}** raw → **{CAL['brier_calibrated']}** calibrated. "
                f"ROC-AUC barely moves ({CAL['roc_auc_raw']} → {CAL['roc_auc_calibrated']}) because calibration keeps the order of borrowers.")
with c2:
    cal = go.Figure()
    cal.add_trace(go.Scatter(x=[0, 0.8], y=[0, 0.8], mode="lines", name="Perfect", line=dict(color="#64748B", dash="dot")))
    for key, label, colr in [("reliability_raw", "Raw score", ROSE), ("reliability_calibrated", "Calibrated", GREEN)]:
        d = pd.DataFrame(CAL[key])
        cal.add_trace(go.Scatter(x=d["predicted"], y=d["observed"], mode="lines+markers", name=label, line=dict(color=colr, width=3)))
    cal.update_layout(title="Predicted vs actual default rate (10 equal-sized groups)", xaxis_title="Average prediction",
                      yaxis_title="Share that actually defaulted")
    st.plotly_chart(style_fig(cal, 340), width="stretch")
st.caption("Points on the dotted line mean 'when the model says 20%, about 20% default'. This is a prototype check on one dataset, not a validated lending model.")
disclaimer()
