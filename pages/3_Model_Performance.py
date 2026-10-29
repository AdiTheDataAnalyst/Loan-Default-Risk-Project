"""Page 3: compare the models using real results saved by training."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import app_data as ad
from src.theme import (AMBER, BLUE, ROSE, TEAL, VIOLET, hero, kpi_row,
                       section, style_fig)

hero("Model Performance", "How the three trained algorithms compare, measured on data they never saw.",
     badge="Evaluation")
ad.stop_if_missing("train")

M = ad.load_metrics()
models, final = M["models"], M["final_model"]
model_names = [name for name in models if name != "Dummy Baseline"]
COLORS = {"Logistic Regression": TEAL, "Random Forest": VIOLET, "XGBoost": AMBER}
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
for name in model_names:
    r = models[name]
    t = r["test"]
    rows.append({"Model": ("★ " if name == final else "") + name, "PR-AUC": t["pr_auc"], "ROC-AUC": t["roc_auc"],
                 "Precision": t["precision"], "Recall": t["recall"], "F1": t["f1"]})
table = pd.DataFrame(rows)
st.dataframe(table, hide_index=True, width="stretch",
             column_config={c: st.column_config.NumberColumn(format="%.3f") for c in ["PR-AUC", "ROC-AUC", "Precision", "Recall", "F1"]})
st.caption("★ = selected model.")

# ---------------- confusion matrix ----------------
section("Confusion matrix and report", "Pick a model to see exactly where it is right and wrong at its own threshold.")
choice = st.selectbox("Model", model_names, index=model_names.index(final))
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
        st.markdown("No hyperparameter search was run for this model.")
