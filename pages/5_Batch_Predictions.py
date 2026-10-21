"""Page 5: upload a CSV, validate it, score every valid row, download the results."""
import io

import pandas as pd
import plotly.express as px
import streamlit as st

from src import app_data as ad
from src.predict import REQUIRED_COLUMNS, ValidationError, load_schema, score, validate_batch
from src.theme import AMBER, BLUE, GREEN, ROSE, TEAL, VIOLET, disclaimer, hero, insight, kpi_row, section, style_fig

MAX_ROWS, MAX_MB = 50_000, 25

hero("Batch Predictions", "Upload a CSV of borrowers, check it against the expected format, score every valid row and download the results.",
     badge="Many borrowers")
ad.stop_if_missing("train", "calibrate", "app_data")
bundle, schema = ad.get_bundle(), load_schema()

# ---------- 1. format ----------
section("1. Expected format", "One row per borrower. Extra columns (such as LoanID) are kept and passed through to the results.")
spec = [{"Column": c, "Type": "number", "Accepted values": f"{schema['numeric'][c]['min']:g} to {schema['numeric'][c]['max']:g}"}
        for c in REQUIRED_COLUMNS if c in schema["numeric"]]
spec += [{"Column": c, "Type": "text", "Accepted values": ", ".join(schema["categorical"][c])} for c in REQUIRED_COLUMNS if c in schema["categorical"]]
with st.expander("Required columns", expanded=False):
    st.dataframe(pd.DataFrame(spec), hide_index=True, width="stretch")

sample_path = ad.SAMPLE_PATH
c1, c2 = st.columns(2)
if sample_path.exists():
    template = pd.read_csv(sample_path).head(5)[REQUIRED_COLUMNS]
    c1.download_button("Download a 5-row template CSV", template.to_csv(index=False).encode(), "loan_batch_template.csv",
                       "text/csv", width="stretch")
    if c2.button("Try it with 300 sample borrowers", width="stretch"):
        st.session_state["demo_df"] = pd.read_csv(sample_path).drop(columns=["Default"], errors="ignore").head(300)
        st.session_state["demo_name"] = "sample of 300 borrowers"

# ---------- 2. upload ----------
section("2. Upload")
upload = st.file_uploader("CSV file", type=["csv"], help=f"Up to {MAX_ROWS:,} rows and {MAX_MB} MB.")
df, source = None, None
if upload is not None:
    st.session_state.pop("demo_df", None)
    if upload.size > MAX_MB * 1024 * 1024:
        st.error(f"This file is larger than {MAX_MB} MB. Please split it into smaller files.")
        st.stop()
    try:
        df = pd.read_csv(upload)
    except Exception as err:      # unreadable / not a CSV
        st.error(f"Could not read this file as a CSV: {err}")
        st.stop()
    source = upload.name
elif "demo_df" in st.session_state:
    df, source = st.session_state["demo_df"], st.session_state["demo_name"]

if df is None:
    st.info("Upload a CSV, or press **Try it with 300 sample borrowers** to see how this page works.")
    disclaimer()
    st.stop()
if df.empty:
    st.error("The file has headers but no rows.")
    st.stop()
if len(df) > MAX_ROWS:
    st.error(f"This file has {len(df):,} rows. The limit is {MAX_ROWS:,}. Please split it.")
    st.stop()

# ---------- 3. validation ----------
section("3. Validation", f"Checking {source}.")
try:
    valid, issues, n_bad = validate_batch(df, schema)
except ValidationError as err:
    st.error(str(err))
    st.markdown("Compare your column names with the **Required columns** table above. Names must match exactly.")
    st.stop()

kpi_row([("Rows uploaded", f"{len(df):,}", source, BLUE), ("Rows scored", f"{len(valid):,}", "passed every check", GREEN),
         ("Rows rejected", f"{n_bad:,}", "see problems below" if n_bad else "none", ROSE if n_bad else TEAL)], mini=True)
if n_bad:
    st.warning(f"{n_bad:,} row(s) have problems and were left out of scoring. The rest were scored.")
    with st.expander(f"Problems found ({len(issues):,} listed)", expanded=True):
        st.dataframe(issues, hide_index=True, width="stretch")
        st.download_button("Download problem list (CSV)", issues.to_csv(index=False).encode(), "validation_problems.csv", "text/csv")
if valid.empty:
    st.error("No valid rows to score.")
    st.stop()

# ---------- 4. results ----------
scored = score(bundle, valid)
scored.insert(0, "FileRow", scored.index + 2)
scored["RiskFlag"] = scored["ReviewFlag"].map({True: "Flag for review", False: "Not flagged"})
n_flag = int(scored["ReviewFlag"].sum())

section("4. Results", f"Scored with {bundle['model_name']}. Estimated default rate is calibrated; the flag uses the review threshold.")
kpi_row([("Flagged for review", f"{n_flag:,}", f"{n_flag / len(scored) * 100:.1f}% of scored rows", ROSE),
         ("Average estimated default rate", f"{scored['EstimatedDefaultRate'].mean() * 100:.1f}%", "across scored rows", AMBER),
         ("Highest estimate", f"{scored['EstimatedDefaultRate'].max() * 100:.1f}%", "single borrower", VIOLET)], mini=True)

hist = px.histogram(scored, x="EstimatedDefaultRate", nbins=30, color="RiskFlag",
                    color_discrete_map={"Flag for review": ROSE, "Not flagged": TEAL}, title="Distribution of estimated default rates")
hist.update_xaxes(tickformat=".0%")
st.plotly_chart(style_fig(hist, 340), width="stretch")

show = scored.drop(columns=["ReviewFlag"]).sort_values("EstimatedDefaultRate", ascending=False)
st.dataframe(show.head(1000), hide_index=True, width="stretch",
             column_config={"EstimatedDefaultRate": st.column_config.ProgressColumn("EstimatedDefaultRate", min_value=0, max_value=1, format="%.3f")})
if len(show) > 1000:
    st.caption("Showing the 1,000 highest-risk rows. The download contains all scored rows.")

buffer = io.StringIO()
show.sort_values("FileRow").to_csv(buffer, index=False)
st.download_button("Download results (CSV)", buffer.getvalue().encode(), "loan_risk_results.csv", "text/csv", type="primary")
insight("<b>Reading the columns:</b> <code>ModelScore</code> ranks borrowers, <code>EstimatedDefaultRate</code> is the calibrated estimate, "
        "<code>RiskFlag</code> marks scores at or above the review threshold. None of these is a decision to approve or reject.", TEAL)
disclaimer()
