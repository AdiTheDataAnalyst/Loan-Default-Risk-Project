"""Entry point. Run with:  streamlit run app.py

It sets the page style once, builds the sidebar (logo, then navigation), and runs the selected page.
"""
import streamlit as st

st.set_page_config(page_title="Loan Risk Intelligence", page_icon="🏦", layout="wide",
                   initial_sidebar_state="expanded")

from src.theme import apply_theme, sidebar_brand, sidebar_note  # noqa: E402  (after set_page_config on purpose)

apply_theme()

pages = [
    st.Page("pages/1_Executive_Dashboard.py", title="Executive Dashboard", icon=":material/dashboard:", default=True),
    st.Page("pages/2_Exploratory_Analysis.py", title="Exploratory Analysis", icon=":material/insights:"),
    st.Page("pages/3_Model_Performance.py", title="Model Performance", icon=":material/query_stats:"),
    st.Page("pages/4_Loan_Risk_Prediction.py", title="Loan Risk Prediction", icon=":material/person_search:"),
    st.Page("pages/5_Batch_Predictions.py", title="Batch Predictions", icon=":material/upload_file:"),
    st.Page("pages/6_Explainable_AI.py", title="Explainable AI", icon=":material/psychology:"),
    st.Page("pages/7_About_the_Project.py", title="About the Project", icon=":material/info:"),
]
navigation = st.navigation(pages, position="hidden")   # we draw our own menu so the logo can sit on top

sidebar_brand()
with st.sidebar:
    for page in pages:
        st.page_link(page)
sidebar_note()

navigation.run()
