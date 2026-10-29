# Loan Risk Intelligence Platform

An interactive Streamlit app for exploring loan default risk with machine learning and individual risk predictions.

## Overview

This educational project explores a loan-default dataset and compares several machine-learning approaches. The app includes an executive dashboard, exploratory analysis, model evaluation, and individual risk predictions.

The included dataset contains **255,347 rows** and **18 columns**, with an observed default rate of **11.61%**. The dataset does not specify a currency for monetary values.

## Features

- Executive dashboard with three summary KPIs: loans analysed, observed default rate, and selected-model PR-AUC.
- Exploratory analysis of borrower characteristics and observed default rates.
- Comparison of a dummy baseline, Logistic Regression, Random Forest, and XGBoost.
- Individual risk estimates with explanations.
- Reproducible training and evaluation pipeline.

## Run locally

Requires Python and pip. From the project directory, create and activate a virtual environment, then install the dependencies.

**Windows PowerShell**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app.py
```

**macOS / Linux**

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run app.py
```

The repository includes trained model and report artifacts required by the app. To retrain the models and regenerate the artifacts, make sure the source CSV is present in `data/raw/`, then run:

```bash
python -m src.run_all
```

Run the test suite with:

```bash
python -m pytest -q
```

## Notebooks

The `notebooks/` folder contains three step-by-step Jupyter notebooks:

1. `01_Data_Cleaning.ipynb` — inspect the raw data, handle duplicate and missing records, and save a cleaned dataset to `data/processed/`.
2. `02_Visualization.ipynb` — explore loan and default patterns using the cleaned dataset.
3. `03_ML_Model.ipynb` — train and compare Logistic Regression, Random Forest, and XGBoost models on a stratified sample.

Run them in order from the project root. Install dependencies with `python -m pip install -r requirements.txt`, then launch Jupyter with `jupyter notebook`.

## Data

The dataset is the [Loan Default dataset by nikhil1e9 on Kaggle](https://www.kaggle.com/datasets/nikhil1e9/loan-default). The raw CSV is included in `data/raw/`, and a smaller sample is provided in `data/sample/`.

Check the dataset's current license and terms before reusing or redistributing it. The project drops the unique `LoanID` identifier from model features and uses `Default` as the target.

## Project structure

```text
Loan-Default-Risk-Project/
├── notebooks/           Data cleaning, visualization, and model training notebooks
├── data/                Dataset
├── app.py               Streamlit app entry point
├── models/              Trained model files
├── requirements.txt     Python dependencies
└── README.md            Project documentation
```

## Method and limitations

Preprocessing is fitted inside each model pipeline using training data only. The data is split into stratified training, validation, and test sets; model selection uses validation PR-AUC, while the held-out test set is reserved for final evaluation. The decision threshold is selected using an F2 score, and predicted scores are calibrated using validation data.

This dataset appears synthetic, and its results may not generalize to real borrowers. The models identify associations, not causes. Some features are sensitive, and the included fairness analysis is not a substitute for legal or domain review. **This is an educational prototype, not a lending decision system.** Do not use its estimates to approve or reject real loans.

## Built with

Python, Streamlit, pandas, scikit-learn, XGBoost, SHAP, Plotly, and pytest.
