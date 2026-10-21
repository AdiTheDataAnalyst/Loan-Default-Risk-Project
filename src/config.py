"""Central settings: paths, column names and constants used across the project."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"

RANDOM_STATE = 42          # same seed = same results every run
TARGET = "Default"
ID_COLUMNS = ["LoanID"]    # identifiers carry no meaning, so they are dropped

NUMERIC_COLUMNS = [
    "Age", "Income", "LoanAmount", "CreditScore", "MonthsEmployed",
    "NumCreditLines", "InterestRate", "LoanTerm", "DTIRatio",
]
CATEGORICAL_COLUMNS = [
    "Education", "EmploymentType", "MaritalStatus", "HasMortgage",
    "HasDependents", "LoanPurpose", "HasCoSigner",
]
# Features we create ourselves (see preprocessing.add_features)
ENGINEERED_COLUMNS = ["LoanToIncome", "MonthlyPaymentToIncome"]

# A missed default costs a lender far more than a false alarm,
# so we pick the threshold using F-beta with beta=2 (recall counts 2x as much as precision).
F_BETA = 2
