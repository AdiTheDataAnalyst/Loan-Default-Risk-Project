"""Run the whole pipeline in the right order:   python -m src.run_all"""
from src import app_data, calibrate, explain, fairness, train

STEPS = [("1/5 Train and compare models", train), ("2/5 Calibrate scores", calibrate),
         ("3/5 SHAP explanations", explain), ("4/5 App summary files", app_data), ("5/5 Fairness check", fairness)]

if __name__ == "__main__":
    for title, module in STEPS:
        print(f"\n{'=' * 8} {title} {'=' * 8}")
        module.main()
    print("\nAll done. Start the app with:  streamlit run app.py")
