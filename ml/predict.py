"""
Simple interactive CLI to score a single case with the trained model.

Run from anywhere:  python ml/predict.py
"""

import pandas as pd
import joblib
from pathlib import Path

MODEL_PATH = Path(__file__).resolve().parent / "cybercrime_model.pkl"


def main():
    model = joblib.load(MODEL_PATH)

    print("================================")
    print("CYBERCRIME RISK PREDICTOR")
    print("================================")

    amount = float(input("Enter amount involved (Rs.): "))
    complaint_hour = int(input("Enter complaint hour (0-23): "))
    withdrawal_hour = int(input("Enter expected withdrawal hour (0-23): "))
    distance_km = float(input("Enter distance from complaint to ATM (km): "))

    new_case = pd.DataFrame([{
        "amount": amount,
        "complaint_hour": complaint_hour,
        "withdrawal_hour": withdrawal_hour,
        "distance_km": distance_km,
    }])

    probability = model.predict_proba(new_case)[0][1]
    risk_percentage = probability * 100

    if risk_percentage >= 70:
        risk_level = "HIGH"
    elif risk_percentage >= 40:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"

    print("\n========== RESULT ==========")
    print(f"Predicted Risk: {risk_percentage:.2f}%")
    print(f"Risk Level: {risk_level}")
    print("============================")


if __name__ == "__main__":
    main()
