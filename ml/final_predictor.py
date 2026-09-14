"""
Combines the trained ML model's probability with each location's
historical geographic risk score to rank the most likely withdrawal
hotspots for a new complaint.

Run from anywhere:  python ml/final_predictor.py
"""

import pandas as pd
import joblib
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = Path(__file__).resolve().parent / "cybercrime_model.pkl"
HOTSPOTS_PATH = BASE_DIR / "data" / "hotspot_risk_scores.csv"


def main():
    model = joblib.load(MODEL_PATH)
    hotspots = pd.read_csv(HOTSPOTS_PATH)

    print("==========================================")
    print("CYBERCRIME WITHDRAWAL HOTSPOT PREDICTOR")
    print("==========================================")

    amount = float(input("Amount involved (Rs.): "))
    complaint_hour = int(input("Complaint hour (0-23): "))
    withdrawal_hour = int(input("Expected withdrawal hour (0-23): "))
    distance_km = float(input("Distance to suspected ATM (km): "))

    new_case = pd.DataFrame([{
        "amount": amount,
        "complaint_hour": complaint_hour,
        "withdrawal_hour": withdrawal_hour,
        "distance_km": distance_km,
    }])

    ml_probability = model.predict_proba(new_case)[0][1] * 100

    hotspots = hotspots.copy()
    hotspots["final_risk"] = (
        hotspots["risk_score"] * 0.6 + ml_probability * 0.4
    )
    hotspots = hotspots.sort_values("final_risk", ascending=False)

    print("\n========== TOP PREDICTED HOTSPOTS ==========\n")
    for _, row in hotspots.head(5).iterrows():
        score = row["final_risk"]
        if score >= 70:
            level = "HIGH"
        elif score >= 40:
            level = "MEDIUM"
        else:
            level = "LOW"

        print(f'{row["withdrawal_location"]:14} Risk: {score:.2f}/100   Level: {level}')

    top = hotspots.iloc[0]
    print("\n==========================================")
    print("MOST LIKELY WITHDRAWAL LOCATION")
    print("==========================================")
    print(f'Location : {top["withdrawal_location"]}')
    print(f'Latitude : {top["latitude"]:.4f}')
    print(f'Longitude: {top["longitude"]:.4f}')
    print(f'Risk     : {top["final_risk"]:.2f}/100')
    print("==========================================")


if __name__ == "__main__":
    main()
