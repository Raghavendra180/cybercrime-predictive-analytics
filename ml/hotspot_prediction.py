"""
Quick command-line hotspot report, computed directly from raw complaints
(a lighter-weight alternative to risk_scoring.py's full weighted score).

Run from anywhere:  python ml/hotspot_prediction.py
"""

import pandas as pd
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "cybercrime_data.csv"


def main():
    data = pd.read_csv(DATA_PATH)

    print("========================================")
    print("PREDICTED WITHDRAWAL HOTSPOTS")
    print("========================================")

    hotspots = data.groupby("withdrawal_location").agg(
        total_cases=("complaint_id", "count"),
        actual_withdrawals=("withdrawal_occurred", "sum"),
        average_amount=("amount", "mean"),
    ).reset_index()

    hotspots["risk_score"] = (
        hotspots["actual_withdrawals"] / hotspots["total_cases"]
    ) * 100

    hotspots = hotspots.sort_values(by="risk_score", ascending=False)

    for _, row in hotspots.iterrows():
        risk = row["risk_score"]
        if risk >= 70:
            risk_level = "HIGH"
        elif risk >= 40:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"

        print(f'{row["withdrawal_location"]:14} {risk_level:8} {risk:.2f}%')

    print("\n========================================")
    print("HOTSPOT ANALYSIS COMPLETE")
    print("========================================")


if __name__ == "__main__":
    main()
