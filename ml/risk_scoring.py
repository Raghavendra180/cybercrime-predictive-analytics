"""
Geographic risk scoring for withdrawal hotspots.

Aggregates the complaints dataset by withdrawal location and produces a
0-100 risk score per location, used by the dashboard's GIS heatmap and
alerting logic. Also derives the highest-risk time window per location.

Run from anywhere:  python ml/risk_scoring.py
"""

import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_IN = BASE_DIR / "data" / "cybercrime_data.csv"
DATA_OUT = BASE_DIR / "data" / "hotspot_risk_scores.csv"


def get_risk_level(score):
    if score >= 70:
        return "HIGH"
    elif score >= 40:
        return "MEDIUM"
    else:
        return "LOW"


def compute_risk_scores(data: pd.DataFrame) -> pd.DataFrame:
    locations = data.groupby("withdrawal_location").agg(
        withdrawal_city=("withdrawal_city", "first"),
        total_cases=("complaint_id", "count"),
        withdrawals=("withdrawal_occurred", "sum"),
        average_amount=("amount", "mean"),
        average_distance=("distance_km", "mean"),
        latitude=("withdrawal_lat", "mean"),
        longitude=("withdrawal_lon", "mean"),
    ).reset_index()

    # Historical withdrawal rate
    locations["withdrawal_rate"] = (
        locations["withdrawals"] / locations["total_cases"]
    ) * 100

    # Normalize amount (higher average amount -> higher risk)
    max_amount = locations["average_amount"].max()
    locations["amount_score"] = (locations["average_amount"] / max_amount) * 100

    # Normalize distance (closer to the complaint origin -> higher risk,
    # since fraudsters/mules typically cash out near the victim's area)
    max_distance = locations["average_distance"].max()
    locations["distance_score"] = (
        1 - (locations["average_distance"] / max_distance)
    ) * 100

    # Volume score: locations seeing more total complaints carry more
    # weight, since they represent a bigger sample / bigger threat surface
    max_cases = locations["total_cases"].max()
    locations["volume_score"] = (locations["total_cases"] / max_cases) * 100

    # Final weighted risk score
    locations["risk_score"] = (
        locations["withdrawal_rate"] * 0.45
        + locations["amount_score"] * 0.20
        + locations["distance_score"] * 0.20
        + locations["volume_score"] * 0.15
    )

    locations["risk_level"] = locations["risk_score"].apply(get_risk_level)
    locations["historical_withdrawal_rate"] = locations["withdrawal_rate"]

    # Peak risk hour per location (most common withdrawal hour among
    # cases that actually resulted in a withdrawal)
    withdrawn = data[data["withdrawal_occurred"] == 1]
    peak_hour = (
        withdrawn.groupby("withdrawal_location")["withdrawal_hour"]
        .agg(lambda s: s.mode().iloc[0] if not s.mode().empty else -1)
        .rename("peak_risk_hour")
    )
    locations = locations.merge(peak_hour, on="withdrawal_location", how="left")
    locations["peak_risk_hour"] = locations["peak_risk_hour"].fillna(-1).astype(int)

    locations = locations.sort_values("risk_score", ascending=False)

    dashboard_cols = [
        "withdrawal_location",
        "withdrawal_city",
        "latitude",
        "longitude",
        "risk_score",
        "risk_level",
        "total_cases",
        "withdrawals",
        "average_amount",
        "average_distance",
        "historical_withdrawal_rate",
        "peak_risk_hour",
    ]
    return locations[dashboard_cols]


def main():
    data = pd.read_csv(DATA_IN)
    dashboard_data = compute_risk_scores(data)
    dashboard_data.to_csv(DATA_OUT, index=False)

    print("==========================================")
    print("GEOGRAPHIC RISK ANALYSIS")
    print("==========================================")
    for _, row in dashboard_data.iterrows():
        print(
            f'{row["withdrawal_location"]:14} '
            f'{row["withdrawal_city"]:12} '
            f'Risk: {row["risk_score"]:.2f}/100  '
            f'{row["risk_level"]:6} '
            f'Peak hr: {row["peak_risk_hour"]:02d}:00'
        )
    print("\n==========================================")
    print(f"Geographic risk data saved: {DATA_OUT}")
    print("==========================================")


if __name__ == "__main__":
    main()
