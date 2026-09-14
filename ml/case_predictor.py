"""
Case-level ranked location predictor.

Given a new complaint's characteristics (city, crime type, amount,
complaint hour), ranks the candidate withdrawal locations in that city by
likelihood of being the actual cash-out point, and returns a Top-K list
with confidence, risk level, expected time window, and a short
explanation for each — instead of a single undifferentiated probability.

This does NOT claim to predict one exact ATM with certainty. It produces
a probabilistic ranking, as required by the problem statement.
"""

from dataclasses import dataclass, field
from typing import List

import pandas as pd


@dataclass
class LocationPrediction:
    rank: int
    location: str
    city: str
    latitude: float
    longitude: float
    confidence: float          # 0-100
    risk_level: str            # HIGH / MEDIUM / LOW
    window_start: int          # hour, 0-23
    window_end: int            # hour, 0-23
    total_cases: int
    historical_withdrawal_rate: float
    average_amount: float
    peak_risk_hour: int
    explanation: List[str] = field(default_factory=list)

    @property
    def time_window(self) -> str:
        return f"{self.window_start:02d}:00\u2013{self.window_end:02d}:00"


def _circular_hour_distance(h1: int, h2: int) -> float:
    if h1 < 0 or h2 < 0:
        return 12.0
    d = abs(h1 - h2)
    return min(d, 24 - d)


def _risk_level(score: float) -> str:
    if score >= 70:
        return "HIGH"
    elif score >= 40:
        return "MEDIUM"
    return "LOW"


def generate_predictions(
    city: str,
    crime_type: str,
    amount: float,
    complaint_hour: int,
    risk_scores: pd.DataFrame,
    complaints: pd.DataFrame,
    top_k: int = 3,
) -> List[LocationPrediction]:
    candidates = risk_scores[risk_scores["withdrawal_city"] == city].copy()
    if candidates.empty:
        return []

    # Historical withdrawal rate for this specific crime type (falls back
    # to the overall rate if the crime type has too few samples).
    crime_rates = complaints.groupby("crime_type")["withdrawal_occurred"].mean()
    crime_rate = float(crime_rates.get(crime_type, complaints["withdrawal_occurred"].mean()))

    scored_rows = []
    for _, row in candidates.iterrows():
        risk_component = row["risk_score"]  # already 0-100

        avg_amount = max(row["average_amount"], 1.0)
        amount_diff_ratio = abs(amount - avg_amount) / avg_amount
        amount_similarity = max(0.0, 100 - amount_diff_ratio * 100)

        peak_hour = int(row["peak_risk_hour"])
        hour_dist = _circular_hour_distance(complaint_hour, peak_hour)
        time_similarity = max(0.0, 100 - (hour_dist / 12) * 100)

        crime_component = crime_rate * 100

        composite = (
            0.35 * risk_component
            + 0.20 * amount_similarity
            + 0.20 * time_similarity
            + 0.25 * crime_component
        )
        composite = max(0.0, min(100.0, composite))

        scored_rows.append({
            "row": row,
            "score": composite,
            "amount_similarity": amount_similarity,
            "time_similarity": time_similarity,
            "crime_rate": crime_rate,
            "peak_hour": peak_hour,
        })

    scored_rows.sort(key=lambda r: r["score"], reverse=True)
    top = scored_rows[:top_k]

    predictions = []
    for i, item in enumerate(top, start=1):
        row = item["row"]
        score = item["score"]
        peak_hour = item["peak_hour"] if item["peak_hour"] >= 0 else complaint_hour
        window_start = (peak_hour - 1) % 24
        window_end = (peak_hour + 2) % 24

        reasons = []
        reasons.append(
            f"Historical withdrawal rate at this location is "
            f"{row['historical_withdrawal_rate']:.0f}% across {int(row['total_cases'])} linked complaints."
        )
        if item["crime_rate"] > 0.5:
            reasons.append(
                f"'{crime_type}' cases historically end in a withdrawal "
                f"{item['crime_rate']*100:.0f}% of the time."
            )
        if item["amount_similarity"] >= 60:
            reasons.append(
                f"Reported amount (\u20b9{amount:,.0f}) is close to the typical amount "
                f"seen here (\u20b9{row['average_amount']:,.0f})."
            )
        if item["time_similarity"] >= 50 and peak_hour >= 0:
            reasons.append(
                f"Complaint hour is close to this location's historical peak "
                f"cash-out hour ({peak_hour:02d}:00)."
            )
        if not reasons:
            reasons.append("Limited historical similarity; ranked mainly on baseline location risk.")

        predictions.append(LocationPrediction(
            rank=i,
            location=row["withdrawal_location"],
            city=row["withdrawal_city"],
            latitude=float(row["latitude"]),
            longitude=float(row["longitude"]),
            confidence=round(score, 1),
            risk_level=_risk_level(score),
            window_start=window_start,
            window_end=window_end,
            total_cases=int(row["total_cases"]),
            historical_withdrawal_rate=float(row["historical_withdrawal_rate"]),
            average_amount=float(row["average_amount"]),
            peak_risk_hour=peak_hour,
            explanation=reasons[:3],
        ))

    return predictions
