"""
Synthetic data generator for the Cybercrime Predictive Intelligence prototype.

Generates a realistic (but fully synthetic) complaints dataset where every
withdrawal location is geographically tied to the city it belongs to, and
withdrawal likelihood depends on crime type, time-of-day and distance -
so the downstream ML model and the GIS heatmap have real signal to learn
from / visualise, instead of random noise.
"""

import csv
import random
from datetime import datetime, timedelta
from pathlib import Path

random.seed(42)

NUMBER_OF_RECORDS = 6000

OUTPUT_PATH = Path(__file__).resolve().parent / "cybercrime_data.csv"

CRIME_TYPES = [
    "UPI Fraud",
    "Phishing",
    "Online Shopping Fraud",
    "Investment Fraud",
    "Job Scam",
    "Loan Scam",
    "SIM Swap Fraud",
    "KYC Update Fraud",
]

# Relative likelihood that a given crime type ends in a physical cash
# withdrawal (some scams, e.g. investment fraud, are more likely to move
# money through mule accounts before a cash-out; others, e.g. UPI fraud,
# are withdrawn almost immediately).
CRIME_WITHDRAWAL_WEIGHT = {
    "UPI Fraud": 0.78,
    "Phishing": 0.62,
    "Online Shopping Fraud": 0.45,
    "Investment Fraud": 0.55,
    "Job Scam": 0.50,
    "Loan Scam": 0.40,
    "SIM Swap Fraud": 0.82,
    "KYC Update Fraud": 0.70,
}

# city -> (state, lat, lon)
CITIES = {
    "Amritsar": ("Punjab", 31.6340, 74.8723),
    "Ludhiana": ("Punjab", 30.9010, 75.8573),
    "Jalandhar": ("Punjab", 31.3260, 75.5762),
    "Chandigarh": ("Chandigarh", 30.7333, 76.7794),
    "Delhi": ("Delhi", 28.6139, 77.2090),
    "Noida": ("Uttar Pradesh", 28.5355, 77.3910),
    "Jaipur": ("Rajasthan", 26.9124, 75.7873),
    "Lucknow": ("Uttar Pradesh", 26.8467, 80.9462),
    "Mumbai": ("Maharashtra", 19.0760, 72.8777),
    "Bengaluru": ("Karnataka", 12.9716, 77.5946),
    "Hyderabad": ("Telangana", 17.3850, 78.4867),
    "Patna": ("Bihar", 25.5941, 85.1376),
}

LOCATION_TYPES = ["ATM", "BANK", "CSC"]  # CSC = Common Service Centre / cash agent

# High-risk hours: fraud money is typically moved and withdrawn quickly
# after the fraud call/message, often outside normal banking hours.
HIGH_RISK_HOURS = set(list(range(19, 24)) + list(range(0, 3)))

start_date = datetime(2026, 1, 1)


def build_withdrawal_points():
    """Create a fixed set of withdrawal locations per city, each with its
    own coordinates jittered a few km from the city centre so the map
    shows distinct, realistic points instead of one averaged blob."""
    points = []
    for city, (state, lat, lon) in CITIES.items():
        n_points = random.randint(4, 6)
        for i in range(n_points):
            loc_type = random.choice(LOCATION_TYPES)
            code = f"{loc_type}-{city[:3].upper()}-{i+1:02d}"
            # jitter roughly within ~8km
            d_lat = random.uniform(-0.07, 0.07)
            d_lon = random.uniform(-0.07, 0.07)
            points.append({
                "code": code,
                "city": city,
                "state": state,
                "latitude": round(lat + d_lat, 6),
                "longitude": round(lon + d_lon, 6),
            })
    return points


def weighted_hour():
    """Complaint hour, skewed towards evening/night when scams peak."""
    weights = []
    for h in range(24):
        if h in HIGH_RISK_HOURS:
            weights.append(3.0)
        elif 9 <= h <= 18:
            weights.append(1.4)
        else:
            weights.append(1.0)
    return random.choices(range(24), weights=weights)[0]


def generate():
    withdrawal_points = build_withdrawal_points()
    by_city = {}
    for p in withdrawal_points:
        by_city.setdefault(p["city"], []).append(p)

    rows = []
    for i in range(1, NUMBER_OF_RECORDS + 1):
        city = random.choice(list(CITIES.keys()))
        state, city_lat, city_lon = CITIES[city]

        crime_type = random.choice(CRIME_TYPES)

        complaint_datetime = start_date + timedelta(
            days=random.randint(0, 270),
            hours=weighted_hour(),
            minutes=random.randint(0, 59),
        )
        complaint_hour = complaint_datetime.hour

        withdrawal_delay = random.randint(1, 6)
        withdrawal_hour = (complaint_hour + withdrawal_delay) % 24

        amount = int(random.lognormvariate(11.2, 0.7))
        amount = max(2000, min(amount, 500000))

        withdrawal_point = random.choice(by_city[city])
        distance = round(random.uniform(0.3, 18.0), 2)

        # ---- probability model driving withdrawal_occurred ----
        base = CRIME_WITHDRAWAL_WEIGHT[crime_type]
        hour_boost = 0.15 if complaint_hour in HIGH_RISK_HOURS else 0.0
        distance_penalty = -0.15 if distance > 10 else (0.05 if distance < 3 else 0.0)
        amount_boost = 0.08 if amount > 100000 else 0.0
        weekend_boost = 0.05 if complaint_datetime.weekday() >= 5 else 0.0

        prob = base + hour_boost + distance_penalty + amount_boost + weekend_boost
        prob = max(0.03, min(0.97, prob))

        withdrawal_occurred = 1 if random.random() < prob else 0

        rows.append({
            "complaint_id": f"C{i:05d}",
            "crime_type": crime_type,
            "city": city,
            "state": state,
            "latitude": city_lat,
            "longitude": city_lon,
            "amount": amount,
            "complaint_hour": complaint_hour,
            "day_of_week": complaint_datetime.strftime("%A"),
            "complaint_date": complaint_datetime.strftime("%Y-%m-%d"),
            "withdrawal_location": withdrawal_point["code"],
            "withdrawal_city": withdrawal_point["city"],
            "withdrawal_lat": withdrawal_point["latitude"],
            "withdrawal_lon": withdrawal_point["longitude"],
            "withdrawal_hour": withdrawal_hour,
            "distance_km": distance,
            "withdrawal_occurred": withdrawal_occurred,
        })

    columns = list(rows[0].keys())
    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    print("SUCCESS!")
    print(f"Created {OUTPUT_PATH.name}")
    print(f"Total records: {len(rows)}")
    print(f"Withdrawal points generated: {len(withdrawal_points)}")


if __name__ == "__main__":
    generate()
