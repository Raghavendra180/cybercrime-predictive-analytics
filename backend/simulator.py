import asyncio
import random
from datetime import datetime, timezone
import httpx

MOCK_ATMS = [
    {"name": "ATM - Connaught Place Inner Circle", "lat": 28.6315, "lon": 77.2167},
    {"name": "ATM - Karol Bagh Market", "lat": 28.6514, "lon": 77.1907},
    {"name": "ATM - Sector 18 Commercial Hub Noida", "lat": 28.5708, "lon": 77.3261},
    {"name": "ATM - Cyber City Rapid Metro Gurgaon", "lat": 28.4900, "lon": 77.0880},
    {"name": "ATM - Laxmi Nagar Vikas Marg", "lat": 28.6304, "lon": 77.2774},
    {"name": "ATM - Saket District Centre", "lat": 28.5284, "lon": 77.2185},
    {"name": "ATM - Rohini Sector 8", "lat": 28.7041, "lon": 77.1025}
]

FRAUD_TYPES = ["UPI Phishing", "Card Cloning", "SIM Swap", "Investment Scam", "Impersonation"]
BANKS = ["SBI", "HDFC", "ICICI", "Axis Bank", "Punjab National Bank"]

async def run_live_feed(api_url: str = "http://127.0.0.1:8000/api/v1/complaints"):
    print("[*] Simulator started: Pushing cybercrime reports to backend...")
    async with httpx.AsyncClient() as client:
        while True:
            await asyncio.sleep(random.uniform(3.5, 7.0))
            selected_atm = random.choice(MOCK_ATMS)
            
            payload = {
                "complaint_id": f"NCR-{datetime.now().strftime('%H%M%S')}-{random.randint(100, 999)}",
                "victim_location": "Delhi-NCR Region",
                "incident_timestamp": datetime.now(timezone.utc).isoformat(),
                "fraud_type": random.choice(FRAUD_TYPES),
                "amount_lost": float(random.choice([12000, 25000, 48000, 95000, 140000, 260000])),
                "mule_account_bank": random.choice(BANKS),
                "suspected_atm_lat": selected_atm["lat"] + random.uniform(-0.003, 0.003),
                "suspected_atm_lon": selected_atm["lon"] + random.uniform(-0.003, 0.003),
                "confidence_score": round(random.uniform(0.72, 0.98), 2)
            }
            
            try:
                res = await client.post(api_url, json=payload, timeout=5)
                if res.status_code == 200:
                    print(f"[+] Ingested Complaint: {payload['complaint_id']} | ₹{payload['amount_lost']:,.0f} at {selected_atm['name']}")
            except Exception as e:
                print(f"[-] Backend connection error: {e}")

if __name__ == "__main__":
    asyncio.run(run_live_feed())