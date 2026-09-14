import asyncio
import csv
import hashlib
import json
import math
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sklearn.cluster import DBSCAN

app = FastAPI(title="Cybercrime Predictive Analytics Engine - LEA Real-Time Hub")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Path pointing to your master historical dataset
CSV_PATH = Path(__file__).resolve().parent.parent / "data" / "cybercrime_data.csv"

complaints_db: List[dict] = []
interventions_db: List[dict] = []
active_connections: List[WebSocket] = []

class CyberComplaint(BaseModel):
    complaint_id: str
    victim_location: str
    incident_timestamp: str
    fraud_type: str
    amount_lost: float
    mule_account_bank: str
    suspected_atm_lat: float
    suspected_atm_lon: float
    confidence_score: Optional[float] = 0.0

class InterventionOrder(BaseModel):
    complaint_id: str
    target_bank: str
    action_type: str  # "FREEZE_ACCOUNT_API" or "DISPATCH_PATROL_VAN"
    field_unit_id: Optional[str] = "PCR-NCR-UNIT-01"

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_text(json.dumps(message))
            except Exception:
                pass

manager = ConnectionManager()

def compute_intervention_window(amount: float, fraud_type: str) -> dict:
    """Predictive cash-out velocity based on transaction size & scam vector."""
    base_window_minutes = 45.0
    if amount > 100000:
        base_window_minutes = 20.0
    elif fraud_type in ["UPI Phishing", "SIM Swap"]:
        base_window_minutes = 15.0

    return {
        "intervention_deadline_mins": base_window_minutes,
        "urgency": "CRITICAL" if base_window_minutes <= 20 else "HIGH",
    }

def calculate_hotspot_clusters(complaints: list, eps_km: float = 3.0, min_samples: int = 2) -> list:
    """Dynamic DBSCAN clustering incorporating exponential temporal risk decay."""
    if len(complaints) < min_samples:
        return []

    now = datetime.now(timezone.utc)
    active_complaints = []
    decay_lambda = math.log(2) / 30.0  # 30-minute half life

    for c in complaints:
        try:
            ingest_dt = datetime.fromisoformat(c["ingested_at"])
            elapsed_mins = max(0.0, (now - ingest_dt).total_seconds() / 60.0)
        except Exception:
            elapsed_mins = 2.0

        decay_factor = math.exp(-decay_lambda * elapsed_mins)
        if decay_factor > 0.10:  # Active within the critical 90-minute window
            c_copy = dict(c)
            c_copy["effective_risk_weight"] = c["amount_lost"] * decay_factor
            c_copy["decay_factor"] = round(decay_factor, 3)
            active_complaints.append(c_copy)

    if len(active_complaints) < min_samples:
        return []

    coords = np.array([[c["suspected_atm_lat"], c["suspected_atm_lon"]] for c in active_complaints])
    kms_per_radian = 6371.0088
    epsilon = eps_km / kms_per_radian

    db = DBSCAN(eps=epsilon, min_samples=min_samples, metric="haversine")
    labels = db.fit_predict(np.radians(coords))

    clusters = []
    for cluster_id in set(labels):
        if cluster_id == -1:
            continue
        indices = np.where(labels == cluster_id)[0]
        pts = coords[indices]
        effective_funds = sum(active_complaints[i]["effective_risk_weight"] for i in indices)
        raw_funds = sum(active_complaints[i]["amount_lost"] for i in indices)

        clusters.append({
            "hotspot_id": f"NCR-CLUSTER-{cluster_id + 1}",
            "latitude": float(np.mean(pts[:, 0])),
            "longitude": float(np.mean(pts[:, 1])),
            "active_complaint_count": len(indices),
            "total_funds_at_risk": raw_funds,
            "effective_decayed_risk": round(effective_funds, 2),
            "threat_level": "RED" if effective_funds >= 80000 else "AMBER",
        })
    return clusters

@app.websocket("/ws/stream")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)

@app.post("/api/v1/complaints")
async def create_complaint(complaint: CyberComplaint):
    risk_profile = compute_intervention_window(complaint.amount_lost, complaint.fraud_type)
    complaint_data = complaint.model_dump()
    complaint_data.update(risk_profile)
    complaint_data["ingested_at"] = datetime.now(timezone.utc).isoformat()
    
    # --- 🔗 BLOCKCHAIN LAYER ADDITION ---
    raw_data = f"{complaint.complaint_id}{complaint.amount_lost}{complaint_data['ingested_at']}"
    tx_hash = hashlib.sha256(raw_data.encode('utf-8')).hexdigest()
    complaint_data["blockchain_tx_hash"] = tx_hash
    # ------------------------------------
    
    # 1. Add to live memory queue
    complaints_db.append(complaint_data)

    # 2. PERMANENT DATABASE SAVE (Appending to your master CSV)
    try:
        with open(CSV_PATH, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                complaint_data["complaint_id"],
                complaint_data["fraud_type"],
                complaint_data["amount_lost"],
                "Delhi",
                "Delhi NCR",
                datetime.now().strftime("%Y-%m-%d"),
                datetime.now().hour,
                (datetime.now().hour + 1) % 24,
                round(random.uniform(1.5, 8.5), 2),
                f"ATM - {complaint_data['mule_account_bank']}",
                "Delhi",
                0
            ])
    except Exception as e:
        print(f"Failed to save to CSV: {e}")

    await manager.broadcast({
        "event_type": "NEW_COMPLAINT",
        "data": complaint_data
    })
    return {"status": "SUCCESS", "complaint": complaint_data}

@app.get("/api/v1/complaints/recent")
async def get_recent_complaints(limit: int = 50):
    return complaints_db[-limit:][::-1]

@app.get("/api/v1/hotspots/active")
async def get_active_hotspots():
    recent = complaints_db[-100:]
    return calculate_hotspot_clusters(recent)

@app.post("/api/v1/intervene")
async def trigger_intervention(order: InterventionOrder):
    record = order.model_dump()
    record["dispatched_at"] = datetime.now(timezone.utc).isoformat()
    record["status"] = "SUCCESS_TRANSMITTED"
    interventions_db.append(record)

    await manager.broadcast({
        "event_type": "INTERVENTION_TRIGGERED",
        "data": record
    })
    return {"status": "SUCCESS", "record": record}

@app.get("/api/v1/interventions")
async def get_interventions():
    return interventions_db[-20:][::-1]

# 🌟 NEW STATS ENDPOINT FOR DASHBOARD SCALING 🌟
@app.get("/api/v1/stats")
async def get_system_stats():
    return {"total_live_ingested": len(complaints_db)}