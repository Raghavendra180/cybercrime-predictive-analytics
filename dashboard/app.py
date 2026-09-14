"""
Cybercrime Predictive Intelligence Platform - Real-Time LEA Command HUD
All deliverables (a-d) preserved with clean, high-contrast dark visual aesthetics.
Fixes CartoDB tile requirements, use_container_width, and pickle version warnings.
"""

import os
import sys
import warnings
from datetime import datetime
from pathlib import Path

# Suppress unpickler version mismatches gracefully
try:
    from sklearn.exceptions import InconsistentVersionWarning
    warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
except ImportError:
    pass

import folium
import joblib
import pandas as pd
import requests
import streamlit as st
from folium.plugins import HeatMap
from streamlit_folium import st_folium

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
ML_DIR = BASE_DIR / "ml"
BACKEND_URL = "http://127.0.0.1:8000"

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from ml.case_predictor import generate_predictions  # noqa: E402
from ml import validation_engine as ve  # noqa: E402

st.set_page_config(
    page_title="Cybercrime Intelligence System | LEA Command",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------
# Clean & Modern LEA UI Styling (Dark Theme Scaffolding)
# --------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, sans-serif;
    }
    
    .block-container { 
        padding-top: 1.2rem;
        padding-bottom: 2rem;
    }
    
    /* Sleek KPI Metric Cards */
    div[data-testid="stMetric"] {
        background: #11141c;
        border: 1px solid #1f2633;
        border-radius: 8px;
        padding: 14px 18px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25);
    }
    div[data-testid="stMetric"] label {
        font-size: 0.82rem !important;
        font-weight: 500 !important;
        color: #8b9bb4 !important;
        letter-spacing: 0.03em;
        text-transform: uppercase;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.6rem !important;
        font-weight: 700 !important;
        color: #f1f5f9 !important;
    }

    /* Status Badges */
    .risk-badge {
        display: inline-flex;
        align-items: center;
        padding: 2px 10px;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 600;
        font-family: 'JetBrains Mono', monospace;
        letter-spacing: 0.04em;
        text-transform: uppercase;
    }
    .risk-high { background: rgba(239, 68, 68, 0.15); color: #ef4444; border: 1px solid rgba(239, 68, 68, 0.4); }
    .risk-medium { background: rgba(245, 158, 11, 0.15); color: #f59e0b; border: 1px solid rgba(245, 158, 11, 0.4); }
    .risk-low { background: rgba(16, 185, 129, 0.15); color: #10b981; border: 1px solid rgba(16, 185, 129, 0.4); }

    /* Tactical HUD Incident Cards */
    .intel-card {
        background: #11141c;
        border: 1px solid #1e2430;
        border-left: 3px solid #3b82f6;
        border-radius: 6px;
        padding: 12px 14px;
        margin-bottom: 10px;
        transition: border 0.15s ease-in-out;
    }
    .intel-card.critical {
        border-left: 3px solid #ef4444;
        background: linear-gradient(90deg, rgba(239,68,68,0.06) 0%, #11141c 100%);
    }
    .intel-card:hover {
        border-color: #3b82f6;
    }
    .intel-mono {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.85rem;
        color: #cbd5e1;
    }
    .intel-title {
        font-size: 0.95rem;
        font-weight: 600;
        color: #ffffff;
        margin-bottom: 4px;
    }
    
    /* Clean Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        border-bottom: 1px solid #1f2633;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        border-radius: 6px 6px 0 0;
        padding: 8px 16px;
        color: #8b9bb4;
        font-size: 0.9rem;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1e2433 !important;
        color: #60a5fa !important;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------
# Cached File Ingestion
# --------------------------------------------------------------------------
@st.cache_data
def load_complaints():
    return pd.read_csv(DATA_DIR / "cybercrime_data.csv")

@st.cache_data
def load_risk_scores():
    return pd.read_csv(DATA_DIR / "hotspot_risk_scores.csv")

@st.cache_resource
def load_model():
    model_path = ML_DIR / "cybercrime_model.pkl"
    if model_path.exists():
        return joblib.load(model_path)
    return None

def risk_badge(level: str) -> str:
    cls = {"HIGH": "risk-high", "MEDIUM": "risk-medium", "LOW": "risk-low"}.get(level, "risk-low")
    return f'<span class="risk-badge {cls}">{level}</span>'

try:
    complaints = load_complaints()
    risk_scores = load_risk_scores()
except FileNotFoundError as e:
    st.error(
        "Required data files are missing. Run `python data/generate_data.py` "
        "and `python ml/risk_scoring.py` first, then reload.\n\n"
        f"Details: {e}"
    )
    st.stop()

model = load_model()

# --------------------------------------------------------------------------
# Sidebar: Direct Operational Command Form
# --------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🛡️ I4C / LEA Command Unit")
    st.caption("PS 26184 | Team Sonic Coders")
    st.divider()

    st.markdown("#### ⚡ Dispatch Live Complaint")
    st.caption("Simulates real-time 1930 portal ingestion to immediately update threat predictions.")

    with st.form("manual_entry_form", clear_on_submit=True):
        c_id = st.text_input("Complaint Reference", value=f"CYB-{datetime.now().strftime('%H%M%S')}")
        f_type = st.selectbox("Crime Type", ["UPI Phishing", "Card Cloning", "SIM Swap", "Investment Scam", "Impersonation"])
        amt = st.number_input("Amount Defrauded (₹)", min_value=1000, value=75000, step=5000)
        bank = st.selectbox("Mule Account Bank", ["SBI", "HDFC", "ICICI", "Axis Bank", "Punjab National Bank"])
        lat = st.number_input("Suspected Lat", value=28.6315, format="%.4f")
        lon = st.number_input("Suspected Lon", value=77.2167, format="%.4f")

        submitted = st.form_submit_button("🚨 Dispatch Alert", width="stretch")
        if submitted:
            payload = {
                "complaint_id": c_id,
                "victim_location": "Delhi Field Desk",
                "incident_timestamp": datetime.now().isoformat(),
                "fraud_type": f_type,
                "amount_lost": float(amt),
                "mule_account_bank": bank,
                "suspected_atm_lat": lat,
                "suspected_atm_lon": lon,
                "confidence_score": 0.94,
            }
            try:
                res = requests.post(f"{BACKEND_URL}/api/v1/complaints", json=payload, timeout=2)
                if res.status_code == 200:
                    st.toast(f"Complaint {c_id} broadcasted!", icon="⚡")
                else:
                    st.error("Failed to transmit incident.")
            except Exception as e:
                st.error(f"Backend offline: {e}")

    st.divider()
    st.markdown("#### Operational Services")
    st.markdown("🟢 **Predictive Engine:** Online")
    st.markdown("🟢 **Spatial Clustering:** Online")
    st.markdown("🟢 **Live Ingestion Feed:** Active")
    st.markdown(f"🟢 **ML Random Forest:** {'Ready' if model is not None else 'Missing'}")

    if st.button("🔄 Clear System Cache", width="stretch"):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()

# --------------------------------------------------------------------------
# Top Header Banner
# --------------------------------------------------------------------------
st.markdown("## 🛡️ Cybercrime Predictive Intelligence Platform")
st.caption("Predictive Analytics Framework for Cybercrime Complaints to Forecast Likely Cash Withdrawal Locations in Advance")
st.markdown("---")

# --------------------------------------------------------------------------
# Top Persistent Telemetry Row (Safe Auto-Refreshing)
# --------------------------------------------------------------------------
@st.fragment(run_every="4s")
def render_top_kpis():
    try:
        r_stats = requests.get(f"{BACKEND_URL}/api/v1/stats", timeout=2)
        live_ingested_count = r_stats.json().get("total_live_ingested", 0)
    except Exception:
        live_ingested_count = 0
        
    # Dynamically adds live backend cases to base CSV cases
    dynamic_total_cases = len(complaints) + live_ingested_count
        
    total_locations = risk_scores["withdrawal_location"].nunique()
    high_risk_locations = int((risk_scores["risk_level"] == "HIGH").sum())
    total_withdrawals = int(complaints["withdrawal_occurred"].sum())
    withdrawal_rate = (total_withdrawals / len(complaints)) * 100 if len(complaints) else 0

    top_col1, top_col2, top_col3, top_col4, top_col5 = st.columns(5)
    
    top_col1.metric("Master Database Cases", f"{dynamic_total_cases:,}")
    top_col2.metric("Hotspots Monitored", total_locations)
    top_col3.metric("High-Risk Hotspots", high_risk_locations)
    top_col4.metric("Recorded Cash-Outs", f"{total_withdrawals:,}")
    top_col5.metric("Avg Cash-Out Rate", f"{withdrawal_rate:.1f}%")

render_top_kpis()
st.write("")

# --------------------------------------------------------------------------
# Tab Navigation
# --------------------------------------------------------------------------
(
    tab_live,
    tab_heatmap,
    tab_predict,
    tab_validate,
    tab_lea,
    tab_alerts,
    tab_analytics,
) = st.tabs(
    [
        "🔴 Live LEA Command Room",
        "🗺️ GIS Risk Heatmap",
        "🤖 Predictive Engine",
        "🔁 Closed-Loop Validation",
        "🧑‍💼 Law Enforcement View",
        "🚨 Alerts & Notifications",
        "📊 Intelligence Summary",
    ]
)

# ==========================================================================
# TAB 1: LIVE REAL-TIME COMMAND ROOM
# ==========================================================================
with tab_live:
    @st.fragment(run_every="4s")
    def render_realtime_hud():
        # 1. Ask backend for the TRUE total of live cases processed
        try:
            r_stats = requests.get(f"{BACKEND_URL}/api/v1/stats", timeout=2)
            true_live_count = r_stats.json().get("total_live_ingested", 0)
        except Exception:
            true_live_count = 0

        # 2. Get the last 50 cases just for drawing the map smoothly
        try:
            r_c = requests.get(f"{BACKEND_URL}/api/v1/complaints/recent?limit=50", timeout=2)
            live_complaints = r_c.json() if r_c.status_code == 200 else []
        except Exception:
            live_complaints = []

        try:
            r_h = requests.get(f"{BACKEND_URL}/api/v1/hotspots/active", timeout=2)
            live_hotspots = r_h.json() if r_h.status_code == 200 else []
        except Exception:
            live_hotspots = []

        kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
        total_live_funds = sum(c.get("amount_lost", 0) for c in live_complaints)
        crit_cases = sum(1 for c in live_complaints if c.get("urgency") == "CRITICAL")

        # Now this shows the TRUE count (51, 52...) instead of stopping at 50
        kpi_col1.metric("Streaming Complaints", true_live_count)
        kpi_col2.metric("Live Funds at Risk", f"₹{total_live_funds:,.0f}")
        kpi_col3.metric("Golden-Hour Interventions", crit_cases)
        kpi_col4.metric("Active Spatial Clusters", len(live_hotspots))

        st.write("")

        map_section, ticker_section = st.columns([2, 1])

        with map_section:
            st.markdown("##### 📍 Live Spatial Clustering & Mule Cash-Out Forecasting (Delhi-NCR)")
            live_map = folium.Map(location=[28.6250, 77.2150], zoom_start=11, tiles="OpenStreetMap")

            for hs in live_hotspots:
                folium.Circle(
                    location=[hs["latitude"], hs["longitude"]],
                    radius=1500,
                    color="#ef4444" if hs["threat_level"] == "RED" else "#f59e0b",
                    fill=True,
                    fill_opacity=0.35,
                    popup=folium.Popup(
                        f"<b>{hs['hotspot_id']}</b><br>"
                        f"Linked Complaints: {hs['active_complaint_count']}<br>"
                        f"Nominal Funds: ₹{hs['total_funds_at_risk']:,.0f}<br>"
                        f"Decayed Threat: ₹{hs['effective_decayed_risk']:,.0f}",
                        max_width=240,
                    ),
                ).add_to(live_map)

            for c in live_complaints:
                is_crit = c.get("urgency") == "CRITICAL"
                folium.CircleMarker(
                    location=[c["suspected_atm_lat"], c["suspected_atm_lon"]],
                    radius=6,
                    color="#ef4444" if is_crit else "#3b82f6",
                    fill=True,
                    fill_color="#ef4444" if is_crit else "#3b82f6",
                    fill_opacity=0.9,
                    tooltip=f"{c['complaint_id']} | {c['fraud_type']} | ₹{c['amount_lost']:,.0f}",
                ).add_to(live_map)

            st_folium(live_map, width=None, height=520, key="live_map_widget", returned_objects=[])

        with ticker_section:
            st.markdown("##### ⚡ Live Intervention Queue")
            if not live_complaints:
                st.info("Awaiting live simulated feed or manual broadcast...")
            else:
                for idx, c in enumerate(live_complaints[:4]):
                    is_crit = c.get("urgency") == "CRITICAL"
                    crit_class = "critical" if is_crit else ""
                    badge_style = "risk-high" if is_crit else "risk-medium"
                    
                    # Ensure Blockchain hash displays safely
                    tx_hash = c.get("blockchain_tx_hash", "0x000...WAITING")[:15]

                    st.markdown(
                        f"""
                        <div class="intel-card {crit_class}">
                            <div style="display:flex; justify-content:space-between; align-items:center;">
                                <span class="intel-title">{c['complaint_id']}</span>
                                <span class="risk-badge {badge_style}">{c.get('urgency', 'HIGH')}</span>
                            </div>
                            <div class="intel-mono" style="margin-top:4px; margin-bottom:4px; color:#94a3b8;">
                                🔗 <b>Tx Hash:</b> <code>{tx_hash}...</code>
                            </div>
                            <div class="intel-mono">
                                ₹{c['amount_lost']:,.0f} • {c['fraud_type']}<br>
                                Target: {c.get('mule_account_bank', 'Unknown')}<br>
                                Window: <b>{c.get('intervention_deadline_mins', 30)} mins</b> remaining
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    btn_c1, btn_c2 = st.columns(2)
                    with btn_c1:
                        if st.button("🚨 Dispatch Van", key=f"btn_pcr_{c['complaint_id']}_{idx}", width="stretch"):
                            requests.post(
                                f"{BACKEND_URL}/api/v1/intervene",
                                json={
                                    "complaint_id": c["complaint_id"],
                                    "target_bank": c.get("mule_account_bank", "Unknown"),
                                    "action_type": "DISPATCH_PATROL_VAN",
                                    "field_unit_id": "PCR-DELHI-04",
                                },
                            )
                            st.toast(f"PCR Van dispatched for {c['complaint_id']}!", icon="🚔")
                    with btn_c2:
                        if st.button("🔒 Freeze API", key=f"btn_frz_{c['complaint_id']}_{idx}", width="stretch"):
                            requests.post(
                                f"{BACKEND_URL}/api/v1/intervene",
                                json={
                                    "complaint_id": c["complaint_id"],
                                    "target_bank": c.get("mule_account_bank", "Unknown"),
                                    "action_type": "FREEZE_ACCOUNT_API",
                                    "field_unit_id": "NPCI-GATEWAY-1930",
                                },
                            )
                            st.toast(f"Fund freeze request sent to {c.get('mule_account_bank')}!", icon="🛡️")

        if live_complaints:
            st.markdown("##### 📋 Telemetry Record Ledger")
            df_feed = pd.DataFrame(live_complaints)
            cols = ["complaint_id", "blockchain_tx_hash", "fraud_type", "amount_lost", "mule_account_bank", "urgency", "intervention_deadline_mins", "ingested_at"]
            st.dataframe(df_feed[[c for c in cols if c in df_feed.columns]], width="stretch", hide_index=True)

    render_realtime_hud()

# ==========================================================================
# TAB 2: GIS RISK HEATMAP
# ==========================================================================
with tab_heatmap:
    st.subheader("Historical GIS Risk Heatmap")
    st.caption("Drill-down filters by location, city, and static baseline risk tier.")

    f1, f2, f3 = st.columns(3)
    with f1:
        city_filter = st.selectbox(
            "Filter by city", ["All"] + sorted(risk_scores["withdrawal_city"].unique().tolist()),
            key="heatmap_city",
        )
    with f2:
        risk_filter = st.selectbox("Filter by risk level", ["All", "HIGH", "MEDIUM", "LOW"], key="heatmap_risk")
    with f3:
        show_heat_layer = st.checkbox("Show density heat layer", value=True)

    filtered_risk = risk_scores.copy()
    if city_filter != "All":
        filtered_risk = filtered_risk[filtered_risk["withdrawal_city"] == city_filter]
    if risk_filter != "All":
        filtered_risk = filtered_risk[filtered_risk["risk_level"] == risk_filter]

    if filtered_risk.empty:
        st.warning("No locations match the selected filters.")
    else:
        center_lat = filtered_risk["latitude"].mean()
        center_lon = filtered_risk["longitude"].mean()
        zoom = 11 if city_filter != "All" else 5

        risk_map = folium.Map(location=[center_lat, center_lon], zoom_start=zoom, tiles="OpenStreetMap")

        if show_heat_layer:
            heat_data = filtered_risk[["latitude", "longitude", "risk_score"]].values.tolist()
            HeatMap(heat_data, radius=28, blur=20, max_zoom=12).add_to(risk_map)

        color_map = {"HIGH": "red", "MEDIUM": "orange", "LOW": "green"}
        for _, row in filtered_risk.iterrows():
            color = color_map.get(row["risk_level"], "blue")
            folium.CircleMarker(
                location=[row["latitude"], row["longitude"]],
                radius=9,
                popup=folium.Popup(
                    f"<b>{row['withdrawal_location']}</b> ({row['withdrawal_city']})<br>"
                    f"Risk Score: {row['risk_score']:.1f}/100<br>"
                    f"Risk Level: {row['risk_level']}<br>"
                    f"Historical Withdrawal Rate: {row['historical_withdrawal_rate']:.1f}%<br>"
                    f"Peak Risk Hour: {row['peak_risk_hour']:02d}:00",
                    max_width=250,
                ),
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.85,
                weight=2,
            ).add_to(risk_map)

        st_folium(risk_map, width=None, height=540, key="risk_heatmap")

    st.subheader("🔥 Ranked Hotspot Ledger")
    hotspots_display = filtered_risk.sort_values("risk_score", ascending=False)
    st.dataframe(
        hotspots_display[
            ["withdrawal_location", "withdrawal_city", "risk_score", "risk_level",
             "historical_withdrawal_rate", "average_amount", "average_distance", "peak_risk_hour"]
        ],
        width="stretch",
        hide_index=True,
    )

# ==========================================================================
# TAB 3: PREDICTIVE ANALYTICS ENGINE
# ==========================================================================
with tab_predict:
    st.subheader("Predictive Analytics Engine")
    st.caption("Estimate the likelihood of cash withdrawal for a new/incoming complaint via Random Forest.")

    if model is None:
        st.error("ML model not found. Run `python ml/train_model.py` to generate `ml/cybercrime_model.pkl`.")
    else:
        c1, c2 = st.columns(2)
        with c1:
            amount = st.number_input("💰 Transaction Amount (₹)", min_value=1000, max_value=1000000, value=100000, step=1000)
            complaint_hour = st.slider("🕐 Complaint Hour", 0, 23, 18)
        with c2:
            withdrawal_hour = st.slider("🕐 Expected Withdrawal Hour", 0, 23, 20)
            distance = st.number_input("📍 Distance from Complaint Location (km)", min_value=0.0, max_value=100.0, value=5.0, step=0.5)

        predict_clicked = st.button("🔮 Evaluate Prediction Model", width="stretch")

        if predict_clicked:
            input_data = pd.DataFrame({
                "amount": [amount],
                "complaint_hour": [complaint_hour],
                "withdrawal_hour": [withdrawal_hour],
                "distance_km": [distance],
            })
            probability = model.predict_proba(input_data)[0][1]
            risk_percentage = probability * 100

            if risk_percentage >= 70:
                risk_level = "HIGH"
            elif risk_percentage >= 40:
                risk_level = "MEDIUM"
            else:
                risk_level = "LOW"

            st.session_state["last_prediction"] = {
                "amount": amount,
                "complaint_hour": complaint_hour,
                "withdrawal_hour": withdrawal_hour,
                "distance": distance,
                "risk_percentage": risk_percentage,
                "risk_level": risk_level,
                "timestamp": datetime.now().strftime("%d-%m-%Y %H:%M:%S"),
            }

        result = st.session_state.get("last_prediction")
        if result:
            st.divider()
            rc1, rc2 = st.columns([1, 2])
            with rc1:
                st.metric("Withdrawal Risk", f"{result['risk_percentage']:.2f}%")
                st.markdown(risk_badge(result["risk_level"]), unsafe_allow_html=True)
            with rc2:
                st.progress(int(result["risk_percentage"]), text=f"Prediction Confidence: {result['risk_percentage']:.2f}%")

            if result["risk_level"] == "HIGH":
                st.error("🚨 High likelihood of cash withdrawal. Recommendation: notify nearest withdrawal points and coordinate account freeze.")
            elif result["risk_level"] == "MEDIUM":
                st.warning("⚠️ Moderate likelihood of cash withdrawal. Enhanced monitoring advised.")
            else:
                st.success("✅ Low probability pattern detected. Routine logging recommended.")

    st.divider()
    st.subheader("📊 Spatial Risk Comparison")
    chart_data = risk_scores[["withdrawal_location", "risk_score"]].set_index("withdrawal_location")
    st.bar_chart(chart_data, y="risk_score")

# ==========================================================================
# TAB 4: CLOSED-LOOP VALIDATION ENGINE
# ==========================================================================
with tab_validate:
    st.subheader("Closed-Loop Validation Engine")
    st.caption("PREDICT → (simulated) EVENT → VALIDATE → METRICS. Evaluates system accuracy under synthetic stress testing.")

    if "sim_log" not in st.session_state:
        st.session_state["sim_log"] = []
    if "validation_history" not in st.session_state:
        st.session_state["validation_history"] = []
    if "case_counter" not in st.session_state:
        st.session_state["case_counter"] = 0

    def _log(line: str):
        ts = datetime.now().strftime("%H:%M:%S")
        st.session_state["sim_log"].append(f"{ts}  {line}")

    st.markdown("##### 1. Create Controlled Scenario")
    cc1, cc2, cc3, cc4 = st.columns(4)
    with cc1:
        sim_city = st.selectbox("City", sorted(risk_scores["withdrawal_city"].unique().tolist()), key="sim_city")
    with cc2:
        sim_crime = st.selectbox("Crime Type", sorted(complaints["crime_type"].unique().tolist()), key="sim_crime")
    with cc3:
        sim_amount = st.number_input("Amount (₹)", min_value=1000, max_value=1000000, value=75000, step=1000, key="sim_amount")
    with cc4:
        sim_hour = st.slider("Complaint Hour", 0, 23, 20, key="sim_hour")

    sim_scenario = st.selectbox("Simulation Scenario", ve.SCENARIOS, key="sim_scenario")

    btn1, btn2, btn3 = st.columns(3)
    generate_clicked = btn1.button("🔮 Forecast Withdrawal Point", width="stretch")
    simulate_clicked = btn2.button("🔁 Simulate Event & Validate", width="stretch")
    reset_clicked = btn3.button("🗑️ Reset Session History", width="stretch")

    if reset_clicked:
        st.session_state["sim_log"] = []
        st.session_state["validation_history"] = []
        st.session_state["case_counter"] = 0
        st.session_state.pop("sim_case", None)
        st.session_state.pop("sim_predictions", None)
        st.rerun()

    if generate_clicked:
        st.session_state["case_counter"] += 1
        case_id = f"CASE-{st.session_state['case_counter']:04d}"
        preds = generate_predictions(
            city=sim_city, crime_type=sim_crime, amount=sim_amount, complaint_hour=sim_hour,
            risk_scores=risk_scores, complaints=complaints, top_k=3,
        )
        st.session_state["sim_case"] = {
            "case_id": case_id, "city": sim_city, "crime_type": sim_crime,
            "amount": sim_amount, "complaint_hour": sim_hour, "created_at": datetime.now(),
        }
        st.session_state["sim_predictions"] = preds
        _log(f"Complaint {case_id} generated — {sim_crime}, ₹{sim_amount:,.0f}")

    case = st.session_state.get("sim_case")
    preds = st.session_state.get("sim_predictions")

    if case and preds:
        st.markdown(f"##### 2. Ranked Forecast — {case['case_id']}")
        for p in preds:
            with st.container(border=True):
                pc1, pc2, pc3, pc4 = st.columns([2, 1, 1, 1])
                pc1.markdown(f"**#{p.rank} — {p.location}**  \n{p.city}")
                pc2.metric("Confidence", f"{p.confidence:.0f}%")
                pc3.markdown(risk_badge(p.risk_level), unsafe_allow_html=True)
                pc4.write(f"⏱️ {p.time_window}")

        st.markdown("##### 3. Simulated Verification Result")
        if simulate_clicked:
            candidates = risk_scores[risk_scores["withdrawal_city"] == case["city"]]
            event = ve.simulate_event(preds, candidates, sim_scenario)
            result = ve.validate(preds, event)
            st.session_state["validation_history"].append({
                "case_id": case["case_id"], "scenario": sim_scenario,
                "predictions": preds, "validation": result,
            })
            verdict = "✓ TOP-1 HIT" if result.top1_hit else ("✓ TOP-3 HIT" if result.topk_hit else "✕ MISS")
            _log(f"Validation for {case['case_id']}: Actual at {result.actual_location} → {verdict}")
            st.session_state["last_result"] = result

        result = st.session_state.get("last_result")
        if result and st.session_state["validation_history"] and \
           st.session_state["validation_history"][-1]["case_id"] == case["case_id"]:
            r1, r2, r3, r4 = st.columns(4)
            r1.metric("Top-1 Precision", "✓ HIT" if result.top1_hit else "✕ MISS")
            r2.metric("Top-3 Precision", "✓ HIT" if result.topk_hit else "✕ MISS")
            r3.metric("Spatial Error", f"{result.geo_error_km} km")
            r4.metric("Window Accuracy", "✓ HIT" if result.time_window_hit else "✕ MISS")

    st.divider()
    st.markdown("##### 📈 Session Benchmark Summary")
    metrics = ve.summarize_history(st.session_state["validation_history"])
    if metrics["count"] > 0:
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Evaluated Cases", metrics["count"])
        m2.metric("Top-1 Accuracy", f"{metrics['top1_accuracy']:.0f}%")
        m3.metric("Top-3 Accuracy", f"{metrics['topk_accuracy']:.0f}%")
        m4.metric("Mean Geo-Offset", f"{metrics['avg_geo_error_km']:.2f} km")
        m5.metric("Window Precision", f"{metrics['time_window_accuracy']:.0f}%")

# ==========================================================================
# TAB 5: LAW ENFORCEMENT INTERFACE
# ==========================================================================
with tab_lea:
    st.subheader("Law Enforcement Case Explorer")
    st.caption("Investigator portal for audit logging and field verification.")

    fc1, fc2, fc3, fc4 = st.columns(4)
    with fc1:
        lea_city = st.selectbox("City", ["All"] + sorted(complaints["city"].unique().tolist()), key="lea_city")
    with fc2:
        lea_crime = st.selectbox("Crime Type", ["All"] + sorted(complaints["crime_type"].unique().tolist()), key="lea_crime")
    with fc3:
        lea_status = st.selectbox("Status", ["All", "Withdrawal Occurred", "Not Recorded"], key="lea_status")
    with fc4:
        lea_hour_range = st.slider("Hour Window", 0, 23, (0, 23), key="lea_hour_range")

    filtered_complaints = complaints.copy()
    if lea_city != "All":
        filtered_complaints = filtered_complaints[filtered_complaints["city"] == lea_city]
    if lea_crime != "All":
        filtered_complaints = filtered_complaints[filtered_complaints["crime_type"] == lea_crime]
    if lea_status == "Withdrawal Occurred":
        filtered_complaints = filtered_complaints[filtered_complaints["withdrawal_occurred"] == 1]
    elif lea_status == "Not Recorded":
        filtered_complaints = filtered_complaints[filtered_complaints["withdrawal_occurred"] == 0]

    filtered_complaints = filtered_complaints[
        filtered_complaints["complaint_hour"].between(lea_hour_range[0], lea_hour_range[1])
    ]

    st.dataframe(filtered_complaints, width="stretch", hide_index=True, height=280)

# ==========================================================================
# TAB 6: ALERTS & NOTIFICATIONS
# ==========================================================================
with tab_alerts:
    st.subheader("Alert & Automated Trigger Matrix")
    st.caption("Active operational channels dispatched to I4C portal, NPCI gateways, and state field cells.")

    top_alerts = risk_scores.sort_values("risk_score", ascending=False).head(5)
    for _, row in top_alerts.iterrows():
        box = st.error if row["risk_level"] == "HIGH" else st.warning
        with box(f"{row['risk_level']} — {row['withdrawal_location']} ({row['withdrawal_city']}) | Risk Score: {row['risk_score']:.1f}%"):
            pass
        d1, d2, d3 = st.columns(3)
        d1.write(f"📍 **Coordinates:** {row['latitude']:.4f}, {row['longitude']:.4f}")
        d2.write(f"💰 **Avg Transaction:** ₹{row['average_amount']:,.0f}")
        d3.write(f"⏱️ **Peak Vulnerability:** {row['peak_risk_hour']:02d}:00")
        st.divider()

# ==========================================================================
# TAB 7: INTELLIGENCE SUMMARY & METRICS
# ==========================================================================
with tab_analytics:
    st.subheader("Operational Intelligence Summary")

    s1, s2 = st.columns(2)
    with s1:
        st.markdown("##### 📌 Historical Case Outcomes")
        w_cases = int(complaints["withdrawal_occurred"].sum())
        nw_cases = len(complaints) - w_cases
        st.metric("Total Confirmed Cash-Outs", w_cases)
        st.metric("Prevented / Unrecorded Cash-Outs", nw_cases)
    with s2:
        st.markdown("##### 🚦 Monitored Risk Tier Breakdown")
        st.bar_chart(risk_scores["risk_level"].value_counts())