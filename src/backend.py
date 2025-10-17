"""
MomentumAI Backend – Render-ready version
"""
import os
import math
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
import requests
from flask_cors import CORS

# --- Flask App Setup ---
app = Flask(__name__)

# Allow CORS for your deployed frontend + localhost
CORS(app, resources={r"/*": {"origins": [
    "https://momentumai-frontendv11.onrender.com",
    "https://momentumai-frontend.onrender.com",
    "http://localhost:3000"
]}})

# --- Environment Variables ---
GOOGLE_SCRIPT_URL = os.environ.get(
    "GOOGLE_SCRIPT_URL",
    "https://script.google.com/macros/s/AKfycbzKry-uh7HtLAQD_NolGX82xWeY2K8xZG9UjgOC_mmdNI7DpclWhGlesff_Qwe_jSau/exec"
)
DATA_FOLDER_PATH = os.environ.get("DATA_FOLDER_PATH", "data")
DATA_FILENAME = os.environ.get(
    "DATA_FILENAME", "Career Mode player datasets - FIFA 15-22.xlsx"
)

player_data = None

# --- Position Metrics & Default Weights ---
POSITION_METRICS = {
    'GK': ['goalkeeping_diving', 'goalkeeping_handling', 'goalkeeping_kicking', 'goalkeeping_positioning', 'goalkeeping_reflexes', 'goalkeeping_speed'],
    'CB': ['pace','passing','defending','power_strength','power_jumping','mentality_interceptions','mentality_positioning','defending_marking_awareness','defending_standing_tackle','defending_sliding_tackle','attacking_crossing'],
    'LB': ['pace','passing','defending','power_strength','power_jumping','mentality_interceptions','mentality_positioning','defending_marking_awareness','defending_standing_tackle','defending_sliding_tackle','attacking_crossing'],
    'RB': ['pace','passing','defending','power_strength','power_jumping','mentality_interceptions','mentality_positioning','defending_marking_awareness','defending_standing_tackle','defending_sliding_tackle','attacking_crossing'],
    'CDM': ['pace','passing','dribbling','defending','physic','mentality_interceptions','mentality_positioning','defending_standing_tackle','defending_sliding_tackle'],
    'CM': ['pace','passing','dribbling','defending','physic','power_stamina','power_strength','mentality_vision'],
    'CAM': ['pace','shooting','passing','dribbling','power_long_shots','mentality_vision','mentality_penalties'],
    'LW': ['pace','shooting','passing','dribbling','power_long_shots','skill_curve','skill_fk_accuracy','movement_acceleration','movement_sprint_speed'],
    'RW': ['pace','shooting','passing','dribbling','power_long_shots','skill_curve','skill_fk_accuracy','movement_acceleration','movement_sprint_speed'],
    'ST': ['pace','shooting','dribbling','power_shot_power','power_jumping','power_strength','mentality_positioning','attacking_finishing'],
    'CF': ['pace','shooting','passing','dribbling','power_shot_power','power_jumping','mentality_positioning','mentality_composure']
}

POSITION_WEIGHTS = {
    'GK': {'goalkeeping_diving': 20,'goalkeeping_handling': 20,'goalkeeping_kicking': 20,'goalkeeping_positioning': 20,'goalkeeping_reflexes': 20},
    'CB': {'defending':50,'physic':20,'pace':10,'passing':10,'dribbling':10},
    'LB': {'pace':30,'passing':20,'defending':15,'physic':10,'dribbling':25},
    'RB': {'pace':30,'passing':20,'defending':15,'physic':10,'dribbling':25},
    'CDM': {'defending':40,'passing':20,'physic':15,'pace':15,'dribbling':10},
    'CM': {'passing':30,'dribbling':20,'defending':15,'pace':15,'shooting':10,'physic':10},
    'CAM': {'passing':30,'dribbling':25,'shooting':25,'pace':10,'physic':10},
    'LW': {'pace':35,'dribbling':30,'shooting':20,'passing':15},
    'RW': {'pace':35,'dribbling':30,'shooting':20,'passing':15},
    'ST': {'shooting':40,'pace':25,'dribbling':20,'physic':15},
    'CF': {'shooting':30,'passing':25,'dribbling':25,'pace':20}
}

# --- Initialize App & Load Dataset ---
def initialize_app():
    global player_data
    fp = os.path.join(DATA_FOLDER_PATH, DATA_FILENAME)
    if not os.path.exists(fp):
        raise FileNotFoundError(f"Dataset not found at {fp}")

    player_data = pd.read_excel(fp)
    # Convert numeric columns
    for col in ['overall','potential','age','value_eur','pace','shooting','passing','dribbling','defending','physic','wage_eur']:
        if col in player_data.columns:
            player_data[col] = pd.to_numeric(player_data[col], errors='coerce').fillna(0)

    # Handle missing columns
    if 'physic' not in player_data.columns:
        player_data['physic'] = player_data.get('physicality',0)
    if 'value_eur' not in player_data.columns:
        player_data['value_eur'] = player_data.get('value_eur',0)

    print(f"✅ Dataset loaded. Total players: {len(player_data)}")

# --- Helper Functions ---
def calculate_momentum_score(player_attributes, weights):
    total_weight = sum(weights.values())
    if total_weight == 0: return 0
    return sum(player_attributes.get(k.lower(),0) * (w/total_weight) for k,w in weights.items())

def sanitize_player_data(players_list):
    clean_list = []
    for player in players_list:
        clean_player = {}
        for k,v in player.items():
            if isinstance(v,float) and (math.isnan(v) or math.isinf(v)):
                clean_player[k] = None
            elif pd.isna(v):
                clean_player[k] = None
            else:
                clean_player[k] = v.item() if isinstance(v,np.generic) else v
        clean_list.append(clean_player)
    return clean_list

# --- Routes ---

@app.route("/api/submit_demo", methods=["POST","OPTIONS"])
def submit_demo():
    if request.method == "OPTIONS": return "",200
    try:
        data = request.json
        if not data: return jsonify({"success":False,"message":"No form data provided."}),400
        response = requests.post(GOOGLE_SCRIPT_URL,json=data)
        response.raise_for_status()
        return jsonify({"success":True,"message":"Form submitted successfully."}),200
    except requests.exceptions.RequestException as e:
        print(f"Error forwarding to Google Apps Script: {e}")
        return jsonify({"success":False,"message":"Error submitting form."}),500

@app.route("/api/search_player", methods=["POST","OPTIONS"])
def api_search_player():
    if request.method=="OPTIONS": return "",200
    payload = request.json or {}
    name = (payload.get("player_name") or "").strip().lower()
    if not name: return jsonify({"error":"player_name is required"}),400

    df = player_data.copy()
    mask = False
    for col in ["short_name","long_name","player_name"]:
        if col in df.columns:
            mask = mask | df[col].astype(str).str.lower().str.contains(name)
    results = df[mask].head(10)

    if results.empty: return jsonify([])
    out = []
    for _, row in results.iterrows():
        weekly_wage = row.get('wage_eur',0)
        yearly_wage = weekly_wage*52 if weekly_wage else 0
        full_attributes = {
            'Overall': row.get('overall'),
            'Potential': row.get('potential'),
            'Age': row.get('age'),
            'Pace': row.get('pace'),
            'Shooting': row.get('shooting'),
            'Passing': row.get('passing'),
            'Dribbling': row.get('dribbling'),
            'Defending': row.get('defending'),
            'Physicality': row.get('physic'),
            'Club': row.get('club_name'),
            'League': row.get('league_name'),
            'Value (GBP)': row.get('value_eur'),
            'Wage (YEARLY GBP)': yearly_wage,
        }
        out.append({
            "short_name": row.get("short_name","N/A"),
            "club_position": row.get("club_position","N/A"),
            "overall": int(row.get("overall",0)),
            "potential": int(row.get("potential",0)),
            "value_eur": int(row.get("value_eur",0)),
            "full_attributes": full_attributes,
            "player_face_url": row.get('player_face_url','https://via.placeholder.com/150'),
            "min_value_eur": row.get('value_eur')*0.8,
            "max_value_eur": row.get('value_eur')*1.2,
        })
    return jsonify(sanitize_player_data(out))

# --- Main Execution ---
if __name__=="__main__":
    print("🚀 Initializing backend...")
    initialize_app()
    port = int(os.environ.get("PORT",5000))
    app.run(host="0.0.0.0",port=port,debug=False)