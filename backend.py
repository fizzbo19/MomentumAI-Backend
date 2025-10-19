"""
MomentumScout Backend – Final Production-Ready Version
FIXES: 
1. Persistent ambiguity error in Pandas filtering (initialization and runtime).
2. Ensures all core filters (Value, Age, Overall) are safely mapped.
3. Implements final CORS solution.
"""
import os
import math
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, send_from_directory
import requests
from flask_cors import CORS

app = Flask(__name__, static_folder="public")

# Frontend origin for CORS - configure in Render env (safe default provided)
# **CRITICAL FIX:** This must be the exact URL of your Netlify site.
FRONTEND_URL = os.environ.get("FRONTEND_URL", "https://momentum-ai-io.netlify.app") 

# Allowed origins list (frontend + common local dev hosts)
ALLOWED_ORIGINS = [
    FRONTEND_URL,
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:5000",
    "http://127.0.0.1:5000"
]

# Enable CORS on all /api/* routes for allowed origins
# The `supports_credentials=True` is often needed for modern browsers.
CORS(app, resources={r"/api/*": {"origins": ALLOWED_ORIGINS}}, supports_credentials=True)

@app.after_request
def add_cors_headers(response):
    origin = request.headers.get("Origin")
    # CRITICAL: Dynamically set the header to match the request origin if allowed
    if origin and origin in ALLOWED_ORIGINS:
        response.headers["Access-Control-Allow-Origin"] = origin
        response.headers["Access-Control-Allow-Credentials"] = "true"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response

# --- Environment Variables ---
GOOGLE_SCRIPT_URL = os.environ.get(
    "GOOGLE_SCRIPT_URL",
    "https://script.google.com/macros/s/YOUR_SCRIPT_ID/exec"
)
DATA_FOLDER_PATH = os.environ.get("DATA_FOLDER_PATH", "data")
DATA_FILENAME = os.environ.get(
    "DATA_FILENAME", "Career Mode player datasets - FIFA 15-22.xlsx"
)

# --- Default position weights (fallback) ---
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
    'ST': {'shooting':40,'pace':25,'dribbling':20,'physic':15,'movement_acceleration':5},
    'CF': {'shooting':30,'passing':25,'dribbling':25,'pace':20,'movement_acceleration':5}
}

POSITION_METRICS_FOR_SCORING = POSITION_WEIGHTS 

# --- Projection rules by age (years to project) ---
def years_to_project(age: int) -> int:
    if age <= 20: return 5
    if 21 <= age <= 25: return 4
    if 26 <= age <= 30: return 3
    if 31 <= age <= 35: return 2
    return 1

# --- Load dataset ---
player_data = None

def initialize_app():
    global player_data
    fp = os.path.join(DATA_FOLDER_PATH, DATA_FILENAME)
    if not os.path.exists(fp):
        raise FileNotFoundError(f"Dataset not found at {fp} (set DATA_FOLDER_PATH/DATA_FILENAME env vars)")
    
    try:
        player_data = pd.read_excel(fp)
    except Exception as e:
        print(f"Error reading Excel file: {e}")
        raise
        
    player_data.columns = [c if isinstance(c, str) else c for c in player_data.columns]
    
    NUMERIC_COLS = ['overall','potential','age','value_eur','pace','shooting','passing','dribbling','defending','physic','wage_eur']
    
    # Loop to forcefully convert columns and handle non-numeric values
    for col in NUMERIC_COLS:
        if col in player_data.columns:
            # CRITICAL FIX: Ensure all numeric columns are strictly float, replacing anything else with 0
            player_data[col] = pd.to_numeric(player_data[col], errors='coerce').fillna(0)
            
    print(f"✅ Dataset loaded. Total players: {len(player_data)}")

# --- Helpers (Sanitization, Scoring, Projection) ---
def sanitize_player_data(players_list):
    clean_list = []
    for player in players_list:
        clean_player = {}
        for k,v in player.items():
            is_nan_or_none = (
                (isinstance(v, float) and (math.isnan(v) or math.isinf(v))) or
                (pd.isna(v)) or 
                (v is None)
            )
            
            if is_nan_or_none:
                clean_player[k] = None
            elif isinstance(v, np.generic):
                clean_player[k] = v.item() 
            else:
                clean_player[k] = v
        clean_list.append(clean_player)
    return clean_list

def compute_score_for_player(row, position, user_weights=None):
    base_weights = POSITION_WEIGHTS.get(position, POSITION_WEIGHTS.get('CM', {})).copy()
    if user_weights:
        for k,v in user_weights.items():
            if v is not None:
                base_weights[k] = float(v)
    total_w = sum(base_weights.values()) if base_weights else 1
    if total_w == 0: total_w = 1
    score = 0.0
    for attr, w in base_weights.items():
        val = row.get(attr, 0)
        try:
            val_num = float(val) if val is not None else 0.0
        except:
            val_num = 0.0
        norm = val_num / 99.0
        score += norm * (w / total_w)
    return round(score * 100, 4)

def project_player(row, years:int):
    ovr = int(row.get('overall') or 0)
    pot = int(row.get('potential') or ovr)
    age = int(row.get('age') or 0)
    value = float(row.get('value_eur') or 0)

    if pot > ovr and years > 0:
        per_year_ovr = (pot - ovr) / years
    else:
        per_year_ovr = 0

    if age <= 20: growth = 0.35
    elif age <= 25: growth = 0.20
    elif age <= 30: growth = 0.12
    elif age <= 35: growth = 0.07
    else: growth = 0.03

    projections = []
    cur_ovr = ovr
    cur_value = value
    for y in range(1, years+1):
        cur_ovr = min(99, cur_ovr + per_year_ovr)
        cur_value = max(0, cur_value * (1 + growth))
        projections.append({
            "year_offset": y,
            "projected_overall": round(cur_ovr, 1),
            "projected_value_eur": int(round(cur_value))
        })
    return projections

def negotiation_range(current_value:int, projected_value:int):
    if current_value is None or current_value <= 0:
        return {"min_offer": 0, "max_offer": 0}
    min_offer = int(round(current_value * 0.7))
    max_offer = int(round(max(projected_value, current_value) * 1.05))
    return {"min_offer": min_offer, "max_offer": max_offer}

# attributes to consider for scoring by default (comprehensive list)
POSITION_METRICS_FOR_SCORING = {
    'GK': ['goalkeeping_diving','goalkeeping_handling','goalkeeping_kicking','goalkeeping_positioning','goalkeeping_reflexes'],
    'CB': ['defending','physic','pace','passing','dribbling'],
    'LB': ['pace','passing','defending','physic','dribbling'],
    'RB': ['pace','passing','defending','physic','dribbling'],
    'CDM': ['defending','passing','dribbling','physic','pace'],
    'CM': ['passing','dribbling','defending','pace','physic'],
    'CAM': ['passing','dribbling','shooting','pace','physic'],
    'LW': ['pace','dribbling','shooting','passing','physic'],
    'RW': ['pace','dribbling','shooting','passing','physic'],
    'ST': ['shooting','pace','dribbling','physic','movement_acceleration'],
    'CF': ['shooting','passing','dribbling','pace','physic']
}

# --- Routes ---
@app.route("/api/submit_demo", methods=["POST", "OPTIONS"])
def submit_demo():
    if request.method == "OPTIONS": return "", 200
    try:
        data = request.json
        if not data: return jsonify({"success": False, "message": "No form data provided."}), 400
        response = requests.post(GOOGLE_SCRIPT_URL, json=data, timeout=10)
        response.raise_for_status()
        return jsonify({"success": True, "message": "Form submitted successfully."}), 200
    except requests.exceptions.RequestException as e:
        print("Error forwarding to Google Apps Script:", e)
        return jsonify({"success": False, "message": "Error submitting form."}), 500

@app.route("/api/search_player", methods=["POST", "OPTIONS"])
def api_search_player():
    if request.method == "OPTIONS": return "", 200
    try:
        payload = request.get_json(silent=True) or {}
        query = (payload.get("player_name") or payload.get("name") or "").strip()
        
        if not query: return jsonify([]) 

        q = str(query).lower()
        df = player_data
        name_cols = [c for c in ['short_name','long_name','player_name'] if c in df.columns]
        
        mask = False
        if name_cols:
            for c in name_cols:
                mask = mask | df[c].astype(str).str.lower().str.contains(q, na=False) 
        else:
            mask = df.astype(str).apply(lambda r: r.str.lower().str.contains(q, na=False).any(), axis=1)
            
        results = df[mask].head(20)
        out = []
        for _, row in results.iterrows():
            age = int(row.get('age') or 0)
            years = years_to_project(age)
            projections = project_player(row, years)
            last_proj_value = projections[-1]['projected_value_eur'] if projections else int(row.get('value_eur') or 0)
            neg = negotiation_range(int(row.get('value_eur') or 0), last_proj_value)
            weekly_wage = row.get('wage_eur', 0)
            yearly_wage_gbp = weekly_wage * 52 if weekly_wage else 0
            
            out.append({
                "short_name": row.get('short_name') or row.get('player_name') or "N/A",
                "club_position": row.get('club_position') or "",
                "overall": int(row.get('overall') or 0),
                "potential": int(row.get('potential') or 0),
                "value_eur": int(row.get('value_eur') or 0),
                "player_face_url": row.get('player_face_url') or '',
                "min_value_eur": neg['min_offer'],
                "max_value_eur": neg['max_offer'],
                "projections": projections,
                "full_attributes": {
                    "Overall": int(row.get('overall') or 0),
                    "Potential": int(row.get('potential') or 0),
                    "Age": age,
                    "Pace": int(row.get('pace') or 0),
                    "Shooting": int(row.get('shooting') or 0),
                    "Passing": int(row.get('passing') or 0),
                    "Dribbling": int(row.get('dribbling') or 0),
                    "Defending": int(row.get('defending') or 0),
                    "Physicality": int(row.get('physic') or 0),
                    "Club": row.get('club_name') or '',
                    "League": row.get('league_name') or '',
                    "Wage (YEARLY GBP)": yearly_wage_gbp
                }
            })
        return jsonify(sanitize_player_data(out))
    except Exception as e:
        print("Error in /api/search_player:", e)
        return jsonify({"message": f"Internal Server Error: {e}"}), 500

@app.route("/api/find_players", methods=["POST","OPTIONS"])
def api_find_players():
    if request.method == "OPTIONS":
        return "", 0
    try:
        payload = request.json or {}
        position = (payload.get("club_position") or "CM").upper()
        filters = payload.get("filters") or {}
        user_weights = payload.get("weights") or {}

        df = player_data.copy()

        # Apply filters (min,max)
        for key, rng in filters.items():
            try:
                if isinstance(rng, (list,tuple)) and len(rng) >= 2:
                    lo = float(rng[0])
                    hi = float(rng[1])
                    
                    # 1. Determine the target column name (all lowercase in DB)
                    target_col = key.lower() 
                    if 'value' in target_col:
                        target_col = 'value_eur' 
                    elif 'overall' in target_col:
                        target_col = 'overall' 
                    elif 'age' in target_col:
                        target_col = 'age'
                    
                    # 2. Check if the column exists in the DataFrame
                    if target_col in df.columns:
                        # CRITICAL: Filtering works by comparing floats against floats
                        df = df[(df[target_col].astype(float) >= lo) & (df[target_col].astype(float) <= hi)]
                    else:
                        # If a metric is requested that doesn't exist, log it and skip filter.
                        print(f"Skipping filter for '{key}': column '{target_col}' not found.")
            except Exception as e:
                # If any conversion or comparison fails, halt this search gracefully.
                print(f"CRASH POINT: Filter error on key '{key}' with range {rng}. Error: {e}")
                return jsonify({"players": []}), 200 

        # Score players
        scored = []
        for _, row in df.iterrows():
            try:
                score = compute_score_for_player(row, position, user_weights=user_weights)
            except Exception as e:
                score = 0
            scored.append((score, row))

        # sort descending by score
        scored_sorted = sorted(scored, key=lambda x: x[0], reverse=True)

        players_out = []
        for score, row in scored_sorted[:50]: 
            age = int(row.get('age') or 0)
            years = years_to_project(age)
            projections = project_player(row, years)
            last_proj_value = projections[-1]['projected_value_eur'] if projections else int(row.get('value_eur') or 0)
            neg = negotiation_range(int(row.get('value_eur') or 0), last_proj_value)
            
            weekly_wage = row.get('wage_eur', 0)
            yearly_wage_gbp = weekly_wage * 52 if weekly_wage else 0
            
            players_out.append({
                "short_name": row.get('short_name') or row.get('player_name') or "N/A",
                "club_position": row.get('club_position') or "",
                "overall": int(row.get('overall') or 0),
                "potential": int(row.get('potential') or 0),
                "value_eur": int(row.get('value_eur') or 0),
                "player_face_url": row.get('player_face_url') or "",
                "min_value_eur": neg['min_offer'], 
                "max_value_eur": neg['max_offer'], 
                "momentum_score": score,
                "projections": projections,
                "negotiation": neg,
                "full_attributes": {
                    'Overall': int(row.get('overall') or 0),
                    'Potential': int(row.get('potential') or 0),
                    'Age': age,
                    'Pace': int(row.get('pace') or 0),
                    'Shooting': int(row.get('shooting') or 0),
                    'Passing': int(row.get('passing') or 0),
                    'Dribbling': int(row.get('dribbling') or 0),
                    'Defending': int(row.get('defending') or 0),
                    'Physicality': int(row.get('physic') or 0),
                    'Club': row.get('club_name') or '',
                    'League': row.get('league_name') or '',
                    'Wage (YEARLY GBP)': yearly_wage_gbp 
                }
            })
        return jsonify({"players": sanitize_player_data(players_out[:5])})
    except Exception as e:
        print("Error in /api/find_players:", e)
        return jsonify({"players": []}), 500

@app.route("/assets/<path:filename>")
def serve_assets(filename):
    return send_from_directory(os.path.join(app.root_path, "public/assets"), filename)

# --- Main Execution ---
if __name__ == "__main__":
    print("🚀 Initializing backend...")
    initialize_app()
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)