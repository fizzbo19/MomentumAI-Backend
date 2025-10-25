# backend.py
"""
MomentumScout Backend – FC26-ready, fully JSON-exposing all fields
- Supports CSV/Excel datasets (FC26)
- Normalizes columns for frontend compatibility
- Computes player scores, projections, and negotiation ranges
- Robust CORS and preflight handling
"""

import os
import math
import json
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, send_from_directory, make_response
from flask_cors import CORS
from utils import project_player, negotiation_range, compute_score_for_player, clean_json, years_to_project


# ----------------- App & CORS -----------------
app = Flask(__name__, static_folder="public", static_url_path="/public")

DEFAULT_FRONTEND = "https://momentum-ai-io.netlify.app"
FRONTEND_URL = os.environ.get("FRONTEND_URL", DEFAULT_FRONTEND)

ALLOWED_ORIGINS = [
    FRONTEND_URL,
    "https://momentumai-frontend.onrender.com",
    "http://localhost:3000",
    "http://127.0.0.1:3000"
]

CORS(app, resources={r"/api/*": {"origins": ALLOWED_ORIGINS, "supports_credentials": True}})

# ----------------- Dataset -----------------
DATA_FOLDER_PATH = os.environ.get("DATA_FOLDER_PATH", "data")
DATA_FILENAME = os.environ.get("DATA_FILENAME", "data/FC26_MomentumScout.csv")
player_data = None

# ----------------- Default Position Weights -----------------
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

# ----------------- Utilities -----------------
def years_to_project(age: int) -> int:
    if age <= 20: return 5
    if 21 <= age <= 25: return 4
    if 26 <= age <= 30: return 3
    if 31 <= age <= 35: return 2
    return 1

def is_nan_like(v):
    try:
        return (isinstance(v, float) and (math.isnan(v) or math.isinf(v))) or (pd.isna(v))
    except Exception:
        return False

def clean_json_value(v):
    if isinstance(v, (np.integer,)): return int(v)
    if isinstance(v, (np.floating,)): 
        if math.isnan(float(v)) or math.isinf(v): return None
        return float(v)
    if isinstance(v, np.bool_): return bool(v)
    if is_nan_like(v): return None
    return v

def clean_json(data):
    if isinstance(data, dict):
        return {k: clean_json(data[k]) for k in data}
    if isinstance(data, list):
        return [clean_json(v) for v in data]
    return clean_json_value(data)

def compute_score_for_player(row, position, user_weights=None):
    if hasattr(row, "to_dict"): row_dict = row.to_dict()
    else: row_dict = dict(row)
    base_weights = POSITION_WEIGHTS.get(position, POSITION_WEIGHTS.get('CM', {})).copy()
    if user_weights:
        for k,v in user_weights.items():
            if v is not None:
                try: base_weights[k] = float(v)
                except: pass
    total_w = sum(base_weights.values()) if base_weights else 1
    if total_w == 0: total_w = 1
    score = 0.0
    for attr, w in base_weights.items():
        val = row_dict.get(attr,0)
        try: val_num = float(val) if val is not None else 0.0
        except: val_num=0.0
        norm = val_num / 99.0
        score += norm * (w / total_w)
    return round(score * 100,4)

def project_player(player_row, years=3):
    """
    Returns a list of projected attributes for each year.
    Every numeric attribute in FC26 dataset is projected.
    """
    attrs = player_row.to_dict()
    
    # Identify numeric attributes
    numeric_attrs = {}
    for k, v in attrs.items():
        try:
            numeric_attrs[k] = float(v)
        except:
            continue

    projections = []

    # Define growth/decline rates heuristically based on age
    age = int(attrs.get('age') or 20)
    if age <= 20: growth_factor = 1.12; overall_delta = 1.8
    elif age <= 25: growth_factor = 1.08; overall_delta = 1.2
    elif age <= 30: growth_factor = 1.05; overall_delta = 0.5
    elif age <= 35: growth_factor = 1.03; overall_delta = -0.5
    else: growth_factor = 1.01; overall_delta = -1.0

    # Project each year
    current_attrs = numeric_attrs.copy()
    for year in range(1, years + 1):
        projected = {}
        for k, val in current_attrs.items():
            # Slight increase/decrease for numeric stats
            if k.lower() in ['overall','potential']:
                projected[k] = max(40, min(99, round(val + overall_delta)))
            elif 'value' in k.lower():
                projected[k] = round(val * growth_factor)
            else:
                # Other numeric attributes: small random +/- 2% per year (can be improved)
                projected[k] = round(val * (1 + 0.02 * (1 if year % 2 == 0 else -1)))
        projected['year'] = year
        projections.append(projected)
        current_attrs = projected.copy()

    return projections



def negotiation_range(current_value:int, projected_value:int):
    if current_value is None or current_value<=0: return {"min_offer":0,"max_offer":0}
    min_offer = int(round(current_value*0.7))
    max_offer = int(round(max(projected_value,current_value)*1.05))
    return {"min_offer":min_offer,"max_offer":max_offer}

# ----------------- Initialize Dataset -----------------
def initialize_app():
    global player_data
    fp = os.path.join(DATA_FOLDER_PATH, DATA_FILENAME)
    if not os.path.exists(fp):
        raise FileNotFoundError(f"Dataset not found at {fp}")

    try:
        if fp.lower().endswith('.csv'): player_data = pd.read_csv(fp, encoding='utf-8-sig')
        else: player_data = pd.read_excel(fp)
    except Exception as e:
        print("Error reading dataset:", e)
        raise

    player_data.columns = [str(c).strip() for c in player_data.columns]
    if 'player_name' not in player_data.columns and 'long_name' in player_data.columns:
        player_data['player_name'] = player_data['long_name']

    NUMERIC_COLS = ['overall','potential','age','value_eur','pace','shooting','passing','dribbling','defending','physic','wage_eur']
    for col in NUMERIC_COLS:
        if col in player_data.columns: player_data[col] = pd.to_numeric(player_data[col], errors='coerce').fillna(0)

    print(f"✅ Dataset loaded. Total players: {len(player_data)}")
    print(player_data.head(3)[[c for c in ['short_name','player_name','overall','club_name'] if c in player_data.columns]])

# ----------------- Routes -----------------
@app.before_request
def handle_options_preflight():
    if request.method=="OPTIONS" and request.path.startswith("/api/"):
        resp=make_response("")
        origin=request.headers.get("Origin","")
        if origin in ALLOWED_ORIGINS:
            resp.headers["Access-Control-Allow-Origin"]=origin
            resp.headers["Access-Control-Allow-Credentials"]="true"
            resp.headers["Access-Control-Allow-Headers"]="Content-Type, Authorization"
            resp.headers["Access-Control-Allow-Methods"]="GET, POST, OPTIONS"
        return resp

@app.after_request
def add_cors_headers(response):
    origin=request.headers.get("Origin")
    if origin and origin in ALLOWED_ORIGINS:
        response.headers["Access-Control-Allow-Origin"]=origin
        response.headers["Access-Control-Allow-Credentials"]="true"
        response.headers["Access-Control-Allow-Headers"]="Content-Type, Authorization"
        response.headers["Access-Control-Allow-Methods"]="GET, POST, OPTIONS"
    return response

@app.route("/api/debug_dataset", methods=["GET"])
def debug_dataset():
    if player_data is None: return jsonify({"loaded":False,"message":"No dataset loaded"}),500
    return jsonify({
        "loaded": True,
        "rows": len(player_data),
        "columns": list(player_data.columns),
        "sample": clean_json(player_data.head(5).to_dict(orient='records'))
    })

app = Flask(__name__)

# Assume `player_data` is your FC26 DataFrame loaded elsewhere
# player_data = pd.read_csv("fc26_players.csv")

@app.route("/api/search_player", methods=["POST","OPTIONS"])
def api_search_player():
    if request.method == "OPTIONS":
        return "", 200
    try:
        payload = request.get_json(silent=True) or {}
        query = (payload.get("player_name") or payload.get("name") or "").strip()
        if not query:
            return jsonify([])

        df = player_data
        name_cols = [c for c in ['short_name','long_name','player_name'] if c in df.columns]
        mask = pd.Series([False]*len(df))
        for c in name_cols:
            mask = mask | df[c].astype(str).str.lower().str.contains(query.lower(), na=False)

        results = df[mask].head(20)
        out = []
        for _, row in results.iterrows():
            age = int(row.get('age') or 0)
            years = years_to_project(age)
            projections = project_player(row, years)
            last_proj_val = projections[-1].get('value_eur', int(row.get('value_eur') or 0)) if projections else int(row.get('value_eur') or 0)
            neg = negotiation_range(int(row.get('value_eur') or 0), last_proj_val)
            score = compute_score_for_player(row, row.get('club_position') or 'CM')

            player_json = clean_json(row.to_dict())
            player_json.update({
                "projections": projections,
                "negotiation": neg,
                "momentum_score": score,
                "full_attributes": row.to_dict()  # Include all FC26 attributes
            })
            out.append(player_json)

        return jsonify(out)

    except Exception as e:
        print("Error in /api/search_player:", e)
        return jsonify({"message": f"Internal Server Error: {e}"}), 500


@app.route("/api/find_players", methods=["POST","OPTIONS"])
def api_find_players():
    if request.method == "OPTIONS":
        return "", 200
    try:
        payload = request.get_json(silent=True) or {}
        position = (payload.get("club_position") or payload.get("position") or "CM").upper()
        filters = payload.get("filters") or {}
        user_weights = payload.get("weights") or {}
        df = player_data.copy()

        # Apply filters
        for key, rng in filters.items():
            try:
                if isinstance(rng, (list, tuple)) and len(rng) >= 2:
                    lo, hi = float(rng[0]), float(rng[1])
                    target_col = key.lower()
                    if 'value' in target_col: target_col = 'value_eur'
                    elif 'overall' in target_col: target_col = 'overall'
                    elif 'age' in target_col: target_col = 'age'
                    if target_col in df.columns:
                        df = df[
                            (pd.to_numeric(df[target_col], errors='coerce').fillna(0) >= lo) &
                            (pd.to_numeric(df[target_col], errors='coerce').fillna(0) <= hi)
                        ]
            except Exception as e:
                print(f"Filter error on {key}: {e}")

        # Score players
        scored = []
        for _, row in df.iterrows():
            try:
                score = compute_score_for_player(row, position, user_weights=user_weights)
            except:
                score = 0
            scored.append((score, row))

        # Sort by score descending
        scored_sorted = sorted(scored, key=lambda x: x[0], reverse=True)

        players_out = []
        for score, row in scored_sorted[:50]:
            age = int(row.get('age') or 0)
            years = years_to_project(age)
            projections = project_player(row, years)
            last_proj_val = projections[-1].get('value_eur', int(row.get('value_eur') or 0)) if projections else int(row.get('value_eur') or 0)
            neg = negotiation_range(int(row.get('value_eur') or 0), last_proj_val)

            player_json = clean_json(row.to_dict())
            player_json.update({
                "projections": projections,
                "negotiation": neg,
                "momentum_score": score,
                "full_attributes": row.to_dict()  # Include all FC26 attributes
            })
            players_out.append(player_json)

        return jsonify({"players": players_out[:5]})

    except Exception as e:
        print("Error in /api/find_players:", e)
        return jsonify({"players": [], "message": f"Internal Server Error: {e}"}), 500



@app.route("/api/submit_demo", methods=["POST"])
def submit_demo():
    data = request.get_json(silent=True)
    if not data: return jsonify({"success": False, "message":"No data received"}),400
    full_name=data.get("fullName")
    email=data.get("email")
    org=data.get("organization")
    print(f"📩 Demo request from {full_name} ({email}) - {org}")
    return jsonify({"success": True, "message":"Demo request received!"})

@app.route("/assets/<path:filename>")
def serve_assets(filename):
    return send_from_directory(os.path.join(app.root_path,"public/assets"), filename)

# ----------------- Main -----------------
if __name__=="__main__":
    print("🚀 Initializing backend...")
    initialize_app()
    port=int(os.environ.get("PORT",5000))
    app.run(host="0.0.0.0", port=port, debug=False)
