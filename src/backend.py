# backend.py
"""
MomentumScout Backend – Consolidated & FC26-ready
- Supports CSV or Excel player datasets (FC26 example).
- Normalizes key column names (e.g. long_name -> player_name).
- Robust CORS + preflight handling for frontend origins.
- Sanitizes NaN/Inf/numpy types before JSON responses.
"""
import os
import math
import json
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, send_from_directory, make_response
from flask_cors import CORS

app = Flask(__name__, static_folder="public", static_url_path="/public")

# FRONTEND URLS - update FRONTEND_URL env var on Render if different
DEFAULT_FRONTEND = "https://momentum-ai-io.netlify.app"
FRONTEND_URL = os.environ.get("FRONTEND_URL", DEFAULT_FRONTEND)

ALLOWED_ORIGINS = [
    FRONTEND_URL,
    "https://momentumai-frontend.onrender.com",
    "http://localhost:3000",
    "http://127.0.0.1:3000"
]

# CORS for /api/*
CORS(app, resources={r"/api/*": {"origins": ALLOWED_ORIGINS, "supports_credentials": True}})

# Env / dataset defaults
DATA_FOLDER_PATH = os.environ.get("DATA_FOLDER_PATH", "data")
DATA_FILENAME = os.environ.get("DATA_FILENAME", "data/FC26_MomentumScout.csv")

# Default weights (fallback)
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

# projection rules
def years_to_project(age: int) -> int:
    if age <= 20: return 5
    if 21 <= age <= 25: return 4
    if 26 <= age <= 30: return 3
    if 31 <= age <= 35: return 2
    return 1

# dataset holder
player_data = None

def initialize_app():
    """Load CSV or Excel dataset and normalize common fields for FC26 compatibility."""
    global player_data
    fp = os.path.join(DATA_FOLDER_PATH, DATA_FILENAME)
    if not os.path.exists(fp):
        raise FileNotFoundError(f"Dataset not found at {fp} (set DATA_FOLDER_PATH/DATA_FILENAME env vars)")

    try:
        if fp.lower().endswith('.csv'):
            # sometimes CSV contains weird encodings; utf-8-sig helps
            player_data = pd.read_csv(fp, encoding='utf-8-sig')
        else:
            player_data = pd.read_excel(fp)
    except Exception as e:
        print("Error reading dataset:", e)
        raise

    # Normalize column names (trim and lower-case where helpful)
    player_data.columns = [str(c).strip() for c in player_data.columns]

    # Add aliases expected by older code
    # FC26 has 'long_name' and 'short_name' — ensure 'player_name' exists
    if 'player_name' not in player_data.columns and 'long_name' in player_data.columns:
        player_data['player_name'] = player_data['long_name']

    # Ensure numeric columns coerced
    NUMERIC_COLS = [
        'overall','potential','age','value_eur','pace','shooting',
        'passing','dribbling','defending','physic','wage_eur'
    ]
    for col in NUMERIC_COLS:
        if col in player_data.columns:
            player_data[col] = pd.to_numeric(player_data[col], errors='coerce').fillna(0)

    print(f"✅ Dataset loaded. Total players: {len(player_data)}")
    # optional quick head for debug
    print(player_data.head(3)[[c for c in ['short_name','player_name','overall','club_name'] if c in player_data.columns]])

# ----------------- Helpers -----------------
def is_nan_like(v):
    try:
        return (isinstance(v, float) and (math.isnan(v) or math.isinf(v))) or (pd.isna(v))
    except Exception:
        return False

def clean_json_value(v):
    """Converts numpy types, nan/inf to JSON-friendly types."""
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        if math.isnan(float(v)) or math.isinf(float(v)):
            return None
        return float(v)
    if isinstance(v, np.bool_):
        return bool(v)
    if is_nan_like(v):
        return None
    return v

def clean_json(data):
    """Recursively clean dict/list values for JSON serialization."""
    if isinstance(data, dict):
        return {k: clean_json(data[k]) for k in data}
    if isinstance(data, list):
        return [clean_json(v) for v in data]
    return clean_json_value(data)

def compute_score_for_player(row, position, user_weights=None):
    """Score a player row (row can be pandas Series or dict-like)."""
    # convert to dictionary for safe .get usage
    if hasattr(row, "to_dict"):
        row_dict = row.to_dict()
    else:
        row_dict = dict(row)
    base_weights = POSITION_WEIGHTS.get(position, POSITION_WEIGHTS.get('CM', {})).copy()
    if user_weights:
        for k, v in user_weights.items():
            if v is not None:
                try:
                    base_weights[k] = float(v)
                except Exception:
                    pass
    total_w = sum(base_weights.values()) if base_weights else 1
    if total_w == 0: total_w = 1
    score = 0.0
    for attr, w in base_weights.items():
        val = row_dict.get(attr, 0)
        try:
            val_num = float(val) if val is not None else 0.0
        except:
            val_num = 0.0
        norm = val_num / 99.0
        score += norm * (w / total_w)
    return round(score * 100, 4)

def project_player(row_like, years:int):
    """Return list of projections given a row-like (dict or Series) or simple values."""
    if hasattr(row_like, "get") or hasattr(row_like, "to_dict"):
        r = row_like.to_dict() if hasattr(row_like, "to_dict") else dict(row_like)
        ovr = int(r.get('overall') or 0)
        pot = int(r.get('potential') or ovr)
        age = int(r.get('age') or 0)
        value = float(r.get('value_eur') or 0)
    else:
        # assume row_like is tuple/list [overall, potential, age, value]
        ovr = int(row_like[0] or 0)
        pot = int(row_like[1] or ovr)
        age = int(row_like[2] or 0)
        value = float(row_like[3] or 0)

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

# ----------------- Routes -----------------

@app.route('/api/submit_demo', methods=['POST'])
def submit_demo():
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"success": False, "message": "No data received"}), 400
    full_name = data.get("fullName")
    email = data.get("email")
    organization = data.get("organization")
    print(f"📩 Demo request from {full_name} ({email}) - {organization}")
    return jsonify({"success": True, "message": "Demo request received!"})

# OPTIONS preflight handler (ensures 200 for preflight)
@app.before_request
def handle_options_preflight():
    if request.method == "OPTIONS" and request.path.startswith("/api/"):
        resp = make_response("")
        origin = request.headers.get("Origin", "")
        if origin in ALLOWED_ORIGINS:
            resp.headers["Access-Control-Allow-Origin"] = origin
            resp.headers["Access-Control-Allow-Credentials"] = "true"
            resp.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
            resp.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        return resp

@app.after_request
def add_cors_headers(response):
    origin = request.headers.get("Origin")
    if origin and origin in ALLOWED_ORIGINS:
        response.headers["Access-Control-Allow-Origin"] = origin
        response.headers["Access-Control-Allow-Credentials"] = "true"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response

@app.route("/api/debug_dataset", methods=["GET"])
def debug_dataset():
    """Small debugging endpoint to confirm dataset is loaded and fields exist."""
    if player_data is None:
        return jsonify({"loaded": False, "message": "No dataset loaded"}), 500
    cols = list(player_data.columns)
    sample = player_data.head(5).to_dict(orient='records')
    sample_clean = clean_json(sample)
    return jsonify({"loaded": True, "rows": len(player_data), "columns": cols, "sample": sample_clean})

@app.route("/api/search_player", methods=["POST", "OPTIONS"])
def api_search_player():
    if request.method == "OPTIONS":
        return "", 200
    try:
        payload = request.get_json(silent=True) or {}
        query = (payload.get("player_name") or payload.get("name") or "").strip()
        if not query:
            return jsonify([])
        q = str(query).lower()
        df = player_data
        name_cols = [c for c in ['short_name', 'long_name', 'player_name'] if c in df.columns]
        # build mask
        mask = pd.Series([False]*len(df))
        if name_cols:
            for c in name_cols:
                mask = mask | df[c].astype(str).str.lower().str.contains(q, na=False)
        else:
            mask = df.astype(str).apply(lambda r: r.str.lower().str.contains(q, na=False).any(), axis=1)

        results = df[mask].head(20)
        out = []
        for _, row in results.iterrows():
            # row is a Series
            age = int(row.get('age') or 0)
            years = years_to_project(age)
            projections = project_player(row, years)
            last_proj_value = projections[-1]['projected_value_eur'] if projections else int(row.get('value_eur') or 0)
            neg = negotiation_range(int(row.get('value_eur') or 0), last_proj_value)
            weekly_wage = row.get('wage_eur', 0)
            yearly_wage_gbp = weekly_wage * 52 if weekly_wage else 0

            full_attrs = {
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
                "full_attributes": full_attrs,
                "raw_attributes": clean_json(row.to_dict())
            })

        return jsonify(clean_json(out))
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

        # Apply filters (min,max)
        for key, rng in filters.items():
            try:
                if isinstance(rng, (list, tuple)) and len(rng) >= 2:
                    lo = float(rng[0])
                    hi = float(rng[1])
                    target_col = key.lower()
                    if 'value' in target_col:
                        target_col = 'value_eur'
                    elif 'overall' in target_col:
                        target_col = 'overall'
                    elif 'age' in target_col:
                        target_col = 'age'
                    if target_col in df.columns:
                        df = df[(pd.to_numeric(df[target_col], errors='coerce').fillna(0).astype(float) >= lo) & (pd.to_numeric(df[target_col], errors='coerce').fillna(0).astype(float) <= hi)]
                    else:
                        print(f"Skipping filter for '{key}': column '{target_col}' not found.")
            except Exception as e:
                print(f"CRASH POINT: Filter error on key '{key}' with range {rng}. Error: {e}")
                return jsonify({"players": []}), 200

        # Score players
        scored = []
        for _, row in df.iterrows():
            try:
                score = compute_score_for_player(row, position, user_weights=user_weights)
            except Exception:
                score = 0
            scored.append((score, row))

        scored_sorted = sorted(scored, key=lambda x: x[0], reverse=True)

        players_out = []
        for score, row in scored_sorted[:50]:
            age = int(row.get('age') or 0)
            years = years_to_project(age)
            projections = project_player(row, years) or []
            last_proj_value = projections[-1]['projected_value_eur'] if projections else int(row.get('value_eur') or 0)
            neg = negotiation_range(int(row.get('value_eur') or 0), last_proj_value)
            weekly_wage = row.get('wage_eur', 0)
            yearly_wage_gbp = weekly_wage * 52 if weekly_wage else 0

            full_attrs = {
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
                "full_attributes": full_attrs,
                "raw_attributes": clean_json(row.to_dict())
            })

        return jsonify({"players": clean_json(players_out[:5])})
    except Exception as e:
        print("Error in /api/find_players:", e)
        return jsonify({"players": [], "message": f"Internal Server Error: {e}"}), 500

@app.route("/assets/<path:filename>")
def serve_assets(filename):
    return send_from_directory(os.path.join(app.root_path, "public/assets"), filename)

# --- Main ---
if __name__ == "__main__":
    print("🚀 Initializing backend...")
    initialize_app()
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
