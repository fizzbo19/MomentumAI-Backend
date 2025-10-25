# backend.py
"""
MomentumScout Backend – FC26-ready
Fully JSON-safe, with CORS, clean data handling, and error protection.
"""

import os
import math
import json
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# ----------------- App & CORS -----------------
app = Flask(__name__, static_folder="public", static_url_path="/public")

ALLOWED_ORIGINS = [
    "https://momentumai-frontend.onrender.com",
    "https://momentum-ai-io.netlify.app",
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

# ----------------- Helper & Safe Functions -----------------

def is_nan_like(v):
    try:
        return (isinstance(v, float) and (math.isnan(v) or math.isinf(v))) or pd.isna(v)
    except Exception:
        return False

def clean_json_value(v):
    """Convert NaN, inf, and numpy types to JSON-safe primitives"""
    if isinstance(v, (np.integer,)): return int(v)
    if isinstance(v, (np.floating, float)):
        if math.isnan(v) or math.isinf(v): return None
        return float(v)
    if isinstance(v, np.bool_): return bool(v)
    if is_nan_like(v): return None
    return v

def clean_json(data):
    """Recursively clean dict/list structures for JSON serialization."""
    if isinstance(data, dict):
        return {k: clean_json(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [clean_json(v) for v in data]
    else:
        return clean_json_value(data)

def safe_int(val, default=0):
    try:
        if val is None or (isinstance(val, float) and (math.isnan(val) or math.isinf(val))):
            return default
        return int(val)
    except:
        return default

def safe_float(val, default=0.0):
    try:
        if val is None or (isinstance(val, float) and (math.isnan(val) or math.isinf(val))):
            return default
        return float(val)
    except:
        return default

# ----------------- Core Computations -----------------
def years_to_project(age: int) -> int:
    if age <= 20: return 5
    if 21 <= age <= 25: return 4
    if 26 <= age <= 30: return 3
    if 31 <= age <= 35: return 2
    return 1

def compute_score_for_player(row, position, user_weights=None):
    base_weights = POSITION_WEIGHTS.get(position, POSITION_WEIGHTS['CM']).copy()
    if user_weights:
        for k, v in user_weights.items():
            try: base_weights[k] = float(v)
            except: pass

    total_w = sum(base_weights.values()) or 1
    score = 0.0
    for attr, w in base_weights.items():
        val = safe_float(row.get(attr, 0))
        norm = val / 99.0
        score += norm * (w / total_w)
    return round(score * 100, 2)

def project_player(player_row, years=3):
    attrs = player_row.to_dict()
    age = safe_int(attrs.get('age'), 20)

    if age <= 20: growth, delta = 1.12, 1.8
    elif age <= 25: growth, delta = 1.08, 1.2
    elif age <= 30: growth, delta = 1.05, 0.5
    elif age <= 35: growth, delta = 1.03, -0.5
    else: growth, delta = 1.01, -1.0

    projections = []
    current_value = safe_float(attrs.get('value_eur'), 0)
    for i in range(1, years + 1):
        current_value *= growth
        projections.append({
            "year": i,
            "projected_value_eur": round(current_value),
            "overall": min(99, max(40, safe_float(attrs.get('overall'), 0) + delta * i))
        })
    return projections

def negotiation_range(current_value, projected_value):
    cur = safe_int(current_value, 0)
    proj = safe_int(projected_value, cur)
    if cur <= 0:
        return {"min_offer": 0, "max_offer": 0}
    return {
        "min_offer": int(cur * 0.7),
        "max_offer": int(max(proj, cur) * 1.05)
    }

# ----------------- Dataset Initialization -----------------
def initialize_app():
    global player_data
    fp = os.path.join(DATA_FOLDER_PATH, DATA_FILENAME)
    if not os.path.exists(fp):
        raise FileNotFoundError(f"Dataset not found at {fp}")

    try:
        if fp.lower().endswith('.csv'):
            player_data = pd.read_csv(fp, encoding='utf-8-sig')
        else:
            player_data = pd.read_excel(fp)
    except Exception as e:
        print("Error loading dataset:", e)
        raise

    player_data.columns = [str(c).strip() for c in player_data.columns]
    if 'player_name' not in player_data.columns and 'long_name' in player_data.columns:
        player_data['player_name'] = player_data['long_name']

    numeric_cols = ['overall','potential','age','value_eur','pace','shooting','passing','dribbling','defending','physic','wage_eur']
    for col in numeric_cols:
        if col in player_data.columns:
            player_data[col] = pd.to_numeric(player_data[col], errors='coerce').fillna(0)

    print(f"✅ Dataset loaded successfully: {len(player_data)} players")

# ----------------- API Routes -----------------
@app.route("/api/search_player", methods=["POST", "OPTIONS"])
def api_search_player():
    if request.method == "OPTIONS":
        return "", 200
    try:
        payload = request.get_json(silent=True) or {}
        query = (payload.get("player_name") or payload.get("name") or "").strip().lower()
        if not query:
            return jsonify([])

        df = player_data.copy()
        mask = pd.Series(False, index=df.index)
        for c in ['short_name', 'long_name', 'player_name']:
            if c in df.columns:
                mask |= df[c].astype(str).str.lower().str.contains(query, na=False)

        results = df[mask].head(20)
        out = []
        for _, row in results.iterrows():
            age = safe_int(row.get('age'))
            years = years_to_project(age)
            projections = project_player(row, years)
            current_val = safe_int(row.get('value_eur'))
            last_proj_val = safe_int(projections[-1].get('projected_value_eur', current_val))
            neg = negotiation_range(current_val, last_proj_val)
            score = compute_score_for_player(row, row.get('club_position') or 'CM')

            player_json = row.to_dict()
            player_json.update({
                "projections": projections,
                "negotiation": neg,
                "momentum_score": score
            })
            out.append(clean_json(player_json))

        return jsonify(out)

    except Exception as e:
        print("Error in /api/search_player:", e)
        return jsonify({"message": f"Internal Server Error: {e}"}), 500


@app.route("/api/find_players", methods=["POST", "OPTIONS"])
def api_find_players():
    if request.method == "OPTIONS":
        return "", 200
    try:
        payload = request.get_json(silent=True) or {}
        position = (payload.get("position") or "CM").upper()
        filters = payload.get("filters") or {}
        user_weights = payload.get("weights") or {}
        df = player_data.copy()

        for key, rng in filters.items():
            try:
                if isinstance(rng, (list, tuple)) and len(rng) >= 2:
                    lo, hi = float(rng[0]), float(rng[1])
                    if key in df.columns:
                        df = df[
                            (pd.to_numeric(df[key], errors='coerce').fillna(0) >= lo) &
                            (pd.to_numeric(df[key], errors='coerce').fillna(0) <= hi)
                        ]
            except Exception as e:
                print(f"Filter error on {key}: {e}")

        scored = []
        for _, row in df.iterrows():
            score = compute_score_for_player(row, position, user_weights)
            scored.append((score, row))

        scored_sorted = sorted(scored, key=lambda x: x[0], reverse=True)
        players_out = []
        for score, row in scored_sorted[:50]:
            age = safe_int(row.get('age'))
            years = years_to_project(age)
            projections = project_player(row, years)
            current_val = safe_int(row.get('value_eur'))
            last_proj_val = safe_int(projections[-1].get('projected_value_eur', current_val))
            neg = negotiation_range(current_val, last_proj_val)

            player_json = row.to_dict()
            player_json.update({
                "projections": projections,
                "negotiation": neg,
                "momentum_score": score
            })
            players_out.append(clean_json(player_json))

        return jsonify({"players": players_out[:5]})

    except Exception as e:
        print("Error in /api/find_players:", e)
        return jsonify({"players": [], "message": f"Internal Server Error: {e}"}), 500


@app.route("/api/submit_demo", methods=["POST"])
def submit_demo():
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"success": False, "message": "No data received"}), 400
    print(f"📩 Demo request from {data.get('fullName')} ({data.get('email')}) - {data.get('organization')}")
    return jsonify({"success": True, "message": "Demo request received!"})


@app.route("/assets/<path:filename>")
def serve_assets(filename):
    return send_from_directory(os.path.join(app.root_path, "public/assets"), filename)

# ----------------- Main -----------------
if __name__ == "__main__":
    print("🚀 Initializing MomentumScout Backend...")
    initialize_app()
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
