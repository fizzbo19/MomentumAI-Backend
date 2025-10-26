# backend.py
"""
MomentumScout Backend (FC26 Edition)
------------------------------------
✅ Supports FC26 dataset (CSV)
✅ Handles NaN values safely
✅ Includes projections, negotiation, and player scoring
✅ Full CORS support for Render + frontend
"""

import os
import math
import json
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# ------------------ Flask Setup ------------------
app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})  # Open for testing, tighten later

# ------------------ Config ------------------
DATA_FILENAME = os.environ.get("DATA_FILENAME", "data/FC26_MomentumScout.csv")
DATA_FOLDER_PATH = os.environ.get("DATA_FOLDER_PATH", "data")

# Resolve absolute path
DATA_PATH = (
    DATA_FILENAME if os.path.isabs(DATA_FILENAME)
    else os.path.join(DATA_FOLDER_PATH, os.path.basename(DATA_FILENAME))
)

player_data = None

# ------------------ Helpers ------------------
def safe_int(val, default=0):
    try:
        if val is None or (isinstance(val, float) and math.isnan(val)):
            return default
        return int(val)
    except:
        return default

def safe_float(val, default=0.0):
    try:
        if val is None or (isinstance(val, float) and math.isnan(val)):
            return default
        return float(val)
    except:
        return default

def clean_json_value(v):
    if isinstance(v, (np.integer,)): return int(v)
    if isinstance(v, (np.floating,)): return None if math.isnan(v) else float(v)
    if isinstance(v, np.bool_): return bool(v)
    return v

def clean_json(d):
    if isinstance(d, dict):
        return {k: clean_json(v) for k, v in d.items()}
    elif isinstance(d, list):
        return [clean_json(v) for v in d]
    elif isinstance(d, float):
        return None if math.isnan(d) else d
    else:
        return d

# ------------------ Core Logic ------------------
def years_to_project(age: int) -> int:
    if age <= 20: return 5
    elif age <= 25: return 4
    elif age <= 30: return 3
    elif age <= 35: return 2
    return 1

def compute_score_for_player(row, position="CM", user_weights=None):
    base_weights = {
        'GK': {'goalkeeping_diving':20, 'goalkeeping_reflexes':20, 'goalkeeping_positioning':20},
        'CB': {'defending':50,'physic':25,'pace':10,'passing':10,'dribbling':5},
        'LB': {'pace':30,'defending':25,'passing':20,'physic':15,'dribbling':10},
        'RB': {'pace':30,'defending':25,'passing':20,'physic':15,'dribbling':10},
        'CDM': {'defending':35,'passing':25,'physic':25,'pace':10,'dribbling':5},
        'CM': {'passing':30,'dribbling':25,'defending':15,'shooting':15,'pace':10},
        'CAM': {'passing':30,'dribbling':30,'shooting':25,'pace':15},
        'LW': {'pace':35,'dribbling':30,'shooting':20,'passing':15},
        'RW': {'pace':35,'dribbling':30,'shooting':20,'passing':15},
        'ST': {'shooting':40,'pace':25,'dribbling':20,'physic':15}
    }.get(position, {})

    if user_weights:
        base_weights.update(user_weights)

    total_w = sum(base_weights.values()) or 1
    score = 0.0
    for attr, weight in base_weights.items():
        val = safe_float(row.get(attr), 0.0)
        score += (val / 100.0) * (weight / total_w)
    return round(score * 100, 2)

def project_player(player_row, years=3):
    """Simulates player growth or decline"""
    attrs = player_row.to_dict()
    current_val = safe_float(attrs.get('value_eur'), 0)
    overall = safe_float(attrs.get('overall'), 70)
    age = safe_int(attrs.get('age'), 22)

    projections = []
    for y in range(1, years + 1):
        if age < 26:
            overall += 1.2
            current_val *= 1.10
        elif 26 <= age < 31:
            overall += 0.3
            current_val *= 1.05
        else:
            overall -= 0.5
            current_val *= 0.97

        projections.append({
            "year": y,
            "projected_overall": round(overall, 1),
            "projected_value_eur": int(current_val)
        })
        age += 1

    return projections

def negotiation_range(current_value: int, projected_value: int):
    current_value = safe_int(current_value, 0)
    projected_value = safe_int(projected_value, current_value)
    if current_value <= 0:
        return {"min_offer": 0, "max_offer": 0}
    return {
        "min_offer": int(current_value * 0.7),
        "max_offer": int(max(current_value, projected_value) * 1.1)
    }

# ------------------ Dataset Loader ------------------
def initialize_app():
    global player_data
    print(f"📂 Loading dataset from {DATA_PATH}")
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")

    try:
        player_data = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    except Exception as e:
        print("❌ Error loading dataset:", e)
        raise
        # 🧠 Add Sofifa player image URLs (for face display)
    if "sofifa_id" in player_data.columns:
        player_data["player_face_url"] = player_data["sofifa_id"].apply(
            lambda x: f"https://cdn.sofifa.net/players/{int(x)//1000:03d}/{int(x)%1000:03d}/24.webp"
            if pd.notna(x) else None
        )
    else:
        # If FC26 data doesn’t have sofifa_id, try fallback with player_url
        if "player_url" in player_data.columns:
            player_data["player_face_url"] = player_data["player_url"].apply(
                lambda url: f"https://cdn.sofifa.net{url.split('/')[-2]}/{url.split('/')[-1]}/24.webp"
                if isinstance(url, str) and "sofifa.com" in url else None
            )


    player_data.columns = [c.strip() for c in player_data.columns]
    if 'player_name' not in player_data.columns and 'long_name' in player_data.columns:
        player_data['player_name'] = player_data['long_name']

    print(f"✅ Loaded {len(player_data)} players from FC26_MomentumScout.csv")
    print(player_data.head(3)[['short_name', 'club_name', 'overall']])

# ------------------ API Routes ------------------
@app.route("/api/search_player", methods=["POST", "OPTIONS"])
def api_search_player():
    if request.method == "OPTIONS":
        return "", 200
    try:
        payload = request.get_json(silent=True) or {}
        query = (payload.get("player_name") or payload.get("name") or "").strip().lower()
        if not query:
            return jsonify({"players": []})

        # ✅ Make sure player_face_url exists
        global player_data
        df = player_data.copy()
        if "player_face_url" not in df.columns or df["player_face_url"].isnull().all():
            df["player_face_url"] = df["sofifa_id"].apply(
                lambda x: f"https://cdn.sofifa.net/players/{int(x)//1000:03d}/{int(x)%1000:03d}/24.webp"
                if pd.notna(x) else ""
            )

        # ✅ Search by multiple name fields
        mask = (
            df["short_name"].astype(str).str.lower().str.contains(query, na=False)
            | df["long_name"].astype(str).str.lower().str.contains(query, na=False)
            | df["player_name"].astype(str).str.lower().str.contains(query, na=False)
        )
        results = df[mask].head(15)

        out = []
        for _, row in results.iterrows():
            age = safe_int(row.get("age"), 0)
            projections = project_player(row, years_to_project(age))
            current_val = safe_int(row.get("value_eur"), 0)
            last_proj_val = projections[-1]["projected_value_eur"] if projections else current_val
            neg = negotiation_range(current_val, last_proj_val)
            score = compute_score_for_player(row, row.get("club_position") or "CM")

            player_json = clean_json(row.to_dict())

            # ✅ Add all key FIFA attributes with new FC 26 naming
            attributes = {
                "Acceleration": safe_int(row.get("movement_acceleration"), 0),
                "Sprint Speed": safe_int(row.get("movement_sprint_speed"), 0),
                "Agility": safe_int(row.get("movement_agility"), 0),
                "Balance": safe_int(row.get("movement_balance"), 0),
                "Dribbling": safe_int(row.get("skill_dribbling"), 0),
                "Ball Control": safe_int(row.get("skill_ball_control"), 0),
                "Finishing": safe_int(row.get("attacking_finishing"), 0),
                "Short Passing": safe_int(row.get("attacking_short_passing"), 0),
                "Long Passing": safe_int(row.get("skill_long_passing"), 0),
                "Shot Power": safe_int(row.get("power_shot_power"), 0),
                "Stamina": safe_int(row.get("power_stamina"), 0),
                "Strength": safe_int(row.get("power_strength"), 0),
                "Reactions": safe_int(row.get("movement_reactions"), 0),
                "Vision": safe_int(row.get("mentality_vision"), 0),
                "Composure": safe_int(row.get("mentality_composure"), 0),
                "Interceptions": safe_int(row.get("mentality_interceptions"), 0),
                "Standing Tackle": safe_int(row.get("defending_standing_tackle"), 0),
                "Sliding Tackle": safe_int(row.get("defending_sliding_tackle"), 0),
            }

            player_json.update({
                "momentum_score": score,
                "negotiation": neg,
                "projections": projections,
                "full_attributes": attributes,
                "player_face_url": row.get("player_face_url", ""),
            })
            out.append(player_json)

        # ✅ Return consistent JSON for frontend
        return jsonify({"players": out})

    except Exception as e:
        print("❌ Error in /api/search_player:", e)
        return jsonify({"message": f"Internal Server Error: {e}"}), 500




@app.route("/api/find_players", methods=["POST", "OPTIONS"])
def api_find_players():
    if request.method == "OPTIONS":
        return "", 200
    try:
        payload = request.get_json(silent=True) or {}
        position = (payload.get("club_position") or payload.get("position") or "CM").upper()
        filters = payload.get("filters") or {}
        df = player_data.copy()

        # ✅ Ensure player_face_url exists
        if "player_face_url" not in df.columns or df["player_face_url"].isnull().all():
            df["player_face_url"] = df["sofifa_id"].apply(
                lambda x: f"https://cdn.sofifa.net/players/{int(x)//1000:03d}/{int(x)%1000:03d}/24.webp"
                if pd.notna(x) else ""
            )

        # Apply numeric filters
        for key, rng in filters.items():
            if isinstance(rng, (list, tuple)) and len(rng) >= 2:
                lo, hi = safe_float(rng[0]), safe_float(rng[1])
                if key in df.columns:
                    df = df[(df[key] >= lo) & (df[key] <= hi)]

        # Build players list with projections, negotiation, attributes
        players = []
        for _, row in df.iterrows():
            score = compute_score_for_player(row, position)
            age = safe_int(row.get("age"), 0)
            projections = project_player(row, years_to_project(age))
            current_val = safe_int(row.get("value_eur"), 0)
            last_proj_val = projections[-1]["projected_value_eur"] if projections else current_val
            neg = negotiation_range(current_val, last_proj_val)

            # ✅ Build full attributes dynamically
            attributes = {
                "Acceleration": safe_int(row.get("movement_acceleration"), 0),
                "Sprint Speed": safe_int(row.get("movement_sprint_speed"), 0),
                "Agility": safe_int(row.get("movement_agility"), 0),
                "Balance": safe_int(row.get("movement_balance"), 0),
                "Dribbling": safe_int(row.get("skill_dribbling"), 0),
                "Ball Control": safe_int(row.get("skill_ball_control"), 0),
                "Finishing": safe_int(row.get("attacking_finishing"), 0),
                "Short Passing": safe_int(row.get("attacking_short_passing"), 0),
                "Long Passing": safe_int(row.get("skill_long_passing"), 0),
                "Shot Power": safe_int(row.get("power_shot_power"), 0),
                "Stamina": safe_int(row.get("power_stamina"), 0),
                "Strength": safe_int(row.get("power_strength"), 0),
                "Reactions": safe_int(row.get("movement_reactions"), 0),
                "Vision": safe_int(row.get("mentality_vision"), 0),
                "Composure": safe_int(row.get("mentality_composure"), 0),
                "Interceptions": safe_int(row.get("mentality_interceptions"), 0),
                "Standing Tackle": safe_int(row.get("defending_standing_tackle"), 0),
                "Sliding Tackle": safe_int(row.get("defending_sliding_tackle"), 0),
            }

            player_json = clean_json(row.to_dict())
            player_json.update({
                "momentum_score": score,
                "negotiation": neg,
                "projections": projections,
                "full_attributes": attributes,
                "player_face_url": row.get("player_face_url", ""),
            })
            players.append(player_json)

        # Sort by momentum score and return top 10
        sorted_players = sorted(players, key=lambda x: x["momentum_score"], reverse=True)
        return jsonify({"players": sorted_players[:10]})

    except Exception as e:
        print("❌ Error in /api/find_players:", e)
        return jsonify({"players": [], "message": f"Internal Server Error: {e}"}), 500



# ------------------ Startup ------------------
if __name__ == "__main__":
    print("🚀 Starting MomentumScout Backend...")
    initialize_app()
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
