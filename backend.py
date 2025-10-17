"""
MomentumAI Backend – Render-ready version with proper CORS
"""
import os
import math
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, send_from_directory
import requests
from flask_cors import CORS

# --- Flask App Setup ---
app = Flask(__name__, static_folder="public")

app = Flask(__name__, static_folder="public")

CORS(app, resources={r"/*": {"origins": [
    "https://momentumai-frontend.onrender.com",
    "http://localhost:5000",  # local testing
    "http://127.0.0.1:5000"
]}}, supports_credentials=True)

# Enable CORS for your frontend only (replace with your frontend URL)
frontend_url = os.environ.get("FRONTEND_URL", "https://momentumai-frontend.onrender.com")
CORS(app, resources={r"/api/*": {"origins": frontend_url}})

# --- Environment Variables ---
GOOGLE_SCRIPT_URL = os.environ.get(
    "GOOGLE_SCRIPT_URL",
    "https://script.google.com/macros/s/YOUR_SCRIPT_ID/exec"
)
DATA_FOLDER_PATH = os.environ.get("DATA_FOLDER_PATH", "data")
DATA_FILENAME = os.environ.get(
    "DATA_FILENAME", "Career Mode player datasets - FIFA 15-22.xlsx"
)

player_data = None

# --- Initialize App & Load Dataset ---
def initialize_app():
    global player_data
    fp = os.path.join(DATA_FOLDER_PATH, DATA_FILENAME)
    if not os.path.exists(fp):
        raise FileNotFoundError(f"Dataset not found at {fp}")

    player_data = pd.read_excel(fp)
    for col in ['overall','potential','age','value_eur','pace','shooting','passing','dribbling','defending','physic','wage_eur']:
        if col in player_data.columns:
            player_data[col] = pd.to_numeric(player_data[col], errors='coerce').fillna(0)
    print(f"✅ Dataset loaded. Total players: {len(player_data)}")

# --- Helper Functions ---
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
@app.route("/api/submit_demo", methods=["POST", "OPTIONS"])
def submit_demo():
    if request.method == "OPTIONS":
        return "", 200
    try:
        data = request.json
        if not data:
            return jsonify({"success": False, "message": "No form data provided."}), 400
        response = requests.post(GOOGLE_SCRIPT_URL, json=data)
        response.raise_for_status()
        return jsonify({"success": True, "message": "Form submitted successfully."}), 200
    except requests.exceptions.RequestException as e:
        print(f"Error forwarding to Google Apps Script: {e}")
        return jsonify({"success": False, "message": "Error submitting form."}), 500

@app.route("/api/search_player", methods=["POST", "OPTIONS"])
def search_player():
    if request.method == "OPTIONS":
        return "", 200
    try:
        req_data = request.json
        player_name = req_data.get("player_name", "").lower()
        if not player_name:
            return jsonify([])

        filtered_players = player_data[player_data['short_name'].str.lower().str.contains(player_name)]
        result = sanitize_player_data(filtered_players.to_dict(orient='records'))
        return jsonify(result)
    except Exception as e:
        print("Error in search_player:", e)
        return jsonify([]), 500

@app.route("/assets/<path:filename>")
def serve_assets(filename):
    return send_from_directory(os.path.join(app.root_path, "public/assets"), filename)

# --- Main Execution ---
if __name__ == "__main__":
    print("🚀 Initializing backend...")
    initialize_app()
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
