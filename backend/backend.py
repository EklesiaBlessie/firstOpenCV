"""
backend_server.py

Lightweight Flask backend for the Jetson Nano Ball Sort leaderboard.
Scores are stored permanently in a local JSON file (leaderboard.json),
so the leaderboard survives restarts of both the server and the game.

Endpoints
---------
POST /add_score
    Body (JSON): {"name": "Alice", "moves": 23}
    Saves one score entry.

GET /get_leaderboard
    Returns the top 5 scores, best (fewest moves) first.

Run it with:
    python3 backend_server.py

It listens on 0.0.0.0:5050, so the game can reach it either at
127.0.0.1:5050 (same device) or at the Jetson's network IP (other devices).
"""

import json
import os
import threading

from flask import Flask, request, jsonify

DB_FILE = "leaderboard.json"
TOP_N = 5

# Simple lock so two requests can't write to the file at the exact same time.
_file_lock = threading.Lock()

app = Flask(__name__)


def load_scores():
    """Read all saved scores from disk. Returns an empty list if the file
    doesn't exist yet or is unreadable."""
    if not os.path.exists(DB_FILE):
        return []
    try:
        with open(DB_FILE, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, ValueError):
        return []


def save_scores(scores):
    """Write the full list of scores back to disk."""
    with open(DB_FILE, "w") as f:
        json.dump(scores, f, indent=2)


@app.route("/add_score", methods=["POST"])
def add_score():
    data = request.get_json(silent=True)

    if not data or "name" not in data or "moves" not in data:
        return jsonify({"status": "error", "message": "Send JSON with 'name' and 'moves'"}), 400

    name = str(data["name"]).strip()[:20]
    if not name:
        name = "Player"

    try:
        moves = int(data["moves"])
    except (TypeError, ValueError):
        return jsonify({"status": "error", "message": "'moves' must be a whole number"}), 400

    with _file_lock:
        scores = load_scores()
        scores.append({"name": name, "moves": moves})
        save_scores(scores)

    return jsonify({"status": "ok", "message": "Score saved"}), 200


@app.route("/get_leaderboard", methods=["GET"])
def get_leaderboard():
    with _file_lock:
        scores = load_scores()

    # Fewer moves is a better score, so sort from smallest to largest.
    top_scores = sorted(scores, key=lambda s: s.get("moves", 999999))[:TOP_N]
    return jsonify({"status": "ok", "leaderboard": top_scores}), 200


if __name__ == "__main__":
    print(f"Ball Sort leaderboard server starting on port 5050, saving to '{DB_FILE}' ...")
    app.run(host="0.0.0.0", port=5050, debug=False, threaded=True, load_dotenv=False)
    