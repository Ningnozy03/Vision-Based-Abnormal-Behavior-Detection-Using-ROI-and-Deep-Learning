# api_server.py
import os
import json
import time
import csv
from flask import send_file

from flask import Flask, Response, jsonify, request, send_from_directory
from flask_cors import CORS

try:
    from .shared_state import status, recent_events, latest_jpg_lock
    import src.shared_state as shared_state
except ImportError:
    from shared_state import status, recent_events, latest_jpg_lock
    import shared_state


# ===== CONFIG =====
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "..", "outputs")
CONFIG_PATH = os.path.join(BASE_DIR, "..", "configs", "config.json")

LATEST_JPG_PATH = os.path.join(BASE_DIR, "..", "outputs", "live", "latest.jpg")
# ===== Flask App =====

app = Flask(__name__)
CORS(app)

# ===== Utils =====
def load_config():
    if not os.path.exists(CONFIG_PATH):
        return {}
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def save_config(cfg):
    os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)

# ===== MJPEG Stream (from AI shared_state) =====
def gen_frames():
    while True:
        if not os.path.exists(LATEST_JPG_PATH):
            time.sleep(0.05)
            continue

        try:
            with open(LATEST_JPG_PATH, "rb") as f:
                jpg = f.read()
        except Exception:
            time.sleep(0.02)
            continue

        if not jpg:
            time.sleep(0.02)
            continue

        # ✅ เพิ่ม Content-Length
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n"
            + f"Content-Length: {len(jpg)}\r\n\r\n".encode()
            + jpg
            + b"\r\n"
        )

        time.sleep(0.03)

@app.route("/stream")
def stream():
    return Response(
        gen_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )
# ===== Status =====
@app.route("/status")
def get_status():
    return jsonify({
        "connected": status.get("connected", False),
        "recording": status.get("recording", False),
        "last_event": status.get("last_event"),
    })

# ===== Events =====
@app.route("/events")
def get_events():
    csv_path = os.path.join(OUTPUT_DIR, "logs", "events.csv")

    if not os.path.exists(csv_path):
        return jsonify([])

    events = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            events.append(row)

    events = list(reversed(events))[:20]

    return jsonify(events)

# ===== Config =====
@app.route("/config", methods=["GET", "POST"])
def config_api():
    if request.method == "GET":
        return jsonify(load_config())

    data = request.json or {}
    cfg = load_config()
    cfg.update(data)
    save_config(cfg)
    return jsonify({"ok": True, "config": cfg})

# ===== Files (snapshots / clips) =====
@app.route("/files/<path:filename>")
def files(filename):
    full_path = os.path.join(OUTPUT_DIR, filename)
    return send_file(full_path, conditional=True)

@app.route("/debug/frame")
def debug_frame():
    with latest_jpg_lock:
        jpg = shared_state.latest_jpg
    return jsonify({
        "has_jpg": jpg is not None,
        "jpg_len": 0 if jpg is None else len(jpg),
        "shared_state_file": shared_state.__file__,
    })


if __name__ == "__main__":
    print("🚀 API Server running on http://0.0.0.0:8000")
    app.run(host="0.0.0.0", port=8000, threaded=True)
