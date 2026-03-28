"""
app.py — OccupAI Flask API v4.2
=================================
Run:
  python app.py

Auth Endpoints:
  POST /auth/register   → create driver account
  POST /auth/login      → get JWT token
  GET  /auth/me         → current user info
  PUT  /auth/profile    → update profile
  POST /auth/logout     → logout

API Endpoints:
  GET  /api/status      → system status (public)
  GET  /api/stats       → occupancy stats (public)
  GET  /api/occupancy   → full state (protected)
  GET  /api/snapshot    → camera frame (protected)
  GET  /api/history     → occupancy history (protected)
  GET  /api/predictions → hourly forecast (protected)
  POST /api/slots       → save slots (admin/owner)
  POST /api/slots/auto  → auto-detect slots (admin/owner)
"""

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from flask_jwt_extended import JWTManager, jwt_required, get_jwt
import cv2
import numpy as np
from ultralytics import YOLO
import threading
import time
import base64
import json
import os
from collections import deque
from datetime import datetime, timedelta

from auth import auth_bp
from database import get_db

app = Flask(__name__, static_folder='static')
CORS(app)

# ╔══════════════════════════════════════════════╗
# ║          JWT CONFIGURATION                  ║
# ╚══════════════════════════════════════════════╝
app.config["JWT_SECRET_KEY"]            = os.getenv("JWT_SECRET_KEY", "occupai_secret_change_this_2027")
app.config["JWT_ACCESS_TOKEN_EXPIRES"]  = timedelta(hours=24)
app.config["JWT_REFRESH_TOKEN_EXPIRES"] = timedelta(days=30)

jwt = JWTManager(app)
app.register_blueprint(auth_bp)

# ╔══════════════════════════════════════════════╗
# ║        CAMERA SETTINGS                      ║
# ╚══════════════════════════════════════════════╝
CAM_SOURCE = os.getenv("CAM_SOURCE", "webcam")

if CAM_SOURCE == "wifi":
    CAM_URL     = os.getenv("RTSP_URL", "rtsp://admin:password@192.168.1.100:554/stream")
    CAM_BACKEND = cv2.CAP_FFMPEG
elif CAM_SOURCE == "droidcam":
    CAM_URL     = "http://localhost:4747/video"
    CAM_BACKEND = cv2.CAP_FFMPEG
else:
    CAM_URL     = 0
    CAM_BACKEND = cv2.CAP_ANY

# ╔══════════════════════════════════════════════╗
# ║         PERFORMANCE SETTINGS                ║
# ╚══════════════════════════════════════════════╝
FEED_W        = 480
FEED_H        = 360
IMGSZ         = 256
YOLO_SKIP     = 5
JPEG_QUALITY  = 45
SNAPSHOT_RATE = 0.1
SLOTS_RELOAD  = 60

# ╔══════════════════════════════════════════════╗
# ║              CONFIGURATION                  ║
# ╚══════════════════════════════════════════════╝
VEHICLE_CLS  = {2, 3, 5, 7}
CONF_THRESH  = 0.25
IOU_THRESH   = 0.15

LOT_WIDTH_M  = 10.0
LOT_HEIGHT_M = 8.0
CAR_W_M      = 2.3
CAR_L_M      = 4.5
SLOT_MAR_M   = 0.3

PX_PER_M_X  = FEED_W / LOT_WIDTH_M
PX_PER_M_Y  = FEED_H / LOT_HEIGHT_M
SLOT_W_PX   = int(CAR_W_M  * PX_PER_M_X)
SLOT_H_PX   = int(CAR_L_M  * PX_PER_M_Y)
SLOT_MAR_PX = int(SLOT_MAR_M * min(PX_PER_M_X, PX_PER_M_Y))

HISTORY_LEN = 100

# ╔══════════════════════════════════════════════╗
# ║           SHARED STATE                      ║
# ╚══════════════════════════════════════════════╝
state = {
    "occupied": 0, "free": 0, "total": 0,
    "occupancy_pct": 0.0, "slot_states": [],
    "slots": [], "yolo_boxes": [],
    "fps": 0.0, "timestamp": "",
    "lot_full": False, "running": False,
}
snap               = {"frame_b64": "", "timestamp": ""}
history            = deque(maxlen=HISTORY_LEN)
state_lock         = threading.Lock()
snap_lock          = threading.Lock()
_latest_frame      = None
_latest_frame_lock = threading.Lock()


# ╔══════════════════════════════════════════════╗
# ║         DETECTION HELPERS                   ║
# ╚══════════════════════════════════════════════╝
def compute_iou(a, b):
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if not inter: return 0.0
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / union if union else 0.0


def smart_fit_slots(occupied_boxes):
    occ_map = np.zeros((FEED_H, FEED_W), dtype=np.uint8)
    buf = 15
    for (x1, y1, x2, y2) in occupied_boxes:
        occ_map[max(0, y1-buf):min(FEED_H, y2+buf),
                max(0, x1-buf):min(FEED_W, x2+buf)] = 1
    slots = []; taken = np.zeros((FEED_H, FEED_W), dtype=np.uint8)
    step_x = SLOT_W_PX + SLOT_MAR_PX
    step_y = SLOT_H_PX + SLOT_MAR_PX
    pad = 10
    for ry in range(pad, FEED_H - SLOT_H_PX - pad, step_y):
        for cx in range(pad, FEED_W - SLOT_W_PX - pad, step_x):
            x1, y1, x2, y2 = cx, ry, cx + SLOT_W_PX, ry + SLOT_H_PX
            if occ_map[y1:y2, x1:x2].sum() == 0 and taken[y1:y2, x1:x2].sum() == 0:
                slots.append([x1, y1, x2, y2])
                taken[y1:y2, x1:x2] = 1
    return slots


# ── Slots: load from DB (replaces slots.json) ──
def load_slots():
    try:
        conn = get_db()
        cur  = conn.cursor()
        cur.execute("SELECT slots FROM slot_config ORDER BY slot_id DESC LIMIT 1")
        row = cur.fetchone()
        cur.close(); conn.close()
        return row["slots"] if row else []
    except Exception as e:
        print(f"⚠ load_slots error: {e}")
        return []


# ── Slots: save to DB ──
def save_slots(slots):
    try:
        conn = get_db()
        cur  = conn.cursor()
        cur.execute("""
            UPDATE slot_config
            SET slots = %s, updated_at = %s
            WHERE slot_id = (SELECT slot_id FROM slot_config ORDER BY slot_id DESC LIMIT 1)
        """, (json.dumps(slots), datetime.utcnow()))
        conn.commit()
        cur.close(); conn.close()
    except Exception as e:
        print(f"⚠ save_slots error: {e}")


# ── Log occupancy to DB ──
def log_occupancy(occupied, free, total, pct, lot_full):
    try:
        conn = get_db()
        cur  = conn.cursor()
        cur.execute("""
            INSERT INTO parking_logs (occupied, free, total, occupancy_pct, lot_full)
            VALUES (%s, %s, %s, %s, %s)
        """, (occupied, free, total, round(pct, 1), lot_full))
        conn.commit()
        cur.close(); conn.close()
    except Exception as e:
        print(f"⚠ log_occupancy error: {e}")


# ╔══════════════════════════════════════════════╗
# ║     SNAPSHOT ENCODER THREAD                 ║
# ╚══════════════════════════════════════════════╝
def snapshot_encoder_loop():
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
    while True:
        time.sleep(SNAPSHOT_RATE)
        with _latest_frame_lock:
            frame = _latest_frame
        if frame is None:
            continue
        try:
            _, buf = cv2.imencode('.jpg', frame, encode_params)
            b64    = base64.b64encode(buf).decode('utf-8')
            ts     = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with snap_lock:
                snap["frame_b64"] = b64
                snap["timestamp"] = ts
        except Exception:
            pass


# ╔══════════════════════════════════════════════╗
# ║         DETECTION THREAD                    ║
# ╚══════════════════════════════════════════════╝
def detection_loop():
    global _latest_frame
    print("Loading YOLO model...")
    model = YOLO("yolov8n.pt")
    model(np.zeros((FEED_H, FEED_W, 3), dtype=np.uint8), imgsz=IMGSZ, verbose=False)
    print("Model ready.")

    cap = cv2.VideoCapture(CAM_URL, CAM_BACKEND)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FPS, 30)

    if not cap.isOpened():
        print("ERROR: Cannot open camera.")
        return

    print(f"Camera OK! ({CAM_SOURCE} {FEED_W}x{FEED_H})")

    saved_slots = load_slots()
    frame_idx   = 0
    yolo_boxes  = []
    fps_t       = time.time()
    fps_n       = 0
    fps_val     = 0.0

    with state_lock:
        state["running"] = True

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.005)
            continue

        frame      = cv2.resize(frame, (FEED_W, FEED_H))
        frame_idx += 1
        fps_n     += 1

        now = time.time()
        if now - fps_t >= 1.0:
            fps_val = fps_n / (now - fps_t)
            fps_n   = 0
            fps_t   = now

        with _latest_frame_lock:
            _latest_frame = frame

        if frame_idx % YOLO_SKIP == 0:
            res        = model(frame, imgsz=IMGSZ, verbose=False)[0]
            yolo_boxes = []
            if res.boxes is not None:
                for r in res.boxes:
                    if int(r.cls[0]) in VEHICLE_CLS and float(r.conf[0]) > CONF_THRESH:
                        x1, y1, x2, y2 = map(int, r.xyxy[0])
                        yolo_boxes.append([x1, y1, x2, y2])

        if frame_idx % SLOTS_RELOAD == 0:
            saved_slots = load_slots()

        active_slots = saved_slots if saved_slots else smart_fit_slots(yolo_boxes)
        occupied     = 0
        slot_states  = []
        for (sx1, sy1, sx2, sy2) in active_slots:
            occ = any(
                compute_iou((sx1, sy1, sx2, sy2), tuple(vb)) > IOU_THRESH
                for vb in yolo_boxes
            )
            slot_states.append(occ)
            occupied += int(occ)

        total    = len(active_slots)
        free     = total - occupied
        pct      = (occupied / total * 100) if total else 0
        lot_full = total > 0 and free == 0
        ts       = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Log to DB + in-memory history every 30 frames (~1s)
        if frame_idx % 30 == 0:
            history.append({
                "time":     ts,
                "occupied": occupied,
                "total":    total,
                "pct":      round(pct, 1)
            })
            log_occupancy(occupied, free, total, pct, lot_full)

        with state_lock:
            state.update({
                "occupied":      occupied,
                "free":          free,
                "total":         total,
                "occupancy_pct": round(pct, 1),
                "slot_states":   slot_states,
                "slots":         active_slots,
                "yolo_boxes":    yolo_boxes,
                "fps":           round(fps_val, 1),
                "timestamp":     ts,
                "lot_full":      lot_full,
            })

    cap.release()


# ╔══════════════════════════════════════════════╗
# ║        API ROUTES — PUBLIC                  ║
# ╚══════════════════════════════════════════════╝
@app.route('/api/status')
def api_status():
    with state_lock:
        running = state["running"]
        fps     = state["fps"]
    return jsonify({
        "status":   "online" if running else "starting",
        "camera":   CAM_SOURCE,
        "fps":      fps,
        "version":  "4.2",
        "system":   "OccupAI Monitor",
        "location": "105 Peñafrancia Ave, Naga City",
    })


@app.route('/api/stats')
def api_stats():
    """Public — anyone can check available slots"""
    with state_lock:
        return jsonify({
            "occupied":      state["occupied"],
            "free":          state["free"],
            "total":         state["total"],
            "occupancy_pct": state["occupancy_pct"],
            "lot_full":      state["lot_full"],
            "timestamp":     state["timestamp"],
            "fps":           state["fps"],
        })


# ╔══════════════════════════════════════════════╗
# ║     API ROUTES — PROTECTED (need JWT)       ║
# ╚══════════════════════════════════════════════╝
@app.route('/api/occupancy')
@jwt_required()
def api_occupancy():
    with state_lock:
        d = dict(state)
    d.pop("running", None)
    return jsonify(d)


@app.route('/api/snapshot')
@jwt_required()
def api_snapshot():
    with snap_lock:
        return jsonify({
            "image":     snap["frame_b64"],
            "timestamp": snap["timestamp"],
        })


@app.route('/api/history')
@jwt_required()
def api_history():
    return jsonify(list(history))


@app.route('/api/predictions')
@jwt_required()
def api_predictions():
    hour_dist = {
        6:0.03, 7:0.12, 8:0.18, 9:0.10, 10:0.06, 11:0.05,
        12:0.10, 13:0.08, 14:0.05, 15:0.04, 16:0.05,
        17:0.08, 18:0.04, 19:0.02
    }
    with state_lock:
        avg = state["total"] if state["total"] > 0 else 20
    hourly    = {str(h): round(avg * p) for h, p in hour_dist.items()}
    peak_hour = max(hour_dist, key=hour_dist.get)
    return jsonify({
        "peak_hour":  peak_hour,
        "peak_label": f"{peak_hour}:00 - {peak_hour+1}:00",
        "hourly_est": hourly,
        "busy_days":  ["Monday", "Tuesday", "Wednesday", "Thursday"],
        "quiet_days": ["Saturday", "Sunday"],
    })


@app.route('/api/slots', methods=['POST'])
@jwt_required()
def api_save_slots():
    """Admin/Owner only"""
    claims = get_jwt()
    if claims.get("role") not in ["admin", "owner"]:
        return jsonify({"success": False, "message": "Admin or Owner access required."}), 403
    data  = request.get_json()
    slots = data.get('slots', [])
    save_slots(slots)
    return jsonify({"success": True, "saved": len(slots)})


@app.route('/api/slots/auto', methods=['POST'])
@jwt_required()
def api_auto_slots():
    """Admin/Owner only"""
    claims = get_jwt()
    if claims.get("role") not in ["admin", "owner"]:
        return jsonify({"success": False, "message": "Admin or Owner access required."}), 403
    with state_lock:
        boxes = list(state["yolo_boxes"])
    auto = smart_fit_slots(boxes)
    save_slots(auto)
    return jsonify({"success": True, "slots": auto, "count": len(auto)})


@app.route('/')
def index():
    return send_from_directory('static', 'index.html')


# ╔══════════════════════════════════════════════╗
# ║                   MAIN                      ║
# ╚══════════════════════════════════════════════╝
if __name__ == '__main__':
    t1 = threading.Thread(target=detection_loop, daemon=True)
    t1.start()
    t2 = threading.Thread(target=snapshot_encoder_loop, daemon=True)
    t2.start()

    print("\n╔══════════════════════════════════════════╗")
    print("║     OccupAI Flask API  v4.2 + Auth       ║")
    print("╠══════════════════════════════════════════╣")
    print("║  Dashboard  : http://localhost:5000      ║")
    print("║  Register   : POST /auth/register        ║")
    print("║  Login      : POST /auth/login           ║")
    print("║  Profile    : GET  /auth/me              ║")
    print("╠══════════════════════════════════════════╣")
    print(f"║  Camera     : {CAM_SOURCE.upper():<27} ║")
    print("╚══════════════════════════════════════════╝\n")

    app.run(
        host='0.0.0.0',
        port=5000,
        debug=False,
        threaded=True,
        use_reloader=False,
    )