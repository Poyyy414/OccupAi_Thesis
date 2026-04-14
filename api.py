"""
api.py — OccupAI REST API Routes
==================================
All JSON API endpoints separated from app.py.

NOTE: /api/stats, /api/snapshot, /api/predictions all live HERE only.
      They must NOT be re-defined in app.py (causes route conflicts).
"""

from flask import Blueprint, jsonify, session
from database import get_db
from datetime import datetime

api_bp = Blueprint("api", __name__, url_prefix="/api")

# ── Injected from app.py ──
_state      = None
_snap       = None
_history    = None
_state_lock = None
_snap_lock  = None


def init_api(state, snap, history, state_lock, snap_lock):
    global _state, _snap, _history, _state_lock, _snap_lock
    _state      = state
    _snap       = snap
    _history    = history
    _state_lock = state_lock
    _snap_lock  = snap_lock


# ── Health check (public) ──
@api_bp.route("/status")
def api_status():
    with _state_lock:
        return jsonify({
            "status":   "online" if _state["running"] else "starting",
            "camera":   "webcam",
            "fps":      _state["fps"],
            "version":  "4.4",
            "location": "105 Peñafrancia Ave, Naga City"
        })


# ── Current occupancy (public) ──
@api_bp.route("/stats")
def api_stats():
    with _state_lock:
        return jsonify({
            "occupied":      _state["occupied"],
            "free":          _state["free"],
            "total":         _state["total"],
            "occupancy_pct": _state["occupancy_pct"],
            "lot_full":      _state["lot_full"],
            "timestamp":     _state["timestamp"],
            "fps":           _state["fps"]
        })


# ── Latest camera frame (protected) ──
@api_bp.route("/snapshot")
def api_snapshot():
    if "user_id" not in session:
        return jsonify({"error": "unauthorized"}), 401
    with _snap_lock:
        b64 = _snap["frame_b64"]
        ts  = _snap["timestamp"]
    print(f"DEBUG /api/snapshot: len={len(b64)}, ts={ts!r}")
    return jsonify({"image": b64, "timestamp": ts})


# ── History log (protected) ──
@api_bp.route("/history")
def api_history():
    if "user_id" not in session:
        return jsonify({"error": "unauthorized"}), 401
    return jsonify(list(_history))


# ── Full occupancy + slot detail (protected) ──
@api_bp.route("/occupancy")
def api_occupancy():
    if "user_id" not in session:
        return jsonify({"error": "unauthorized"}), 401
    with _state_lock:
        return jsonify({
            "occupied":      _state["occupied"],
            "free":          _state["free"],
            "total":         _state["total"],
            "occupancy_pct": _state["occupancy_pct"],
            "lot_full":      _state["lot_full"],
            "slot_states":   _state["slot_states"],
            "fps":           _state["fps"],
            "timestamp":     _state["timestamp"]
        })


# ── Peak hour predictions (protected) ──
@api_bp.route("/predictions")
def api_predictions():
    if "user_id" not in session:
        return jsonify({"error": "unauthorized"}), 401
    try:
        conn = get_db(); cur = conn.cursor()
        cur.execute("""
            SELECT EXTRACT(HOUR FROM created_at) AS hour,
                   AVG(occupancy_pct) AS avg_pct
            FROM parking_logs
            WHERE created_at >= NOW() - INTERVAL '7 days'
            GROUP BY hour ORDER BY hour
        """)
        rows = cur.fetchall(); cur.close(); conn.close()
        hourly = {str(int(r["hour"])): round(float(r["avg_pct"]), 1) for r in rows}
        for h in range(24):
            hourly.setdefault(str(h), 0.0)
        peak_hour  = max(hourly, key=lambda h: hourly[h])
        peak_val   = hourly[peak_hour]
        busy_days  = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][:3]
        quiet_days = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][5:]
        return jsonify({
            "hourly_est": hourly,
            "peak_hour":  int(peak_hour),
            "peak_label": f"{peak_hour}:00 ({peak_val:.0f}%)",
            "busy_days":  busy_days,
            "quiet_days": quiet_days
        })
    except Exception as e:
        print(f"⚠ predictions: {e}")
        hourly = {str(h): 0.0 for h in range(24)}
        return jsonify({
            "hourly_est": hourly,
            "peak_hour":  8,
            "peak_label": "N/A",
            "busy_days":  [],
            "quiet_days": []
        })