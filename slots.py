"""
slots.py — Slot Management API Routes
"""

from flask import Blueprint, jsonify, session, request
import json
from datetime import datetime
from database import get_db

slots_bp = Blueprint("slots", __name__, url_prefix="/api/slots")

_state      = None
_state_lock = None


def init_slots(state, state_lock, smart_fit_fn):
    global _state, _state_lock, _smart_fit
    _state      = state
    _state_lock = state_lock
    _smart_fit  = smart_fit_fn


def _save_slots(slots):
    try:
        conn = get_db(); cur = conn.cursor()
        cur.execute("""
            UPDATE slot_config SET slots=%s, updated_at=%s
            WHERE slot_id=(
                SELECT slot_id FROM slot_config
                ORDER BY slot_id DESC LIMIT 1)""",
            (json.dumps(slots), datetime.utcnow()))
        conn.commit(); cur.close(); conn.close()
    except Exception as e:
        print(f"⚠ save_slots: {e}")


@slots_bp.route("/auto", methods=["POST"])
def api_auto_slots():
    """POST /api/slots/auto — auto-generate slots from YOLO boxes"""
    if "user_id" not in session:
        return jsonify({"success": False, "message": "Login required."}), 401
    if session.get("role") not in ["admin", "owner"]:
        return jsonify({"success": False, "message": "Admin or Owner only."}), 403

    with _state_lock:
        boxes = list(_state["yolo_boxes"])

    auto = _smart_fit(boxes)
    _save_slots(auto)
    return jsonify({"success": True, "slots": auto, "count": len(auto)})


@slots_bp.route("/save", methods=["POST"])
def api_save_slots():
    """POST /api/slots/save — save custom slots"""
    if "user_id" not in session:
        return jsonify({"success": False, "message": "Login required."}), 401
    if session.get("role") not in ["admin", "owner"]:
        return jsonify({"success": False, "message": "Admin or Owner only."}), 403

    data  = request.get_json()
    slots = data.get("slots", [])
    _save_slots(slots)
    return jsonify({"success": True, "count": len(slots)})


@slots_bp.route("/load", methods=["GET"])
def api_load_slots():
    """GET /api/slots/load — get current slot config"""
    if "user_id" not in session:
        return jsonify({"error": "unauthorized"}), 401
    try:
        conn = get_db(); cur = conn.cursor()
        cur.execute("SELECT slots FROM slot_config ORDER BY slot_id DESC LIMIT 1")
        row = cur.fetchone(); cur.close(); conn.close()
        return jsonify({"slots": row["slots"] if row else [], "count": len(row["slots"]) if row else 0})
    except Exception as e:
        return jsonify({"error": str(e)}), 500