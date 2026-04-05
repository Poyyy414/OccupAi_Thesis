"""
app.py — OccupAI Flask API v4.3 + Jinja2 Templates
=====================================================
Run: python app.py

Pages (Jinja2):
  GET  /              → redirect to /login or /dashboard
  GET  /login         → login page
  GET  /register      → register page
  GET  /dashboard     → driver dashboard (session required)
  GET  /admin         → admin dashboard (admin/owner only)
  POST /do-login      → form login handler
  POST /do-register   → form register handler
  GET  /logout        → logout handler

API (JSON):
  GET  /api/status    → public
  GET  /api/stats     → public
  GET  /api/snapshot  → session protected
  GET  /api/history   → session protected
  GET  /api/predictions → public
  POST /api/slots/auto → admin/owner only
"""

from flask import (Flask, render_template, request, redirect,
                   url_for, session, jsonify, flash)
from flask_cors import CORS
from flask_jwt_extended import JWTManager
import cv2
import numpy as np
from ultralytics import YOLO
import bcrypt
import threading
import time
import base64
import json
import os
from collections import deque
from datetime import datetime, timedelta
from functools import wraps

from auth import auth_bp
from database import get_db

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "occupai_session_secret_2027")
app.permanent_session_lifetime = timedelta(hours=24)
CORS(app)

app.config["JWT_SECRET_KEY"]           = os.getenv("JWT_SECRET_KEY", "occupai_jwt_2027")
app.config["JWT_ACCESS_TOKEN_EXPIRES"] = timedelta(hours=24)
jwt = JWTManager(app)
app.register_blueprint(auth_bp)

# ── Camera ──
CAM_SOURCE = os.getenv("CAM_SOURCE", "webcam")
if CAM_SOURCE == "wifi":
    CAM_URL = os.getenv("RTSP_URL", "rtsp://admin:password@192.168.1.100:554/stream")
    CAM_BACKEND = cv2.CAP_FFMPEG
elif CAM_SOURCE == "droidcam":
    CAM_URL = "http://localhost:4747/video"
    CAM_BACKEND = cv2.CAP_FFMPEG
else:
    CAM_URL = 0
    CAM_BACKEND = cv2.CAP_ANY

# ── Performance ──
FEED_W=320; FEED_H=240; IMGSZ=160; YOLO_SKIP=10
JPEG_QUALITY=30; SNAPSHOT_RATE=0.5; SLOTS_RELOAD=60

# ── Lot config ──
VEHICLE_CLS={2,3,5,7}; CONF_THRESH=0.25; IOU_THRESH=0.15
LOT_WIDTH_M=10.0; LOT_HEIGHT_M=8.0; CAR_W_M=2.3; CAR_L_M=4.5; SLOT_MAR_M=0.3
PX_PER_M_X=FEED_W/LOT_WIDTH_M; PX_PER_M_Y=FEED_H/LOT_HEIGHT_M
SLOT_W_PX=int(CAR_W_M*PX_PER_M_X); SLOT_H_PX=int(CAR_L_M*PX_PER_M_Y)
SLOT_MAR_PX=int(SLOT_MAR_M*min(PX_PER_M_X,PX_PER_M_Y)); HISTORY_LEN=100

# ── Shared state ──
state={"occupied":0,"free":0,"total":0,"occupancy_pct":0.0,"slot_states":[],
       "slots":[],"yolo_boxes":[],"fps":0.0,"timestamp":"","lot_full":False,"running":False}
snap={"frame_b64":"","timestamp":""}
history=deque(maxlen=HISTORY_LEN)
state_lock=threading.Lock(); snap_lock=threading.Lock()
_latest_frame=None; _latest_frame_lock=threading.Lock()


# ── Auth decorators ──
def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if "user_id" not in session:
            return redirect(url_for("login_page"))
        return f(*args, **kwargs)
    return decorated

def admin_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if "user_id" not in session:
            return redirect(url_for("login_page"))
        if session.get("role") not in ["admin","owner"]:
            return redirect(url_for("dashboard_page"))
        return f(*args, **kwargs)
    return decorated


# ── Detection helpers ──
def compute_iou(a,b):
    ix1=max(a[0],b[0]); iy1=max(a[1],b[1]); ix2=min(a[2],b[2]); iy2=min(a[3],b[3])
    inter=max(0,ix2-ix1)*max(0,iy2-iy1)
    if not inter: return 0.0
    union=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter
    return inter/union if union else 0.0

def smart_fit_slots(occupied_boxes):
    occ_map=np.zeros((FEED_H,FEED_W),dtype=np.uint8); buf=15
    for (x1,y1,x2,y2) in occupied_boxes:
        occ_map[max(0,y1-buf):min(FEED_H,y2+buf),max(0,x1-buf):min(FEED_W,x2+buf)]=1
    slots=[]; taken=np.zeros((FEED_H,FEED_W),dtype=np.uint8)
    step_x=SLOT_W_PX+SLOT_MAR_PX; step_y=SLOT_H_PX+SLOT_MAR_PX; pad=10
    for ry in range(pad,FEED_H-SLOT_H_PX-pad,step_y):
        for cx in range(pad,FEED_W-SLOT_W_PX-pad,step_x):
            x1,y1,x2,y2=cx,ry,cx+SLOT_W_PX,ry+SLOT_H_PX
            if occ_map[y1:y2,x1:x2].sum()==0 and taken[y1:y2,x1:x2].sum()==0:
                slots.append([x1,y1,x2,y2]); taken[y1:y2,x1:x2]=1
    return slots

def load_slots():
    try:
        conn=get_db(); cur=conn.cursor()
        cur.execute("SELECT slots FROM slot_config ORDER BY slot_id DESC LIMIT 1")
        row=cur.fetchone(); cur.close(); conn.close()
        return row["slots"] if row else []
    except Exception as e:
        print(f"⚠ load_slots: {e}"); return []

def save_slots(slots):
    try:
        conn=get_db(); cur=conn.cursor()
        cur.execute("""UPDATE slot_config SET slots=%s,updated_at=%s
            WHERE slot_id=(SELECT slot_id FROM slot_config ORDER BY slot_id DESC LIMIT 1)""",
            (json.dumps(slots),datetime.utcnow()))
        conn.commit(); cur.close(); conn.close()
    except Exception as e:
        print(f"⚠ save_slots: {e}")

def log_occupancy(occupied,free,total,pct,lot_full):
    try:
        conn=get_db(); cur=conn.cursor()
        cur.execute("INSERT INTO parking_logs(occupied,free,total,occupancy_pct,lot_full) VALUES(%s,%s,%s,%s,%s)",
            (occupied,free,total,round(pct,1),lot_full))
        conn.commit(); cur.close(); conn.close()
    except Exception as e:
        print(f"⚠ log_occupancy: {e}")


# ── Snapshot thread ──
def snapshot_encoder_loop():
    encode_params=[cv2.IMWRITE_JPEG_QUALITY,JPEG_QUALITY]
    while True:
        time.sleep(SNAPSHOT_RATE)
        with _latest_frame_lock: frame=_latest_frame
        if frame is None: continue
        try:
            _,buf=cv2.imencode('.jpg',frame,encode_params)
            b64=base64.b64encode(buf).decode('utf-8')
            ts=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with snap_lock: snap["frame_b64"]=b64; snap["timestamp"]=ts
        except: pass


# ── Detection thread ──
def detection_loop():
    global _latest_frame
    print("Loading YOLO model...")
    model=YOLO("yolov8n.pt")
    model(np.zeros((FEED_H,FEED_W,3),dtype=np.uint8),imgsz=IMGSZ,verbose=False)
    print("Model ready.")
    cap=cv2.VideoCapture(CAM_URL,CAM_BACKEND)
    cap.set(cv2.CAP_PROP_BUFFERSIZE,1); cap.set(cv2.CAP_PROP_FPS,30)
    if not cap.isOpened(): print("ERROR: Cannot open camera."); return
    print(f"Camera OK! ({CAM_SOURCE} {FEED_W}x{FEED_H})")
    saved_slots=load_slots(); frame_idx=0; yolo_boxes=[]
    fps_t=time.time(); fps_n=0; fps_val=0.0
    with state_lock: state["running"]=True
    while True:
        ret,frame=cap.read()
        if not ret: time.sleep(0.005); continue
        frame=cv2.resize(frame,(FEED_W,FEED_H))
        frame_idx+=1; fps_n+=1
        now=time.time()
        if now-fps_t>=1.0: fps_val=fps_n/(now-fps_t); fps_n=0; fps_t=now
        with _latest_frame_lock: _latest_frame=frame
        if frame_idx%YOLO_SKIP==0:
            res=model(frame,imgsz=IMGSZ,verbose=False)[0]; yolo_boxes=[]
            if res.boxes is not None:
                for r in res.boxes:
                    if int(r.cls[0]) in VEHICLE_CLS and float(r.conf[0])>CONF_THRESH:
                        x1,y1,x2,y2=map(int,r.xyxy[0]); yolo_boxes.append([x1,y1,x2,y2])
        if frame_idx%SLOTS_RELOAD==0: saved_slots=load_slots()
        active_slots=saved_slots if saved_slots else smart_fit_slots(yolo_boxes)
        occupied=0; slot_states=[]
        for (sx1,sy1,sx2,sy2) in active_slots:
            occ=any(compute_iou((sx1,sy1,sx2,sy2),tuple(vb))>IOU_THRESH for vb in yolo_boxes)
            slot_states.append(occ); occupied+=int(occ)
        total=len(active_slots); free=total-occupied
        pct=(occupied/total*100) if total else 0
        lot_full=total>0 and free==0
        ts=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if frame_idx%30==0:
            history.append({"time":ts,"occupied":occupied,"total":total,"pct":round(pct,1)})
            log_occupancy(occupied,free,total,pct,lot_full)
        with state_lock:
            state.update({"occupied":occupied,"free":free,"total":total,
                "occupancy_pct":round(pct,1),"slot_states":slot_states,
                "slots":active_slots,"yolo_boxes":yolo_boxes,
                "fps":round(fps_val,1),"timestamp":ts,"lot_full":lot_full})
    cap.release()


# ╔══════════════════════════════════════════════╗
# ║        PAGE ROUTES (Jinja2)                 ║
# ╚══════════════════════════════════════════════╝
@app.route("/")
def index():
    if "user_id" in session:
        return redirect(url_for("admin_page") if session.get("role") in ["admin","owner"]
                        else url_for("dashboard_page"))
    return redirect(url_for("login_page"))

@app.route("/login")
def login_page():
    if "user_id" in session: return redirect(url_for("index"))
    return render_template("login.html")

@app.route("/register")
def register_page():
    if "user_id" in session: return redirect(url_for("index"))
    return render_template("register.html")

@app.route("/dashboard")
@login_required
def dashboard_page():
    with state_lock: s=dict(state)
    return render_template("dashboard.html", user=session,
        occupied=s["occupied"], free=s["free"], total=s["total"],
        occupancy_pct=s["occupancy_pct"], lot_full=s["lot_full"],
        timestamp=s["timestamp"])

@app.route("/admin")
@admin_required
def admin_page():
    with state_lock: s=dict(state)
    return render_template("admin.html", user=session,
        occupied=s["occupied"], free=s["free"], total=s["total"],
        occupancy_pct=s["occupancy_pct"], lot_full=s["lot_full"],
        fps=s["fps"], timestamp=s["timestamp"],
        yolo_count=len(s["yolo_boxes"]), slot_count=len(s["slots"]))


# ── Form handlers ──
@app.route("/do-login", methods=["POST"])
def do_login():
    email    = request.form.get("email","").strip().lower()
    password = request.form.get("password","")
    try:
        conn=get_db(); cur=conn.cursor()
        cur.execute("""SELECT user_id,first_name,last_name,full_name,
            email,password_hash,role,is_active FROM users WHERE email=%s""",(email,))
        user=cur.fetchone(); cur.close(); conn.close()
        if not user:
            flash("Email not found. Please register first.","error")
            return redirect(url_for("login_page"))
        if not user["is_active"]:
            flash("Account disabled. Contact administrator.","error")
            return redirect(url_for("login_page"))
        if not bcrypt.checkpw(password.encode(),user["password_hash"].encode()):
            flash("Incorrect password.","error")
            return redirect(url_for("login_page"))
        session.permanent=True
        session.update({"user_id":user["user_id"],"first_name":user["first_name"],
            "last_name":user["last_name"],"full_name":user["full_name"],
            "email":user["email"],"role":user["role"]})
        conn2=get_db(); cur2=conn2.cursor()
        cur2.execute("UPDATE users SET last_login=%s WHERE user_id=%s",
                     (datetime.utcnow(),user["user_id"]))
        conn2.commit(); cur2.close(); conn2.close()
        return redirect(url_for("admin_page") if user["role"] in ["admin","owner"]
                        else url_for("dashboard_page"))
    except Exception as e:
        flash(f"Login error: {e}","error")
        return redirect(url_for("login_page"))


@app.route("/do-register", methods=["POST"])
def do_register():
    first_name=request.form.get("first_name","").strip()
    last_name =request.form.get("last_name","").strip()
    email     =request.form.get("email","").strip().lower()
    password  =request.form.get("password","")
    if not all([first_name,last_name,email,password]):
        flash("All fields are required.","error")
        return redirect(url_for("register_page"))
    if len(password)<6:
        flash("Password must be at least 6 characters.","error")
        return redirect(url_for("register_page"))
    pw_hash=bcrypt.hashpw(password.encode(),bcrypt.gensalt()).decode()
    try:
        conn=get_db(); cur=conn.cursor()
        cur.execute("SELECT user_id FROM users WHERE email=%s",(email,))
        if cur.fetchone():
            flash("Email already registered. Please login.","error")
            cur.close(); conn.close()
            return redirect(url_for("register_page"))
        cur.execute("""INSERT INTO users(first_name,last_name,email,password_hash,role)
            VALUES(%s,%s,%s,%s,'driver') RETURNING user_id""",
            (first_name,last_name,email,pw_hash))
        new_id=cur.fetchone()["user_id"]
        cur.execute("INSERT INTO drivers(user_id) VALUES(%s)",(new_id,))
        conn.commit(); cur.close(); conn.close()
        flash("Account created! Please login.","success")
        return redirect(url_for("login_page"))
    except Exception as e:
        flash(f"Registration error: {e}","error")
        return redirect(url_for("register_page"))


@app.route("/logout")
def do_logout():
    session.clear()
    return redirect(url_for("login_page"))


# ╔══════════════════════════════════════════════╗
# ║        API ROUTES (JSON polling)            ║
# ╚══════════════════════════════════════════════╝
@app.route("/api/status")
def api_status():
    with state_lock:
        return jsonify({"status":"online" if state["running"] else "starting",
            "camera":CAM_SOURCE,"fps":state["fps"],"version":"4.3",
            "location":"105 Peñafrancia Ave, Naga City"})

@app.route("/api/stats")
def api_stats():
    with state_lock:
        return jsonify({"occupied":state["occupied"],"free":state["free"],
            "total":state["total"],"occupancy_pct":state["occupancy_pct"],
            "lot_full":state["lot_full"],"timestamp":state["timestamp"],"fps":state["fps"]})

@app.route("/api/snapshot")
def api_snapshot():
    if "user_id" not in session: return jsonify({"error":"unauthorized"}),401
    with snap_lock:
        return jsonify({"image":snap["frame_b64"],"timestamp":snap["timestamp"]})

@app.route("/api/history")
def api_history():
    if "user_id" not in session: return jsonify({"error":"unauthorized"}),401
    return jsonify(list(history))

@app.route("/api/predictions")
def api_predictions():
    hour_dist={6:0.03,7:0.12,8:0.18,9:0.10,10:0.06,11:0.05,
               12:0.10,13:0.08,14:0.05,15:0.04,16:0.05,17:0.08,18:0.04,19:0.02}
    with state_lock: avg=state["total"] if state["total"]>0 else 20
    hourly={str(h):round(avg*p) for h,p in hour_dist.items()}
    peak=max(hour_dist,key=hour_dist.get)
    return jsonify({"peak_hour":peak,"peak_label":f"{peak}:00-{peak+1}:00",
        "hourly_est":hourly,"busy_days":["Monday","Tuesday","Wednesday","Thursday"],
        "quiet_days":["Saturday","Sunday"]})

@app.route("/api/slots/auto", methods=["POST"])
def api_auto_slots():
    if "user_id" not in session or session.get("role") not in ["admin","owner"]:
        return jsonify({"success":False,"message":"Admin or Owner only."}),403
    with state_lock: boxes=list(state["yolo_boxes"])
    auto=smart_fit_slots(boxes); save_slots(auto)
    return jsonify({"success":True,"slots":auto,"count":len(auto)})


if __name__=="__main__":
    threading.Thread(target=detection_loop,daemon=True).start()
    threading.Thread(target=snapshot_encoder_loop,daemon=True).start()
    print("\n╔══════════════════════════════════════╗")
    print("║   OccupAI  v4.3  Flask + Jinja2      ║")
    print("╠══════════════════════════════════════╣")
    print("║  http://localhost:5000               ║")
    print("╚══════════════════════════════════════╝\n")
    app.run(host="0.0.0.0",port=5000,debug=False,threaded=True,use_reloader=False)