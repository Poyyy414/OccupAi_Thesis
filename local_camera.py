"""
local_camera.py
Run this on your LOCAL PC to push webcam frames to Render.
"""
import cv2
import base64
import requests
import time

SERVER_URL = "https://occupai-thesis-1.onrender.com/api/push-frame"
CAM_TOKEN  = "occupai_cam_2027"
PUSH_FPS   = 15

def wake_server():
    print("Waking up Render server...")
    for attempt in range(20):
        try:
            r = requests.get(
                "https://occupai-thesis-1.onrender.com/login",
                timeout=15
            )
            if r.status_code < 500:
                print(f"✅ Server is awake! ({r.status_code})")
                return True
        except Exception as e:
            print(f"  Attempt {attempt+1}/20 — waiting... ({e})")
        time.sleep(5)
    print("❌ Could not wake server.")
    return False

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

time.sleep(2)

if not cap.isOpened():
    print("ERROR: Cannot open webcam")
    exit(1)

actual_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
actual_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
print(f"Camera resolution: {int(actual_w)}x{int(actual_h)}")

if not wake_server():
    cap.release()
    exit(1)

print(f"Streaming to Render at {PUSH_FPS} FPS... Press Ctrl+C to stop")

session = requests.Session()

while True:
    start = time.time()

    ret, frame = cap.read()
    if not ret or frame is None:
        print("Warning: failed to grab frame, retrying...")
        time.sleep(0.1)
        continue

    display_frame = cv2.resize(frame, (1280, 720))

    _, buf = cv2.imencode('.jpg', display_frame,
                          [cv2.IMWRITE_JPEG_QUALITY, 75])
    b64 = base64.b64encode(buf).decode('utf-8')

    try:
        r = session.post(SERVER_URL,
            json={"frame": b64},
            headers={"X-Cam-Token": CAM_TOKEN},
            timeout=10
        )
        print(f"Pushed frame → {r.status_code}")
    except requests.exceptions.ConnectionError:
        print("Server unreachable, retrying in 5s...")
        time.sleep(5)
        continue
    except requests.exceptions.Timeout:
        print("Timeout — retrying in 3s...")
        time.sleep(3)
        continue
    except Exception as e:
        print(f"Error: {e}")

    elapsed = time.time() - start
    sleep_time = (1.0 / PUSH_FPS) - elapsed
    if sleep_time > 0:
        time.sleep(sleep_time)

cap.release()