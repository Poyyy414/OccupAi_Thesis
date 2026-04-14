"""
local_camera.py
Run this on your LOCAL PC to push webcam frames to Render.
"""
import cv2
import base64
import requests
import time

SERVER_URL = "https://occupai-thesis.onrender.com/api/push-frame"
CAM_TOKEN  = "occupai_cam_2027"   # must match CAM_TOKEN env var on Render
PUSH_FPS   = 2                    # frames per second to push

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  256)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 192)

time.sleep(2)  # camera warm-up

if not cap.isOpened():
    print("ERROR: Cannot open webcam")
    exit(1)

print("Streaming to Render... Press Ctrl+C to stop")

INTERVAL    = 1.0 / PUSH_FPS
retry_delay = 1.0   # backs off on repeated errors, resets on success

while True:
    loop_start = time.time()

    ret, frame = cap.read()
    if not ret or frame is None:
        print("Warning: failed to grab frame, retrying...")
        time.sleep(0.5)
        continue

    _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 40])
    b64    = base64.b64encode(buf).decode('utf-8')

    try:
        r = requests.post(
            SERVER_URL,
            json={"frame": b64},
            headers={"X-Cam-Token": CAM_TOKEN},
            timeout=5
        )
        print(f"Pushed frame → {r.status_code}")
        retry_delay = 1.0  # reset backoff on success
    except requests.exceptions.ConnectionError:
        print(f"Connection error — retrying in {retry_delay:.0f}s")
        time.sleep(retry_delay)
        retry_delay = min(retry_delay * 2, 30)  # cap at 30s backoff
        continue
    except Exception as e:
        print(f"Error: {e}")

    # sleep for the remainder of the interval
    elapsed = time.time() - loop_start
    sleep_for = INTERVAL - elapsed
    if sleep_for > 0:
        time.sleep(sleep_for)

cap.release()