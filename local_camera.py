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
PUSH_FPS   = 2   # push 2 frames per second (adjust as needed)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  256)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 192)

if not cap.isOpened():
    print("ERROR: Cannot open webcam")
    exit(1)

print("Streaming to Render... Press Ctrl+C to stop")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    _, buf   = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 40])
    b64      = base64.b64encode(buf).decode('utf-8')

    try:
        r = requests.post(SERVER_URL,
            json={"frame": b64},
            headers={"X-Cam-Token": CAM_TOKEN},
            timeout=5
        )
        print(f"Pushed frame → {r.status_code}")
    except Exception as e:
        print(f"Error: {e}")

    time.sleep(1.0 / PUSH_FPS)

cap.release()