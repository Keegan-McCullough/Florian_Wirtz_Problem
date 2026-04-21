import cv2
import redis
import pickle
import json
import os

r = redis.Redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379/0"))

cap = cv2.VideoCapture(video_path)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO + tracker step
    frame, detections = model_process(frame)

    # push to Redis
    r.set("frame:latest", pickle.dumps(frame))
    r.set("detections:latest", json.dumps(detections))