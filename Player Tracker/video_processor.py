from ultralytics import YOLO
import cv2
import numpy as np
from dotenv import load_dotenv
import os
from collections import defaultdict
from mss import mss


load_dotenv()  # loads .env into environment variables

sct = mss()

#Left side of the screen for width 1920 and height 1080
monitor = {
    "top": 0,
    "left": 0,
    "width": 960,
    "height": 1080
}

yolo = YOLO("yolov8n.pt")

'''if not camera_url:
    raise ValueError("CAMERA_URL not set")
cap = cv2.VideoCapture()
success, frame = cap.read()
        if not success:
            break'''

track_history = defaultdict(lambda: [])
while True:
    img = np.array(sct.grab(monitor))
    frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    results = yolo.track(frame, persist=True, classes=[0], verbose=False)

    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xywh.cpu()
        track_ids = results[0].boxes.id.int().cpu().tolist()

  
        for box, track_id in zip(boxes, track_ids):
            x, y, w, h = box
            track = track_history[track_id]
            
            center_point = (float(x), float(y))
            track.append(center_point)
            
            # Limit history to last 30 frames (to avoid messy screen)
            if len(track) > 30:
                track.pop(0)

            # Draw the movement trail
            points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [points], isClosed=False, color=(0, 255, 0), thickness=2)
            
            top_left_x = int(x - w / 2)
            top_left_y = int(y - h / 2)
            bottom_right_x = int(x + w / 2)
            bottom_right_y = int(y + h / 2)
            
            cv2.rectangle(frame, (top_left_x, top_left_y), 
                          (bottom_right_x, bottom_right_y), (0, 0, 255), 2)
            
            cv2.putText(frame, f"ID: {track_id}", (top_left_x, top_left_y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("Soccer Player Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

#cap.release()

cv2.destroyAllWindows()