from ultralytics import YOLO
import cv2
import numpy as np
from dotenv import load_dotenv
import os
from collections import defaultdict

def yolo_tracking(frame_queue):
    yolo = YOLO("yolov8m.pt")
    yolo.export(format="openvino")
    ov_model = YOLO("yolov8m_openvino_model/")
    track_history = defaultdict(lambda: [])

    while True:
        if not frame_queue.empty():
            frame = frame_queue.get()

            results = ov_model.track(frame, imgsz=640, conf = 0.1, iou=0.5, persist=True, tracker="bytetrack.yaml", classes=[0], verbose=False)

            if results and results[0].boxes.id is not None:
                boxes = results[0].boxes.xywh.cpu()
                track_ids = results[0].boxes.id.int().cpu().tolist()

                for box, track_id in zip(boxes, track_ids):
                    x, y, w, h = box
                    track = track_history[track_id]
                    center_point = (float(x), float(y))
                    track.append(center_point)
                    if len(track) > 30:
                        track.pop(0)

                    # Draw movement trail
                    points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                    cv2.polylines(frame, [points], isClosed=False, color=(0, 255, 0), thickness=2)

                    # Draw bounding box
                    top_left = (int(x - w/2), int(y - h/2))
                    bottom_right = (int(x + w/2), int(y + h/2))
                    cv2.rectangle(frame, top_left, bottom_right, (0, 0, 255), 2)
                    cv2.putText(frame, f"ID: {track_id}", (top_left[0], top_left[1]-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Display results
            cv2.imshow("Soccer Player Tracking", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cv2.destroyAllWindows()
