# PlayerTracker/tracker.py
from ultralytics import YOLO
import cv2
import numpy as np
from collections import defaultdict
from setup import TrackingSetup


def yolo_tracking(frame_queue, setup: TrackingSetup, ov_model):
    track_history = defaultdict(list)

    while True:
        if frame_queue.empty():
            continue

        frame = frame_queue.get()
        results = ov_model.track(frame, imgsz=640, conf=0.01, iou=0.5,
                                  persist=True, tracker="custom_botsort.yaml",
                                  classes=[0], verbose=False)


        if results and results[0].boxes.id is not None:
            boxes = results[0].boxes.xywh.cpu()
            track_ids = results[0].boxes.id.int().cpu().tolist()

            for box, track_id in zip(boxes, track_ids):
                x, y, w, h = box

                if not setup.point_in_boundary(float(x), float(y)):
                    continue
                if setup.selected_ids and track_id not in setup.selected_ids:
                    continue

                track_history[track_id].append((float(x), float(y)))
                if len(track_history[track_id]) > 90:
                    track_history[track_id].pop(0)

                tl = (int(x - w/2), int(y - h/2))
                br = (int(x + w/2), int(y + h/2))
                cv2.rectangle(frame, tl, br, (0, 0, 255), 2)
                cv2.putText(frame, f"ID: {track_id}" , (tl[0], tl[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


        if setup.boundary_complete:
            pts = np.array(setup.boundary_points, np.int32)
            cv2.polylines(frame, [pts], True, (0, 255, 255), 1)

        cv2.imshow("Soccer Player Tracking", cv2.UMat(frame))
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()