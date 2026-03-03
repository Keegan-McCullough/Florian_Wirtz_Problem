# PlayerTracker/setup.py
import cv2
import numpy as np
from collections import defaultdict

class TrackingSetup:
    def __init__(self):
        self.boundary_points = []
        self.boundary_complete = False
        self.selected_ids = set()
        self.current_detections = {}  # id -> (x, y, w, h)
        self.setup_complete = False

    def draw_ui(self, frame):
        display = frame.copy()

        # Draw boundary
        if len(self.boundary_points) > 0:
            pts = np.array(self.boundary_points, np.int32)
            cv2.polylines(display, [pts], self.boundary_complete, (0, 255, 255), 2)

            for pt in self.boundary_points:
                cv2.circle(display, pt, 5, (0, 255, 255), -1)

        # Draw current detections
        for track_id, (x, y, w, h) in self.current_detections.items():
            tl = (int(x - w/2), int(y - h/2))
            br = (int(x + w/2), int(y + h/2))
            color = (0, 255, 0) if track_id in self.selected_ids else (100, 100, 100)
            cv2.rectangle(display, tl, br, color, 2)
            cv2.putText(display, f"ID:{track_id} [click]", (tl[0], tl[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        instructions = [
            "LEFT CLICK: Add boundary point",
            "RIGHT CLICK: Close boundary / Select IDs",
            "Middle click ID box: Toggle track",
            "ENTER: Confirm and start tracking",
            "R: Reset"
        ]
        for i, line in enumerate(instructions):
            cv2.putText(display, line, (10, 20 + i * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return display

    def mouse_callback(self, event, x, y, frame, flags):
        if event == cv2.EVENT_LBUTTONDOWN:
            if not self.boundary_complete:
                self.boundary_points.append((x, y))

            else:
                # Toggle ID selection
                for track_id, (bx, by, bw, bh) in self.current_detections.items():
                    tl = (int(bx - bw/2), int(by - bh/2))
                    br = (int(bx + bw/2), int(by + bh/2))
                    if tl[0] <= x <= br[0] and tl[1] <= y <= br[1]:
                        if track_id in self.selected_ids:
                            self.selected_ids.discard(track_id)
                        else:
                            self.selected_ids.add(track_id)
                self.draw_ui(frame)

        elif event == cv2.EVENT_RBUTTONDOWN:
            # Close the boundary
            if len(self.boundary_points) >= 3:
                self.boundary_complete = True

    def point_in_boundary(self, x, y):
        #Check if a point is inside the drawn boundary polygon.
        if not self.boundary_complete or len(self.boundary_points) < 3:
            return True  # No boundary = track everything
        pts = np.array(self.boundary_points, np.int32)
        return cv2.pointPolygonTest(pts, (float(x), float(y)), False) >= 0


def run_setup(frame_queue, ov_model) -> TrackingSetup:
    
    #Returns a configured TrackingSetup when the user hits ENTER.
    setup = TrackingSetup()
    cv2.namedWindow("Setup")
    cv2.setMouseCallback("Setup", setup.mouse_callback, frame_queue.get())

    while not setup.setup_complete:
        if frame_queue.empty():
            continue

        frame = frame_queue.get()

        # Run detection during setup so user can see and select IDs
        results = ov_model.track(frame, imgsz=640, conf=0.5, iou=0.1,
                                  persist=True, tracker="custom_botsort.yaml",
                                  classes=[0], verbose=False)

        if results and results[0].boxes.id is not None:
            boxes = results[0].boxes.xywh.cpu()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            setup.current_detections = {
                tid: tuple(box.tolist()) for tid, box in zip(track_ids, boxes)
            }

        display = setup.draw_ui(frame)
        cv2.imshow("Setup", display)

        key = cv2.waitKey(1) & 0xFF
        if key == 13:  # ENTER — confirm
            setup.setup_complete = True
        elif key == ord("r"):  # Reset
            setup.boundary_points = []
            setup.boundary_complete = False
            setup.selected_ids = set()

    cv2.destroyWindow("Setup")
    return setup