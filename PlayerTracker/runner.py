# PlayerTracker/runner.py
import threading
from queue import Queue
from video_processor import screen_capture
from tracker import yolo_tracking
from setup import run_setup
from ultralytics import YOLO

if __name__ == "__main__":
    frame_queue = Queue(maxsize=10)
    
    ov_model = YOLO("yolov8m_openvino_model/")

    # Start capture thread so setup has live frames to work with
    capture_thread = threading.Thread(target=screen_capture, args=(frame_queue,), daemon=True)
    capture_thread.start()

    # Blocking setup phase
    setup = run_setup(frame_queue, ov_model)

    print(f"Tracking IDs: {setup.selected_ids or 'ALL within boundary'}")
    print(f"Boundary set: {setup.boundary_complete}")

    # Hand off to main tracking loop
    yolo_tracking(frame_queue, setup, ov_model)