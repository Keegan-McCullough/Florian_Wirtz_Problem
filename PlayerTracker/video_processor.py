# PlayerTracker/video_processor.py
from ultralytics import YOLO
import cv2
import numpy as np
from collections import defaultdict
from mss import mss

def screen_capture(frame_queue, monitor_index=1, max_queue_size=15):
    sct = mss()
    monitor = sct.monitors[monitor_index]

    # want only the left side right now
    left_half = {
        "top": monitor["top"] + 50,
        "left": monitor["left"], 
        "width": monitor["width"] // 2,
        "height": monitor["height"] - 200
    }

    frame_id = 0

    while True:
        img = np.array(sct.grab(left_half))
        frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        frame = cv2.convertScaleAbs(frame, alpha=1.0)

        data = pickle.dumps(frame)
        
        pipe = frame_queue.pipelne()
        pipe.set(f"frame:{frame_id}", data, ex=5)
        pipe.set("latest_frame_id", frame_id)
        pipe.execute
        frame_id += 1

        # Display capture optional
        #cv2.imshow("Recording", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()