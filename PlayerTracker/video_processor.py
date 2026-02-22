from ultralytics import YOLO
import cv2
import numpy as np
from collections import defaultdict
from mss import mss


def screen_capture(frame_queue, monitor_index=1, max_queue_size=10):
    sct = mss()
    monitor = sct.monitors[monitor_index]

    # want only the left side right now
    left_half = {
        "top": monitor["top"],
        "left": monitor["left"], 
        "width": monitor["width"] // 2,
        "height": monitor["height"]
    }

    while True:
        img = np.array(sct.grab(left_half))
        frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        frame = cv2.convertScaleAbs(frame, alpha=1.2, beta=20)

        # Drop frame if queue is full
        if frame_queue.qsize() < max_queue_size:
            frame_queue.put(frame)
        else:
            # Optional: skip frame
            pass

        # Display capture (optional)
        cv2.imshow("Recording", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()

