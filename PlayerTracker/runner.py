from multiprocessing import Process, Queue
import time
from tracker import yolo_tracking
from video_processor import screen_capture

if __name__ == "__main__":
    frame_queue = Queue(maxsize=10)

    # Start capture process
    capture_process = Process(target=screen_capture, args=(frame_queue,))
    capture_process.start()

    # Start YOLO tracking process
    tracking_process = Process(target=yolo_tracking, args=(frame_queue,))
    tracking_process.start()

    # Wait for both to finish
    capture_process.join()
    tracking_process.join()
