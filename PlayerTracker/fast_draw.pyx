# PlayerTracker/fast_draw.pyx
import numpy as np
cimport numpy as np
import cv2

# Define types for parameters to get C-speed
def draw_trails(np.uint8_t[:, :, :] frame, dict track_history, int max_len=30):
    cdef int track_id
    cdef list track
    
    for track_id, track in track_history.items():
        if len(track) > 1:
            # Type-optimized conversion of track points
            points = np.array(track, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [points], False, (0, 255, 0), 2)