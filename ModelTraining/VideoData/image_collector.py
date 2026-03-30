import cv2
import os

video_path = 'england_epl/2014-2015/2015-02-21 - 18-00 Crystal Palace 1 - 2 Arsenal/1_720p.mkv'
output_dir = 'train'
frame_interval = 30  # Extract 1 frame every 30 frames (roughly 1 per second)

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("failed to get capture")
count = 0
saved_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Only save every Nth frame to ensure dataset diversity
    if count % frame_interval == 0:
        file_name = os.path.join(output_dir, f"CPA_frame_{saved_count:05d}.jpg")
        cv2.imwrite(file_name, frame)
        saved_count += 1
        
    count += 1

cap.release()