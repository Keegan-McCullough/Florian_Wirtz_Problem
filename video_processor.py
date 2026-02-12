from ultralytics import YOLO
import cv2
import sqlite3
import threading
import queue
import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import asyncio
from collections import deque
import traceback
from dotenv import load_dotenv
import os
from mss import mss


load_dotenv()  # loads .env into environment variables

sct = mss()

monitor = sct.monitors[1]  # main screen

yolo = YOLO("yolov8n")
# prolly be the web cam video
camera_url = os.getenv("CAMERA_URL")

if not camera_url:
    raise ValueError("CAMERA_URL not set")
cap = cv2.VideoCapture()

while True:
    img = np.array(sct.grab(monitor))
    frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    # keep if you want to see the live screen
    #cv2.imshow("Live Screen", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()