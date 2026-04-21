# PlayerTracker/runner.py
import threading
from queue import Queue
from video_processor import screen_capture
from tracker import yolo_tracking
from setup import run_setup
from redis_store import RedisTrackerStore
from ultralytics import YOLO
import wandb
import os
import redis

if __name__ == "__main__":
    frame_queue = redis.Redis(host="localhost", port=6380, password="Liverpool2020!",)

    redis_store = RedisTrackerStore()
    redis_store.clear_project_cache()

    # Uncomment to load in a new model and change version
    #run = wandb.init()
    #artifact = run.use_artifact('florian_wirtz_problem/florian-wirtz-problem/wirtz-tracking-model:v0', type='model')
    #artifact_dir = artifact.download()

    # The artifact folder will contain both the .pt and the openvino_version
    if os.name == "nt":
        model_path = os.path.join("artifacts", "wirtz-tracking-model-v0", "best_openvino_model")
    else:
        model_path = os.path.join("artifacts", "wirtz-tracking-model-v0", "soccer_model.pt")
    ov_model = YOLO(model_path, task="detect")
    
    #ov_model = YOLO("best_openvino_model", task="detect")

    # Start capture thread so setup has live frames to work with
    capture_thread = threading.Thread(target=screen_capture, args=(frame_queue,), daemon=True)
    capture_thread.start()

    # Blocking setup phase
    setup = run_setup(frame_queue, ov_model, redis_store=redis_store)

    print(f"Tracking IDs: {setup.selected_ids or 'ALL within boundary'}")
    print(f"Boundary set: {setup.boundary_complete}")
    print(f"Redis connected: {redis_store.enabled}")

    # Hand off to main tracking loop
    yolo_tracking(frame_queue, setup, ov_model, redis_store=redis_store)