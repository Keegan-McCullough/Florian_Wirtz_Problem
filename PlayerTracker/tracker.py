# PlayerTracker/tracker.py
import cv2
import numpy as np
from setup import TrackingSetup
from redis_store import RedisTrackerStore

def yolo_tracking(frame_queue, setup: TrackingSetup, ov_model, redis_store: RedisTrackerStore | None = None):
    local_id_map = {}
    
    while True:
        if frame_queue.empty():
            continue

        frame = frame_queue.get()
        results = ov_model.track(frame, imgsz=640, conf=0.5, iou=0.8,
                                  persist=True, tracker="custom_botsort.yaml",
                                  classes=[0], verbose=False)

        if results and results[0].boxes.id is not None:
            boxes = results[0].boxes.xywh.cpu()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            confidences = results[0].boxes.conf.cpu().tolist()


            # Get selected IDs once per frame for efficiency
            redis_selected_ids = redis_store.get_selected_ids() if redis_store else None
            active_selected_ids = redis_selected_ids if redis_selected_ids is not None else setup.selected_ids

            for box, track_id, conf in zip(boxes, track_ids, confidences):
                x, y, w, h = box

                if not setup.point_in_boundary(float(x), float(y)):
                    continue
                
                if active_selected_ids and track_id not in active_selected_ids:
                    continue
                
                if redis_store:
                    remote_map = redis_store.get_id_mapping()
                    local_id_map.update({int(k): int(v) for k, v in remote_map.items()})

                if track_id in local_id_map:
                    perm_id = local_id_map[track_id]
                else:
                    # Attempt to assign a new permanent slot (1-22)
                    perm_id = redis_store.find_available_slot(x, y, 22) if redis_store else None
        
                    redis_store.set_id_mapping(track_id, perm_id)
                    local_id_map[track_id] = perm_id 

                # Update Redis with only the latest position
                if redis_store:
                    redis_store.update_player_data(track_id=perm_id, x=float(x), y=float(y), conf=float(conf))

                
                # Simple bounding box visualization
                tl = (int(x - w/2), int(y - h/2))
                br = (int(x + w/2), int(y + h/2))
                cv2.rectangle(frame, tl, br, (0, 0, 255), 2)
                cv2.putText(frame, f"ID: {track_id}" , (tl[0], tl[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        if setup.boundary_complete:
            pts = np.array(setup.boundary_points, np.int32)
            cv2.polylines(frame, [pts], True, (0, 255, 255), 1)

        cv2.imshow("Soccer Player Tracking", cv2.UMat(frame))
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()