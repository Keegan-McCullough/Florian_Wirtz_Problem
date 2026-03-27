# PlayerTracker/redis_store.py
import json
import os
import time
from typing import Iterable, Optional
import redis
import math

class RedisTrackerStore:
    def __init__(self, redis_url: Optional[str] = None, key_prefix: str = "player_tracker"):
        self.redis_url = redis_url or os.getenv("REDIS_URL", "redis://localhost:6379/0")
        self.key_prefix = key_prefix
        self.enabled = True

        try:
            self.client = redis.Redis.from_url(self.redis_url, decode_responses=True)
            self.client.ping()
        except redis.RedisError:
            self.client = None
            self.enabled = False

    def _key(self, suffix: str) -> str:
        return f"{self.key_prefix}:{suffix}"
    
    def update_player_data(self, track_id: int, x: float, y: float, conf: float) -> None:
        if not self.enabled:
            return

        payload = {
            "track_id": int(track_id),
            "x": float(x),
            "y": float(y),
            "conf": float(conf),
            "ts": time.time(),
        }
        serialized = json.dumps(payload)

        try:
            # HSET ensures only the most recent data exists for track_id
            self.client.hset(self._key("latest_positions"), str(track_id), serialized)
            self.client.sadd(self._key("active_track_ids"), int(track_id))
        except redis.RedisError:
            self.enabled = False

    def save_setup(self, selected_ids: Iterable[int], boundary_points, boundary_complete: bool) -> None:
        if not self.enabled: return
        payload = {
            "selected_ids": list(selected_ids),
            "boundary_points": list(boundary_points),
            "boundary_complete": bool(boundary_complete),
            "updated_at": time.time(),
        }
        try:
            self.client.set(self._key("setup"), json.dumps(payload))
            self.client.set(self._key("selected_ids"), json.dumps(list(selected_ids)))
        except redis.RedisError:
            self.enabled = False

    def get_selected_ids(self) -> Optional[set[int]]:
        if not self.enabled: return None
        try:
            raw = self.client.get(self._key("selected_ids"))
            if not raw: return None
            values = json.loads(raw)
            return {int(value) for value in values} if isinstance(values, list) else None
        except (redis.RedisError, ValueError, TypeError):
            self.enabled = False
            return None

    def get_id_mapping(self) -> dict:
        if not self.enabled: 
            return {}
        return self.client.hgetall(self._key("id_map"))

    def set_id_mapping(self, yolo_id: int, perm_id: int):
        if not self.enabled: 
            return
        self.client.hset(self._key("id_map"), str(yolo_id), str(perm_id))

    def find_available_slot(self, x, y, max_slots: int = 22) -> Optional[int]:
        if not self.enabled: 
            return None
        mappings = {int(k): json.loads(v) for k, v in self.client.hgetall(self._key("latest_positions")).items()}
        mapped = self.get_id_mapping()
        used_slots = set(int(v) for v in mapped.values())
        
        for slot in range(1, max_slots + 1):
            if slot not in used_slots:
                return slot
        
        best_id = None
        min_score = float('inf')
        current_ts = time.time()

        for slot, data in mappings.items():
            if (data['ts']) > (current_ts - 0.5):
                continue
            # Euclidean distance for closest in space
            dist = math.sqrt((x - data['x'])**2 + (y - data['y'])**2)
            
            # Time Component (How long has it been since we saw them?)
            time_diff = current_ts - data['ts']
            
            # Score Formula: Distance divided by time (or a weighted sum)
            # We want high time_diff and low dist to result in a low score
            # If they've been gone a long time, the 'penalty' for distance decreases
            score = dist + time_diff
            
            if score < min_score:
                min_score = score
                best_id = slot
        return best_id

    def clear_project_cache(self):
        if not self.enabled: return
        pattern = f"{self.key_prefix}:*"
        cursor = 0
        while True:
            cursor, keys = self.client.scan(cursor=cursor, match=pattern)
            if keys:
                self.client.delete(*keys)
            if cursor == 0: break