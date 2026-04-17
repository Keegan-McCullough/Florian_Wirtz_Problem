# PlayerTracker/redis_store.py
import json
import os
import time
from typing import Iterable, Optional
import redis
import math
import numpy as np

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
            # We want low time_diff and low dist to result in a low score
            # If they've been gone a long time, the 'penalty' for distance decreases
            score = (.7 * dist) + (.3 * (time_diff/3))
            
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

    def store_embedding(self, perm_id: int, embedding: np.ndarray) -> None:
        if not self.enabled:
            return

        # Ensure embedding is a list for JSON serialization
        if isinstance(embedding, np.ndarray):
            embedding = embedding.tolist()

        try:
            # We store this in a hash where the key is the permanent player ID
            self.client.hset(self._key("player_embeddings"), str(perm_id), json.dumps(embedding))
        except redis.RedisError:
            self.enabled = False

    def find_match_by_embedding(self, current_embedding: np.ndarray, threshold: float = 0.75) -> Optional[int]:
        if not self.enabled:
            return None

        try:
            all_embeddings = self.client.hgetall(self._key("player_embeddings"))
            if not all_embeddings:
                return None

            best_match_id = None
            max_similarity = -1.0

            # Normalize the current embedding once for Cosine Similarity calculation
            norm_current = current_embedding / np.linalg.norm(current_embedding)

            for perm_id, stored_raw in all_embeddings.items():
                stored_vec = np.array(json.loads(stored_raw))
                
                # Calculate Cosine Similarity
                norm_stored = stored_vec / np.linalg.norm(stored_vec)
                similarity = np.dot(norm_current, norm_stored)

                if similarity > max_similarity:
                    max_similarity = similarity
                    best_match_id = int(perm_id)

            return best_match_id if max_similarity >= threshold else None

        except (redis.RedisError, ValueError, TypeError):
            self.enabled = False
            return None