#PlayerTracker/inspect_redis.py
import argparse
import json
import os
from datetime import datetime, timedelta, timezone
from typing import Any

import redis


def try_parse_json(value: str) -> Any:
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return value


def human_time(epoch_seconds: float | int) -> str:
    dt_utc = datetime.fromtimestamp(float(epoch_seconds), tz=timezone.utc)
    dt_local = dt_utc.astimezone()
    return dt_local.isoformat(timespec="seconds")


def human_ttl(ttl_seconds: int) -> str:
    if ttl_seconds == -1:
        return "no expiry"
    if ttl_seconds == -2:
        return "key not found"
    return str(timedelta(seconds=max(0, ttl_seconds)))


def normalize_time_fields(value: Any) -> Any:
    if isinstance(value, dict):
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            normalized[key] = normalize_time_fields(item)
            if key in {"updated_at", "ts"} and isinstance(item, (int, float)):
                try:
                    normalized[f"{key}_human"] = human_time(item)
                except (OverflowError, OSError, ValueError):
                    pass
        return normalized

    if isinstance(value, list):
        return [normalize_time_fields(item) for item in value]

    return value


def read_key(client: redis.Redis, key: str, parse_json: bool) -> dict:
    key_type = client.type(key)
    ttl = client.ttl(key)

    if key_type == "string":
        raw = client.get(key)
        value = try_parse_json(raw) if parse_json and raw is not None else raw
    elif key_type == "list":
        raw_list = client.lrange(key, 0, -1)
        value = [try_parse_json(item) for item in raw_list] if parse_json else raw_list
    elif key_type == "set":
        raw_set = sorted(client.smembers(key))
        value = [try_parse_json(item) for item in raw_set] if parse_json else raw_set
    elif key_type == "hash":
        raw_hash = client.hgetall(key)
        if parse_json:
            value = {field: try_parse_json(val) for field, val in raw_hash.items()}
        else:
            value = raw_hash
    elif key_type == "zset":
        items = client.zrange(key, 0, -1, withscores=True)
        value = [{"member": member, "score": score} for member, score in items]
    else:
        value = None

    value = normalize_time_fields(value)

    return {
        "key": key,
        "type": key_type,
        "ttl_seconds": ttl,
        "ttl_human": human_ttl(ttl),
        "value": value,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect Redis keys and values for PlayerTracker.")
    parser.add_argument("--redis-url", default=os.getenv("REDIS_URL", "redis://localhost:6379/0"))
    parser.add_argument("--pattern", default="player_tracker:latest_positions")
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--no-json-parse", action="store_true", help="Do not parse JSON strings")
    args = parser.parse_args()

    client = redis.Redis.from_url(args.redis_url, decode_responses=True)

    try:
        client.ping()
    except redis.RedisError as exc:
        raise SystemExit(f"Redis connection failed: {exc}")

    cursor = 0
    keys: list[str] = []

    while True:
        cursor, batch = client.scan(cursor=cursor, match=args.pattern, count=min(args.limit, 1000))
        keys.extend(batch)

        if cursor == 0 or len(keys) >= args.limit:
            break

    keys = sorted(keys)[: args.limit]

    output = {
        "redis_url": args.redis_url,
        "pattern": args.pattern,
        "count": len(keys),
        "items": [read_key(client, key, parse_json=not args.no_json_parse) for key in keys],
    }

    print(json.dumps(output, indent=2, default=str))


if __name__ == "__main__":
    main()
