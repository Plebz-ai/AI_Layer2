import aioredis
import json
import os

REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")

async def get_redis():
    return await aioredis.from_url(REDIS_URL, decode_responses=True)

async def get_session(session_id: str):
    redis = await get_redis()
    data = await redis.get(f"session:{session_id}")
    if data:
        return json.loads(data)
    return None

async def set_session(session_id: str, session_data: dict):
    redis = await get_redis()
    await redis.set(f"session:{session_id}", json.dumps(session_data), ex=3600)

async def delete_session(session_id: str):
    redis = await get_redis()
    await redis.delete(f"session:{session_id}") 