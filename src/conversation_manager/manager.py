import redis
import json

class ConversationManager:
    def __init__(self, redis_url):
        self.redis = redis.StrictRedis.from_url(redis_url)

    def save_context(self, session_id, context):
        self.redis.set(session_id, json.dumps(context))

    def get_context(self, session_id):
        context = self.redis.get(session_id)
        return json.loads(context) if context else {}

    def delete_context(self, session_id):
        self.redis.delete(session_id)