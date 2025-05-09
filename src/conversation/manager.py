from typing import Dict, List, Optional, AsyncGenerator
import asyncio
import json
import os
import logging
from datetime import datetime

# Optional Redis dependency
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logging.warning("Redis not available - using in-memory conversation storage")

class ConversationManager:
    def __init__(self):
        # Try to initialize Redis if available
        self.use_redis = False
        
        if REDIS_AVAILABLE:
            try:
                redis_host = os.getenv("REDIS_HOST", "localhost")
                redis_port = int(os.getenv("REDIS_PORT", 6379))
                redis_password = os.getenv("REDIS_PASSWORD", "")
                
                self.redis = redis.Redis(
                    host=redis_host,
                    port=redis_port,
                    password=redis_password,
                    decode_responses=True
                )
                
                # Test connection
                self.redis.ping()
                self.use_redis = True
                logging.info(f"Redis connected at {redis_host}:{redis_port}")
            except Exception as e:
                logging.warning(f"Redis connection failed: {str(e)}")
        
        # If Redis not available, use in-memory storage
        if not self.use_redis:
            self.memory_storage = {}
            logging.info("Using in-memory conversation storage")
        
        # Conversation settings
        self.max_history = 20
        self.message_ttl = 3600  # 1 hour

    async def start_conversation(
        self,
        user_id: str,
        character_id: str,
        is_custom: bool = False
    ) -> str:
        """Start a new conversation."""
        try:
            conversation_id = f"{user_id}_{character_id}_{datetime.now().timestamp()}"
            
            # Initialize conversation data
            conversation_data = {
                "user_id": user_id,
                "character_id": character_id,
                "is_custom": is_custom,
                "started_at": datetime.now().isoformat(),
                "messages": [],
                "status": "active"
            }
            
            # Store in storage
            await self._store_conversation(conversation_id, conversation_data)
            
            return conversation_id
            
        except Exception as e:
            logging.error(f"Error starting conversation: {str(e)}")
            raise

    async def add_message(
        self,
        conversation_id: str,
        content: str,
        sender: str,
        message_type: str = "text"
    ) -> Dict:
        """Add a message to the conversation."""
        try:
            # Get conversation data
            conversation_data = await self._get_conversation(conversation_id)
            if not conversation_data:
                logging.warning(f"Conversation {conversation_id} not found, creating new")
                conversation_data = {
                    "user_id": "unknown",
                    "character_id": "unknown",
                    "is_custom": False,
                    "started_at": datetime.now().isoformat(),
                    "messages": [],
                    "status": "active"
                }

            # Create message
            message = {
                "id": f"msg_{datetime.now().timestamp()}",
                "content": content,
                "sender": sender,
                "type": message_type,
                "timestamp": datetime.now().isoformat()
            }

            # Add to conversation
            conversation_data["messages"].append(message)
            
            # Trim history if needed
            if len(conversation_data["messages"]) > self.max_history:
                conversation_data["messages"] = conversation_data["messages"][-self.max_history:]

            # Update storage
            await self._store_conversation(conversation_id, conversation_data)
            
            return message
            
        except Exception as e:
            logging.error(f"Error adding message: {str(e)}")
            return {
                "id": f"error_{datetime.now().timestamp()}",
                "content": "Failed to save message",
                "sender": sender,
                "type": message_type,
                "timestamp": datetime.now().isoformat()
            }

    async def get_conversation_history(
        self,
        conversation_id: str,
        limit: Optional[int] = None
    ) -> List[Dict]:
        """Get conversation history."""
        try:
            conversation_data = await self._get_conversation(conversation_id)
            if not conversation_data:
                logging.warning(f"Conversation {conversation_id} not found, returning empty history")
                return []

            messages = conversation_data.get("messages", [])
            if limit:
                messages = messages[-limit:]
                
            return messages
            
        except Exception as e:
            logging.error(f"Error getting conversation history: {str(e)}")
            return []

    async def end_conversation(self, conversation_id: str) -> None:
        """End a conversation."""
        try:
            conversation_data = await self._get_conversation(conversation_id)
            if not conversation_data:
                logging.warning(f"Cannot end non-existent conversation: {conversation_id}")
                return

            conversation_data["status"] = "ended"
            conversation_data["ended_at"] = datetime.now().isoformat()
            
            await self._store_conversation(conversation_id, conversation_data)
            
        except Exception as e:
            logging.error(f"Error ending conversation: {str(e)}")

    async def _store_conversation(self, conversation_id: str, data: Dict) -> None:
        """Store conversation data."""
        try:
            if self.use_redis:
                key = f"conversation:{conversation_id}"
                self.redis.setex(
                    key,
                    self.message_ttl,
                    json.dumps(data)
                )
            else:
                # Store in memory
                self.memory_storage[conversation_id] = data
        except Exception as e:
            logging.error(f"Error storing conversation: {str(e)}")

    async def _get_conversation(self, conversation_id: str) -> Optional[Dict]:
        """Get conversation data."""
        try:
            if self.use_redis:
                key = f"conversation:{conversation_id}"
                data = self.redis.get(key)
                return json.loads(data) if data else None
            else:
                # Get from memory
                return self.memory_storage.get(conversation_id)
        except Exception as e:
            logging.error(f"Error getting conversation: {str(e)}")
            return None

    async def cleanup_old_conversations(self) -> None:
        """Clean up old conversations."""
        try:
            # Redis automatically handles TTL
            pass
        except Exception as e:
            print(f"Error cleaning up conversations: {str(e)}")
            raise 