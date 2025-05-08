from typing import Dict, List, Optional, AsyncGenerator
import asyncio
import json
import os
from datetime import datetime
import redis
from dotenv import load_dotenv

load_dotenv()

class ConversationManager:
    def __init__(self):
        # Initialize Redis connection
        self.redis = redis.Redis(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=int(os.getenv("REDIS_PORT", 6379)),
            password=os.getenv("REDIS_PASSWORD", ""),
            decode_responses=True
        )
        
        # Conversation settings
        self.max_history = 10
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
            
            # Store in Redis
            await self._store_conversation(conversation_id, conversation_data)
            
            return conversation_id
            
        except Exception as e:
            print(f"Error starting conversation: {str(e)}")
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
                raise ValueError(f"Conversation {conversation_id} not found")

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

            # Update Redis
            await self._store_conversation(conversation_id, conversation_data)
            
            return message
            
        except Exception as e:
            print(f"Error adding message: {str(e)}")
            raise

    async def get_conversation_history(
        self,
        conversation_id: str,
        limit: Optional[int] = None
    ) -> List[Dict]:
        """Get conversation history."""
        try:
            conversation_data = await self._get_conversation(conversation_id)
            if not conversation_data:
                raise ValueError(f"Conversation {conversation_id} not found")

            messages = conversation_data["messages"]
            if limit:
                messages = messages[-limit:]
                
            return messages
            
        except Exception as e:
            print(f"Error getting conversation history: {str(e)}")
            raise

    async def end_conversation(self, conversation_id: str) -> None:
        """End a conversation."""
        try:
            conversation_data = await self._get_conversation(conversation_id)
            if not conversation_data:
                raise ValueError(f"Conversation {conversation_id} not found")

            conversation_data["status"] = "ended"
            conversation_data["ended_at"] = datetime.now().isoformat()
            
            await self._store_conversation(conversation_id, conversation_data)
            
        except Exception as e:
            print(f"Error ending conversation: {str(e)}")
            raise

    async def _store_conversation(self, conversation_id: str, data: Dict) -> None:
        """Store conversation data in Redis."""
        try:
            key = f"conversation:{conversation_id}"
            await self.redis.setex(
                key,
                self.message_ttl,
                json.dumps(data)
            )
        except Exception as e:
            print(f"Error storing conversation: {str(e)}")
            raise

    async def _get_conversation(self, conversation_id: str) -> Optional[Dict]:
        """Get conversation data from Redis."""
        try:
            key = f"conversation:{conversation_id}"
            data = await self.redis.get(key)
            return json.loads(data) if data else None
        except Exception as e:
            print(f"Error getting conversation: {str(e)}")
            raise

    async def cleanup_old_conversations(self) -> None:
        """Clean up old conversations."""
        try:
            # Redis automatically handles TTL
            pass
        except Exception as e:
            print(f"Error cleaning up conversations: {str(e)}")
            raise 