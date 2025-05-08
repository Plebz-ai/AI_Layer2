from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import json
import asyncio
from typing import Dict, List, Optional
import os
from dotenv import load_dotenv

from speech.service import SpeechService
from llm.service import LLMService
from conversation.manager import ConversationManager

load_dotenv()

app = FastAPI(title="Character AI Platform")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
speech_service = SpeechService()
llm_service = LLMService()
conversation_manager = ConversationManager()

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}

    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket

    def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]

    async def send_message(self, message: str, client_id: str):
        if client_id in self.active_connections:
            await self.active_connections[client_id].send_text(message)

manager = ConnectionManager()

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await manager.connect(websocket, client_id)
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            
            # Handle different message types
            if message["type"] == "start_conversation":
                conversation_id = await conversation_manager.start_conversation(
                    client_id,
                    message["character_id"],
                    message.get("is_custom", False)
                )
                await manager.send_message(
                    json.dumps({
                        "type": "conversation_started",
                        "conversation_id": conversation_id
                    }),
                    client_id
                )
                
            elif message["type"] == "text_message":
                # Get conversation history
                history = await conversation_manager.get_conversation_history(
                    message["conversation_id"]
                )
                
                # Generate response
                if message.get("is_custom", False):
                    response = await llm_service.generate_custom_character_response(
                        message["character_config"],
                        message["content"],
                        history
                    )
                else:
                    response = await llm_service.generate_predefined_response(
                        message["character_id"],
                        message["content"],
                        history
                    )
                
                # Add messages to conversation
                await conversation_manager.add_message(
                    message["conversation_id"],
                    message["content"],
                    "user"
                )
                await conversation_manager.add_message(
                    message["conversation_id"],
                    response,
                    "character"
                )
                
                # Send response
                await manager.send_message(
                    json.dumps({
                        "type": "text_response",
                        "content": response
                    }),
                    client_id
                )
                
            elif message["type"] == "voice_message":
                # Convert speech to text
                text = await speech_service.speech_to_text_stream(
                    message["audio_data"]
                )
                
                # Get conversation history
                history = await conversation_manager.get_conversation_history(
                    message["conversation_id"]
                )
                
                # Generate response
                if message.get("is_custom", False):
                    response = await llm_service.generate_custom_character_response(
                        message["character_config"],
                        text,
                        history
                    )
                else:
                    response = await llm_service.generate_predefined_response(
                        message["character_id"],
                        text,
                        history
                    )
                
                # Add messages to conversation
                await conversation_manager.add_message(
                    message["conversation_id"],
                    text,
                    "user",
                    "voice"
                )
                await conversation_manager.add_message(
                    message["conversation_id"],
                    response,
                    "character",
                    "voice"
                )
                
                # Convert response to speech
                audio_stream = await speech_service.text_to_speech_stream(
                    response,
                    message["voice_id"]
                )
                
                # Send audio response
                async for audio_chunk in audio_stream:
                    await manager.send_message(
                        json.dumps({
                            "type": "voice_response",
                            "audio_data": audio_chunk
                        }),
                        client_id
                    )
                    
    except WebSocketDisconnect:
        manager.disconnect(client_id)
    except Exception as e:
        print(f"Error in websocket: {str(e)}")
        await manager.send_message(
            json.dumps({
                "type": "error",
                "message": str(e)
            }),
            client_id
        )

@app.get("/characters")
async def get_characters():
    """Get list of available characters."""
    try:
        return llm_service.predefined_characters
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/voices")
async def get_voices():
    """Get list of available voices."""
    try:
        return await speech_service.get_available_voices()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        reload=True
    ) 