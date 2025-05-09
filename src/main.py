from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import json
import asyncio
from typing import Dict, List, Optional
import os
from dotenv import load_dotenv
import base64

from speech.service import SpeechService
from llm.service import LLMService
from conversation.manager import ConversationManager

from pydantic import BaseModel

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
                text = await speech_service.speech_to_text(
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
                audio_data = await speech_service.text_to_speech(
                    response,
                    message.get("voice_id", "en-US-JennyNeural")
                )
                
                # Send audio response
                await manager.send_message(
                    json.dumps({
                        "type": "voice_response",
                        "audio_data": base64.b64encode(audio_data).decode('utf-8')
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
        return await speech_service.get_available_voices_azure()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# New REST endpoints for HTTP-based AI service integration
class ChatRequest(BaseModel):
    character_id: int
    content: str

class ChatResponse(BaseModel):
    text: str

@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest):
    # Generate chat response (no history)
    text = await llm_service.generate_predefined_response(
        str(req.character_id), req.content, []
    )
    return ChatResponse(text=text)

class STTRequest(BaseModel):
    audio_data: str  # base64-encoded audio

class STTResponse(BaseModel):
    transcript: str

@app.post("/api/speech-to-text", response_model=STTResponse)
async def stt_endpoint(req: STTRequest):
    audio_bytes = base64.b64decode(req.audio_data)
    transcript = await speech_service.speech_to_text(audio_bytes)
    return STTResponse(transcript=transcript)

class TTSRequest(BaseModel):
    text: str
    voice_name: Optional[str] = "en-US-JennyNeural"

class TTSResponse(BaseModel):
    audio_data: str  # base64-encoded audio

@app.post("/api/text-to-speech", response_model=TTSResponse)
async def tts_endpoint(req: TTSRequest):
    audio_bytes = await speech_service.text_to_speech(req.text, req.voice_name)
    audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')
    return TTSResponse(audio_data=audio_b64)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", 5000)),
        reload=True
    ) 