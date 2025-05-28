# TTS Service Logic (Text to Speech)

import os
import logging
import httpx
from fastapi import FastAPI, Request, Response, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse

# --- CONFIG ---
ELEVENLABS_API_KEY = os.environ.get("ELEVENLABS_API_KEY", "")
# Use a free ElevenLabs voice (Rachel, voice_id: 21m00Tcm4TlvDq8ikWAM)
ELEVENLABS_VOICE_ID = os.environ.get("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")  # Rachel (free)
ELEVENLABS_MODEL_ID = os.environ.get("ELEVENLABS_MODEL_ID", "eleven_multilingual_v2")
ELEVENLABS_URL = f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVENLABS_VOICE_ID}/stream"

app = FastAPI()
logger = logging.getLogger("tts_service")
logging.basicConfig(level=logging.INFO)

async def elevenlabs_stream(text: str):
    headers = {
        "xi-api-key": ELEVENLABS_API_KEY,
        "Content-Type": "application/json",
        "Accept": "audio/mpeg",
    }
    payload = {
        "text": text,
        "model_id": ELEVENLABS_MODEL_ID,
        "voice_settings": {"stability": 0.5, "similarity_boost": 0.5},
    }
    try:
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream("POST", ELEVENLABS_URL, headers=headers, json=payload) as resp:
                if resp.status_code != 200:
                    error_body = await resp.aread()
                    logger.error(f"TTS error: {resp.status_code} {error_body.decode(errors='ignore')}")
                    if b'free_users_not_allowed' in error_body:
                        logger.error("The selected ElevenLabs voice is not available for free users. Please use a free voice.")
                    yield b""
                    return
                async for chunk in resp.aiter_bytes():
                    yield chunk
    except Exception as e:
        logger.error(f"ElevenLabs TTS connection failed: {e}")
        yield b""

@app.post("/stream-text-to-speech")
async def stream_text_to_speech(request: Request):
    data = await request.json()
    text = data.get("text", "")
    if not text:
        return Response(content=b"", status_code=400)
    return StreamingResponse(elevenlabs_stream(text), media_type="audio/mpeg")

@app.websocket("/ws/stream-text-to-speech")
async def websocket_text_to_speech(ws: WebSocket):
    await ws.accept()
    logger.info("[TTS WS] Client connected")
    try:
        while True:
            try:
                data = await ws.receive_text()
            except WebSocketDisconnect:
                logger.info("[TTS WS] Client disconnected")
                break
            except Exception as e:
                logger.error(f"[TTS WS] Receive error: {e}")
                break
            # Stream to ElevenLabs
            async for chunk in elevenlabs_stream(data):
                await ws.send_bytes(chunk)
    except Exception as e:
        logger.error(f"[TTS WS] Unexpected error: {e}")
    finally:
        await ws.close()
        logger.info("[TTS WS] Connection closed") 