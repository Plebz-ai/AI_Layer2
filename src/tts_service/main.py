from dotenv import load_dotenv
import os
load_dotenv(dotenv_path='../../.env')
from fastapi import FastAPI, Request, Response, HTTPException
from pydantic import BaseModel
from service import text_to_speech, stream_text_to_speech
from fastapi.responses import StreamingResponse
import asyncio
import base64
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("tts_service")

app = FastAPI(title="TTS Service - Text to Speech")

class TTSRequest(BaseModel):
    text: str
    voice_type: str = "predefined"  # male, female, predefined

class TTSResponse(BaseModel):
    audio_data: str  # base64-encoded audio

@app.post("/text-to-speech", response_model=TTSResponse)
async def text_to_speech_endpoint(req: TTSRequest):
    try:
        audio_bytes = await text_to_speech(req.text, req.voice_type)
        audio_data = base64.b64encode(audio_bytes).decode("utf-8")
        return TTSResponse(audio_data=audio_data)
    except Exception as e:
        logger.error(f"TTS error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/stream-text-to-speech")
async def stream_text_to_speech_endpoint(req: TTSRequest):
    try:
        async def audio_stream():
            async for chunk in stream_text_to_speech(req.text, req.voice_type):
                yield base64.b64encode(chunk).decode("utf-8")
                await asyncio.sleep(0.01)
        return StreamingResponse(audio_stream(), media_type="text/plain")
    except Exception as e:
        logger.error(f"TTS streaming error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.info(f"Request: {request.method} {request.url}")
    try:
        response = await call_next(request)
        logger.info(f"Response status: {response.status_code}")
        return response
    except Exception as e:
        logger.error(f"Error: {e}")
        raise 