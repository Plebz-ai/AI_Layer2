from dotenv import load_dotenv
import os
load_dotenv(dotenv_path='../../.env')
from fastapi import FastAPI, Request, Response, HTTPException
from pydantic import BaseModel
from service import speech_to_text, stream_deepgram
from fastapi.responses import StreamingResponse
import asyncio
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("stt_service")

app = FastAPI(title="STT Service - Speech to Text")

class STTRequest(BaseModel):
    audio_data: str  # base64-encoded audio

class STTResponse(BaseModel):
    transcript: str

@app.post("/speech-to-text", response_model=STTResponse)
async def speech_to_text_endpoint(req: STTRequest):
    try:
        transcript = speech_to_text(req.audio_data)
        return STTResponse(transcript=transcript)
    except Exception as e:
        logger.error(f"STT error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/stream-speech-to-text")
async def stream_speech_to_text(request: Request):
    try:
        async def audio_chunk_iter():
            async for chunk in request.stream():
                yield chunk
        return StreamingResponse(stream_deepgram(audio_chunk_iter()), media_type="text/plain")
    except Exception as e:
        logger.error(f"STT streaming error: {e}")
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