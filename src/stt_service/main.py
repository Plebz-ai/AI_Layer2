from dotenv import load_dotenv
import os
import sys
import time
import uuid
from fastapi import FastAPI, Request, Response, HTTPException, Header, Depends
from pydantic import BaseModel
from service import speech_to_text, stream_deepgram
from fastapi.responses import StreamingResponse
import asyncio
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("stt_service")

app = FastAPI(title="STT Service - Speech to Text")

# Startup check for required env vars
if not os.getenv("DEEPGRAM_API_KEY"):
    logger.fatal("[STT] Missing required environment variable: DEEPGRAM_API_KEY")
    sys.exit(1)

INTERNAL_API_KEY = os.getenv("INTERNAL_API_KEY", "changeme-internal-key")

class STTRequest(BaseModel):
    audio_data: str  # base64-encoded audio

class STTResponse(BaseModel):
    transcript: str

async def verify_internal_api_key(x_internal_api_key: str = Header(...)):
    if x_internal_api_key != INTERNAL_API_KEY:
        logger.error(f"[STT] Invalid or missing internal API key: {x_internal_api_key}")
        raise HTTPException(status_code=403, detail="Forbidden: invalid internal API key")

@app.post("/speech-to-text", response_model=STTResponse, dependencies=[Depends(verify_internal_api_key)])
async def speech_to_text_endpoint(req: STTRequest, request: Request):
    request_id = getattr(request.state, 'request_id', 'unknown')
    logger.info(f"[request_id={request_id}] /speech-to-text payload: [audio_data omitted, length={len(req.audio_data)}]")
    if not os.getenv("DEEPGRAM_API_KEY"):
        logger.warning(f"[request_id={request_id}] /speech-to-text called but DEEPGRAM_API_KEY is missing!")
    try:
        transcript = await speech_to_text(req.audio_data)
        logger.info(f"[request_id={request_id}] /speech-to-text response: transcript length={len(transcript)}")
        return STTResponse(transcript=transcript)
    except Exception as e:
        import traceback
        logger.error(f"[request_id={request_id}] STT error: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/stream-speech-to-text", dependencies=[Depends(verify_internal_api_key)])
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
    if not os.getenv("DEEPGRAM_API_KEY"):
        logger.warning("[STT] /health called but DEEPGRAM_API_KEY is missing!")
        return {"status": "error", "error": "Missing DEEPGRAM_API_KEY"}, 500
    return {"status": "ok"}

@app.middleware("http")
async def log_requests(request: Request, call_next):
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    logger.info(f"[request_id={request_id}] Request: {request.method} {request.url}")
    start = time.time()
    try:
        response = await call_next(request)
        latency = (time.time() - start) * 1000
        logger.info(f"[request_id={request_id}] Response status: {response.status_code} | Latency: {latency:.2f}ms")
        return response
    except Exception as e:
        import traceback
        logger.error(f"[request_id={request_id}] Error: {e}\n{traceback.format_exc()}")
        raise 