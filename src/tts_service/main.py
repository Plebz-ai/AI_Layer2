from dotenv import load_dotenv
import os
import sys
from fastapi import FastAPI, Request, Response, HTTPException, Header, Depends
from pydantic import BaseModel
from service import text_to_speech, stream_text_to_speech, get_voice_config
from fastapi.responses import StreamingResponse
import asyncio
import base64
import logging
import time
import uuid

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("tts_service")

app = FastAPI(title="TTS Service - Text to Speech")

# Startup check for required env vars
if not os.getenv("ELEVENLABS_API_KEY"):
    logger.fatal("[TTS] Missing required environment variable: ELEVENLABS_API_KEY")
    sys.exit(1)

INTERNAL_API_KEY = os.getenv("INTERNAL_API_KEY", "changeme-internal-key")

class TTSRequest(BaseModel):
    text: str
    voice_type: str = "predefined"  # male, female, predefined

class TTSResponse(BaseModel):
    audio_data: str  # base64-encoded audio

async def verify_internal_api_key(x_internal_api_key: str = Header(...)):
    if x_internal_api_key != INTERNAL_API_KEY:
        logger.error(f"[TTS] Invalid or missing internal API key: {x_internal_api_key}")
        raise HTTPException(status_code=403, detail="Forbidden: invalid internal API key")

@app.post("/text-to-speech", response_model=TTSResponse, dependencies=[Depends(verify_internal_api_key)])
async def text_to_speech_endpoint(req: TTSRequest, request: Request):
    request_id = getattr(request.state, 'request_id', 'unknown')
    logger.info(f"[request_id={request_id}] /text-to-speech payload: text length={len(req.text)}, voice_type={req.voice_type}")
    if not os.getenv("ELEVENLABS_API_KEY"):
        logger.warning(f"[request_id={request_id}] /text-to-speech called but ELEVENLABS_API_KEY is missing!")
    try:
        # Validate voice_type/model_id/voice_id
        try:
            get_voice_config(req.voice_type)
        except Exception as e:
            logger.error(f"[request_id={request_id}] Invalid TTS config: {e}")
            raise HTTPException(status_code=400, detail=str(e))
        audio_bytes = await text_to_speech(req.text, req.voice_type)
        logger.info(f"[request_id={request_id}] /text-to-speech response: audio bytes length={len(audio_bytes)}")
        audio_data = base64.b64encode(audio_bytes).decode("utf-8")
        return TTSResponse(audio_data=audio_data)
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        logger.error(f"[request_id={request_id}] TTS error: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/stream-text-to-speech", dependencies=[Depends(verify_internal_api_key)])
async def stream_text_to_speech_endpoint(req: TTSRequest):
    try:
        try:
            get_voice_config(req.voice_type)
        except Exception as e:
            logger.error(f"Invalid TTS config: {e}")
            raise HTTPException(status_code=400, detail=str(e))
        async def audio_stream():
            async for chunk in stream_text_to_speech(req.text, req.voice_type):
                yield base64.b64encode(chunk).decode("utf-8")
                await asyncio.sleep(0.01)
        return StreamingResponse(audio_stream(), media_type="text/plain")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"TTS streaming error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    if not os.getenv("ELEVENLABS_API_KEY"):
        logger.warning("[TTS] /health called but ELEVENLABS_API_KEY is missing!")
        return {"status": "error", "error": "Missing ELEVENLABS_API_KEY"}, 500
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