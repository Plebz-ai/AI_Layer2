from dotenv import load_dotenv
import os
import sys
import time
import uuid
from fastapi import Request, Response, HTTPException, Header, Depends
from pydantic import BaseModel
from service import app  
from fastapi.responses import StreamingResponse
import asyncio
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("stt_service")


if not os.getenv("DEEPGRAM_API_KEY"):
    logger.fatal("[STT] Missing required environment variable: DEEPGRAM_API_KEY")
    sys.exit(1)

INTERNAL_API_KEY = os.getenv("INTERNAL_API_KEY", "changeme-internal-key")
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
if not DEEPGRAM_API_KEY:
    print("[STT_SERVICE] WARNING: DEEPGRAM_API_KEY not set!", flush=True)

TTS_ONLY = os.getenv("TTS_ONLY", "0") == "1"
VAD_STT_ONLY = os.getenv("VAD_STT_ONLY", "0") == "1"
LLM_ONLY = os.getenv("LLM_ONLY", "0") == "1"
print(f"[STT_SERVICE] TTS_ONLY={TTS_ONLY}, VAD_STT_ONLY={VAD_STT_ONLY}, LLM_ONLY={LLM_ONLY}", flush=True)

class STTRequest(BaseModel):
    audio_data: str

class STTResponse(BaseModel):
    transcript: str

async def verify_internal_api_key(x_internal_api_key: str = Header(...)):
    if x_internal_api_key != INTERNAL_API_KEY:
        logger.error(f"[STT] Invalid or missing internal API key: {x_internal_api_key}")
        raise HTTPException(status_code=403, detail="Forbidden: invalid internal API key")

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