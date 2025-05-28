from dotenv import load_dotenv
import os
import sys
from fastapi import FastAPI, Request, Response, HTTPException, Header, Depends, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
import logging
import time
import uuid
from service import elevenlabs_stream

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("tts_service")

app = FastAPI(title="TTS Service - Text to Speech")

# Startup check for required env vars
if not os.getenv("ELEVENLABS_API_KEY"):
    logger.fatal("[TTS] Missing required environment variable: ELEVENLABS_API_KEY")
    sys.exit(1)

INTERNAL_API_KEY = os.getenv("INTERNAL_API_KEY", "changeme-internal-key")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
if not ELEVENLABS_API_KEY:
    print("[TTS_SERVICE] WARNING: ELEVENLABS_API_KEY not set!", flush=True)

TTS_ONLY = os.getenv("TTS_ONLY", "0") == "1"
VAD_STT_ONLY = os.getenv("VAD_STT_ONLY", "0") == "1"
LLM_ONLY = os.getenv("LLM_ONLY", "0") == "1"
print(f"[TTS_SERVICE] TTS_ONLY={TTS_ONLY}, VAD_STT_ONLY={VAD_STT_ONLY}, LLM_ONLY={LLM_ONLY}", flush=True)

async def verify_internal_api_key(x_internal_api_key: str = Header(...)):
    if x_internal_api_key != INTERNAL_API_KEY:
        logger.error(f"[TTS] Invalid or missing internal API key: {x_internal_api_key}")
        raise HTTPException(status_code=403, detail="Forbidden: invalid internal API key")

@app.post("/stream-text-to-speech", dependencies=[Depends(verify_internal_api_key)])
async def stream_text_to_speech_endpoint(request: Request):
    data = await request.json()
    text = data.get("text", "")
    if not text:
        return Response(content=b"", status_code=400)
    return StreamingResponse(elevenlabs_stream(text), media_type="audio/mpeg")

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