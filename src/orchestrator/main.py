from dotenv import load_dotenv
import os
from fastapi import FastAPI, HTTPException, Request, status, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
from typing import Optional
from service import orchestrate_interaction, router as voice_router
from voice_ws import router as voice_ws_router
import logging
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
import httpx
import time
import uuid
import webrtcvad
import asyncio
import json
import sys

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("orchestrator")

app = FastAPI(title="AI Orchestrator Service")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("[STARTUP] main.py loaded", file=sys.stderr)

try:
    from service import orchestrate_interaction, router as voice_router
    print("[STARTUP] service router loaded", file=sys.stderr)
except Exception as e:
    print(f"[STARTUP ERROR] service import failed: {e}", file=sys.stderr)
    raise

try:
    from voice_ws import router as voice_ws_router
    print("[STARTUP] voice_ws router loaded", file=sys.stderr)
except Exception as e:
    print(f"[STARTUP ERROR] voice_ws import failed: {e}", file=sys.stderr)
    raise

app.include_router(voice_router)
print("[STARTUP] Registered voice_router", file=sys.stderr)
app.include_router(voice_ws_router)
print("[STARTUP] Registered voice_ws_router", file=sys.stderr)

class OrchestratorRequest(BaseModel):
    user_input: str
    character_details: dict
    mode: str  # 'chat' or 'voice'
    audio_data: Optional[str] = None  # base64, for voice mode

class OrchestratorResponse(BaseModel):
    response: str
    audio_data: Optional[str] = None

ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
if not ELEVENLABS_API_KEY:
    print("[ORCHESTRATOR] WARNING: ELEVENLABS_API_KEY not set!", flush=True)

TTS_ONLY = os.getenv("TTS_ONLY", "0") == "1"
VAD_STT_ONLY = os.getenv("VAD_STT_ONLY", "0") == "1"
LLM_ONLY = os.getenv("LLM_ONLY", "0") == "1"
print(f"[ORCHESTRATOR] TTS_ONLY={TTS_ONLY}, VAD_STT_ONLY={VAD_STT_ONLY}, LLM_ONLY={LLM_ONLY}", flush=True)

@app.post("/interact", response_model=OrchestratorResponse)
async def interact(req: OrchestratorRequest, request: Request):
    """
    POST /interact
    - For text chat: send {user_input: str, character_details: dict, mode: "chat"}
    - For voice: send {audio_data: base64-encoded PCM 16kHz mono, character_details: dict, mode: "voice"}
    At least one of user_input or audio_data must be provided.
    """
    request_id = getattr(request.state, 'request_id', 'unknown')
    logger.info(f"[request_id={request_id}] /interact payload: user_input length={len(req.user_input)}, character_details keys={list(req.character_details.keys())}, mode={req.mode}, audio_data length={len(req.audio_data) if req.audio_data else 0}")
    import traceback
    # Strict input validation
    errors = []
    if not isinstance(req.user_input, str):
        errors.append("user_input must be a string")
    if not isinstance(req.character_details, dict):
        errors.append("character_details must be an object")
    if not isinstance(req.mode, str):
        errors.append("mode must be a string")
    if req.audio_data is not None and not isinstance(req.audio_data, str):
        errors.append("audio_data must be a string or null")
    if (not req.user_input or len(req.user_input.strip()) == 0) and (not req.audio_data or len(req.audio_data.strip()) == 0):
        errors.append("Either user_input (text) or audio_data (voice) must be provided.")
    # If mode is voice, check audio_data is valid base64
    if req.mode == "voice" and req.audio_data:
        import base64
        try:
            base64.b64decode(req.audio_data)
        except Exception:
            errors.append("audio_data must be valid base64-encoded PCM bytes for voice mode.")
    if errors:
        logger.error(f"[request_id={request_id}] Validation error(s) in /interact: {errors}")
        return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST, content={"detail": errors})
    try:
        result = await orchestrate_interaction(
            user_input=req.user_input,
            character_details=req.character_details,
            mode=req.mode,
            audio_data=req.audio_data
        )
        logger.info(f"[request_id={request_id}] /interact response: response length={len(result.get('response',''))}, audio_data length={len(result.get('audio_data') or '')}")
        if result.get("error"):
            return JSONResponse(status_code=500, content={"response": result.get("response", "Sorry, something went wrong."), "audio_data": result.get("audio_data"), "error": result["error"]})
        return OrchestratorResponse(**result)
    except Exception as e:
        logger.error(f"[request_id={request_id}] Exception in /interact: {e}")
        logger.error(traceback.format_exc())
        return JSONResponse(status_code=500, content={"response": "Sorry, something went wrong.", "audio_data": None, "error": {"exception": str(e)}})

@app.get("/health")
async def health():
    # Check downstream services
    services = {
        "llm1": "http://llm1_service:8001/health",
        "llm2": "http://llm2_service:8002/health",
        "stt": "http://stt_service:8003/health",
        "tts": "http://tts_service:8004/health",
    }
    results = {}
    unhealthy = []
    async with httpx.AsyncClient(timeout=2.0) as client:
        for name, url in services.items():
            try:
                resp = await client.get(url)
                if resp.status_code == 200:
                    results[name] = "ok"
                else:
                    results[name] = f"error: {resp.text}"
                    unhealthy.append(name)
            except Exception as e:
                results[name] = f"error: {str(e)}"
                unhealthy.append(name)
    status_code = 200 if not unhealthy else 500
    if unhealthy:
        logger.warning(f"[Orchestrator] /health: unhealthy services: {unhealthy}")
    return {"status": "ok" if not unhealthy else "error", "unhealthy": unhealthy, "services": results}, status_code

@app.options("/interact")
async def options_interact():
    return JSONResponse(status_code=200, content={})

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

@app.post("/stream-speech-to-text")
async def stream_speech_to_text(request: Request):
    """
    POST /stream-speech-to-text
    - Expects raw PCM 16kHz mono audio stream in the request body (not JSON, not base64).
    - Proxies the stream to the STT service and streams back the transcript.
    - For real-time/streaming voice, use the WebSocket endpoint instead.
    """
    request_id = getattr(request.state, 'request_id', 'unknown')
    logger.info(f"[request_id={request_id}] /stream-speech-to-text called (method={request.method}) headers={dict(request.headers)}")
    stt_url = "http://stt_service:8003/stream-speech-to-text"
    start = time.time()
    INTERNAL_API_KEY = os.getenv("INTERNAL_API_KEY", "changeme-internal-key")
    async with httpx.AsyncClient(timeout=None) as client:
        try:
            async def proxy():
                headers = {"x-internal-api-key": INTERNAL_API_KEY}
                async with client.stream("POST", stt_url, content=request.stream(), headers=headers) as resp:
                    async for chunk in resp.aiter_bytes():
                        yield chunk
            response = StreamingResponse(proxy(), media_type="text/plain")
            # Add CORS headers explicitly for streaming POST
            response.headers["Access-Control-Allow-Origin"] = "*"
            response.headers["Access-Control-Allow-Methods"] = "POST, OPTIONS"
            response.headers["Access-Control-Allow-Headers"] = "Content-Type"
            latency = (time.time() - start) * 1000
            logger.info(f"[request_id={request_id}] /stream-speech-to-text setup complete | Latency: {latency:.2f}ms")
            return response
        except Exception as e:
            logger.error(f"[request_id={request_id}] /stream-speech-to-text error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

@app.post("/stream-text-to-speech")
async def stream_text_to_speech(request: Request):
    request_id = getattr(request.state, 'request_id', 'unknown')
    logger.info(f"[request_id={request_id}] /stream-text-to-speech called")
    tts_url = "http://tts_service:8004/stream-text-to-speech"
    start = time.time()
    async with httpx.AsyncClient(timeout=None) as client:
        try:
            async def proxy():
                async with client.stream("POST", tts_url, content=await request.body()) as resp:
                    async for chunk in resp.aiter_bytes():
                        yield chunk
            response = StreamingResponse(proxy(), media_type="text/plain")
            latency = (time.time() - start) * 1000
            logger.info(f"[request_id={request_id}] /stream-text-to-speech setup complete | Latency: {latency:.2f}ms")
            return response
        except Exception as e:
            logger.error(f"[request_id={request_id}] /stream-text-to-speech error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/ws/health")
async def ws_health(websocket: WebSocket):
    await websocket.accept()
    await websocket.send_text("ok")
    await websocket.close()

@app.websocket("/ws/echo")
async def ws_echo(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_bytes()
            await websocket.send_bytes(data)
    except Exception as e:
        print(f"[WS ECHO] Connection closed or error: {e}")
    finally:
        await websocket.close() 