import json
import logging
import uuid
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Request
from starlette.websockets import WebSocketState
import base64
import httpx
import os
import asyncio
from utils.redis_session import get_session, set_session, delete_session
from speech.vad import is_speech
import sys
import numpy as np
import time
import threading
import websockets

router = APIRouter()
logger = logging.getLogger("voice_ws")

# Message types
MSG_TYPE_INIT = "init"
MSG_TYPE_AUDIO_CHUNK = "audio_chunk"
MSG_TYPE_VAD_STATE = "vad_state"
MSG_TYPE_TRANSCRIPT_PARTIAL = "transcript_partial"
MSG_TYPE_TRANSCRIPT_FINAL = "transcript_final"
MSG_TYPE_LLM2_PARTIAL = "llm2_partial"
MSG_TYPE_LLM2_FINAL = "llm2_final"
MSG_TYPE_TTS_CHUNK = "tts_chunk"
MSG_TYPE_TTS_END = "tts_end"
MSG_TYPE_BARGE_IN = "barge_in"
MSG_TYPE_GREETING = "greeting"
MSG_TYPE_ERROR = "error"

# Service URLs (use orchestrator/service.py logic)
STT_URL = os.getenv("STT_URL", "http://stt_service:8003/speech-to-text")
LLM2_URL = os.getenv("LLM2_URL", "http://llm2_service:8002/generate-response")
TTS_STREAM_URL = os.getenv("TTS_STREAM_URL", "http://tts_service:8004/stream-text-to-speech")
STT_STREAM_URL = os.getenv("STT_STREAM_URL", "http://stt_service:8003/stream-speech-to-text")
INTERNAL_API_KEY = os.getenv("INTERNAL_API_KEY", "changeme-internal-key")

VAD_STT_ONLY = os.getenv("VAD_STT_ONLY", "0") == "1"
TTS_ONLY = os.getenv("TTS_ONLY", "0") == "1"
LLM_ONLY = os.getenv("LLM_ONLY", "0") == "1"

print("[STARTUP] voice_ws.py loaded", file=sys.stderr)

# Add buffer dump state
DUMP_LIMIT = 5
received_buffers = {}

@router.on_event("startup")
async def startup_event():
    print("[ORCH] Orchestrator started and listening for WebSocket connections on /ws/voice-session", file=sys.stderr)

@router.get("/ws/voice-session")
async def ws_voice_session_catchall(request: Request):
    print(f"[ORCH] Received non-WebSocket request to /ws/voice-session: method={request.method}, headers={dict(request.headers)}", file=sys.stderr)
    return {"error": "This endpoint only supports WebSocket connections."}

@router.websocket("/ws/voice-session")
async def voice_session_ws(websocket: WebSocket):
    print("[WS] Connection attempt received", file=sys.stderr)
    await websocket.accept()
    session_id = str(uuid.uuid4())
    logger.info(f"[WS] New voice session: {session_id}")
    print(f"[WS] New voice session: {session_id}", file=sys.stderr)
    session_data = {
        "id": session_id,
        "state": {},
        "buffer": bytearray(),
        "llm1_context": None,
        "character_details": None,
        "history": [],
        "tts_playing": False,
    }
    await set_session(session_id, {**session_data, "buffer": ""})
    received_buffers[session_id] = 0
    try:
        # 1. Wait for INIT message with character details
        print(f"[WS {session_id}] Waiting for INIT message", file=sys.stderr)
        init_msg = await websocket.receive_text()
        print(f"[WS {session_id}] Received INIT message: {init_msg}", file=sys.stderr)
        try:
            init_data = json.loads(init_msg)
            if init_data.get("type") != MSG_TYPE_INIT or "characterDetails" not in init_data:
                raise ValueError("First message must be INIT with character_details")
        except Exception as e:
            logger.error(f"[WS {session_id}] Invalid INIT: {e}")
            print(f"[WS {session_id}] Invalid INIT: {e}", file=sys.stderr)
            await websocket.send_json({"type": MSG_TYPE_ERROR, "error": f"Invalid INIT: {e}"})
            await websocket.close()
            print(f"[WS {session_id}] Closed due to invalid INIT", file=sys.stderr)
            return
        session = await get_session(session_id)
        session["character_details"] = init_data["characterDetails"]
        await set_session(session_id, session)
        logger.info(f"[WS {session_id}] Session initialized with character: {init_data['characterDetails']}")
        print(f"[WS {session_id}] Session initialized with character: {init_data['characterDetails']}", file=sys.stderr)
        # 2. Run LLM1 to generate system prompt/context (stub for now)
        llm1_context = f"[SYSTEM_PROMPT for {init_data['characterDetails'].get('name', 'character')}]"
        session["llm1_context"] = llm1_context
        await set_session(session_id, session)
        # 3. Send AI greeting (stub for now)
        greeting_text = f"Hello, I am {init_data['characterDetails'].get('name', 'your assistant')}! How can I help you today?"
        await websocket.send_json({"type": MSG_TYPE_GREETING, "text": greeting_text})
        logger.info(f"[WS {session_id}] Sent greeting: {greeting_text}")
        # --- NEW: Open persistent WebSocket to STT service ---
        stt_ws_url = "ws://stt_service:8003/ws/stream-speech-to-text"
        async with websockets.connect(stt_ws_url, max_size=2**24) as stt_ws:
            async def frontend_to_stt():
                while True:
                    msg = await websocket.receive()
                    if msg["type"] == "websocket.disconnect":
                        logger.info(f"[WS {session_id}] WebSocket disconnected.")
                        break
                    if msg["type"] == "websocket.receive" and "bytes" in msg:
                        audio_chunk = msg["bytes"]
                        await stt_ws.send(audio_chunk)
                        # logger.info(f"[WS {session_id}] Forwarded {len(audio_chunk)} bytes to STT WS")
            async def stt_to_frontend():
                async for stt_msg in stt_ws:
                    try:
                        data = json.loads(stt_msg)
                        if data.get("type") == "transcript":
                            transcript = data["text"]
                            await websocket.send_json({"type": MSG_TYPE_TRANSCRIPT_FINAL, "text": transcript})
                            logger.info(f"[WS {session_id}] Forwarded transcript to frontend: {transcript}")

                            # --- NEW: Call LLM2 for a response ---
                            session = await get_session(session_id)
                            character_details = session.get("character_details", {})
                            history = session.get("history", [])
                            persona_context = session.get("llm1_context", "")
                            llm2_payload = {
                                "user_query": transcript,
                                "persona_context": persona_context,
                                "rules": {},  # TODO: fill with actual rules if available
                                "model": "gpt-4o-mini"
                            }
                            try:
                                async with httpx.AsyncClient() as client:
                                    resp = await client.post(
                                        LLM2_URL,
                                        json=llm2_payload,
                                        headers={"x-internal-api-key": INTERNAL_API_KEY}
                                    )
                                    llm2_response = resp.json().get("response", "")
                            except Exception as e:
                                logger.error(f"[WS {session_id}] Error calling LLM2: {e}")
                                llm2_response = "[Error: LLM2 unavailable]"
                            # Update history
                            history.append({"role": "user", "content": transcript})
                            history.append({"role": "assistant", "content": llm2_response})
                            session["history"] = history
                            await set_session(session_id, session)
                            # Send LLM2 response to frontend
                            await websocket.send_json({"type": MSG_TYPE_LLM2_FINAL, "text": llm2_response})
                            logger.info(f"[WS {session_id}] Forwarded LLM2 response to frontend: {llm2_response}")

                            # --- NEW: Stream TTS audio to frontend ---
                            try:
                                async with httpx.AsyncClient(timeout=None) as client:
                                    tts_headers = {"x-internal-api-key": INTERNAL_API_KEY}
                                    tts_payload = {"text": llm2_response}
                                    tts_url = TTS_STREAM_URL
                                    async with client.stream("POST", tts_url, headers=tts_headers, json=tts_payload) as tts_resp:
                                        if tts_resp.status_code != 200:
                                            error_body = await tts_resp.aread()
                                            logger.error(f"[WS {session_id}] TTS error: {tts_resp.status_code} {error_body.decode(errors='ignore')}")
                                            await websocket.send_json({"type": MSG_TYPE_ERROR, "error": f"TTS error: {tts_resp.status_code}"})
                                        else:
                                            async for chunk in tts_resp.aiter_bytes():
                                                if chunk:
                                                    await websocket.send_bytes(json.dumps({"type": MSG_TYPE_TTS_CHUNK, "audio": base64.b64encode(chunk).decode()} ).encode())
                                            await websocket.send_json({"type": MSG_TYPE_TTS_END})
                                            logger.info(f"[WS {session_id}] Streamed TTS audio to frontend.")
                            except Exception as e:
                                logger.error(f"[WS {session_id}] Error streaming TTS audio: {e}")
                                await websocket.send_json({"type": MSG_TYPE_ERROR, "error": f"TTS streaming error: {e}"})
                        else:
                            await websocket.send_json(data)
                    except Exception as e:
                        logger.error(f"[WS {session_id}] Error parsing STT WS message: {e}")
            await asyncio.gather(frontend_to_stt(), stt_to_frontend())
    except WebSocketDisconnect:
        logger.info(f"[WS {session_id}] Session disconnected (WebSocketDisconnect)")
        print(f"[WS {session_id}] Session disconnected (WebSocketDisconnect)", file=sys.stderr)
    except Exception as e:
        logger.error(f"[WS {session_id}] Error in session: {e}")
        print(f"[WS {session_id}] Error in session: {e}", file=sys.stderr)
        try:
            await websocket.send_json({"type": MSG_TYPE_ERROR, "error": str(e)})
        except Exception:
            pass
    finally:
        logger.info(f"[WS {session_id}] Cleaning up session.")
        print(f"[WS {session_id}] Cleaning up session.", file=sys.stderr)
        await delete_session(session_id)
        if websocket.application_state != WebSocketState.DISCONNECTED:
            try:
                await websocket.close()
                print(f"[WS {session_id}] WebSocket closed in finally block", file=sys.stderr)
            except Exception:
                pass 