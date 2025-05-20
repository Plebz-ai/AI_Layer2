import json
import logging
import uuid
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState
import base64
import httpx
import os
import asyncio
from utils.redis_session import get_session, set_session, delete_session
from speech.vad import is_speech
import sys
import numpy as np

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

@router.websocket("/ws/voice-session")
async def voice_session_ws(websocket: WebSocket):
    await websocket.accept()
    session_id = str(uuid.uuid4())
    logger.info(f"[WS] New voice session: {session_id}")
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
    # Track how many buffers we've dumped for this session
    received_buffers[session_id] = 0
    try:
        # 1. Wait for INIT message with character details
        init_msg = await websocket.receive_text()
        try:
            init_data = json.loads(init_msg)
            if init_data.get("type") != MSG_TYPE_INIT or "character_details" not in init_data:
                raise ValueError("First message must be INIT with character_details")
        except Exception as e:
            logger.error(f"[WS {session_id}] Invalid INIT: {e}")
            await websocket.send_json({"type": MSG_TYPE_ERROR, "error": f"Invalid INIT: {e}"})
            await websocket.close()
            return
        session = await get_session(session_id)
        session["character_details"] = init_data["character_details"]
        await set_session(session_id, session)
        logger.info(f"[WS {session_id}] Session initialized with character: {init_data['character_details']}")
        # 2. Run LLM1 to generate system prompt/context (stub for now)
        llm1_context = f"[SYSTEM_PROMPT for {init_data['character_details'].get('name', 'character')}]"
        session["llm1_context"] = llm1_context
        await set_session(session_id, session)
        # 3. Send AI greeting (stub for now)
        greeting_text = f"Hello, I am {init_data['character_details'].get('name', 'your assistant')}! How can I help you today?"
        await websocket.send_json({"type": MSG_TYPE_GREETING, "text": greeting_text})
        logger.info(f"[WS {session_id}] Sent greeting: {greeting_text}")
        # 4. Main loop: handle audio, VAD, STT, LLM2, TTS, barge-in, etc.
        speaking = False
        silence_counter = 0
        max_silence_chunks = 10
        tts_playing = False
        tts_cancel_event = None
        history = []
        audio_buffer = bytearray()
        async with httpx.AsyncClient(timeout=None) as client:
            while True:
                try:
                    msg = await websocket.receive()
                except Exception as e:
                    logger.error(f"[WS {session_id}] Error receiving message: {e}")
                    break
                if msg["type"] == "websocket.disconnect":
                    logger.info(f"[WS {session_id}] WebSocket disconnected.")
                    break
                if msg["type"] == "websocket.receive":
                    if "bytes" in msg:
                        audio_chunk = msg["bytes"]
                        # Dump first DUMP_LIMIT buffers to disk for inspection
                        if received_buffers[session_id] < DUMP_LIMIT:
                            dump_path = f"/tmp/audio_dump_{session_id}_{received_buffers[session_id]}.raw"
                            with open(dump_path, "wb") as f:
                                f.write(audio_chunk)
                            logger.info(f"[WS {session_id}] Dumped audio buffer to {dump_path} (len={len(audio_chunk)})")
                            received_buffers[session_id] += 1
                        # Log RMS and VAD decision
                        pcm = np.frombuffer(audio_chunk, dtype=np.int16)
                        rms = np.sqrt(np.mean(pcm.astype(np.float32) ** 2)) if pcm.size > 0 else 0
                        try:
                            speech_detected = is_speech(audio_chunk)
                        except Exception as e:
                            logger.error(f"[WS {session_id}] VAD error: {e}")
                            await websocket.send_json({"type": MSG_TYPE_ERROR, "error": f"VAD error: {e}"})
                            continue
                        logger.info(f"[WS {session_id}] Audio frame: len={len(audio_chunk)}, RMS={rms:.2f}, speech_detected={speech_detected}")
                        # --- BARGE-IN LOGIC ---
                        if tts_playing and speech_detected:
                            tts_playing = False
                            if tts_cancel_event:
                                tts_cancel_event.set()
                            logger.info(f"[WS {session_id}] Barge-in: user started speaking during TTS.")
                            await websocket.send_json({"type": MSG_TYPE_BARGE_IN})
                        # --- END BARGE-IN ---
                        if speech_detected:
                            audio_buffer.extend(audio_chunk)
                            silence_counter = 0
                            if not speaking:
                                speaking = True
                                await websocket.send_json({"type": MSG_TYPE_VAD_STATE, "speaking": True})
                        else:
                            if speaking:
                                silence_counter += 1
                                if silence_counter >= max_silence_chunks:
                                    speaking = False
                                    await websocket.send_json({"type": MSG_TYPE_VAD_STATE, "speaking": False})
                                    # --- End of utterance: send to STT ---
                                    if audio_buffer:
                                        audio_b64 = base64.b64encode(audio_buffer).decode("utf-8")
                                        stt_payload = {"audio_data": audio_b64}
                                        # Use streaming STT endpoint for lower latency
                                        logger.info(f"[WS {session_id}] STT request payload: {json.dumps(stt_payload)[:500]}...")
                                        try:
                                            async def audio_stream():
                                                yield base64.b64decode(audio_b64)
                                            async with client.stream("POST", STT_STREAM_URL, content=audio_stream(), headers={"x-internal-api-key": INTERNAL_API_KEY, "Content-Type": "application/octet-stream"}, timeout=30) as stt_resp:
                                                logger.info(f"[WS {session_id}] STT response code: {stt_resp.status_code}")
                                                transcript = ""
                                                async for chunk in stt_resp.aiter_text():
                                                    transcript += chunk
                                                    await websocket.send_json({"type": MSG_TYPE_TRANSCRIPT_PARTIAL, "text": transcript})
                                                await websocket.send_json({"type": MSG_TYPE_TRANSCRIPT_FINAL, "text": transcript})
                                        except Exception as e:
                                            logger.error(f"[WS {session_id}] STT streaming exception: {e}")
                                            await websocket.send_json({"type": MSG_TYPE_ERROR, "error": f"STT streaming exception: {e}"})
                                        audio_buffer = bytearray()
                                        await websocket.send_json({"type": MSG_TYPE_VAD_STATE, "speaking": False})
                                        continue
    except WebSocketDisconnect:
        logger.info(f"[WS {session_id}] Session disconnected (WebSocketDisconnect)")
    except Exception as e:
        logger.error(f"[WS {session_id}] Error in session: {e}")
        try:
            await websocket.send_json({"type": MSG_TYPE_ERROR, "error": str(e)})
        except Exception:
            pass
    finally:
        logger.info(f"[WS {session_id}] Cleaning up session.")
        await delete_session(session_id)
        if websocket.application_state != WebSocketState.DISCONNECTED:
            try:
                await websocket.close()
            except Exception:
                pass 