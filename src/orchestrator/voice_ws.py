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
INTERNAL_API_KEY = os.getenv("INTERNAL_API_KEY", "changeme-internal-key")

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
                                        try:
                                            stt_resp = await client.post(STT_URL, json=stt_payload, headers={"x-internal-api-key": INTERNAL_API_KEY}, timeout=15)
                                            if stt_resp.status_code == 200:
                                                transcript = stt_resp.json().get("transcript", "")
                                                await websocket.send_json({"type": MSG_TYPE_TRANSCRIPT_FINAL, "text": transcript})
                                                logger.info(f"[WS {session_id}] STT transcript: {transcript}")
                                                # --- SESSION HISTORY: Add user message ---
                                                session = await get_session(session_id)
                                                history = session.get("history", [])
                                                history.append({"sender": "user", "content": transcript})
                                                session["history"] = history
                                                await set_session(session_id, session)
                                                # --- LLM2 HANDOFF ---
                                                persona_context = session["llm1_context"]
                                                rules = session["character_details"]
                                                llm2_payload = {
                                                    "user_query": transcript,
                                                    "persona_context": persona_context,
                                                    "rules": rules,
                                                    "history": history,
                                                }
                                                try:
                                                    llm2_resp = await client.post(LLM2_URL, json=llm2_payload, headers={"x-internal-api-key": INTERNAL_API_KEY}, timeout=20)
                                                    if llm2_resp.status_code == 200:
                                                        llm2_text = llm2_resp.json().get("response", "")
                                                        await websocket.send_json({"type": MSG_TYPE_LLM2_FINAL, "text": llm2_text})
                                                        logger.info(f"[WS {session_id}] LLM2 response: {llm2_text}")
                                                        # --- SESSION HISTORY: Add AI message ---
                                                        session = await get_session(session_id)
                                                        history = session.get("history", [])
                                                        history.append({"sender": "character", "content": llm2_text})
                                                        session["history"] = history
                                                        await set_session(session_id, session)
                                                        # --- TTS HANDOFF ---
                                                        tts_payload = {"text": llm2_text, "voice_type": rules.get("voice_type", "predefined")}
                                                        try:
                                                            tts_cancel_event = asyncio.Event()
                                                            tts_playing = True
                                                            async with client.stream("POST", TTS_STREAM_URL, json=tts_payload, headers={"x-internal-api-key": INTERNAL_API_KEY}, timeout=30) as tts_resp:
                                                                if tts_resp.status_code == 200:
                                                                    async for chunk in tts_resp.aiter_text():
                                                                        if tts_cancel_event.is_set():
                                                                            logger.info(f"[WS {session_id}] TTS stream cancelled by barge-in.")
                                                                            break
                                                                        await websocket.send_json({"type": MSG_TYPE_TTS_CHUNK, "audio": chunk})
                                                                    await websocket.send_json({"type": MSG_TYPE_TTS_END})
                                                                else:
                                                                    logger.error(f"[WS {session_id}] TTS error: {tts_resp.text}")
                                                                    await websocket.send_json({"type": MSG_TYPE_ERROR, "error": f"TTS error: {tts_resp.text}"})
                                                            tts_playing = False
                                                        except Exception as e:
                                                            tts_playing = False
                                                            logger.error(f"[WS {session_id}] TTS exception: {e}")
                                                            await websocket.send_json({"type": MSG_TYPE_ERROR, "error": f"TTS exception: {e}"})
                                                        finally:
                                                            tts_cancel_event = None
                                                    else:
                                                        logger.error(f"[WS {session_id}] LLM2 error: {llm2_resp.text}")
                                                        await websocket.send_json({"type": MSG_TYPE_ERROR, "error": f"LLM2 error: {llm2_resp.text}"})
                                                except Exception as e:
                                                    logger.error(f"[WS {session_id}] LLM2 exception: {e}")
                                                    await websocket.send_json({"type": MSG_TYPE_ERROR, "error": f"LLM2 exception: {e}"})
                                        except Exception as e:
                                            logger.error(f"[WS {session_id}] STT exception: {e}")
                                            await websocket.send_json({"type": MSG_TYPE_ERROR, "error": f"STT exception: {e}"})
                                        audio_buffer = bytearray()
                                        await websocket.send_json({"type": MSG_TYPE_VAD_STATE, "speaking": False})
                            elif len(audio_buffer) > 0:
                                # Still collecting silence, add to buffer
                                audio_buffer.extend(audio_chunk)
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