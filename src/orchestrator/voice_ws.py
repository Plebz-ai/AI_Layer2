import json
import logging
import uuid
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from fastapi.concurrency import run_in_threadpool
from starlette.websockets import WebSocketState
from typing import Dict, Any
from speech.vad import is_speech
import base64
import httpx
import os
import asyncio

router = APIRouter()
logger = logging.getLogger("voice_ws")

# Session state store (in-memory for now)
sessions: Dict[str, Dict[str, Any]] = {}

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

@router.websocket("/ws/voice-session")
async def voice_session_ws(websocket: WebSocket):
    await websocket.accept()
    session_id = str(uuid.uuid4())
    logger.info(f"[WS] New voice session: {session_id}")
    sessions[session_id] = {
        "id": session_id,
        "state": {},
        "buffer": bytearray(),
        "vad": None,  # To be filled with VAD state
        "llm1_context": None,
        "character_details": None,
        "history": [],
        "tts_playing": False,
    }
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
        sessions[session_id]["character_details"] = init_data["character_details"]
        logger.info(f"[WS {session_id}] Session initialized with character: {init_data['character_details']}")
        # 2. Run LLM1 to generate system prompt/context (stub for now)
        llm1_context = f"[SYSTEM_PROMPT for {init_data['character_details'].get('name', 'character')}]"
        sessions[session_id]["llm1_context"] = llm1_context
        # 3. Send AI greeting (stub for now)
        greeting_text = f"Hello, I am {init_data['character_details'].get('name', 'your assistant')}! How can I help you today?"
        await websocket.send_json({"type": MSG_TYPE_GREETING, "text": greeting_text})
        logger.info(f"[WS {session_id}] Sent greeting: {greeting_text}")
        # 4. Main loop: handle audio, VAD, STT, LLM2, TTS, barge-in, etc.
        speaking = False
        silence_counter = 0
        max_silence_chunks = 10  # Tune for your chunk size and latency
        tts_playing = False
        tts_cancel_event = None
        history = []
        sessions[session_id]["history"] = history
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
                    try:
                        speech_detected = is_speech(audio_chunk)
                    except Exception as e:
                        logger.error(f"[WS {session_id}] VAD error: {e}")
                        await websocket.send_json({"type": MSG_TYPE_ERROR, "error": f"VAD error: {e}"})
                        continue
                    # --- BARGE-IN LOGIC ---
                    if tts_playing and speech_detected:
                        tts_playing = False
                        if tts_cancel_event:
                            tts_cancel_event.set()
                        logger.info(f"[WS {session_id}] Barge-in: user started speaking during TTS.")
                        await websocket.send_json({"type": MSG_TYPE_BARGE_IN})
                    # --- END BARGE-IN ---
                    if speech_detected:
                        sessions[session_id]["buffer"] += audio_chunk
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
                                audio_bytes = bytes(sessions[session_id]["buffer"])
                                sessions[session_id]["buffer"] = bytearray()
                                silence_counter = 0
                                if audio_bytes:
                                    audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
                                    stt_url = "http://stt_service:8003/speech-to-text"
                                    payload = {"audio_data": audio_b64}
                                    try:
                                        async with httpx.AsyncClient(timeout=10) as client:
                                            try:
                                                resp = await client.post(stt_url, json=payload, timeout=10)
                                            except Exception as e:
                                                logger.error(f"[WS {session_id}] STT service call failed: {e}")
                                                await websocket.send_json({"type": MSG_TYPE_ERROR, "error": f"STT service error: {e}"})
                                                continue
                                            if resp.status_code == 200:
                                                transcript = resp.json().get("transcript", "")
                                                await websocket.send_json({"type": MSG_TYPE_TRANSCRIPT_FINAL, "text": transcript})
                                                logger.info(f"[WS {session_id}] STT transcript: {transcript}")
                                                # --- SESSION HISTORY: Add user message ---
                                                history.append({"sender": "user", "content": transcript})
                                                # --- LLM2 HANDOFF ---
                                                llm2_url = "http://llm2_service:8002/generate-response"
                                                persona_context = sessions[session_id]["llm1_context"]
                                                rules = sessions[session_id]["character_details"]
                                                llm2_payload = {
                                                    "user_query": transcript,
                                                    "persona_context": persona_context,
                                                    "rules": rules,
                                                    "history": history,
                                                }
                                                try:
                                                    llm2_headers = {"x-internal-api-key": os.getenv("INTERNAL_API_KEY", "changeme-internal-key")}
                                                    try:
                                                        llm2_resp = await client.post(llm2_url, json=llm2_payload, headers=llm2_headers, timeout=20)
                                                    except Exception as e:
                                                        logger.error(f"[WS {session_id}] LLM2 service call failed: {e}")
                                                        await websocket.send_json({"type": MSG_TYPE_ERROR, "error": f"LLM2 service error: {e}"})
                                                        continue
                                                    if llm2_resp.status_code == 200:
                                                        llm2_text = llm2_resp.json().get("response", "")
                                                        await websocket.send_json({"type": MSG_TYPE_LLM2_FINAL, "text": llm2_text})
                                                        logger.info(f"[WS {session_id}] LLM2 response: {llm2_text}")
                                                        # --- SESSION HISTORY: Add AI message ---
                                                        history.append({"sender": "character", "content": llm2_text})
                                                        # --- TTS HANDOFF ---
                                                        tts_url = "http://tts_service:8004/stream-text-to-speech"
                                                        voice_type = rules.get("voice_type", "predefined")
                                                        tts_payload = {"text": llm2_text, "voice_type": voice_type}
                                                        try:
                                                            tts_cancel_event = asyncio.Event()
                                                            tts_playing = True
                                                            try:
                                                                async with client.stream("POST", tts_url, json=tts_payload, headers=llm2_headers, timeout=30) as tts_resp:
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
                                                            except Exception as e:
                                                                logger.error(f"[WS {session_id}] TTS streaming error: {e}")
                                                                await websocket.send_json({"type": MSG_TYPE_ERROR, "error": f"TTS streaming error: {e}"})
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
                                            else:
                                                logger.error(f"[WS {session_id}] STT error: {resp.text}")
                                                await websocket.send_json({"type": MSG_TYPE_ERROR, "error": f"STT error: {resp.text}"})
                                    except Exception as e:
                                        logger.error(f"[WS {session_id}] STT exception: {e}")
                                        await websocket.send_json({"type": MSG_TYPE_ERROR, "error": f"STT exception: {e}"})
                        # If not speaking, just ignore silence
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
        sessions.pop(session_id, None)
        if websocket.application_state != WebSocketState.DISCONNECTED:
            try:
                await websocket.close()
            except Exception:
                pass 