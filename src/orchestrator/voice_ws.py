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
            await websocket.send_json({"type": MSG_TYPE_ERROR, "error": f"Invalid INIT: {e}"})
            await websocket.close()
            return
        sessions[session_id]["character_details"] = init_data["character_details"]
        # 2. Run LLM1 to generate system prompt/context (stub for now)
        # TODO: Call LLM1 service here
        llm1_context = f"[SYSTEM_PROMPT for {init_data['character_details'].get('name', 'character')}]"
        sessions[session_id]["llm1_context"] = llm1_context
        # 3. Send AI greeting (stub for now)
        greeting_text = f"Hello, I am {init_data['character_details'].get('name', 'your assistant')}! How can I help you today?"
        await websocket.send_json({"type": MSG_TYPE_GREETING, "text": greeting_text})
        # 4. Main loop: handle audio, VAD, STT, LLM2, TTS, barge-in, etc.
        speaking = False
        silence_counter = 0
        max_silence_chunks = 10  # Tune for your chunk size and latency
        while True:
            msg = await websocket.receive()
            if msg["type"] == "websocket.disconnect":
                break
            if msg["type"] == "websocket.receive":
                if "bytes" in msg:
                    audio_chunk = msg["bytes"]
                    # VAD: detect speech
                    speech_detected = is_speech(audio_chunk)
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
                                # End of utterance detected
                                speaking = False
                                await websocket.send_json({"type": MSG_TYPE_VAD_STATE, "speaking": False})
                                # --- STT HANDOFF ---
                                audio_bytes = bytes(sessions[session_id]["buffer"])
                                sessions[session_id]["buffer"] = bytearray()
                                silence_counter = 0
                                if audio_bytes:
                                    # Encode to base64 for STT REST API
                                    audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
                                    stt_url = "http://stt_service:8003/speech-to-text"
                                    payload = {"audio_data": audio_b64}
                                    try:
                                        async with httpx.AsyncClient() as client:
                                            resp = await client.post(stt_url, json=payload)
                                            if resp.status_code == 200:
                                                transcript = resp.json().get("transcript", "")
                                                await websocket.send_json({"type": MSG_TYPE_TRANSCRIPT_FINAL, "text": transcript})
                                                # TODO: Handoff to LLM2 next
                                            else:
                                                await websocket.send_json({"type": MSG_TYPE_ERROR, "error": f"STT error: {resp.text}"})
                                    except Exception as e:
                                        await websocket.send_json({"type": MSG_TYPE_ERROR, "error": f"STT exception: {e}"})
                        # If not speaking, just ignore silence
                    # TODO: Barge-in logic if TTS is playing
                elif "text" in msg:
                    # Control message (future: barge-in, etc.)
                    pass
    except WebSocketDisconnect:
        logger.info(f"[WS] Session {session_id} disconnected")
    except Exception as e:
        logger.error(f"[WS] Error in session {session_id}: {e}")
        await websocket.send_json({"type": MSG_TYPE_ERROR, "error": str(e)})
    finally:
        sessions.pop(session_id, None)
        if websocket.application_state != WebSocketState.DISCONNECTED:
            await websocket.close() 