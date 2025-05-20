import json
import logging
import uuid
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from fastapi.concurrency import run_in_threadpool
from starlette.websockets import WebSocketState
from typing import Dict, Any

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
        while True:
            msg = await websocket.receive()
            if msg["type"] == "websocket.disconnect":
                break
            if msg["type"] == "websocket.receive":
                if "bytes" in msg:
                    # Audio chunk from user
                    # TODO: VAD, buffer, STT, barge-in, etc.
                    pass
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