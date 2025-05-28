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
        # 4. Main loop: handle audio, VAD, STT, LLM2, TTS, barge-in, etc.
        speaking = False
        silence_counter = 0
        max_silence_chunks = 10
        tts_playing = False
        tts_cancel_event = None
        history = []
        audio_buffer = bytearray()
        audio_buffer_for_stt = bytearray()
        async with httpx.AsyncClient(timeout=None) as client:
            while True:
                try:
                    msg = await websocket.receive()
                    # print(f"[WS {session_id}] Received message in main loop: {{msg}}", file=sys.stderr)  # Commented out to reduce log size
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
                        # Append incoming audio to buffer
                        audio_buffer.extend(audio_chunk)

                        # Process the buffer in 960-byte chunks for VAD
                        while len(audio_buffer) >= 960:
                            vad_chunk = audio_buffer[:960]
                            audio_buffer = audio_buffer[960:] # Keep the rest in buffer

                            pcm = np.frombuffer(vad_chunk, dtype=np.int16)
                            rms = np.sqrt(np.mean(pcm.astype(np.float32) ** 2)) if pcm.size > 0 else 0
                            try:
                                speech_detected = is_speech(bytes(vad_chunk))
                                # logger.info(f"[WS {session_id}] VAD frame: len={len(vad_chunk)}, RMS={rms:.2f}, speech_detected={speech_detected}")  # Commented out to reduce log size
                                # print(f"[WS {session_id}] VAD frame: len={len(vad_chunk)}, RMS={rms:.2f}, speech_detected={speech_detected}", file=sys.stderr)  # Already commented out
                            except Exception as e:
                                logger.error(f"[WS {session_id}] VAD error on 960-byte frame: {e}")
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
                                # When speech is detected, start/continue accumulating audio for STT
                                silence_counter = 0 # Reset silence counter
                                if not speaking:
                                    speaking = True
                                    await websocket.send_json({"type": MSG_TYPE_VAD_STATE, "speaking": True})
                                # The audio is already in audio_buffer (the part that was NOT consumed by VAD)
                                # We will accumulate the audio_buffer after the loop
                            else:
                                if speaking:
                                    # If previously speaking, count silence
                                    silence_counter += 1
                                    if silence_counter >= max_silence_chunks:
                                        # End of utterance: process the accumulated audio_buffer
                                        speaking = False
                                        await websocket.send_json({"type": MSG_TYPE_VAD_STATE, "speaking": False})
                                        # --- End of utterance: send the accumulated audio_buffer to STT ---
                                        if audio_buffer_for_stt:
                                            audio_b64 = base64.b64encode(audio_buffer_for_stt).decode("utf-8")
                                            stt_payload = {"audio_data": audio_b64}
                                            # Use streaming STT endpoint for lower latency
                                            logger.info(f"[WS {session_id}] STT request payload: {json.dumps(stt_payload)[:500]}...")
                                            try:
                                                # Note: This streaming approach might need adjustment to handle the buffered chunks correctly
                                                # Consider sending the whole audio_buffer_for_stt or refactoring STT streaming
                                                async def audio_stream():
                                                    yield base64.b64decode(audio_b64) # This might not work correctly with the accumulated buffer
                                                async with client.stream("POST", STT_STREAM_URL, content=audio_stream(), headers={"x-internal-api-key": INTERNAL_API_KEY, "Content-Type": "application/octet-stream"}, timeout=30) as stt_resp:
                                                    logger.info(f"[WS {session_id}] STT response code: {stt_resp.status_code}")
                                                    transcript = ""
                                                    async for chunk in stt_resp.aiter_text():
                                                        transcript += chunk
                                                        await websocket.send_json({"type": MSG_TYPE_TRANSCRIPT_PARTIAL, "text": transcript})
                                                    await websocket.send_json({"type": MSG_TYPE_TRANSCRIPT_FINAL, "text": transcript})
                                                    # --- LLM2 + TTS PIPELINE ---
                                                    if transcript.strip():
                                                        print(f"[ORCH] Forwarding transcript to LLM2 @ {time.time():.3f}: {transcript}")
                                                        # Call LLM2
                                                        llm2_payload = {
                                                            "user_query": transcript,
                                                            "persona_context": session.get("llm1_context", "You are a helpful AI assistant."),
                                                            "rules": {},
                                                            "model": os.getenv("AZURE_GPT4O_MINI_DEPLOYMENT", "gpt-4o-mini")
                                                        }
                                                        try:
                                                            llm2_start = time.time()
                                                            llm2_resp = await client.post(LLM2_URL, json=llm2_payload, headers={"x-internal-api-key": INTERNAL_API_KEY})
                                                            llm2_data = llm2_resp.json()
                                                            ai_text = llm2_data.get("response", "")
                                                            print(f"[ORCH] LLM2 response @ {time.time():.3f} (latency: {time.time()-llm2_start:.3f}s): {ai_text}")
                                                            await websocket.send_json({"type": MSG_TYPE_LLM2_FINAL, "text": ai_text})
                                                            # Call TTS
                                                            tts_payload = {
                                                                "text": ai_text,
                                                                "voice_type": session["character_details"].get("voice_type", "predefined")
                                                            }
                                                            async with client.stream("POST", TTS_STREAM_URL, json=tts_payload, headers={"x-internal-api-key": INTERNAL_API_KEY}) as tts_resp:
                                                                async for tts_chunk in tts_resp.aiter_bytes():
                                                                    await websocket.send_bytes(tts_chunk)
                                                                await websocket.send_json({"type": MSG_TYPE_TTS_END})
                                                        except Exception as e:
                                                            logger.error(f"[WS {session_id}] LLM2 or TTS pipeline error: {e}")
                                                            await websocket.send_json({"type": MSG_TYPE_ERROR, "error": f"LLM2 or TTS pipeline error: {e}"})
                                            except Exception as e:
                                                logger.error(f"[WS {session_id}] STT streaming exception: {e}")
                                                await websocket.send_json({"type": MSG_TYPE_ERROR, "error": f"STT streaming exception: {e}"})
                                            # Clear the audio buffer for STT after processing
                                            audio_buffer_for_stt = bytearray()
                                        # Always send VAD_STATE False when speaking stops
                                        await websocket.send_json({"type": MSG_TYPE_VAD_STATE, "speaking": False})

                        # After processing all 960-byte VAD frames, accumulate any remaining audio_buffer for STT if speaking
                        if speaking:
                            audio_buffer_for_stt.extend(audio_buffer) # Add the remainder of audio_buffer to stt buffer
                            audio_buffer = bytearray() # Clear the main audio_buffer after processing
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