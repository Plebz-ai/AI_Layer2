# Orchestrator Service Logic

import httpx
import asyncio
import logging
import json
import os
import secrets
from fastapi import APIRouter
import jwt
import time

logger = logging.getLogger("orchestrator")

# Use service names for Docker networking or localhost for standalone
# If running in Docker, use service names, else use localhost
USE_DOCKER = os.environ.get("USE_DOCKER", "true").lower() in ("true", "1", "yes")
HOST_PREFIX = "" if USE_DOCKER else "localhost:"

# Service URLs with configurable host
LLM1_URL = f"http://llm1_service:8001/generate-context" if USE_DOCKER else "http://localhost:8001/generate-context"
LLM2_URL = f"http://llm2_service:8002/generate-response" if USE_DOCKER else "http://localhost:8002/generate-response"
STT_URL = f"http://stt_service:8003/speech-to-text" if USE_DOCKER else "http://localhost:8003/speech-to-text"
TTS_URL = f"http://tts_service:8004/text-to-speech" if USE_DOCKER else "http://localhost:8004/text-to-speech"
TTS_STREAM_URL = f"http://tts_service:8004/stream-text-to-speech" if USE_DOCKER else "http://localhost:8004/stream-text-to-speech"
STT_STREAM_URL = f"http://stt_service:8003/stream-speech-to-text" if USE_DOCKER else "http://localhost:8003/stream-speech-to-text"

LIVEKIT_API_KEY = os.getenv("LIVEKIT_API_KEY", "your_livekit_api_key")
LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET", "your_livekit_api_secret")
LIVEKIT_URL = os.getenv("LIVEKIT_URL", "http://livekit:7880")

INTERNAL_API_KEY = os.getenv("INTERNAL_API_KEY", "changeme-internal-key")

router = APIRouter()

# Circuit breaker state
circuit_breakers = {"llm1": {"failures": 0, "open_until": 0}, "llm2": {"failures": 0, "open_until": 0}, "stt": {"failures": 0, "open_until": 0}, "tts": {"failures": 0, "open_until": 0}}

# Add in-memory cache for LLM1 context per session
llm1_context_cache = {}

@router.post("/voice-call/start")
async def start_voice_call(user_id: str):
    """
    Create a new LiveKit room and return a token for the user to join.
    """
    room_name = f"room-{secrets.token_hex(8)}"
    # TODO: Use LiveKit SDK to create room if needed (LiveKit auto-creates on join)
    # Generate access token for user
    token = generate_livekit_token(user_id, room_name)
    return {"room": room_name, "token": token, "livekit_url": LIVEKIT_URL}

@router.post("/voice-call/join")
async def join_voice_call(user_id: str, room_name: str):
    """
    Join an existing LiveKit room and return a token for the user.
    """
    token = generate_livekit_token(user_id, room_name)
    return {"room": room_name, "token": token, "livekit_url": LIVEKIT_URL}

# --- Token generation helper ---
def generate_livekit_token(user_id, room_name):
    # Generate a JWT for LiveKit using pyjwt
    api_key = LIVEKIT_API_KEY
    api_secret = LIVEKIT_API_SECRET
    now = int(time.time())
    payload = {
        "iss": api_key,
        "sub": user_id,
        "nbf": now,
        "exp": now + 3600,  # 1 hour expiry
        "room": room_name,
        "video": True,
        "audio": True,
        "can_publish": True,
        "can_subscribe": True,
        "can_publish_data": True,
        "can_publish_sources": ["audio"],
        "can_subscribe_sources": ["audio"],
    }
    token = jwt.encode(payload, api_secret, algorithm="HS256")
    return token

# Log all URLs at startup
logger.info(f"[ORCHESTRATOR] Service URLs: LLM1={LLM1_URL}, LLM2={LLM2_URL}, STT={STT_URL}, TTS={TTS_URL}")

async def safe_post(client, url, json, fallback=None, retries=2, request_id=None, step_name=None):
    service_name = None
    if "llm1" in url: service_name = "llm1"
    elif "llm2" in url: service_name = "llm2"
    elif "stt" in url: service_name = "stt"
    elif "tts" in url: service_name = "tts"
    now = time.time()
    if service_name and circuit_breakers[service_name]["open_until"] > now:
        logger.error(f"[request_id={request_id}] [CB] Circuit open for {service_name}, skipping call.")
        return type('DummyResp', (), {"json": lambda self: fallback or {}, "status_code": 503, "text": str(fallback), "error_details": {"status": 503, "message": "Circuit open"}})()
    start = time.time()
    logger.info(f"[request_id={request_id}] [latency] Starting {step_name or url}")
    last_error = None
    for attempt in range(retries):
        try:
            resp = await client.post(url, json=json, timeout=10.0, headers={"x-internal-api-key": INTERNAL_API_KEY})
            latency = (time.time() - start) * 1000
            logger.info(f"[request_id={request_id}] [latency] {step_name or url} attempt {attempt+1}: {latency:.2f}ms, status={resp.status_code}")
            if resp.status_code == 200:
                if service_name:
                    circuit_breakers[service_name]["failures"] = 0
                return resp
            logger.error(f"[request_id={request_id}] Non-200 response from {url}: {resp.status_code}, {resp.text}")
            last_error = {"status": resp.status_code, "body": resp.text}
        except Exception as e:
            logger.error(f"[request_id={request_id}] Exception calling {url}: {str(e)}")
            last_error = {"status": "exception", "message": str(e)}
        if service_name:
            circuit_breakers[service_name]["failures"] += 1
            if circuit_breakers[service_name]["failures"] >= 3:
                circuit_breakers[service_name]["open_until"] = time.time() + 30
                logger.error(f"[request_id={request_id}] [CB] Circuit opened for {service_name} for 30s due to repeated failures.")
                break
        if attempt < retries - 1:
            wait_time = 0.5 * (2 ** attempt)
            logger.info(f"[request_id={request_id}] Retrying in {wait_time}s (attempt {attempt+1}/{retries})")
            await asyncio.sleep(wait_time)
    latency = (time.time() - start) * 1000
    logger.error(f"[request_id={request_id}] [latency] All retries failed for {step_name or url} after {latency:.2f}ms, using fallback")
    class DummyResp:
        def json(self_inner):
            return fallback or {}
        status_code = 500
        text = str(fallback)
        error_details = last_error
    return DummyResp()

async def orchestrate_interaction(user_input: str, character_details: dict, mode: str, audio_data: str = None, session_id: str = None, history: list = None, request_id: str = None):
    pipeline_start = time.time()
    async with httpx.AsyncClient() as client:
        if mode == "chat":
            if not user_input or not character_details:
                return {"response": "Missing user input or character details.", "audio_data": None, "error": {"orchestrator": "Missing required fields."}}
            # Use session_id as cache key if available
            cache_key = session_id or json.dumps(character_details, sort_keys=True)
            if cache_key in llm1_context_cache:
                context, rules = llm1_context_cache[cache_key]
                logging.info(f"[request_id={request_id}] [latency] Using cached LLM1 context for session: {cache_key}")
            else:
                llm1_payload = {"user_input": user_input, "character_details": character_details}
                if session_id:
                    llm1_payload["session_id"] = session_id
                if history:
                    llm1_payload["history"] = history
                logging.info(f"[request_id={request_id}] [latency] LLM1 payload: {json.dumps(llm1_payload)}")
                llm1_start = time.time()
                llm1_resp = await safe_post(client, LLM1_URL, llm1_payload, fallback={"context": "fallback-context", "rules": {}}, request_id=request_id, step_name="LLM1")
                llm1_latency = (time.time() - llm1_start) * 1000
                logging.info(f"[request_id={request_id}] [latency] LLM1 total: {llm1_latency:.2f}ms")
                context = llm1_resp.json().get("context", "fallback-context")
                rules = llm1_resp.json().get("rules", {})
                llm1_error = None
                if getattr(llm1_resp, 'status_code', 200) != 200 or context == "fallback-context":
                    llm1_error = getattr(llm1_resp, 'error_details', None) or llm1_resp.json().get("error") or "LLM1 failed to generate context."
                    logging.error(f"[request_id={request_id}] [latency] LLM1 failed. Error: {llm1_error}, Response: {llm1_resp.json()}")
                    return {"response": "Sorry, the character could not generate context. Please try again later.", "audio_data": None, "error": {"llm1": llm1_error}}
                # Cache the context and rules for this session
                llm1_context_cache[cache_key] = (context, rules)
            model = os.getenv("AZURE_GPT4O_MINI_DEPLOYMENT", "gpt-4o-mini")
            llm2_payload = {"user_query": user_input, "persona_context": context, "rules": rules, "model": model}
            if session_id:
                llm2_payload["session_id"] = session_id
            if history:
                llm2_payload["history"] = history
            logging.info(f"[request_id={request_id}] [latency] LLM2 payload: {json.dumps(llm2_payload)}")
            llm2_start = time.time()
            llm2_resp = await safe_post(client, LLM2_URL, llm2_payload, fallback={"response": "Sorry, something went wrong."}, request_id=request_id, step_name="LLM2")
            llm2_latency = (time.time() - llm2_start) * 1000
            logging.info(f"[request_id={request_id}] [latency] LLM2 total: {llm2_latency:.2f}ms")
            response = llm2_resp.json().get("response", "Sorry, something went wrong.")
            llm2_error = None
            if getattr(llm2_resp, 'status_code', 200) != 200 or not response or not response.strip() or response == "Sorry, something went wrong.":
                llm2_error = getattr(llm2_resp, 'error_details', None) or llm2_resp.json().get("error") or "LLM2 failed to generate response."
                logging.error(f"[request_id={request_id}] [latency] LLM2 failed. Error: {llm2_error}, Response: {llm2_resp.json()}")
                return {"response": "Sorry, the character could not respond. Please try again later.", "audio_data": None, "error": {"llm2": llm2_error}}
            result = {"response": response, "audio_data": None, "error": None}
            pipeline_latency = (time.time() - pipeline_start) * 1000
            logging.info(f"[request_id={request_id}] [latency] Final orchestrator result: {result} | Pipeline total: {pipeline_latency:.2f}ms")
            return result
        elif mode == "voice":
            stt_start = time.time()
            logger.info(f"[request_id={request_id}] [latency] Calling STT: {STT_URL} with audio_data present: {audio_data is not None}")
            stt_resp = await safe_post(client, STT_URL, {"audio_data": audio_data}, fallback={"transcript": ""}, request_id=request_id, step_name="STT")
            stt_latency = (time.time() - stt_start) * 1000
            transcript = stt_resp.json().get("transcript", "")
            stt_error = None
            if getattr(stt_resp, 'status_code', 200) != 200 or not transcript:
                stt_error = getattr(stt_resp, 'error_details', None) or stt_resp.json().get("error") or "STT failed to transcribe audio."
                logger.error(f"[request_id={request_id}] [latency] STT failed. Error: {stt_error}, Response: {stt_resp.json()}")
                return {"response": "Sorry, we could not transcribe your audio. Please try again.", "audio_data": None, "error": {"stt": stt_error}}
            logger.info(f"[request_id={request_id}] [latency] STT response: {stt_resp.json()} | STT total: {stt_latency:.2f}ms")
            llm1_start = time.time()
            llm1_resp = await safe_post(client, LLM1_URL, {"user_input": transcript, "character_details": character_details}, fallback={"context": "fallback-context", "rules": {}}, request_id=request_id, step_name="LLM1")
            llm1_latency = (time.time() - llm1_start) * 1000
            logger.info(f"[request_id={request_id}] [latency] LLM1 response: {llm1_resp.json()} | LLM1 total: {llm1_latency:.2f}ms")
            context = llm1_resp.json().get("context", "fallback-context")
            rules = llm1_resp.json().get("rules", {})
            llm1_error = None
            if getattr(llm1_resp, 'status_code', 200) != 200 or context == "fallback-context":
                llm1_error = getattr(llm1_resp, 'error_details', None) or llm1_resp.json().get("error") or "LLM1 failed to generate context."
                logger.error(f"[request_id={request_id}] [latency] LLM1 failed. Error: {llm1_error}, Response: {llm1_resp.json()}")
                return {"response": "Sorry, the character could not generate context. Please try again later.", "audio_data": None, "error": {"llm1": llm1_error}}
            model = os.getenv("AZURE_GPT4O_MINI_DEPLOYMENT", "gpt-4o-mini")
            llm2_start = time.time()
            logger.info(f"[request_id={request_id}] [latency] Calling LLM2: {LLM2_URL} with user_query: {transcript}, persona_context: {context}, rules: {rules}, model: {model}")
            llm2_resp = await safe_post(client, LLM2_URL, {"user_query": transcript, "persona_context": context, "rules": rules, "model": model}, fallback={"response": "Sorry, something went wrong."}, request_id=request_id, step_name="LLM2")
            llm2_latency = (time.time() - llm2_start) * 1000
            logger.info(f"[request_id={request_id}] [latency] LLM2 response: {llm2_resp.json()} | LLM2 total: {llm2_latency:.2f}ms")
            response = llm2_resp.json().get("response", "Sorry, something went wrong.")
            llm2_error = None
            if getattr(llm2_resp, 'status_code', 200) != 200 or not response or not response.strip() or response == "Sorry, something went wrong.":
                llm2_error = getattr(llm2_resp, 'error_details', None) or llm2_resp.json().get("error") or "LLM2 failed to generate response."
                logger.error(f"[request_id={request_id}] [latency] LLM2 failed. Error: {llm2_error}, Response: {llm2_resp.json()}")
                return {"response": "Sorry, the character could not respond. Please try again later.", "audio_data": None, "error": {"llm2": llm2_error}}
            tts_voice_type = character_details.get("voice_type", "predefined")
            tts_start = time.time()
            logger.info(f"[request_id={request_id}] [latency] Calling TTS: {TTS_URL} with text: {response}, voice_type: {tts_voice_type}")
            tts_resp = await safe_post(client, TTS_URL, {"text": response, "voice_type": tts_voice_type}, fallback={"audio_data": None}, request_id=request_id, step_name="TTS")
            tts_latency = (time.time() - tts_start) * 1000
            logger.info(f"[request_id={request_id}] [latency] TTS response: {tts_resp.json()} | TTS total: {tts_latency:.2f}ms")
            audio_out = tts_resp.json().get("audio_data", None)
            tts_error = None
            if getattr(tts_resp, 'status_code', 200) != 200 or not audio_out:
                tts_error = getattr(tts_resp, 'error_details', None) or tts_resp.json().get("error") or "TTS failed to generate audio."
                logger.error(f"[request_id={request_id}] [latency] TTS failed. Error: {tts_error}, Response: {tts_resp.json()}")
                return {"response": response, "audio_data": None, "error": {"tts": tts_error}}
            result = {"response": response, "audio_data": audio_out, "error": None}
            pipeline_latency = (time.time() - pipeline_start) * 1000
            logger.info(f"[request_id={request_id}] [latency] Final orchestrator result: {result} | Pipeline total: {pipeline_latency:.2f}ms")
            return result
        else:
            logger.info(f"[request_id={request_id}] Invalid mode: {mode}")
            return {"response": "Invalid mode", "audio_data": None, "error": {"orchestrator": "Invalid mode"}} 