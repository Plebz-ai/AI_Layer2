# Orchestrator Service Logic

import httpx
import asyncio
import logging
import json
import os

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

# Log all URLs at startup
logger.info(f"[ORCHESTRATOR] Service URLs: LLM1={LLM1_URL}, LLM2={LLM2_URL}, STT={STT_URL}, TTS={TTS_URL}")

async def safe_post(client, url, json, fallback=None, retries=2):
    logger.info(f"[ORCHESTRATOR] Sending POST request to {url}")
    for attempt in range(retries):
        try:
            resp = await client.post(url, json=json, timeout=10.0)  # Increased timeout
            logger.info(f"[ORCHESTRATOR] Response from {url}: status={resp.status_code}")
            if resp.status_code == 200:
                return resp
            logger.error(f"[ORCHESTRATOR] Non-200 response from {url}: {resp.status_code}, {resp.text}")
        except Exception as e:
            logger.error(f"[ORCHESTRATOR] Exception calling {url}: {str(e)}")
            if attempt < retries - 1:
                wait_time = 0.5 * (2 ** attempt)  # Exponential backoff
                logger.info(f"[ORCHESTRATOR] Retrying in {wait_time}s (attempt {attempt+1}/{retries})")
                await asyncio.sleep(wait_time)
    
    # Fallback
    logger.error(f"[ORCHESTRATOR] All retries failed for {url}, using fallback")
    class DummyResp:
        def json(self_inner):
            return fallback or {}
        status_code = 500
        text = str(fallback)
    return DummyResp()

async def orchestrate_interaction(user_input: str, character_details: dict, mode: str, audio_data: str = None, session_id: str = None, history: list = None):
    async with httpx.AsyncClient() as client:
        if mode == "chat":
            # Validate required fields
            if not user_input or not character_details:
                return {"response": "Missing user input or character details.", "audio_data": None, "error": {"orchestrator": "Missing required fields."}}
            # Send full character_details, session_id, and history to LLM1
            llm1_payload = {"user_input": user_input, "character_details": character_details}
            if session_id:
                llm1_payload["session_id"] = session_id
            if history:
                llm1_payload["history"] = history
            logging.info(f"[AI-Layer2 DEBUG] LLM1 payload: {json.dumps(llm1_payload)}")
            llm1_resp = await safe_post(client, LLM1_URL, llm1_payload, fallback={"context": "fallback-context", "rules": {}})
            context = llm1_resp.json().get("context", "fallback-context")
            rules = llm1_resp.json().get("rules", {})
            llm1_error = None
            if context == "fallback-context":
                llm1_error = llm1_resp.json().get("error") or "LLM1 failed to generate context."
                logging.error(f"[AI-Layer2 ERROR] LLM1 failed. Error: {llm1_error}, Response: {llm1_resp.json()}")
                return {"response": "Sorry, the character could not generate context. Please try again later.", "audio_data": None, "error": {"llm1": llm1_error}}
            # Always use O4-Mini deployment for response
            model = os.getenv("AZURE_GPT4O_MINI_DEPLOYMENT", "gpt-4o-mini")
            llm2_payload = {"user_query": user_input, "persona_context": context, "rules": rules, "model": model}
            if session_id:
                llm2_payload["session_id"] = session_id
            if history:
                llm2_payload["history"] = history
            logging.info(f"[AI-Layer2 DEBUG] LLM2 payload: {json.dumps(llm2_payload)}")
            llm2_resp = await safe_post(client, LLM2_URL, llm2_payload, fallback={"response": "Sorry, something went wrong."})
            response = llm2_resp.json().get("response", "Sorry, something went wrong.")
            # Treat empty responses as errors
            if not response or not response.strip():
                llm2_error = llm2_resp.json().get("error") or "LLM2 returned empty response."
                logging.error(f"[AI-Layer2 ERROR] LLM2 returned empty response. {llm2_error}")
                return {"response": "Sorry, the character could not respond. Please try again later.", "audio_data": None, "error": {"llm2": llm2_error}}
            llm2_error = None
            if response == "Sorry, something went wrong.":
                llm2_error = llm2_resp.json().get("error") or "LLM2 failed to generate response."
                logging.error(f"[AI-Layer2 ERROR] LLM2 failed. Error: {llm2_error}, Response: {llm2_resp.json()}")
                return {"response": "Sorry, the character could not respond. Please try again later.", "audio_data": None, "error": {"llm2": llm2_error}}
            result = {"response": response, "audio_data": None}
            logging.info(f"[AI-Layer2 DEBUG] Final orchestrator result: {result}")
            return result
        elif mode == "voice":
            logger.info(f"[AI-Layer2 DEBUG] Calling STT: {STT_URL} with audio_data present: {audio_data is not None}")
            stt_resp = await safe_post(client, STT_URL, {"audio_data": audio_data}, fallback={"transcript": ""})
            transcript = stt_resp.json().get("transcript", "")
            logger.info(f"[AI-Layer2 DEBUG] STT response: {stt_resp.json()}")
            llm1_resp = await safe_post(client, LLM1_URL, {"user_input": transcript, "character_details": character_details}, fallback={"context": "fallback-context", "rules": {}})
            logger.info(f"[AI-Layer2 DEBUG] LLM1 response: {llm1_resp.json()}")
            context = llm1_resp.json().get("context", "fallback-context")
            rules = llm1_resp.json().get("rules", {})
            # Always use the configured deployment for Azure OpenAI (avoid unsupported models)
            model = os.getenv("AZURE_GPT4O_MINI_DEPLOYMENT", "gpt-4o-mini")
            logger.info(f"[AI-Layer2 DEBUG] Calling LLM2: {LLM2_URL} with user_query: {transcript}, persona_context: {context}, rules: {rules}, model: {model}")
            llm2_resp = await safe_post(client, LLM2_URL, {"user_query": transcript, "persona_context": context, "rules": rules, "model": model}, fallback={"response": "Sorry, something went wrong."})
            logger.info(f"[AI-Layer2 DEBUG] LLM2 response: {llm2_resp.json()}")
            response = llm2_resp.json().get("response", "Sorry, something went wrong.")
            # Treat empty responses as errors in voice mode
            if not response or not response.strip():
                logger.error(f"[AI-Layer2 ERROR] LLM2 returned empty response in voice mode.")
                return {"response": "Sorry, the character could not respond. Please try again later.", "audio_data": None, "error": {"llm2": "LLM2 returned empty response."}}
            tts_voice_type = character_details.get("voice_type", "predefined")
            logger.info(f"[AI-Layer2 DEBUG] Calling TTS: {TTS_URL} with text: {response}, voice_type: {tts_voice_type}")
            tts_resp = await safe_post(client, TTS_URL, {"text": response, "voice_type": tts_voice_type}, fallback={"audio_data": None})
            logger.info(f"[AI-Layer2 DEBUG] TTS response: {tts_resp.json()}")
            audio_out = tts_resp.json().get("audio_data", None)
            result = {"response": response, "audio_data": audio_out}
            logger.info(f"[AI-Layer2 DEBUG] Final orchestrator result: {result}")
            return result
        else:
            logger.info(f"[AI-Layer2 DEBUG] Invalid mode: {mode}")
            return {"response": "Invalid mode", "audio_data": None} 