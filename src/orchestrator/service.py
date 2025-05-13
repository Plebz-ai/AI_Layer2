# Orchestrator Service Logic

import httpx
import asyncio
import logging

logger = logging.getLogger("orchestrator")

LLM1_URL = "http://localhost:8001/generate-context"
LLM2_URL = "http://localhost:8002/generate-response"
STT_URL = "http://localhost:8003/speech-to-text"
TTS_URL = "http://localhost:8004/text-to-speech"
TTS_STREAM_URL = "http://localhost:8004/stream-text-to-speech"
STT_STREAM_URL = "http://localhost:8003/stream-speech-to-text"

async def safe_post(client, url, json, fallback=None, retries=2):
    for attempt in range(retries):
        try:
            resp = await client.post(url, json=json, timeout=3.0)
            if resp.status_code == 200:
                return resp
        except Exception:
            await asyncio.sleep(0.1)
    # Fallback
    class DummyResp:
        def json(self_inner):
            return fallback or {}
        status_code = 500
        text = str(fallback)
    return DummyResp()

async def orchestrate_interaction(user_input: str, character_details: dict, mode: str, audio_data: str = None):
    async with httpx.AsyncClient() as client:
        if mode == "chat":
            # Filter character_details for LLM1: only send 'name' and 'persona'
            filtered_character_details = {}
            if 'name' in character_details:
                filtered_character_details['name'] = character_details['name']
            # Map 'personality' to 'persona' if present
            if 'persona' in character_details:
                filtered_character_details['persona'] = character_details['persona']
            elif 'personality' in character_details:
                filtered_character_details['persona'] = character_details['personality']
            else:
                filtered_character_details['persona'] = 'friendly'
            logger.info(f"[AI-Layer2 DEBUG] Calling LLM1: {LLM1_URL} with user_input: {user_input} and character_details: {filtered_character_details}")
            llm1_resp = await safe_post(client, LLM1_URL, {"user_input": user_input, "character_details": filtered_character_details}, fallback={"context": "fallback-context", "rules": {}})
            logger.info(f"[AI-Layer2 DEBUG] LLM1 status: {getattr(llm1_resp, 'status_code', None)}, text: {getattr(llm1_resp, 'text', None)}")
            logger.info(f"[AI-Layer2 DEBUG] LLM1 response: {llm1_resp.json()}")
            context = llm1_resp.json().get("context", "fallback-context")
            rules = llm1_resp.json().get("rules", {})
            llm1_error = None
            if context == "fallback-context":
                llm1_error = llm1_resp.json().get("error") or "LLM1 failed to generate context."
            model = "O4-mini" if len(user_input) < 50 else "Llama-4"
            logger.info(f"[AI-Layer2 DEBUG] Calling LLM2: {LLM2_URL} with user_query: {user_input}, persona_context: {context}, rules: {rules}, model: {model}")
            llm2_resp = await safe_post(client, LLM2_URL, {"user_query": user_input, "persona_context": context, "rules": rules, "model": model}, fallback={"response": "Sorry, something went wrong."})
            logger.info(f"[AI-Layer2 DEBUG] LLM2 status: {getattr(llm2_resp, 'status_code', None)}, text: {getattr(llm2_resp, 'text', None)}")
            logger.info(f"[AI-Layer2 DEBUG] LLM2 response: {llm2_resp.json()}")
            response = llm2_resp.json().get("response", "Sorry, something went wrong.")
            llm2_error = None
            if response == "Sorry, something went wrong.":
                llm2_error = llm2_resp.json().get("error") or "LLM2 failed to generate response."
            result = {"response": response, "audio_data": None}
            if llm1_error or llm2_error:
                result["error"] = {"llm1": llm1_error, "llm2": llm2_error}
            logger.info(f"[AI-Layer2 DEBUG] Final orchestrator result: {result}")
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
            model = "O4-mini" if len(transcript) < 50 else "Llama-4"
            logger.info(f"[AI-Layer2 DEBUG] Calling LLM2: {LLM2_URL} with user_query: {transcript}, persona_context: {context}, rules: {rules}, model: {model}")
            llm2_resp = await safe_post(client, LLM2_URL, {"user_query": transcript, "persona_context": context, "rules": rules, "model": model}, fallback={"response": "Sorry, something went wrong."})
            logger.info(f"[AI-Layer2 DEBUG] LLM2 response: {llm2_resp.json()}")
            response = llm2_resp.json().get("response", "Sorry, something went wrong.")
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