# LLM1 Service Logic (Prompt/Context Generator)

import os
import logging
from openai import AsyncAzureOpenAI
import traceback
import asyncio
import random

# Update for gpt-4.1-mini deployment
GPT41_MINI_ENDPOINT = os.getenv("AZURE_GPT41_MINI_ENDPOINT", "https://ai-anuragpradeepjha5004ai785724618017.openai.azure.com/")
GPT41_MINI_API_KEY = os.getenv("AZURE_GPT41_MINI_API_KEY")
GPT41_MINI_DEPLOYMENT = os.getenv("AZURE_GPT41_MINI_DEPLOYMENT", "gpt-4.1-mini")
GPT41_MINI_API_VERSION = os.getenv("AZURE_GPT41_MINI_API_VERSION", "2024-12-01-preview")

# Validate required env vars
if not GPT41_MINI_ENDPOINT or not isinstance(GPT41_MINI_ENDPOINT, str):
    raise RuntimeError("Missing or invalid AZURE_GPT41_MINI_ENDPOINT environment variable.")
if not GPT41_MINI_API_KEY or not isinstance(GPT41_MINI_API_KEY, str):
    raise RuntimeError("Missing or invalid AZURE_GPT41_MINI_API_KEY environment variable.")

client = AsyncAzureOpenAI(
    api_version=GPT41_MINI_API_VERSION,
    azure_endpoint=GPT41_MINI_ENDPOINT,
    api_key=GPT41_MINI_API_KEY,
)

# Log environment variables at startup (except API key)
logging.info(f"[LLM1] GPT41_MINI_ENDPOINT={GPT41_MINI_ENDPOINT}")
logging.info(f"[LLM1] GPT41_MINI_DEPLOYMENT={GPT41_MINI_DEPLOYMENT}")
logging.info(f"[LLM1] GPT41_MINI_API_VERSION={GPT41_MINI_API_VERSION}")

async def generate_context(user_input: str, character_details: dict, session_id: str = None, history: list = None, temperature: float = 0.7, top_p: float = 0.95):
    name = character_details.get("name", "Character")
    persona = character_details.get("personality", "default persona")
    voice_type = character_details.get("voice_type", "predefined")
    prompt = f"You are {name}. {character_details.get('description', '')} Your personality traits are: {persona}. Respond in character, being concise and engaging."
    max_retries = 5
    for attempt in range(max_retries):
        try:
            response_params = {
                "messages": [
                    {"role": "system", "content": "You are a friendly, concise conversational partner. Always reply in 1-2 sentences, like a real human chat. Avoid long or formal responses."},
                    {"role": "user", "content": prompt}
                ],
                "max_completion_tokens": 64,  # Lowered for concise context
                "temperature": 0.7,
                "top_p": top_p,
                "model": GPT41_MINI_DEPLOYMENT,
                "stream": True,
            }
            start_time = asyncio.get_event_loop().time()
            response_stream = await client.chat.completions.create(**response_params)
            full_context = ""
            async for chunk in response_stream:
                delta = getattr(chunk.choices[0], 'delta', None)
                if delta and hasattr(delta, 'content') and delta.content:
                    full_context += delta.content
                    logging.info(f"[LLM1] [stream] Partial: {repr(full_context)} @ {asyncio.get_event_loop().time() - start_time:.3f}s")
            logging.info(f"[LLM1] [stream] Final: {repr(full_context)} @ {asyncio.get_event_loop().time() - start_time:.3f}s")
            rules = {
                "persona": persona,
                "style": character_details.get("style", "default"),
                "forbidden_topics": character_details.get("forbidden_topics", []),
                "voice_type": voice_type,
            }
            return {"context": full_context, "rules": rules}
        except Exception as e:
            wait_time = None
            retry_after = None
            if hasattr(e, 'response') and hasattr(e.response, 'headers'):
                retry_after = e.response.headers.get('Retry-After')
                if retry_after:
                    try:
                        wait_time = int(retry_after)
                    except Exception:
                        pass
            if ("429" in str(e) or "RateLimitError" in str(e)) and attempt < max_retries - 1:
                if wait_time is None:
                    wait_time = 2 ** attempt  # fallback exponential backoff
                # Add jitter to avoid thundering herd
                jitter = random.uniform(0, 0.5)
                total_wait = wait_time + jitter
                logging.info(f"[LLM1] Rate limit hit, retrying after {total_wait:.2f} seconds (Retry-After: {retry_after})...")
                await asyncio.sleep(total_wait)
                continue
            logging.error(f"[LLM1] OpenAI call failed (attempt {attempt+1}/{max_retries}): {e}\n{traceback.format_exc()}")
            return {"context": "fallback-context", "rules": {}, "error": str(e)} 