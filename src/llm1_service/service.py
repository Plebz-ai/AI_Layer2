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
    persona = character_details.get("persona", character_details.get("personality", "friendly"))
    description = character_details.get("description", "")
    voice_type = character_details.get("voice_type", "")
    avatar_url = character_details.get("avatar_url", "")
    logging.info(f"[LLM1] generate_context called with session_id={session_id}, user_input={user_input}, character_details={character_details}, history={history}")
    prompt = (
        f"You are {name}, {persona}. {description}\n"
        f"Voice: {voice_type}\nAvatar: {avatar_url}\n"
        f"User: {user_input}"
    )
    max_retries = 5
    for attempt in range(max_retries):
        try:
            response = await client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are a friendly, concise conversational partner. Always reply in 1-2 sentences, like a real human chat. Avoid long or formal responses."},
                    {"role": "user", "content": prompt}
                ],
                max_completion_tokens=64,  # Lowered for concise context
                temperature=0.7,
                top_p=top_p,
                model=GPT41_MINI_DEPLOYMENT,
            )
            # Log rate limit headers if present
            if hasattr(response, 'headers'):
                rl_headers = {k: v for k, v in response.headers.items() if 'ratelimit' in k.lower()}
                logging.info(f"[LLM1] Rate limit headers: {rl_headers}")
            logging.info(f"[LLM1] OpenAI API raw response: {response}")
            if not hasattr(response, 'choices') or not response.choices or not hasattr(response.choices[0], 'message'):
                logging.error(f"[LLM1] OpenAI API returned unexpected response: {response}")
                return {"context": "fallback-context", "rules": {}, "error": "OpenAI API returned unexpected response format."}
            context = response.choices[0].message.content
            rules = {
                "persona": persona,
                "style": character_details.get("style", "default"),
                "forbidden_topics": character_details.get("forbidden_topics", []),
                "voice_type": voice_type,
            }
            return {"context": context, "rules": rules}
        except Exception as e:
            # Check for Retry-After header in the exception if available
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