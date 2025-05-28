# LLM2 Service Logic (Persona/Character Brain)

import os
import logging
from openai import AsyncAzureOpenAI
import openai, httpx
import traceback
import random
import asyncio

print(f"[DEBUG] openai version: {openai.__version__}")
print(f"[DEBUG] httpx version: {httpx.__version__}")

# Update for gpt-4o-mini deployment
GPT4O_MINI_ENDPOINT = os.getenv("AZURE_GPT4O_MINI_ENDPOINT", "https://ai-anuragpradeepjha5004ai785724618017.openai.azure.com/")
GPT4O_MINI_API_KEY = os.getenv("AZURE_GPT4O_MINI_API_KEY")
GPT4O_MINI_DEPLOYMENT = os.getenv("AZURE_GPT4O_MINI_DEPLOYMENT", "gpt-4o-mini")
GPT4O_MINI_API_VERSION = os.getenv("AZURE_GPT4O_MINI_API_VERSION", "2024-12-01-preview")

# Add environment variable for max tokens
MAX_COMPLETION_TOKENS = int(os.getenv("LLM2_MAX_COMPLETION_TOKENS", "1024"))

# Validate required env vars
if not GPT4O_MINI_ENDPOINT or not isinstance(GPT4O_MINI_ENDPOINT, str):
    raise RuntimeError("Missing or invalid AZURE_GPT4O_MINI_ENDPOINT environment variable.")
if not GPT4O_MINI_API_KEY or not isinstance(GPT4O_MINI_API_KEY, str):
    raise RuntimeError("Missing or invalid AZURE_GPT4O_MINI_API_KEY environment variable.")

client = AsyncAzureOpenAI(
    api_version=GPT4O_MINI_API_VERSION,
    azure_endpoint=GPT4O_MINI_ENDPOINT,
    api_key=GPT4O_MINI_API_KEY,
)

# Log environment variables at startup (except API key)
logging.info(f"[LLM2] GPT4O_MINI_ENDPOINT={GPT4O_MINI_ENDPOINT}")
logging.info(f"[LLM2] GPT4O_MINI_DEPLOYMENT={GPT4O_MINI_DEPLOYMENT}")
logging.info(f"[LLM2] GPT4O_MINI_API_VERSION={GPT4O_MINI_API_VERSION}")

async def generate_response(user_query: str, persona_context: str, rules: dict = None, model: str = None, session_id: str = None, history: list = None, temperature: float = 1.0, top_p: float = 1.0):
    logging.info(f"[LLM2] generate_response called with session_id={session_id}, user_query={user_query}")
    messages = [
        {"role": "system", "content": "Reply in a short, natural, conversational way. No more than 2 sentences. Avoid long or formal responses."},
        {"role": "system", "content": persona_context}
    ]
    if rules:
        messages.append({"role": "system", "content": f"Rules: {rules}"})
    if history:
        for msg in history:
            role = "assistant" if msg.get("sender") == "character" else "user"
            messages.append({"role": role, "content": msg.get("content")})
    messages.append({"role": "user", "content": user_query})
    logging.info(f"[LLM2] OpenAI API messages: {messages}")
    max_retries = 5
    for attempt in range(max_retries):
        try:
            logging.info(f"[LLM2] Sending to Azure: model={model or GPT4O_MINI_DEPLOYMENT}, messages={messages}")
            params = {
                "messages": messages,
                "max_completion_tokens": 64,
                "model": model or GPT4O_MINI_DEPLOYMENT,
                "temperature": 0.7,
                "top_p": top_p,
                "stream": True,
            }
            logging.info(f"[LLM2] Outgoing OpenAI params: {params}")
            start_time = asyncio.get_event_loop().time()
            response_stream = await client.chat.completions.create(**params)
            full_reply = ""
            async for chunk in response_stream:
                if not chunk.choices or len(chunk.choices) == 0:
                    continue
                delta = getattr(chunk.choices[0], 'delta', None)
                if delta and hasattr(delta, 'content') and delta.content:
                    full_reply += delta.content
                    logging.info(f"[LLM2] [stream] Partial: {repr(full_reply)} @ {asyncio.get_event_loop().time() - start_time:.3f}s")
            logging.info(f"[LLM2] [stream] Final: {repr(full_reply)} @ {asyncio.get_event_loop().time() - start_time:.3f}s")
            return {"response": full_reply}
        except Exception as e:
            err_str = str(e)
            logging.error(f"[LLM2] OpenAI call failed (attempt {attempt+1}/{max_retries}): {e}\n{traceback.format_exc()}")
            if ("429" in err_str or "RateLimitError" in err_str):
                # Exponential backoff with jitter
                base = 2
                max_wait = 30
                wait_time = min(max_wait, base ** attempt + random.uniform(0, 1))
                logging.warning(f"[LLM2] Rate limit hit (429). Retrying after {wait_time:.2f} seconds... If this persists, consider upgrading your Azure OpenAI quota.")
                await asyncio.sleep(wait_time)
                if attempt < max_retries - 1:
                    continue
                else:
                    logging.error("[LLM2] All retries exhausted due to rate limiting.")
                    return {"response": "Sorry, you are being rate limited by Azure OpenAI. Please wait and try again, or upgrade your quota at https://aka.ms/oai/quotaincrease.", "error": err_str}
            return {"response": "Sorry, something went wrong.", "error": err_str} 