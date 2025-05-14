# LLM2 Service Logic (Persona/Character Brain)

import os
import logging
from openai import AzureOpenAI
import openai, httpx
import traceback

print(f"[DEBUG] openai version: {openai.__version__}")
print(f"[DEBUG] httpx version: {httpx.__version__}")

AZURE_O4MINI_ENDPOINT = os.getenv("AZURE_O4MINI_ENDPOINT")
AZURE_O4MINI_API_KEY = os.getenv("AZURE_O4MINI_API_KEY")
AZURE_O4MINI_DEPLOYMENT = os.getenv("AZURE_O4MINI_DEPLOYMENT", "o4-mini")
AZURE_O4MINI_API_VERSION = os.getenv("AZURE_O4MINI_API_VERSION", "2024-12-01-preview")

# Validate required env vars
if not AZURE_O4MINI_ENDPOINT or not isinstance(AZURE_O4MINI_ENDPOINT, str):
    raise RuntimeError("Missing or invalid AZURE_O4MINI_ENDPOINT environment variable.")
if not AZURE_O4MINI_API_KEY or not isinstance(AZURE_O4MINI_API_KEY, str):
    raise RuntimeError("Missing or invalid AZURE_O4MINI_API_KEY environment variable.")

client = AzureOpenAI(
    api_version=AZURE_O4MINI_API_VERSION,
    azure_endpoint=AZURE_O4MINI_ENDPOINT,
    api_key=AZURE_O4MINI_API_KEY,
)

async def generate_response(user_query: str, persona_context: str, rules: dict = None, model: str = None, session_id: str = None, history: list = None):
    logging.info(f"[LLM2] generate_response called with session_id={session_id}, user_query={user_query}, persona_context={persona_context}, rules={rules}, history={history}")
    # Compose prompt using persona_context, rules, and history
    prompt = f"Persona Context: {persona_context}\n"
    if rules:
        prompt += f"Rules: {rules}\n"
    if history:
        prompt += f"Chat History: {history}\n"
    prompt += f"User: {user_query}"
    try:
        response = await client.chat.completions.create(
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": user_query}
            ],
            max_completion_tokens=256,
            temperature=0.7,
            model=model or AZURE_O4MINI_DEPLOYMENT,
        )
        reply = response.choices[0].message.content
        return {"response": reply}
    except Exception as e:
        logging.error(f"[LLM2] OpenAI call failed: {e}\n{traceback.format_exc()}")
        return {"response": "Sorry, something went wrong.", "error": str(e)} 