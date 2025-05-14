# LLM1 Service Logic (Prompt/Context Generator)

import os
import logging
from openai import AsyncAzureOpenAI
import traceback

# Environment variables for Azure OpenAI GPT-4.1
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4.1")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")

# Validate required env vars
if not AZURE_OPENAI_ENDPOINT or not isinstance(AZURE_OPENAI_ENDPOINT, str):
    raise RuntimeError("Missing or invalid AZURE_OPENAI_ENDPOINT environment variable.")
if not AZURE_OPENAI_API_KEY or not isinstance(AZURE_OPENAI_API_KEY, str):
    raise RuntimeError("Missing or invalid AZURE_OPENAI_API_KEY environment variable.")

client = AsyncAzureOpenAI(
    api_version=AZURE_OPENAI_API_VERSION,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_API_KEY,
)

# Log environment variables at startup (except API key)
logging.info(f"[LLM1] AZURE_OPENAI_ENDPOINT={AZURE_OPENAI_ENDPOINT}")
logging.info(f"[LLM1] AZURE_OPENAI_DEPLOYMENT={AZURE_OPENAI_DEPLOYMENT}")
logging.info(f"[LLM1] AZURE_OPENAI_API_VERSION={AZURE_OPENAI_API_VERSION}")

async def generate_context(user_input: str, character_details: dict, session_id: str = None, history: list = None):
    name = character_details.get("name", "Character")
    persona = character_details.get("persona", character_details.get("personality", "friendly"))
    description = character_details.get("description", "")
    voice_type = character_details.get("voice_type", "")
    avatar_url = character_details.get("avatar_url", "")
    logging.info(f"[LLM1] generate_context called with session_id={session_id}, user_input={user_input}, character_details={character_details}, history={history}")
    # Compose a detailed prompt using all character fields and history
    prompt = (
        f"Generate a system prompt for the following AI character.\n"
        f"Name: {name}\n"
        f"Persona: {persona}\n"
        f"Description: {description}\n"
        f"Voice Type: {voice_type}\n"
        f"Avatar URL: {avatar_url}\n"
    )
    if history:
        prompt += f"\nChat History: {history}\n"
    prompt += f"\nUser: {user_input}"
    logging.info(f"[LLM1] Calling OpenAI with prompt: {prompt}")
    try:
        response = await client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            max_completion_tokens=256,
            temperature=0.7,
            model=AZURE_OPENAI_DEPLOYMENT,
        )
        logging.info(f"[LLM1] OpenAI API raw response: {response}")
        if not hasattr(response, 'choices') or not response.choices or not hasattr(response.choices[0], 'message'):
            logging.error(f"[LLM1] OpenAI API returned unexpected response: {response}")
            return {"context": "fallback-context", "rules": {}, "error": "OpenAI API returned unexpected response format."}
        context = response.choices[0].message.content
        # Example: expand rules for persona, style, forbidden topics, etc.
        rules = {
            "persona": persona,
            "style": character_details.get("style", "default"),
            "forbidden_topics": character_details.get("forbidden_topics", []),
            "voice_type": voice_type,
        }
        return {"context": context, "rules": rules}
    except Exception as e:
        logging.error(f"[LLM1] OpenAI call failed: {e}\n{traceback.format_exc()}")
        return {"context": "fallback-context", "rules": {}, "error": str(e)} 