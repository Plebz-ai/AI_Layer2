# LLM2 Service Logic (Persona/Character Brain)

import os
import logging
from openai import AzureOpenAI
import openai, httpx

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

def generate_response(user_query: str, persona_context: str, rules: dict, model_name: str = None):
    prompt = f"{persona_context}\nUser: {user_query}\n"
    try:
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            max_completion_tokens=256,
            model=model_name or AZURE_O4MINI_DEPLOYMENT
        )
        result = response.choices[0].message.content
    except Exception as e:
        logging.error(f"LLM2 Azure o4-mini error: {e}")
        result = "Fallback response"
    return result 