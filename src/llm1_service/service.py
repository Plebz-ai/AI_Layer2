# LLM1 Service Logic (Prompt/Context Generator)

import os
import logging
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential

AZURE_LLAMA_ENDPOINT = os.getenv("AZURE_LLAMA_ENDPOINT")
AZURE_LLAMA_API_KEY = os.getenv("AZURE_LLAMA_API_KEY")
AZURE_LLAMA_MODEL_NAME = os.getenv("AZURE_LLAMA_MODEL_NAME", "finetuned-model-wmoqzjog")

# Validate required env vars
if not AZURE_LLAMA_ENDPOINT or not isinstance(AZURE_LLAMA_ENDPOINT, str):
    raise RuntimeError("Missing or invalid AZURE_LLAMA_ENDPOINT environment variable.")
if not AZURE_LLAMA_API_KEY or not isinstance(AZURE_LLAMA_API_KEY, str):
    raise RuntimeError("Missing or invalid AZURE_LLAMA_API_KEY environment variable.")

client = ChatCompletionsClient(
    endpoint=AZURE_LLAMA_ENDPOINT,
    credential=AzureKeyCredential(AZURE_LLAMA_API_KEY),
    api_version="2024-05-01-preview"
)

async def generate_context(user_input: str, character_details: dict, session_id: str = None):
    name = character_details.get("name", "Character")
    persona = character_details.get("persona", "friendly")
    prompt = f"Generate a system prompt for {name}, a {persona} AI. User: {user_input}"
    try:
        response = client.complete(
            messages=[
                SystemMessage(content="You are a helpful assistant."),
                UserMessage(content=prompt)
            ],
            max_tokens=256,
            temperature=0.7,
            model=AZURE_LLAMA_MODEL_NAME
        )
        context = response.choices[0].message.content
    except Exception as e:
        logging.error(f"LLM1 Azure Llama error: {e}")
        context = "Fallback context"
    rules = {"persona": persona, "max_length": 256}
    return {"context": context, "rules": rules} 