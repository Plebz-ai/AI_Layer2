from dotenv import load_dotenv
import os
import openai, httpx
print(f"[DEBUG] openai version: {openai.__version__}")
print(f"[DEBUG] httpx version: {httpx.__version__}")
load_dotenv()  # Load .env from current working directory (/app in Docker)

# Validate required env vars at startup
REQUIRED_VARS = ["AZURE_O4MINI_ENDPOINT", "AZURE_O4MINI_API_KEY"]
for var in REQUIRED_VARS:
    if not os.getenv(var):
        import sys
        print(f"[FATAL] Missing required environment variable: {var}", file=sys.stderr)
        exit(1)

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from typing import Optional
from service import generate_response
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("llm2_service")

app = FastAPI(title="LLM2 Service - Character Brain")

class LLM2Request(BaseModel):
    user_query: str
    persona_context: str
    rules: dict
    model: Optional[str] = None

class LLM2Response(BaseModel):
    response: str

@app.post("/generate-response", response_model=LLM2Response)
async def generate_response_endpoint(req: LLM2Request):
    try:
        result = await generate_response(req.user_query, req.persona_context, req.rules, req.model)
        return LLM2Response(response=result["response"])
    except Exception as e:
        logger.error(f"LLM2 error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.info(f"Request: {request.method} {request.url}")
    try:
        response = await call_next(request)
        logger.info(f"Response status: {response.status_code}")
        return response
    except Exception as e:
        logger.error(f"Error: {e}")
        raise 