from dotenv import load_dotenv
import os
load_dotenv()  # Load .env from current working directory (/app in Docker)

# Validate required env vars at startup
REQUIRED_VARS = ["AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_KEY"]
for var in REQUIRED_VARS:
    if not os.getenv(var):
        import sys
        print(f"[FATAL] Missing required environment variable: {var}", file=sys.stderr)
        exit(1)

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from service import generate_context
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("llm1_service")

app = FastAPI(title="LLM1 Service - Prompt/Context Generator")

class LLM1Request(BaseModel):
    user_input: str
    character_details: dict
    session_id: str = None

class LLM1Response(BaseModel):
    context: str
    rules: dict

@app.post("/generate-context", response_model=LLM1Response)
async def generate_context_endpoint(req: LLM1Request):
    try:
        result = await generate_context(req.user_input, req.character_details, req.session_id)
        return LLM1Response(**result)
    except Exception as e:
        logger.error(f"LLM1 error: {e}")
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