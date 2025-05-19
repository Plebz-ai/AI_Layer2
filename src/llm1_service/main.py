import os

# Validate required env vars at startup
REQUIRED_VARS = ["AZURE_GPT41_MINI_ENDPOINT", "AZURE_GPT41_MINI_API_KEY"]
for var in REQUIRED_VARS:
    if not os.getenv(var):
        import sys
        print(f"[FATAL] Missing required environment variable: {var}", file=sys.stderr)
        sys.exit(1)

from fastapi import FastAPI, HTTPException, Request, Header
from pydantic import BaseModel
from service import generate_context
import logging
import time
import uuid
from fastapi import Depends

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

INTERNAL_API_KEY = os.getenv("INTERNAL_API_KEY", "changeme-internal-key")

async def verify_internal_api_key(x_internal_api_key: str = Header(...)):
    if x_internal_api_key != INTERNAL_API_KEY:
        logger.error(f"[LLM1] Invalid or missing internal API key: {x_internal_api_key}")
        raise HTTPException(status_code=403, detail="Forbidden: invalid internal API key")

@app.post("/generate-context", dependencies=[Depends(verify_internal_api_key)], response_model=LLM1Response)
async def generate_context_endpoint(req: LLM1Request, request: Request):
    request_id = getattr(request.state, 'request_id', 'unknown')
    logger.info(f"[request_id={request_id}] /generate-context payload: user_input length={len(req.user_input)}, character_details keys={list(req.character_details.keys())}")
    missing = [var for var in REQUIRED_VARS if not os.getenv(var)]
    if missing:
        logger.warning(f"[request_id={request_id}] /generate-context called but missing: {missing}")
    try:
        result = await generate_context(req.user_input, req.character_details, req.session_id)
        logger.info(f"[request_id={request_id}] /generate-context response: context length={len(result.get('context',''))}, rules keys={list(result.get('rules',{}).keys())}")
        return LLM1Response(**result)
    except Exception as e:
        import traceback
        logger.error(f"[request_id={request_id}] LLM1 error: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    missing = [var for var in REQUIRED_VARS if not os.getenv(var)]
    if missing:
        logger.warning(f"[LLM1] /health called but missing: {missing}")
        return {"status": "error", "error": f"Missing env vars: {missing}"}, 500
    return {"status": "ok"}

@app.middleware("http")
async def log_requests(request: Request, call_next):
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    logger.info(f"[request_id={request_id}] Request: {request.method} {request.url}")
    start = time.time()
    try:
        response = await call_next(request)
        latency = (time.time() - start) * 1000
        logger.info(f"[request_id={request_id}] Response status: {response.status_code} | Latency: {latency:.2f}ms")
        return response
    except Exception as e:
        import traceback
        logger.error(f"[request_id={request_id}] Error: {e}\n{traceback.format_exc()}")
        raise 