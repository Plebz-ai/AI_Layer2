import os
import openai, httpx
print(f"[DEBUG] openai version: {openai.__version__}")
print(f"[DEBUG] httpx version: {httpx.__version__}")

# Validate required env vars at startup
REQUIRED_VARS = ["AZURE_GPT4O_MINI_ENDPOINT", "AZURE_GPT4O_MINI_API_KEY"]
for var in REQUIRED_VARS:
    if not os.getenv(var):
        import sys
        print(f"[FATAL] Missing required environment variable: {var}", file=sys.stderr)
        sys.exit(1)

from fastapi import FastAPI, HTTPException, Request, Header
from pydantic import BaseModel
from typing import Optional
from service import generate_response
import logging
import time
import uuid
from fastapi import Depends

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

INTERNAL_API_KEY = os.getenv("INTERNAL_API_KEY", "changeme-internal-key")

async def verify_internal_api_key(x_internal_api_key: str = Header(...)):
    if x_internal_api_key != INTERNAL_API_KEY:
        logger.error(f"[LLM2] Invalid or missing internal API key: {x_internal_api_key}")
        raise HTTPException(status_code=403, detail="Forbidden: invalid internal API key")

@app.post("/generate-response", dependencies=[Depends(verify_internal_api_key)], response_model=LLM2Response)
async def generate_response_endpoint(req: LLM2Request, request: Request):
    request_id = getattr(request.state, 'request_id', 'unknown')
    logger.info(f"[request_id={request_id}] /generate-response payload: user_query length={len(req.user_query)}, persona_context length={len(req.persona_context)}, rules keys={list(req.rules.keys())}, model={req.model}")
    missing = [var for var in REQUIRED_VARS if not os.getenv(var)]
    if missing:
        logger.warning(f"[request_id={request_id}] /generate-response called but missing: {missing}")
    try:
        result = await generate_response(req.user_query, req.persona_context, req.rules, req.model)
        logger.info(f"[request_id={request_id}] /generate-response response: response length={len(result.get('response',''))}")
        return LLM2Response(response=result["response"])
    except Exception as e:
        import traceback
        logger.error(f"[request_id={request_id}] LLM2 error: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    missing = [var for var in REQUIRED_VARS if not os.getenv(var)]
    if missing:
        logger.warning(f"[LLM2] /health called but missing: {missing}")
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