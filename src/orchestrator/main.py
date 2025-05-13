from dotenv import load_dotenv
import os
load_dotenv(dotenv_path='../../.env')
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from typing import Optional
from service import orchestrate_interaction
import logging
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("orchestrator")

app = FastAPI(title="AI Orchestrator Service")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class OrchestratorRequest(BaseModel):
    user_input: str
    character_details: dict
    mode: str  # 'chat' or 'voice'
    audio_data: Optional[str] = None  # base64, for voice mode

class OrchestratorResponse(BaseModel):
    response: str
    audio_data: Optional[str] = None

@app.post("/interact", response_model=OrchestratorResponse)
async def interact(req: OrchestratorRequest):
    logger.info(f"[AI-Layer2 DEBUG] Received /interact payload: {req.json()}")
    import traceback
    try:
        result = await orchestrate_interaction(
            user_input=req.user_input,
            character_details=req.character_details,
            mode=req.mode,
            audio_data=req.audio_data
        )
        return OrchestratorResponse(**result)
    except Exception as e:
        logger.error(f"Exception in /interact: {e}")
        logger.error(traceback.format_exc())
        return OrchestratorResponse(response="Sorry, something went wrong.", audio_data=None)

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.options("/interact")
async def options_interact():
    return JSONResponse(status_code=200, content={})

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