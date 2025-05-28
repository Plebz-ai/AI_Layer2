# STT Service Logic (Speech to Text)

import base64
import os
import httpx
import asyncio
import json
import logging
from fastapi import FastAPI, Request, HTTPException, APIRouter
from starlette.responses import StreamingResponse
from deepgram import Deepgram, LiveTranscriptionEvents

logger = logging.getLogger("stt_service")

app = FastAPI(title="STT Service")
router = APIRouter()

DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
if not DEEPGRAM_API_KEY:
    logger.fatal("DEEPGRAM_API_KEY not set! Please set it in your .env file.")
    raise RuntimeError("DEEPGRAM_API_KEY not set!")

dg_client = Deepgram(DEEPGRAM_API_KEY)

@router.get("/health")
async def health_check():
    return {"status": "ok"}

@router.post("/stream-speech-to-text")
async def stream_speech_to_text_endpoint(request: Request):
    async def audio_stream_consumer():
        async for chunk in request.stream():
            yield chunk
    return StreamingResponse(stream_deepgram(audio_stream_consumer()), media_type="text/plain")

async def stream_deepgram(audio_chunks):
    # This function streams audio to Deepgram and yields transcripts
    import websockets
    import ssl
    import sys
    import traceback
    ws_url = f"wss://api.deepgram.com/v1/listen?encoding=linear16&sample_rate=16000&punctuate=true"
    headers = [("Authorization", f"Token {DEEPGRAM_API_KEY}")]
    try:
        async with websockets.connect(ws_url, extra_headers=headers, ssl=ssl.create_default_context()) as ws:
            async def sender():
                async for chunk in audio_chunks:
                    await ws.send(chunk)
                await ws.send(json.dumps({"type": "CloseStream"}))
            sender_task = asyncio.create_task(sender())
            while True:
                msg = await ws.recv()
                data = json.loads(msg)
                transcript = data.get("channel", {}).get("alternatives", [{}])[0].get("transcript", "")
                if transcript:
                    yield (transcript + "\n").encode("utf-8")
                if data.get("is_final"):
                    break
            await sender_task
    except Exception as e:
        logger.error(f"Deepgram streaming error: {e}\n{traceback.format_exc()}")
        yield f"[Deepgram error: {e}]".encode("utf-8")

# Note: The verify_internal_api_key dependency is applied in main.py 