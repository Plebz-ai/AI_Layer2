# STT Service Logic (Speech to Text)

import os
import asyncio
import logging
import time
import json
from fastapi import FastAPI, Request, APIRouter
from starlette.responses import StreamingResponse
import websockets

logger = logging.getLogger("stt_service")

app = FastAPI(title="STT Service")
router = APIRouter()

DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
if not DEEPGRAM_API_KEY:
    logger.fatal("DEEPGRAM_API_KEY not set! Please set it in your .env file.")
    raise RuntimeError("DEEPGRAM_API_KEY not set!")

DEEPGRAM_URL = "wss://api.deepgram.com/v1/listen"

@router.get("/health")
async def health_check():
    return {"status": "ok"}

@router.post("/stream-speech-to-text")
async def stream_speech_to_text_endpoint(request: Request):
    async def audio_stream_consumer():
        async for chunk in request.stream():
            yield chunk

    async def stream_deepgram(audio_chunks):
        headers = {"Authorization": f"Token {DEEPGRAM_API_KEY}"}
        params = {
            "model": "nova-3",
            "punctuate": "true",
            "language": "en-US",
            "encoding": "linear16",
            "channels": "1",
            "sample_rate": "16000",
            "interim_results": "true",
            "utterance_end_ms": "300",
        }
        url = DEEPGRAM_URL + "?" + "&".join(f"{k}={v}" for k, v in params.items())
        try:
            async with websockets.connect(url, extra_headers=headers) as ws:
                async def sender():
                    async for chunk in audio_chunks:
                        await ws.send(chunk)
                    await ws.send(b"")  # Signal end of stream
                sender_task = asyncio.create_task(sender())
                try:
                    async for msg in ws:
                        data = json.loads(msg)
                        transcript = data.get("channel", {}).get("alternatives", [{}])[0].get("transcript", "")
                        if transcript:
                            print(f"[Deepgram Transcript @ {time.time():.3f}] {transcript}")
                            yield (transcript + "\n").encode("utf-8")
                except websockets.ConnectionClosed:
                    pass
                await sender_task
        except Exception as e:
            logger.error(f"Deepgram websocket connection failed: {e}")
            yield b"[ERROR] Deepgram connection failed.\n"

    return StreamingResponse(stream_deepgram(audio_stream_consumer()), media_type="text/plain")

# Note: The verify_internal_api_key dependency is applied in main.py 