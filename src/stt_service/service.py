# STT Service Logic (Speech to Text)

import os
import asyncio
import logging
import time
from fastapi import FastAPI, Request, APIRouter
from starlette.responses import StreamingResponse
from deepgram import DeepgramClient, LiveOptions, LiveTranscriptionEvents

logger = logging.getLogger("stt_service")

app = FastAPI(title="STT Service")
router = APIRouter()

DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
if not DEEPGRAM_API_KEY:
    logger.fatal("DEEPGRAM_API_KEY not set! Please set it in your .env file.")
    raise RuntimeError("DEEPGRAM_API_KEY not set!")

deepgram = DeepgramClient(DEEPGRAM_API_KEY)

@router.get("/health")
async def health_check():
    return {"status": "ok"}

@router.post("/stream-speech-to-text")
async def stream_speech_to_text_endpoint(request: Request):
    async def audio_stream_consumer():
        async for chunk in request.stream():
            yield chunk

    async def stream_deepgram(audio_chunks):
        options = LiveOptions(
            model="nova-3",
            punctuate=True,
            language="en-US",
            encoding="linear16",
            channels=1,
            sample_rate=16000,
            interim_results=True,
            utterance_end_ms=300,
        )
        queue = asyncio.Queue()

        async def on_transcript(event):
            transcript = event.channel.alternatives[0].transcript
            if transcript:
                print(f"[Deepgram Transcript @ {time.time():.3f}] {transcript}")
                await queue.put((transcript + "\n").encode("utf-8"))

        async with deepgram.listen.live.v("1") as dg_connection:
            dg_connection.on(LiveTranscriptionEvents.Transcript, on_transcript)
            await dg_connection.start(options)

            async def sender():
                async for chunk in audio_chunks:
                    await dg_connection.send(chunk)
                await dg_connection.finish()

            sender_task = asyncio.create_task(sender())
            while True:
                item = await queue.get()
                if item is None:
                    break
                yield item
            await sender_task

    return StreamingResponse(stream_deepgram(audio_stream_consumer()), media_type="text/plain")

# Note: The verify_internal_api_key dependency is applied in main.py 