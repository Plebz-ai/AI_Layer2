# STT Service Logic (Speech to Text)

import base64
import os
import httpx
import asyncio
import json
import logging

DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
DEEPGRAM_URL = "wss://api.deepgram.com/v1/listen"
DEEPGRAM_REST_URL = "https://api.deepgram.com/v1/listen"

logger = logging.getLogger("stt_service")

async def speech_to_text(audio_data: str):
    try:
        audio_bytes = base64.b64decode(audio_data)
        headers = {
            "Authorization": f"Token {DEEPGRAM_API_KEY}",
            "Content-Type": "audio/wav"
        }
        async with httpx.AsyncClient() as client:
            resp = await client.post(DEEPGRAM_REST_URL, content=audio_bytes, headers=headers)
            resp.raise_for_status()
            result = resp.json()
            transcript = result.get("results", {}).get("channels", [{}])[0].get("alternatives", [{}])[0].get("transcript", "")
            return transcript or "[No transcript found]"
    except Exception as e:
        logger.error(f"Deepgram STT error: {e}")
        return "[Error decoding or transcribing audio]"

async def stream_deepgram(audio_chunk_iterable):
    headers = {"Authorization": f"Token {DEEPGRAM_API_KEY}"}
    async with httpx.AsyncClient() as client:
        async with client.ws_connect(DEEPGRAM_URL, headers=headers) as ws:
            async for chunk in audio_chunk_iterable:
                await ws.send_bytes(chunk)
                msg = await ws.receive_json()
                transcript = msg.get("channel", {}).get("alternatives", [{}])[0].get("transcript", "")
                if transcript:
                    yield transcript
            await ws.aclose() 