# STT Service Logic (Speech to Text)

import base64
import os
import httpx
import asyncio

DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
DEEPGRAM_URL = "wss://api.deepgram.com/v1/listen"

def speech_to_text(audio_data: str):
    # Simulate decoding and STT
    try:
        _ = base64.b64decode(audio_data)
        transcript = "This is a dummy transcript."
    except Exception:
        transcript = "[Error decoding audio]"
    return transcript 

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