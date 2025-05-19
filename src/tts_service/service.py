# TTS Service Logic (Text to Speech)

import base64
import os
import httpx
import asyncio

ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
VOICE_IDS = {
    "male": os.getenv("ELEVENLABS_VOICE_ID_MALE", "VOICE_ID_MALE"),
    "female": os.getenv("ELEVENLABS_VOICE_ID_FEMALE", "VOICE_ID_FEMALE"),
    "predefined": os.getenv("ELEVENLABS_VOICE_ID_PREDEFINED", "VOICE_ID_PREDEFINED")
    # Add more voices as needed, e.g. 'robot', 'child', etc.
}

# Simulate preheating: pool of HTTP clients
PREHEAT_POOL_SIZE = 3
preheated_clients = [httpx.AsyncClient() for _ in range(PREHEAT_POOL_SIZE)]

async def text_to_speech(text: str, voice_type: str = "predefined"):
    voice_id = VOICE_IDS.get(voice_type, VOICE_IDS["predefined"])
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
    headers = {"xi-api-key": ELEVENLABS_API_KEY}
    payload = {"text": text, "voice_settings": {"stability": 0.5, "similarity_boost": 0.5}}
    client = preheated_clients[hash(text) % PREHEAT_POOL_SIZE]
    try:
        resp = await client.post(url, json=payload, headers=headers)
        resp.raise_for_status()
        return resp.content  # Return audio bytes
    except Exception as e:
        import logging
        logging.error(f"TTS error for voice_type={voice_type}, voice_id={voice_id}: {e}")
        # Return a short silent WAV as fallback (1 second of silence, 16kHz mono)
        import wave, io
        buf = io.BytesIO()
        with wave.open(buf, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(b'\x00\x00' * 16000)
        return buf.getvalue()

async def stream_text_to_speech(text: str, voice_type: str = "predefined"):
    # ElevenLabs does not support true streaming, so we chunk the result
    audio = await text_to_speech(text, voice_type)
    chunk_size = 1024
    for i in range(0, len(audio), chunk_size):
        yield audio[i:i+chunk_size]

def text_to_speech_base64(text: str, voice_name: str = "en-US-JennyNeural"):
    # Simulate TTS by returning a dummy base64 string
    dummy_audio = b"audio-bytes"
    audio_b64 = base64.b64encode(dummy_audio).decode("utf-8")
    return audio_b64 