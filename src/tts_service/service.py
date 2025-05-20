# TTS Service Logic (Text to Speech)

import base64
import os
import httpx
import asyncio

ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
VOICE_CONFIGS = {
    "male": {
        "model_id": os.getenv("ELEVENLABS_TTS_MODEL_ID_MALE"),
        "voice_id": os.getenv("ELEVENLABS_VOICE_ID_MALE"),
    },
    "female": {
        "model_id": os.getenv("ELEVENLABS_TTS_MODEL_ID_FEMALE"),
        "voice_id": os.getenv("ELEVENLABS_VOICE_ID_FEMALE"),
    },
    "predefined": {
        "model_id": os.getenv("ELEVENLABS_TTS_MODEL_ID_PREDEFINED"),
        "voice_id": os.getenv("ELEVENLABS_VOICE_ID_PREDEFINED"),
    },
    # Add more voices as needed
}

# Simulate preheating: pool of HTTP clients
PREHEAT_POOL_SIZE = 3
preheated_clients = [httpx.AsyncClient() for _ in range(PREHEAT_POOL_SIZE)]

def get_voice_config(voice_type: str):
    config = VOICE_CONFIGS.get(voice_type)
    if not config or not config["model_id"] or not config["voice_id"]:
        raise ValueError(f"Invalid or missing TTS config for voice_type '{voice_type}'")
    return config

async def text_to_speech(text: str, voice_type: str = "predefined"):
    config = get_voice_config(voice_type)
    model_id = config["model_id"]
    voice_id = config["voice_id"]
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
    headers = {"xi-api-key": ELEVENLABS_API_KEY}
    payload = {
        "text": text,
        "model_id": model_id,
        "voice_settings": {"stability": 0.5, "similarity_boost": 0.5}
    }
    client = preheated_clients[hash(text) % PREHEAT_POOL_SIZE]
    try:
        resp = await client.post(url, json=payload, headers=headers)
        resp.raise_for_status()
        return resp.content  # Return audio bytes
    except Exception as e:
        import logging
        logging.error(f"TTS error for voice_type={voice_type}, model_id={model_id}, voice_id={voice_id}: {e}")
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
    audio = await text_to_speech(text, voice_type)
    chunk_size = 1024
    for i in range(0, len(audio), chunk_size):
        yield audio[i:i+chunk_size]

def text_to_speech_base64(text: str, voice_name: str = "en-US-JennyNeural"):
    # Simulate TTS by returning a dummy base64 string
    dummy_audio = b"audio-bytes"
    audio_b64 = base64.b64encode(dummy_audio).decode("utf-8")
    return audio_b64 