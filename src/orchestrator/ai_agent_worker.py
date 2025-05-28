import os
import asyncio
import logging
import jwt
import time
from livekit import rtc
from service import generate_response
import torch
from silero_vad import VoiceActivityDetector, read_audio
import websockets
import httpx
import base64

LIVEKIT_URL = os.getenv("LIVEKIT_URL", "ws://livekit:7880")
LIVEKIT_API_KEY = os.getenv("LIVEKIT_API_KEY", "your_livekit_api_key")
LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET", "your_livekit_api_secret")

AI_USER_ID = "ai-agent"

# Helper to generate a LiveKit JWT for the AI agent
def generate_livekit_token(user_id, room_name):
    now = int(time.time())
    payload = {
        "iss": LIVEKIT_API_KEY,
        "sub": user_id,
        "nbf": now,
        "exp": now + 3600,
        "room": room_name,
        "video": False,
        "audio": True,
        "can_publish": True,
        "can_subscribe": True,
        "can_publish_data": True,
        "can_publish_sources": ["audio"],
        "can_subscribe_sources": ["audio"],
    }
    token = jwt.encode(payload, LIVEKIT_API_SECRET, algorithm="HS256")
    return token

# Initialize Silero VAD
vad_model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=False)
(get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils

async def run_ai_agent(room_name):
    """
    Connect to LiveKit as the AI agent, subscribe to audio, run STT→LLM→TTS, publish audio.
    """
    token = generate_livekit_token(AI_USER_ID, room_name)
    room = await rtc.connect(LIVEKIT_URL, token)
    logging.info(f"[AI_AGENT] Connected to room {room_name} as {AI_USER_ID}")

    tts_audio_track = None  # Track for current TTS playback
    tts_playing = False

    @room.on("track_subscribed")
    async def on_track(track, publication, participant):
        nonlocal tts_audio_track, tts_playing
        if track.kind == "audio":
            logging.info(f"[AI_AGENT] Subscribed to audio from {participant.identity}")
            # Stream audio to Deepgram STT
            async for transcript, audio_chunk in stream_deepgram_with_audio(track):
                # VAD: Check if user is speaking (barge-in)
                if tts_playing and is_speech(audio_chunk):
                    logging.info(f"[AI_AGENT] Barge-in detected, stopping TTS audio.")
                    if tts_audio_track:
                        await room.local_participant.unpublish_track(tts_audio_track)
                        tts_audio_track = None
                        tts_playing = False
                logging.info(f"[AI_AGENT] Transcript: {transcript}")
                # Streaming/low-latency: As soon as transcript is "final enough", respond
                if transcript and not tts_playing:
                    persona_context = "You are a helpful AI assistant."
                    rules = {"persona": "default", "style": "default", "forbidden_topics": [], "voice_type": "predefined"}
                    llm_result = await generate_response(transcript, persona_context, rules)
                    ai_text = llm_result["response"]
                    logging.info(f"[AI_AGENT] LLM response: {ai_text}")
                    from tts_service.service import text_to_speech
                    audio_bytes = await text_to_speech(ai_text, rules.get("voice_type", "predefined"))
                    tts_audio_track = rtc.LocalAudioTrack(audio_bytes)
                    await room.local_participant.publish_track(tts_audio_track)
                    tts_playing = True
                    logging.info(f"[AI_AGENT] Published TTS audio track for response.")

    # Helper: VAD on audio chunk
    def is_speech(audio_chunk):
        # audio_chunk: bytes, PCM 16kHz mono
        import numpy as np
        import soundfile as sf
        import io
        try:
            # Convert bytes to numpy array
            audio_np, sr = sf.read(io.BytesIO(audio_chunk), dtype='int16')
            speech_timestamps = get_speech_timestamps(audio_np, vad_model, sampling_rate=sr)
            return bool(speech_timestamps)
        except Exception as e:
            logging.error(f"VAD error: {e}")
            return False

    # Helper: Stream Deepgram with audio chunk yield
    async def stream_deepgram_with_audio(track):
        """
        Streams audio from LiveKit track to Deepgram, yields (partial transcript, audio_chunk) pairs.
        """
        import numpy as np
        import soundfile as sf
        import io
        import json
        import asyncio
        import httpx
        import websockets

        DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
        DEEPGRAM_URL = "wss://api.deepgram.com/v1/listen"
        headers = {"Authorization": f"Token {DEEPGRAM_API_KEY}"}

        # Buffer for partial audio
        audio_buffer = b""
        chunk_size = 3200  # ~0.1s of 16kHz 16-bit mono PCM
        async with httpx.AsyncClient() as client:
            async with client.ws_connect(DEEPGRAM_URL, headers=headers) as ws:
                async for audio_chunk in track:
                    # audio_chunk: bytes, PCM 16kHz mono
                    audio_buffer += audio_chunk
                    await ws.send_bytes(audio_chunk)
                    # VAD/barge-in: yield each chunk for VAD
                    try:
                        msg = await ws.receive_json()
                        transcript = msg.get("channel", {}).get("alternatives", [{}])[0].get("transcript", "")
                        yield (transcript, audio_chunk)
                    except Exception as e:
                        logging.error(f"Deepgram streaming error: {e}")
                        continue
                await ws.aclose()

    # Keep the agent running
    while True:
        await asyncio.sleep(1)

# Entry point for running the agent (for testing)
if __name__ == "__main__":
    import sys
    room_name = sys.argv[1] if len(sys.argv) > 1 else "test-room"
    asyncio.run(run_ai_agent(room_name)) 