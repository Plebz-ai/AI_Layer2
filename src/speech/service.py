from typing import AsyncGenerator, Optional
import asyncio
import json
from elevenlabs import generate, stream, set_api_key
from deepgram import Deepgram
import soundfile as sf
import numpy as np
from pydub import AudioSegment
import io
import os
from dotenv import load_dotenv

load_dotenv()

class SpeechService:
    def __init__(self):
        self.elevenlabs_api_key = os.getenv("ELEVENLABS_API_KEY")
        self.deepgram_api_key = os.getenv("DEEPGRAM_API_KEY")
        set_api_key(self.elevenlabs_api_key)
        self.deepgram = Deepgram(self.deepgram_api_key)
        
        # Audio configuration
        self.sample_rate = int(os.getenv("AUDIO_SAMPLE_RATE", "16000"))
        self.channels = int(os.getenv("AUDIO_CHANNELS", "1"))
        self.audio_format = os.getenv("AUDIO_FORMAT", "wav")

    async def text_to_speech_stream(
        self, 
        text: str, 
        voice_id: str,
        stability: float = 0.5,
        similarity_boost: float = 0.75
    ) -> AsyncGenerator[bytes, None]:
        """Stream TTS audio in real-time."""
        try:
            audio_stream = generate(
                text=text,
                voice=voice_id,
                model="eleven_monolingual_v1",
                stream=True,
                stability=stability,
                similarity_boost=similarity_boost
            )
            
            for chunk in audio_stream:
                yield chunk
                
        except Exception as e:
            print(f"Error in TTS streaming: {str(e)}")
            raise

    async def speech_to_text_stream(
        self, 
        audio_stream: AsyncGenerator[bytes, None]
    ) -> AsyncGenerator[str, None]:
        """Stream STT transcription in real-time."""
        try:
            async for audio_chunk in audio_stream:
                # Process audio chunk
                audio_data = await self._process_audio_chunk(audio_chunk)
                
                # Send to Deepgram
                response = await self.deepgram.transcription.prerecorded(
                    audio_data,
                    {
                        "punctuate": True,
                        "model": "nova-2",
                        "language": "en",
                        "smart_format": True,
                    }
                )
                
                if response and "results" in response:
                    transcript = response["results"]["channels"][0]["alternatives"][0]["transcript"]
                    if transcript.strip():
                        yield transcript
                        
        except Exception as e:
            print(f"Error in STT streaming: {str(e)}")
            raise

    async def _process_audio_chunk(self, chunk: bytes) -> bytes:
        """Process audio chunk to ensure correct format."""
        try:
            # Convert to AudioSegment
            audio = AudioSegment.from_file(io.BytesIO(chunk))
            
            # Convert to mono if needed
            if audio.channels != self.channels:
                audio = audio.set_channels(self.channels)
            
            # Set sample rate
            if audio.frame_rate != self.sample_rate:
                audio = audio.set_frame_rate(self.sample_rate)
            
            # Export to bytes
            buffer = io.BytesIO()
            audio.export(buffer, format=self.audio_format)
            return buffer.getvalue()
            
        except Exception as e:
            print(f"Error processing audio chunk: {str(e)}")
            raise

    async def get_available_voices(self) -> list:
        """Get available voices from ElevenLabs."""
        try:
            from elevenlabs import voices
            available_voices = voices()
            return [
                {
                    "voice_id": voice.voice_id,
                    "name": voice.name,
                    "category": voice.category,
                    "description": voice.labels.get("description", ""),
                    "preview_url": voice.preview_url
                }
                for voice in available_voices
            ]
        except Exception as e:
            print(f"Error getting available voices: {str(e)}")
            raise 