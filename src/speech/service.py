from typing import AsyncGenerator, Optional, List, Dict
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
import base64
import logging

load_dotenv()

class SpeechService:
    def __init__(self):
        # --- TTS (Text-to-Speech) config ---
        self.elevenlabs_api_key = os.getenv("ELEVENLABS_API_KEY")
        self.tts_model_id_predefined = os.getenv("ELEVENLABS_TTS_MODEL_ID_PREDEFINED")
        self.voice_id_predefined = os.getenv("ELEVENLABS_VOICE_ID_PREDEFINED")
        self.tts_model_id_male = os.getenv("ELEVENLABS_TTS_MODEL_ID_MALE")
        self.voice_id_male = os.getenv("ELEVENLABS_VOICE_ID_MALE")
        self.tts_model_id_female = os.getenv("ELEVENLABS_TTS_MODEL_ID_FEMALE")
        self.voice_id_female = os.getenv("ELEVENLABS_VOICE_ID_FEMALE")

        # --- STT (Speech-to-Text) config ---
        self.stt_provider = os.getenv("STT_PROVIDER", "elevenlabs")
        self.stt_model_id = os.getenv("STT_MODEL_ID")
        self.stt_model_api_key = os.getenv("STT_MODEL_API_KEY")

        # Set API keys for SDKs
        if self.elevenlabs_api_key:
            set_api_key(self.elevenlabs_api_key)
        if self.stt_provider == "deepgram" and self.stt_model_api_key:
            self.deepgram = Deepgram(self.stt_model_api_key)

        # Audio configuration
        self.sample_rate = int(os.getenv("AUDIO_SAMPLE_RATE", "16000"))
        self.channels = int(os.getenv("AUDIO_CHANNELS", "1"))
        self.audio_format = os.getenv("AUDIO_FORMAT", "wav")

    async def text_to_speech(self, text: str, character_type: str = "predefined") -> bytes:
        """Convert text to speech using ElevenLabs, selecting the correct model/voice for character type."""
        if not self.elevenlabs_api_key:
            logging.warning("No ElevenLabs API key set, using mock TTS.")
            return b"MOCK_AUDIO_DATA"
        # Select model/voice based on character_type
        if character_type == "predefined":
            model_id = self.tts_model_id_predefined
            voice_id = self.voice_id_predefined
        elif character_type == "male":
            model_id = self.tts_model_id_male
            voice_id = self.voice_id_male
        elif character_type == "female":
            model_id = self.tts_model_id_female
            voice_id = self.voice_id_female
        else:
            model_id = self.tts_model_id_predefined
            voice_id = self.voice_id_predefined
        try:
            audio = generate(
                text=text,
                voice=voice_id,
                model=model_id
            )
            if isinstance(audio, bytes):
                return audio
        except Exception as e:
            logging.error(f"Error in ElevenLabs text to speech: {str(e)}")
        return b"MOCK_AUDIO_DATA"

    async def speech_to_text(self, audio_data: bytes) -> str:
        """Convert speech to text using the selected provider/model."""
        if self.stt_provider == "deepgram" and self.stt_model_api_key:
            try:
                response = await self.deepgram.transcription.prerecorded(
                    audio_data,
                    {
                        "model": self.stt_model_id,
                        "language": "en",
                        "punctuate": True,
                        "smart_format": True,
                    }
                )
                if response and "results" in response:
                    transcript = response["results"]["channels"][0]["alternatives"][0]["transcript"]
                    return transcript
            except Exception as e:
                logging.error(f"Error in Deepgram STT: {str(e)}")
        # TODO: Add ElevenLabs STT if needed
        logging.info("Using mock speech-to-text")
        return "This is a mock transcript."

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

    async def analyze_sentiment(self, text: str) -> Dict:
        """Analyze sentiment of text"""
        # Mock implementation
        logging.info("Using mock sentiment analysis")
        return {
            "sentiment": "neutral",
            "confidence_scores": {
                "positive": 0.33,
                "neutral": 0.34,
                "negative": 0.33
            }
        }

    async def get_available_voices_azure(self) -> List[Dict]:
        """Get list of available Azure voices"""
        # Mock voices
        logging.info("Using mock voice list")
        return [
            {
                "name": "en-US-JennyNeural",
                "locale": "en-US",
                "gender": "Female",
                "style": None
            },
            {
                "name": "en-US-GuyNeural",
                "locale": "en-US",
                "gender": "Male",
                "style": None
            }
        ] 