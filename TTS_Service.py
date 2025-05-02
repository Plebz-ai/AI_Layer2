import requests
import logging
import time
import json
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from config import Config
from STT_Service import ElevenLabsClientSingleton

# Configure logging
logger = logging.getLogger("TTS_Service")

class TextToSpeechService:
    """Service class for text-to-speech functionality using ElevenLabs"""
    
    def __init__(self):
        self.api_key = Config.ELEVENLABS_API_KEY
        self.base_url = Config.ELEVENLABS_API_URL
        self.session = ElevenLabsClientSingleton.get_session()
        
        if not self.api_key:
            logger.error("ElevenLabs API key is not set in configuration")

    def get_available_voices(self):
        """Get list of available voices from ElevenLabs"""
        if not self.api_key:
            raise Exception("ElevenLabs API key is not set in configuration")

        headers = {
            "xi-api-key": self.api_key,
            "accept": "application/json"
        }

        try:
            response = self.session.get(
                f"{self.base_url}/voices",
                headers=headers,
                timeout=Config.DEFAULT_TIMEOUT
            )

            if response.status_code != 200:
                logger.error(f"Failed to get voices: {response.status_code}, {response.text}")
                return []

            voices = response.json()
            return voices.get("voices", [])
        except Exception as e:
            logger.error(f"Error getting voices: {str(e)}")
            return []

    def convert_text_to_speech(self, text, voice_id="21m00Tcm4TlvDq8ikWAM"):
        """
        Converts text to speech using ElevenLabs API
        
        Args:
            text (str): The text to convert to speech
            voice_id (str): The ID of the voice to use (default is "Rachel")
            
        Returns:
            bytes: Audio data of synthesized speech
            
        Raises:
            Exception: If the API call fails or returns an error
        """
        if not self.api_key:
            raise Exception("ElevenLabs API key is not set in configuration")

        headers = {
            "xi-api-key": self.api_key,
            "accept": "audio/mpeg",
            "Content-Type": "application/json"
        }

        payload = {
            "text": text,
            "model_id": "eleven_monolingual_v1",
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.75
            }
        }

        try:
            start_time = time.time()
            
            response = self.session.post(
                f"{self.base_url}/text-to-speech/{voice_id}",
                headers=headers,
                json=payload,
                timeout=Config.DEFAULT_TIMEOUT
            )

            elapsed_time = time.time() - start_time
            logger.info(f"TTS conversion completed in {elapsed_time:.2f} seconds")

            if response.status_code != 200:
                logger.error(f"ElevenLabs TTS API error: {response.status_code}, {response.text}")
                raise Exception(f"ElevenLabs TTS API error: {response.status_code}")

            logger.info(f"Successfully converted {len(text)} characters to speech")
            return response.content

        except requests.exceptions.RequestException as e:
            logger.error(f"Network error in TTS conversion: {str(e)}")
            raise Exception(f"Network error in TTS conversion: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error in TTS conversion: {str(e)}")
            raise

# Legacy function for backwards compatibility
def convert_text_to_speech(text, voice_id="21m00Tcm4TlvDq8ikWAM"):
    """Legacy function that delegates to the new service class"""
    service = TextToSpeechService()
    return service.convert_text_to_speech(text, voice_id)