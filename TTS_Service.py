import requests
import logging
import time
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from config import Config

# Configure logging
logger = logging.getLogger("TTS_Service")

class TTSClientSingleton:
    """Singleton class for managing TTS API client sessions"""
    _instance = None
    _session = None
    
    @classmethod
    def get_session(cls):
        """
        Returns a singleton requests session with retry configuration
        to minimize connection overhead and handle transient errors.
        """
        if cls._instance is None or cls._session is None:
            logger.info("Initializing new TTS client session")
            
            # Get retry configuration from centralized config
            retry_config = Config.get_retry_config()
            
            # Configure retry strategy with exponential backoff
            retry_strategy = Retry(
                total=retry_config["total"],
                backoff_factor=retry_config["backoff_factor"],
                status_forcelist=retry_config["status_forcelist"],
                allowed_methods=["POST"]
            )
            
            # Create session with the retry strategy
            adapter = HTTPAdapter(max_retries=retry_strategy)
            cls._session = requests.Session()
            cls._session.mount("https://", adapter)
            cls._instance = cls()
            
        return cls._session


class TextToSpeechService:
    """Service class for text-to-speech functionality"""
    
    def __init__(self):
        self.tts_url = Config.TTS_URL
        self.api_key = Config.DEEPGRAM_API_KEY
        self.session = TTSClientSingleton.get_session()
        
        if not self.tts_url or not self.api_key:
            logger.error("TTS_URL or DEEPGRAM_API_KEY not set in configuration")
    
    def convert_text_to_speech(self, text, voice="en-US"):
        """
        Converts the given text to speech using the TTS service.
        
        Args:
            text (str): The text to convert to speech
            voice (str): The voice ID to use for synthesis (default: en-US)
            
        Returns:
            bytes: Audio data of synthesized speech
            
        Raises:
            Exception: If the API call fails or returns an error
        """
        if not self.tts_url or not self.api_key:
            raise Exception("TTS configuration missing from configuration")
        
        headers = {"Authorization": f"Token {self.api_key}"}
        payload = {"text": text, "voice": voice}
        
        try:
            # Start timer for performance monitoring
            start_time = time.time()
            
            # Make the API call with the session that has retry logic
            response = self.session.post(
                self.tts_url, 
                headers=headers, 
                json=payload,
                timeout=Config.DEFAULT_TIMEOUT
            )
            
            # Log performance metrics
            elapsed_time = time.time() - start_time
            logger.info(f"TTS conversion completed in {elapsed_time:.2f} seconds")
            
            if response.status_code != 200:
                logger.error(f"TTS service error: {response.status_code}, {response.text}")
                raise Exception(f"TTS service error: {response.status_code}")
            
            logger.info(f"Successfully converted {len(text)} characters to speech")
            return response.content
            
        except requests.exceptions.ConnectionError as e:
            logger.error(f"Connection error to TTS service: {str(e)}")
            raise Exception(f"Failed to connect to TTS service: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error in TTS processing: {str(e)}")
            raise


# Legacy function for backwards compatibility
def convert_text_to_speech(text, voice="en-US"):
    """Legacy function that delegates to the new service class"""
    service = TextToSpeechService()
    return service.convert_text_to_speech(text, voice)