import requests
import logging
import time
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from config import Config

# Configure logging
logger = logging.getLogger("STT_Service")

class ElevenLabsClientSingleton:
    """Singleton class for managing ElevenLabs API client sessions"""
    _instance = None
    _session = None
    
    @classmethod
    def get_session(cls):
        if cls._instance is None or cls._session is None:
            logger.info("Initializing new ElevenLabs client session")
            
            retry_config = Config.get_retry_config()
            retry_strategy = Retry(
                total=retry_config["total"],
                backoff_factor=retry_config["backoff_factor"],
                status_forcelist=retry_config["status_forcelist"],
                allowed_methods=["POST", "GET"]
            )
            
            adapter = HTTPAdapter(max_retries=retry_strategy)
            cls._session = requests.Session()
            cls._session.mount("https://", adapter)
            cls._instance = cls()
            
        return cls._session

class SpeechToTextService:
    """Service class for speech-to-text functionality using ElevenLabs"""
    
    def __init__(self):
        self.api_key = Config.ELEVENLABS_API_KEY
        self.base_url = Config.ELEVENLABS_API_URL
        self.session = ElevenLabsClientSingleton.get_session()
        
        if not self.api_key:
            logger.error("ElevenLabs API key is not set in configuration")

    def transcribe_audio(self, audio_file_path):
        """
        Transcribes the audio file using ElevenLabs STT API.
        
        Args:
            audio_file_path (str): Path to the audio file to transcribe
            
        Returns:
            str: The transcribed text
            
        Raises:
            Exception: If the API call fails or returns an error
        """
        if not self.api_key:
            raise Exception("ElevenLabs API key is not set in configuration")

        headers = {
            "xi-api-key": self.api_key,
            "accept": "application/json"
        }
        
        try:
            # Start timer for performance monitoring
            start_time = time.time()
            
            # First, upload the audio file
            upload_url = f"{self.base_url}/speech-to-text"
            
            with open(audio_file_path, "rb") as audio_file:
                files = {
                    "audio": ("audio.wav", audio_file, "audio/wav")
                }
                
                response = self.session.post(
                    upload_url,
                    headers=headers,
                    files=files,
                    timeout=Config.DEFAULT_TIMEOUT
                )

            elapsed_time = time.time() - start_time
            logger.info(f"STT transcription completed in {elapsed_time:.2f} seconds")
            
            if response.status_code != 200:
                logger.error(f"ElevenLabs STT API error: {response.status_code}, {response.text}")
                raise Exception(f"ElevenLabs STT API error: {response.status_code}")

            result = response.json()
            
            # Extract the transcript from the response
            transcript = result.get("text", "")
            
            if transcript:
                sample = transcript[:50] + "..." if len(transcript) > 50 else transcript
                logger.info(f"Transcription success: '{sample}'")
            else:
                logger.warning("Transcription returned empty result")
                
            return transcript
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Network error during transcription: {str(e)}")
            raise Exception(f"Network error during transcription: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error in transcription: {str(e)}")
            raise

# Legacy function for backwards compatibility
def transcribe_audio_with_elevenlabs(audio_file_path):
    """Legacy function that delegates to the new service class"""
    service = SpeechToTextService()
    return service.transcribe_audio(audio_file_path)