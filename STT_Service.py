import requests
import logging
import time
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from config import Config

# Configure logging
logger = logging.getLogger("STT_Service")

class DeepgramClientSingleton:
    """Singleton class for managing Deepgram API client sessions"""
    _instance = None
    _session = None
    
    @classmethod
    def get_session(cls):
        """
        Returns a singleton requests session with retry configuration
        to minimize connection overhead and handle transient errors.
        """
        if cls._instance is None or cls._session is None:
            logger.info("Initializing new Deepgram client session")
            
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


class SpeechToTextService:
    """Service class for speech-to-text functionality"""
    
    def __init__(self):
        self.api_key = Config.DEEPGRAM_API_KEY
        self.stt_url = Config.STT_URL
        self.session = DeepgramClientSingleton.get_session()
        
        if not self.api_key:
            logger.error("Deepgram API key is not set in configuration")
    
    def transcribe_audio(self, audio_file_path):
        """
        Transcribes the audio file using Deepgram's STT API with exponential backoff retry.
        
        Args:
            audio_file_path (str): Path to the audio file to transcribe
            
        Returns:
            str: The transcribed text
            
        Raises:
            Exception: If the API call fails or returns an error
        """
        if not self.api_key:
            raise Exception("Deepgram API key is not set in configuration")

        headers = {
            "Authorization": f"Token {self.api_key}",
            "Content-Type": "audio/wav",
        }
        
        try:
            # Start timer for performance monitoring
            start_time = time.time()
            
            with open(audio_file_path, "rb") as audio_file:
                response = self.session.post(
                    self.stt_url, 
                    headers=headers, 
                    data=audio_file,
                    timeout=Config.DEFAULT_TIMEOUT
                )

            # Log performance metrics
            elapsed_time = time.time() - start_time
            logger.info(f"STT transcription completed in {elapsed_time:.2f} seconds")
            
            if response.status_code != 200:
                logger.error(f"Deepgram STT API error: {response.status_code}, {response.text}")
                raise Exception(f"Deepgram STT API error: {response.status_code}, {response.text}")

            result = response.json()
            transcript = result.get("results", {}).get("channels", [{}])[0].get("alternatives", [{}])[0].get("transcript", "")
            
            # Log a sample of the transcript
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
def transcribe_audio_with_deepgram(audio_file_path):
    """Legacy function that delegates to the new service class"""
    service = SpeechToTextService()
    return service.transcribe_audio(audio_file_path)