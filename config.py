import os
import logging
from dotenv import load_dotenv

# Attempt to load environment variables from .env file if present
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Create a logger
logger = logging.getLogger("AI_Config")

class Config:
    """
    Centralized configuration class for AI services.
    Loads configuration from environment variables with sensible defaults.
    """
    
    # LLM Configuration
    AZURE_ENDPOINT = os.getenv("AZURE_ENDPOINT", "https://finetuned-model-wmoqzjog.eastus2.models.ai.azure.com")
    AZURE_MODEL_NAME = os.getenv("AZURE_MODEL_NAME", "Llama-4-Maverick-17B-128E-Instruct-FP8")
    AZURE_API_KEY = os.getenv("AZURE_API_KEY")
    
    # TTS Configuration
    TTS_URL = os.getenv("TTS_URL")
    
    # STT Configuration 
    DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
    STT_URL = os.getenv("STT_URL", "https://api.deepgram.com/v1/listen")
    
    # ElevenLabs Configuration
    ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
    ELEVENLABS_API_URL = os.getenv("ELEVENLABS_API_URL", "https://api.elevenlabs.io/v1")
    
    # Performance and Scaling Settings
    DEFAULT_TIMEOUT = int(os.getenv("DEFAULT_TIMEOUT", "30"))  # Default timeout in seconds
    MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
    BACKOFF_FACTOR = float(os.getenv("BACKOFF_FACTOR", "0.5"))
    
    # Application Settings
    DEBUG_MODE = os.getenv("DEBUG_MODE", "false").lower() == "true"
    
    @classmethod
    def validate(cls):
        """Validates the configuration and logs any issues"""
        missing_vars = []
        
        if not cls.AZURE_API_KEY:
            missing_vars.append("AZURE_API_KEY")
            
        if not cls.ELEVENLABS_API_KEY:
            missing_vars.append("ELEVENLABS_API_KEY")
            
        if not cls.DEEPGRAM_API_KEY:
            missing_vars.append("DEEPGRAM_API_KEY")
            
        if not cls.TTS_URL:
            missing_vars.append("TTS_URL")
            
        if missing_vars:
            logger.warning(f"Missing environment variables: {', '.join(missing_vars)}")
            return False
        
        logger.info("Configuration validated successfully")
        return True
            
    @classmethod
    def get_retry_config(cls):
        """Returns a dictionary with retry configuration parameters"""
        return {
            "total": cls.MAX_RETRIES,
            "backoff_factor": cls.BACKOFF_FACTOR,
            "status_forcelist": [429, 500, 502, 503, 504]
        }
        
# Validate configuration on import
Config.validate()