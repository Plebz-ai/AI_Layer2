import os

class Config:
    AZURE_LLAMA_ENDPOINT = os.getenv('AZURE_LLAMA_ENDPOINT', 'https://default-llama-endpoint')
    AZURE_API_KEY = os.getenv('AZURE_API_KEY', 'default-api-key')
    ELEVENLABS_API_KEY = os.getenv('ELEVENLABS_API_KEY', 'default-elevenlabs-key')
    REDIS_URL = os.getenv('REDIS_URL', 'redis://localhost:6379/0')