import requests
import os
def convert_text_to_speech(text):
    """
    Converts the given text to speech using the Deepgram TTS service.
    """
    TTS_URL = os.getenv("TTS_URL")
    DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
    headers = {"Authorization": f"Token {DEEPGRAM_API_KEY}"}
    response = requests.post(TTS_URL, headers=headers, json={"text": text, "voice": "en-US"})
    if response.status_code != 200:
        raise Exception(f"TTS service error: {response.status_code}")
    return response.content