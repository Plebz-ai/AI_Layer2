import os
import requests

def transcribe_audio_with_deepgram(audio_file_path):
    """
    Transcribes the audio file using Deepgram's STT API.
    """
    DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
    if not DEEPGRAM_API_KEY:
        raise Exception("Deepgram API key is not set in the environment variables.")

    url = "https://api.deepgram.com/v1/listen"
    headers = {
        "Authorization": f"Token {DEEPGRAM_API_KEY}",
        "Content-Type": "audio/wav",
    }

    with open(audio_file_path, "rb") as audio_file:
        response = requests.post(url, headers=headers, data=audio_file)

    if response.status_code != 200:
        raise Exception(f"Deepgram STT API error: {response.status_code}, {response.text}")

    result = response.json()
    return result.get("results", {}).get("channels", [{}])[0].get("alternatives", [{}])[0].get("transcript", "")