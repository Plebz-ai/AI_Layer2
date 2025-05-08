import requests

class SpeechService:
    def __init__(self, tts_api_key, stt_api_key):
        self.tts_api_key = tts_api_key
        self.stt_api_key = stt_api_key

    def text_to_speech(self, text, voice_type):
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_type}"
        headers = {"xi-api-key": self.tts_api_key, "Content-Type": "application/json"}
        payload = {"text": text}
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        return response.content

    def speech_to_text(self, audio_data):
        url = "https://api.deepgram.com/v1/listen"
        headers = {"Authorization": f"Token {self.stt_api_key}"}
        files = {"file": audio_data}
        response = requests.post(url, headers=headers, files=files)
        response.raise_for_status()
        return response.json()["results"]["channels"][0]["alternatives"][0]["transcript"]