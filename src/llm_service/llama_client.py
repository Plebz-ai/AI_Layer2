import requests

class LLamaClient:
    def __init__(self, endpoint, api_key):
        self.endpoint = endpoint
        self.api_key = api_key

    def generate_response(self, messages):
        url = f"{self.endpoint}/openai/deployments/llama/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        payload = {
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 800,
            "top_p": 0.95,
            "frequency_penalty": 0,
            "presence_penalty": 0
        }
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        return response.json()