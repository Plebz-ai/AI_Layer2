import os
import requests
import redis
from flask import Flask, request, jsonify, Response
from werkzeug.utils import secure_filename
import speech_recognition as sr
from grpc import insecure_channel
from LLM_Layer import get_llm_response
import subprocess
from pydub.utils import mediainfo
import base64

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# External service URLs
LLM_URL = "https://azure-ai-foundry-llama-endpoint.com/process"
# Use Deepgram Aura TTS endpoint
TTS_URL = "https://api.deepgram.com/v1/speak"
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")

# Initialize Redis client
redis_client = redis.StrictRedis(host='redis', port=6379, decode_responses=True)

# Example gRPC setup (assuming a proto file is defined for communication)
channel = insecure_channel('backend:50051')
# grpc_stub = YourServiceStub(channel)  # Replace with actual gRPC stub

@app.route('/process_audio', methods=['POST'])
def process_audio():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400
    audio_file = request.files['audio']
    if audio_file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    filename = secure_filename(audio_file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    audio_file.save(file_path)

    # Send raw audio to Deepgram STT
    with open(file_path, 'rb') as f:
        dg_url = f"https://api.deepgram.com/v1/listen?model=nova-3&language=en-US"
        resp = requests.post(
            dg_url,
            headers={
                'Authorization': f'Token {DEEPGRAM_API_KEY}',
                'Content-Type': 'audio/wav'
            },
            data=f.read()
        )
    if resp.status_code != 200:
        return jsonify({'error': f'Deepgram STT error: {resp.status_code} {resp.text}'}), 500
    result = resp.json()
    transcript = result.get('results', {}).get('channels', [{}])[0].get('alternatives', [{}])[0].get('transcript')
    if not transcript:
        # Return empty transcript instead of error to allow downstream processing
        transcript = ""
    return jsonify({'transcript': transcript})

# Add TTS synthesis endpoint
@app.route('/tts/synthesize', methods=['POST'])
def tts_synthesize():
    data = request.get_json()
    text = data.get('text')
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    # Call Deepgram Aura Text-to-Speech API
    # Default to aura-asteria-en if no model specified
    model = data.get('model', 'aura-asteria-en')
    dg_url = f"{TTS_URL}?model={model}"
    response = requests.post(
        dg_url,
        headers={
            "Authorization": f"Token {DEEPGRAM_API_KEY}",
            "Content-Type": "application/json"
        },
        json={"text": text}
    )
    if response.status_code != 200:
        return jsonify({'error': f'Deepgram TTS error: {response.status_code} {response.text}'}), response.status_code
    # Return MP3 audio stream
    return Response(response.content, mimetype='audio/mpeg')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)