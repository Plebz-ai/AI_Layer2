import os
import json
import requests
import redis
import logging
from flask import Flask, request, jsonify, Response
from werkzeug.utils import secure_filename
import speech_recognition as sr
from grpc import insecure_channel
from LLM_Layer import get_llm_response
import subprocess
from pydub.utils import mediainfo
import base64
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("transcription_service")

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# External service URLs
TTS_URL = "https://api.deepgram.com/v1/speak"
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")

# Initialize Redis client
redis_client = redis.StrictRedis(host='redis', port=6379, decode_responses=True)

# Example gRPC setup (assuming a proto file is defined for communication)
channel = insecure_channel('backend:50051')
# grpc_stub = YourServiceStub(channel)  # Replace with actual gRPC stub

@app.route('/process_audio', methods=['POST'])
def process_audio():
    try:
        # Check if we're getting audio data or a file
        if 'audio' in request.files:
            # Handle file upload
            audio_file = request.files['audio']
            if audio_file.filename == '':
                return jsonify({'error': 'No selected file'}), 400
            filename = secure_filename(audio_file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            audio_file.save(file_path)
            
            with open(file_path, 'rb') as f:
                audio_data = f.read()
        elif request.data:
            # Handle raw binary data
            audio_data = request.data
        else:
            return jsonify({'error': 'No audio data provided'}), 400
        
        # Get session ID from request headers or parameters
        session_id = request.headers.get('X-Session-ID') or request.args.get('session_id')
        character_id = request.headers.get('X-Character-ID') or request.args.get('character_id')
        
        logger.info(f"Processing audio for session {session_id} and character {character_id}")
        
        # Send raw audio to Deepgram STT
        dg_url = f"https://api.deepgram.com/v1/listen?model=nova-3&language=en-US"
        resp = requests.post(
            dg_url,
            headers={
                'Authorization': f'Token {DEEPGRAM_API_KEY}',
                'Content-Type': 'audio/wav'
            },
            data=audio_data
        )
        
        if resp.status_code != 200:
            logger.error(f"Deepgram STT error: {resp.status_code} {resp.text}")
            return jsonify({'error': f'Deepgram STT error: {resp.status_code} {resp.text}'}), 500
        
        result = resp.json()
        transcript = result.get('results', {}).get('channels', [{}])[0].get('alternatives', [{}])[0].get('transcript')
        
        if not transcript:
            transcript = ""
            logger.warning("No transcript detected in speech")
        
        logger.info(f"Speech-to-text result: {transcript}")
            
        # Get character info and conversation history if session ID and character ID are provided
        character_info = None
        conversation_history = []
        
        if session_id and character_id and redis_client:
            try:
                # Try to get character info from Redis
                character_key = f"character:{character_id}"
                character_data = redis_client.get(character_key)
                if character_data:
                    character_info = json.loads(character_data)
                
                # Try to get conversation history from Redis
                history_key = f"conversation:{session_id}"
                history_data = redis_client.lrange(history_key, -10, -1)  # Get last 10 messages
                if history_data:
                    conversation_history = [json.loads(msg) for msg in history_data]
            except Exception as e:
                logger.error(f"Error retrieving data from Redis: {str(e)}")
        
        # Generate LLM response
        ai_response = get_llm_response(transcript, character_info, conversation_history)
        logger.info(f"Generated AI response: {ai_response[:50]}..." if len(ai_response) > 50 else f"Generated AI response: {ai_response}")
        
        # Store the conversation in Redis if possible
        if session_id and redis_client:
            try:
                # Save user message
                user_message = json.dumps({
                    "sender": "user",
                    "content": transcript,
                    "timestamp": int(time.time())
                })
                redis_client.rpush(f"conversation:{session_id}", user_message)
                
                # Save AI response
                ai_message = json.dumps({
                    "sender": "character",
                    "content": ai_response,
                    "timestamp": int(time.time())
                })
                redis_client.rpush(f"conversation:{session_id}", ai_message)
            except Exception as e:
                logger.error(f"Error storing conversation in Redis: {str(e)}")
        
        # Return both the transcript and the AI response
        return jsonify({
            'transcript': transcript,
            'response': ai_response
        })
        
    except Exception as e:
        logger.error(f"Error processing audio: {str(e)}")
        return jsonify({'error': f'Error processing audio: {str(e)}'}), 500

# Add TTS synthesis endpoint
@app.route('/tts/synthesize', methods=['POST'])
def tts_synthesize():
    try:
        data = request.get_json()
        text = data.get('text')
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        logger.info(f"Synthesizing speech for text: {text[:50]}..." if len(text) > 50 else f"Synthesizing speech for text: {text}")
        
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
            logger.error(f"Deepgram TTS error: {response.status_code} {response.text}")
            return jsonify({'error': f'Deepgram TTS error: {response.status_code} {response.text}'}), response.status_code
        
        # Return MP3 audio stream
        return Response(response.content, mimetype='audio/mpeg')
        
    except Exception as e:
        logger.error(f"Error synthesizing speech: {str(e)}")
        return jsonify({'error': f'Error synthesizing speech: {str(e)}'}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'service': 'transcription_service'})

if __name__ == '__main__':
    # Ensure we have the required environment variables
    if not DEEPGRAM_API_KEY:
        logger.warning("DEEPGRAM_API_KEY not set. TTS and STT functionality will not work.")
    
    if not os.getenv("AZURE_API_KEY"):
        logger.warning("AZURE_API_KEY not set. LLM functionality will not work.")
    
    app.run(host='0.0.0.0', port=5000)