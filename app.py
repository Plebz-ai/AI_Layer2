"""
AI Layer Flask Server

This module provides HTTP endpoints for the AI services, allowing the
backend to interact with the AI components via REST API.
"""

import os
import logging
import tempfile
import uuid
from flask import Flask, request, jsonify, send_file
from werkzeug.utils import secure_filename
from ai_service import AIService
from config import Config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("AI_Layer_Flask")

# Initialize Flask app
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Limit upload size to 16MB
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(__file__), 'uploads')

# Create uploads directory if it doesn't exist
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Initialize AI service
ai_service = AIService()

@app.route('/healthz', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "ok"}), 200

@app.route('/process_audio', methods=['POST'])
def process_audio():
    """
    Process audio file through the AI pipeline.
    
    Expects:
    - audio file in the 'audio' field
    - Optional session_id and character_id in form fields
    
    Returns:
    - transcript: The transcribed text
    - response_text: The AI-generated text response
    - audio_data (base64): The synthesized speech response
    """
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file provided"}), 400
    
    audio_file = request.files['audio']
    if audio_file.filename == '':
        return jsonify({"error": "No audio file selected"}), 400
    
    # Get session and character info from request
    session_id = request.form.get('session_id', str(uuid.uuid4()))
    character_id = request.form.get('character_id')
    
    character_info = None
    if character_id:
        # In a real implementation, this would fetch character details from a database
        character_info = {
            "id": character_id,
            "name": "AI Assistant",
            "personality": "Helpful and friendly",
            "voice": "en-US"
        }
    
    # Save the audio file temporarily
    filename = secure_filename(f"{session_id}_{uuid.uuid4()}.wav")
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    audio_file.save(filepath)
    
    try:
        # Process the audio through the AI pipeline
        result = ai_service.process_audio(filepath, character_info)
        
        if not result["success"]:
            return jsonify({"error": result["error"]}), 500
        
        # Return the response - audio_data is already base64 encoded in the AIService
        return jsonify({
            "transcript": result["transcript"],
            "response": result["response_text"],
            "audio_data": result["audio_data"].decode('utf-8') if result["audio_data"] else None
        }), 200
        
    except Exception as e:
        logger.error(f"Error processing audio: {str(e)}")
        return jsonify({"error": f"Error processing audio: {str(e)}"}), 500
    finally:
        # Clean up the temporary audio file
        if os.path.exists(filepath):
            os.remove(filepath)

@app.route('/generate_response', methods=['POST'])
def generate_response():
    """
    Generate an AI response for a given text input.
    
    Expects:
    - text: The text to respond to
    - character_info: Optional character information
    - conversation_history: Optional conversation history
    
    Returns:
    - response: The AI-generated response
    """
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({"error": "No text provided"}), 400
    
    text = data['text']
    character_info = data.get('character_info')
    conversation_history = data.get('conversation_history')
    
    try:
        response = ai_service.generate_response(text, character_info, conversation_history)
        return jsonify({"response": response}), 200
    except Exception as e:
        logger.error(f"Error generating response: {str(e)}")
        return jsonify({"error": f"Error generating response: {str(e)}"}), 500

@app.route('/text_to_speech', methods=['POST'])
def text_to_speech():
    """
    Convert text to speech.
    
    Expects:
    - text: The text to convert to speech
    - voice: Optional voice identifier
    
    Returns:
    - audio file (WAV format)
    """
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({"error": "No text provided"}), 400
    
    text = data['text']
    voice = data.get('voice', 'en-US')
    
    try:
        audio_data = ai_service.text_to_speech(text, voice)
        if not audio_data:
            return jsonify({"error": "Failed to generate speech"}), 500
        
        # Save the audio to a temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
        temp_file.write(audio_data)
        temp_file.close()
        
        return send_file(
            temp_file.name,
            mimetype='audio/wav',
            as_attachment=True,
            download_name='response.wav'
        )
    except Exception as e:
        logger.error(f"Error in text to speech conversion: {str(e)}")
        return jsonify({"error": f"Error in text to speech conversion: {str(e)}"}), 500

@app.route('/speech_to_text', methods=['POST'])
def speech_to_text():
    """
    Convert speech to text.
    
    Expects:
    - audio file in the 'audio' field
    
    Returns:
    - transcript: The transcribed text
    """
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file provided"}), 400
    
    audio_file = request.files['audio']
    if audio_file.filename == '':
        return jsonify({"error": "No audio file selected"}), 400
    
    # Save the audio file temporarily
    filename = secure_filename(f"stt_{uuid.uuid4()}.wav")
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    audio_file.save(filepath)
    
    try:
        # Transcribe the audio
        transcript = ai_service.speech_to_text(filepath)
        if not transcript:
            return jsonify({"error": "Failed to transcribe audio"}), 500
        
        return jsonify({"transcript": transcript}), 200
    except Exception as e:
        logger.error(f"Error transcribing audio: {str(e)}")
        return jsonify({"error": f"Error transcribing audio: {str(e)}"}), 500
    finally:
        # Clean up the temporary audio file
        if os.path.exists(filepath):
            os.remove(filepath)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = Config.DEBUG_MODE
    
    logger.info(f"Starting AI Layer Flask server on port {port}, debug mode: {debug}")
    app.run(host='0.0.0.0', port=port, debug=debug)