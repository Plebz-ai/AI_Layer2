"""
AI Service Orchestration Layer

This module provides a unified interface for interacting with all AI services.
It orchestrates the flow between speech-to-text, language model, and text-to-speech
services to provide a complete conversation experience.
"""

import logging
import os
import base64
import time
from STT_Service import SpeechToTextService
from LLM_Layer import LanguageModelService
from TTS_Service import TextToSpeechService
from config import Config

# Configure logging
logger = logging.getLogger("AI_Service")

class AIService:
    """
    Main service class that orchestrates STT, LLM, and TTS services
    to provide a complete conversation experience.
    """
    
    def __init__(self):
        """Initialize all required services and validate configuration."""
        # Initialize component services
        self.stt_service = SpeechToTextService()
        self.llm_service = LanguageModelService()
        self.tts_service = TextToSpeechService()
        
        # Initialize conversation history
        self.conversation_history = []
        
        # Get available voices
        self.available_voices = self.tts_service.get_available_voices()
        
        if Config.DEBUG_MODE:
            logger.info("AI Service initialized in DEBUG mode")
            if self.available_voices:
                logger.info(f"Found {len(self.available_voices)} available voices")
    
    def get_voice_id(self, voice_type="neutral"):
        """Map voice type to ElevenLabs voice ID"""
        voice_mapping = {
            "neutral": "21m00Tcm4TlvDq8ikWAM",  # Rachel
            "friendly": "MF3mGyEYCl7XYWbV9V6O",  # Bella
            "professional": "AZnzlk1XvdvUeBnXmlld",  # Domi
            "casual": "ThT5KcBeYPX3keUQqHPh"  # Antoni
        }
        return voice_mapping.get(voice_type, "21m00Tcm4TlvDq8ikWAM")
    
    def process_audio(self, audio_file_path, character_info=None):
        """Process audio through the AI pipeline"""
        result = {
            "transcript": None,
            "response_text": None,
            "audio_data": None,
            "success": False,
            "error": None
        }
        
        try:
            # Step 1: Transcribe audio to text
            logger.info(f"Processing audio file: {audio_file_path}")
            transcript = self.stt_service.transcribe_audio(audio_file_path)
            result["transcript"] = transcript
            
            if not transcript:
                result["error"] = "Failed to transcribe audio (empty transcript)"
                return result
            
            # Add user message to conversation history
            self.add_to_history(transcript, sender="user")
            
            # Step 2: Generate response from LLM
            response_text = self.llm_service.get_response(
                transcript, 
                character_info, 
                self.conversation_history
            )
            result["response_text"] = response_text
            
            if not response_text:
                result["error"] = "Failed to generate response from language model"
                return result
            
            # Add AI message to conversation history
            self.add_to_history(response_text, sender="AI")
            
            # Step 3: Convert response to speech
            voice_type = character_info.get("voice_type", "neutral") if character_info else "neutral"
            voice_id = self.get_voice_id(voice_type)
            audio_data = self.tts_service.convert_text_to_speech(response_text, voice_id)
            
            # Encode audio data as base64 for API response
            if audio_data:
                result["audio_data"] = base64.b64encode(audio_data)
            else:
                result["error"] = "Failed to convert response to speech"
                return result
            
            # Mark as successful if we have audio data
            result["success"] = True
            return result
            
        except Exception as e:
            logger.error(f"Error in AI service processing: {str(e)}")
            result["error"] = f"Error processing request: {str(e)}"
            return result
    
    def generate_response(self, text, character_info=None, conversation_history=None):
        """
        Generate a text response from the language model without audio processing.
        
        Args:
            text (str): The text input to respond to
            character_info (dict): Optional character information 
            conversation_history (list): Optional previous messages for context
            
        Returns:
            str: The generated text response
        """
        try:
            return self.llm_service.get_response(text, character_info, conversation_history)
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return f"Error generating response: {str(e)}"
    
    def text_to_speech(self, text, voice="en-US"):
        """
        Convert text to speech without going through the full pipeline.
        
        Args:
            text (str): The text to convert to speech
            voice (str): The voice ID to use
            
        Returns:
            bytes: Audio data of synthesized speech
        """
        try:
            return self.tts_service.convert_text_to_speech(text, voice)
        except Exception as e:
            logger.error(f"Error in text-to-speech conversion: {str(e)}")
            return None
    
    def speech_to_text(self, audio_file_path):
        """
        Convert speech to text without going through the full pipeline.
        
        Args:
            audio_file_path (str): Path to the audio file
            
        Returns:
            str: The transcribed text
        """
        try:
            return self.stt_service.transcribe_audio(audio_file_path)
        except Exception as e:
            logger.error(f"Error in speech-to-text conversion: {str(e)}")
            return None
    
    def clear_conversation_history(self):
        """Clear the conversation history"""
        self.conversation_history = []
        logger.info("Conversation history cleared")
    
    def get_conversation_history(self):
        """Get the current conversation history"""
        return self.conversation_history
    
    def manage_conversation_history(self):
        """Maintain conversation history size and clean up old messages"""
        if len(self.conversation_history) > 10:  # Keep last 10 messages
            self.conversation_history = self.conversation_history[-10:]
    
    def add_to_history(self, message, sender="user"):
        """Add a message to conversation history"""
        self.conversation_history.append({
            "sender": sender,
            "content": message,
            "timestamp": time.time()
        })
        self.manage_conversation_history()
    
    def get_character_voices(self):
        """Get available character voices"""
        return {
            "neutral": "Rachel (Natural)",
            "friendly": "Bella (Friendly)",
            "professional": "Domi (Professional)",
            "casual": "Antoni (Casual)"
        }