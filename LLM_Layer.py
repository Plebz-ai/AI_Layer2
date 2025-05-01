import logging
import time
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import ServiceRequestError, ClientAuthenticationError
from config import Config

# Configure logging
logger = logging.getLogger("LLM_Layer")

class ChatClientSingleton:
    """Singleton class for managing Azure ChatCompletions client"""
    _instance = None
    _client = None
    
    @classmethod
    def get_instance(cls):
        """
        Returns a singleton instance of ChatCompletionsClient to reuse connections
        and minimize TLS handshakes.
        """
        if cls._instance is None or cls._client is None:
            endpoint = Config.AZURE_ENDPOINT
            api_key = Config.AZURE_API_KEY
            
            if not api_key:
                logger.error("AZURE_API_KEY not found in configuration")
                raise ValueError("AZURE_API_KEY is required")
                
            logger.info("Initializing new ChatCompletionsClient instance")
            cls._client = ChatCompletionsClient(
                endpoint=endpoint,
                credential=AzureKeyCredential(api_key),
            )
            cls._instance = cls()
        return cls._client


class LanguageModelService:
    """Service class for language model interactions"""
    
    def __init__(self):
        self.model_name = Config.AZURE_MODEL_NAME
        try:
            self.client = ChatClientSingleton.get_instance()
        except ValueError as e:
            logger.error(f"Failed to initialize LLM client: {str(e)}")
            self.client = None
    
    def get_response(self, transcript, character_info=None, conversation_history=None):
        """
        Sends the transcript to the Language Model and retrieves the response.
        
        Args:
            transcript (str): The user's speech transcript to respond to
            character_info (dict): Optional character information (personality, voice type, etc.)
            conversation_history (list): Optional previous messages for context
            
        Returns:
            str: The AI response text
        """
        # Log the incoming request
        logger.info(f"Processing transcript: {transcript[:50]}..." if len(transcript) > 50 else f"Processing transcript: {transcript}")
        
        if not self.client:
            return "I'm sorry, I'm having trouble connecting to my brain right now. Please try again later."
        
        try:
            # Prepare system message with character context if available
            system_message = "You are a helpful assistant."
            if character_info:
                system_message = f"You are {character_info.get('name', 'an AI assistant')}. "
                system_message += f"Personality: {character_info.get('personality', 'helpful and friendly')}. "
                system_message += "Respond in a conversational manner fitting your personality."
            
            # Prepare messages including history if available
            messages = [SystemMessage(content=system_message)]
            
            # Add conversation history if provided
            if conversation_history:
                for msg in conversation_history[-5:]:  # Use last 5 messages for context
                    if msg.get('sender') == 'user':
                        messages.append(UserMessage(content=msg.get('content')))
                    elif msg.get('sender') == 'character':
                        messages.append(SystemMessage(content=msg.get('content')))
            
            # Add current user message
            messages.append(UserMessage(content=transcript))
            
            # Start timer for performance monitoring
            start_time = time.time()
            
            # Get completion from Azure AI
            response = self.client.complete(
                messages=messages,
                max_tokens=2048,
                temperature=0.8,
                top_p=0.1,
                presence_penalty=0.0,
                frequency_penalty=0.0,
                model=self.model_name,
                timeout=Config.DEFAULT_TIMEOUT
            )
            
            # Log performance metrics
            elapsed_time = time.time() - start_time
            logger.info(f"LLM response generated in {elapsed_time:.2f} seconds")
            
            # Return the AI response text
            return response.choices[0].message.content
            
        except ClientAuthenticationError as e:
            logger.error(f"Authentication error: {str(e)}")
            return "I'm sorry, I'm having trouble accessing my language capabilities. Please check your API key configuration."
        except ServiceRequestError as e:
            logger.error(f"Service request error: {str(e)}")
            return "I'm having trouble reaching the language service. Please check your network connection and try again."
        except Exception as e:
            logger.error(f"Unexpected error in LLM processing: {str(e)}")
            return "I apologize, but I'm experiencing technical difficulties. Please try again later."


# Legacy function for backwards compatibility
def get_llm_response(transcript, character_info=None, conversation_history=None):
    """Legacy function that delegates to the new service class"""
    service = LanguageModelService()
    return service.get_response(transcript, character_info, conversation_history)