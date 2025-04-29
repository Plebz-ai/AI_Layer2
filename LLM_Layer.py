import os
import logging
import time
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import ServiceRequestError, ClientAuthenticationError

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("LLM_Layer")

def get_llm_response(transcript, character_info=None, conversation_history=None):
    """
    Sends the transcript to the Azure AI Foundry Llama model and retrieves the response.
    
    Parameters:
        transcript (str): The user's speech transcript to respond to
        character_info (dict): Optional character information (personality, voice type, etc.)
        conversation_history (list): Optional previous messages for context
    
    Returns:
        str: The AI response text
    """
    # Log the incoming request
    logger.info(f"Processing transcript: {transcript[:50]}..." if len(transcript) > 50 else f"Processing transcript: {transcript}")
    
    try:
        # Get API credentials from environment
        endpoint = os.getenv("AZURE_ENDPOINT", "https://finetuned-model-wmoqzjog.eastus2.models.ai.azure.com")
        model_name = os.getenv("AZURE_MODEL_NAME", "Llama-4-Maverick-17B-128E-Instruct-FP8")
        api_key = os.getenv("AZURE_API_KEY")
        
        # Check if API key is available
        if not api_key:
            logger.error("AZURE_API_KEY not found in environment variables")
            return "I'm sorry, I'm having trouble connecting to my brain right now. Please try again later."
        
        # Prepare system message with character context if available
        system_message = "You are a helpful assistant."
        if character_info:
            system_message = f"You are {character_info.get('name', 'an AI assistant')}. "
            system_message += f"Personality: {character_info.get('personality', 'helpful and friendly')}. "
            system_message += "Respond in a conversational manner fitting your personality."
        
        # Initialize the chat client
        client = ChatCompletionsClient(
            endpoint=endpoint,
            credential=AzureKeyCredential(api_key),
        )
        
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
        response = client.complete(
            messages=messages,
            max_tokens=2048,
            temperature=0.8,
            top_p=0.1,
            presence_penalty=0.0,
            frequency_penalty=0.0,
            model=model_name
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