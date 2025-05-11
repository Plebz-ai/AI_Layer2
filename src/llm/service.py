from typing import List, Dict, Optional
import os
import json
import aiohttp
import asyncio
import logging
from datetime import datetime

# Optional dependencies
try:
    from openai import AsyncOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logging.warning("OpenAI SDK not available - using mock LLM service")

class LLMService:
    def __init__(self):
        # --- Azure OpenAI (o4-mini) ---
        self.azure_openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        self.azure_openai_api_key = os.getenv("AZURE_OPENAI_API_KEY")
        self.azure_openai_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_LLM1", "o4-mini")
        self.azure_openai_api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
        # --- Max tokens config ---
        self.default_max_tokens = int(os.getenv("AZURE_OPENAI_MAX_TOKENS", "512"))
        # Character ID mapping (numeric to string)
        self.character_id_map = {
            "1": "elon_musk",
            "2": "sherlock_holmes",
            "3": "marie_curie"
            # Add more as needed
        }

        # --- Azure Inference (Llama-4-Maverick) ---
        self.azure_inference_endpoint = os.getenv("AZURE_INFERENCE_ENDPOINT")
        self.azure_inference_api_key = os.getenv("AZURE_INFERENCE_API_KEY")
        self.azure_inference_deployment = os.getenv("AZURE_INFERENCE_DEPLOYMENT", "finetuned-model-wmoqzjog")
        self.azure_inference_api_version = os.getenv("AZURE_INFERENCE_API_VERSION", "2024-05-01-preview")

        # Check if Azure OpenAI is available
        self.use_azure_openai = bool(self.azure_openai_endpoint and self.azure_openai_api_key)
        self.use_azure_inference = bool(self.azure_inference_endpoint and self.azure_inference_api_key)

        # OpenAI direct configuration (fallback)
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if OPENAI_AVAILABLE and self.openai_api_key:
            self.openai_client = AsyncOpenAI(api_key=self.openai_api_key)
            self.use_openai = True
            logging.info("OpenAI direct API configured")
        else:
            self.use_openai = False

        if not (self.use_azure_openai or self.use_azure_inference or self.use_openai):
            logging.warning("No LLM services available - using mock implementations")

        # Load predefined characters
        self.predefined_characters = self._load_predefined_characters()
        logging.info(f"Loaded {len(self.predefined_characters)} predefined characters")

    def _load_predefined_characters(self) -> Dict:
        """Load predefined character configurations."""
        try:
            characters_path = "src/llm/data/predefined_characters.json"
            if not os.path.exists(characters_path):
                characters_path = "llm/data/predefined_characters.json"
            with open(characters_path, "r") as f:
                return json.load(f)
        except FileNotFoundError as e:
            logging.error(f"Characters file not found: {str(e)}")
            # Return fallback characters
            return {
                "elon_musk": {
                    "name": "Elon Musk",
                    "description": "Tech entrepreneur and CEO of Tesla and SpaceX",
                    "personality": "Innovative, ambitious, and forward-thinking",
                    "voice_type": "male_voice_1",
                    "system_prompt": "You are Elon Musk, the CEO of Tesla and SpaceX. You are known for your innovative thinking, ambitious goals, and sometimes controversial statements."
                }
            }

    async def _call_azure_openai(self, 
                                 messages: List[Dict], 
                                 model_name: str = None,
                                 temperature: float = 1.0,
                                 max_tokens: int = None,
                                 top_p: float = None,
                                 presence_penalty: float = None,
                                 frequency_penalty: float = None) -> str:
        """Call Azure OpenAI (o4-mini) API. Only include parameters if not default or explicitly set."""
        if not self.use_azure_openai:
            raise ValueError("Azure OpenAI not configured")
        try:
            deployment = model_name or self.azure_openai_deployment
            url = f"{self.azure_openai_endpoint}/openai/deployments/{deployment}/chat/completions?api-version={self.azure_openai_api_version}"
            async with aiohttp.ClientSession() as session:
                headers = {
                    "Content-Type": "application/json",
                    "api-key": self.azure_openai_api_key
                }
                if max_tokens is None:
                    max_tokens = self.default_max_tokens
                if max_tokens < 128:
                    logging.warning(f"[AzureOpenAI] max_completion_tokens is very low: {max_tokens}. Forcing to 128.")
                    max_tokens = 128
                payload = {
                    "messages": messages,
                    "max_completion_tokens": max_tokens
                }
                if temperature is not None and temperature != 1.0:
                    payload["temperature"] = temperature
                if top_p is not None:
                    payload["top_p"] = top_p
                elif os.getenv("AZURE_OPENAI_ENABLE_TOP_P", "false").lower() == "true":
                    payload["top_p"] = 0.95
                if presence_penalty is not None:
                    payload["presence_penalty"] = presence_penalty
                elif os.getenv("AZURE_OPENAI_ENABLE_PRESENCE_PENALTY", "false").lower() == "true":
                    payload["presence_penalty"] = 0.6
                if frequency_penalty is not None:
                    payload["frequency_penalty"] = frequency_penalty
                elif os.getenv("AZURE_OPENAI_ENABLE_FREQUENCY_PENALTY", "false").lower() == "true":
                    payload["frequency_penalty"] = 0.3
                logging.info(f"[AzureOpenAI] Payload: {json.dumps(payload)[:500]}")
                async with session.post(url, headers=headers, json=payload) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise Exception(f"Azure OpenAI API error ({response.status}): {error_text}")
                    result = await response.json()
                    logging.info(f"[AzureOpenAI] Full response: {json.dumps(result)[:500]}")
                    content = result["choices"][0]["message"]["content"]
                    if not content:
                        logging.warning("[AzureOpenAI] Empty response from Azure OpenAI, using fallback.")
                        return None
                    return content
        except Exception as e:
            logging.error(f"Error calling Azure OpenAI: {str(e)}")
            return None

    async def _call_azure_inference(self, 
                                    messages: List[Dict], 
                                    temperature: float = 0.7,
                                    max_tokens: int = 150) -> str:
        """Call Azure Inference (Llama-4-Maverick) API."""
        if not self.use_azure_inference:
            raise ValueError("Azure Inference not configured")
        try:
            deployment = self.azure_inference_deployment
            url = f"{self.azure_inference_endpoint}/v1/chat/completions?api-version={self.azure_inference_api_version}"
            async with aiohttp.ClientSession() as session:
                headers = {
                    "Content-Type": "application/json",
                    "api-key": self.azure_inference_api_key
                }
                payload = {
                    "messages": messages,
                    "temperature": temperature,
                    "max_completion_tokens": max_tokens,
                    "frequency_penalty": 0.3,
                    "presence_penalty": 0.6,
                    "model": deployment
                }
                async with session.post(url, headers=headers, json=payload) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise Exception(f"Azure Inference API error ({response.status}): {error_text}")
                    result = await response.json()
                    return result["choices"][0]["message"]["content"]
        except Exception as e:
            logging.error(f"Error calling Azure Inference: {str(e)}")
            raise

    async def _call_openai_direct(self, 
                                 messages: List[Dict], 
                                 model: str = "gpt-4-turbo-preview",
                                 temperature: float = 1.0,
                                 max_tokens: int = 150) -> str:
        """Call OpenAI API directly (fallback if Azure not available)."""
        if not self.use_openai:
            raise ValueError("OpenAI direct API not configured")
        try:
            response = await self.openai_client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                presence_penalty=0.6,
                frequency_penalty=0.3
            )
            return response.choices[0].message.content
        except Exception as e:
            logging.error(f"Error calling OpenAI direct: {str(e)}")
            raise

    async def generate_predefined_response(
        self,
        character_id: str,
        user_message: str,
        conversation_history: List[Dict] = None
    ) -> str:
        """Generate response for a predefined character using o4-mini."""
        try:
            # Map numeric ID to string key if needed
            character_key = self.character_id_map.get(str(character_id), character_id)
            character = self.predefined_characters.get(character_key)
            if not character:
                logging.warning(f"Character {character_id} not found, using fallback")
                character = next(iter(self.predefined_characters.values()))
            messages = [
                {"role": "system", "content": character["system_prompt"]},
            ]
            if conversation_history:
                for msg in conversation_history[-5:]:
                    role = "assistant" if msg.get("sender") == "character" else "user"
                    messages.append({"role": role, "content": msg.get("content", "")})
            messages.append({"role": "user", "content": user_message})
            response = None
            # Allow per-character LLM params
            temp = character.get("temperature", 1.0)
            top_p = character.get("top_p")
            presence_penalty = character.get("presence_penalty")
            frequency_penalty = character.get("frequency_penalty")
            if self.use_azure_openai:
                try:
                    response = await self._call_azure_openai(
                        messages,
                        temperature=temp,
                        max_tokens=self.default_max_tokens,
                        top_p=top_p,
                        presence_penalty=presence_penalty,
                        frequency_penalty=frequency_penalty
                    )
                except Exception as e:
                    logging.error(f"Azure OpenAI failed, trying fallback: {str(e)}")
            if not response and self.use_openai:
                try:
                    response = await self._call_openai_direct(messages, max_tokens=self.default_max_tokens)
                except Exception as e:
                    logging.error(f"OpenAI direct failed: {str(e)}")
            if not response:
                logging.info("Using mock response generator")
                response = f"This is a mock response to: {user_message}. Character: {character['name']}."
            return response
        except Exception as e:
            logging.error(f"Error generating predefined response: {str(e)}")
            return f"I'm sorry, I encountered an error. Please try again later."

    async def generate_custom_character_response(
        self,
        character_config: Dict,
        user_message: str,
        conversation_history: List[Dict] = None
    ) -> str:
        """Generate response for a custom character using Llama-4-Maverick."""
        try:
            character_details = f"""
Name: {character_config.get('name', 'Unknown')}
Description: {character_config.get('description', 'No description')}
Personality: {character_config.get('personality', 'No personality defined')}
Voice Type: {character_config.get('voice_type', 'default')}
"""
            prompt_messages = [
                {"role": "system", "content": "You are a prompt engineering expert. Create a detailed system prompt for a character."},
                {"role": "user", "content": f"""
Create a detailed prompt for an AI character with the following traits:
{character_details}

The prompt should:
1. Start with \"You are [character name].\"
2. Include detailed personality traits and speaking style
3. Give background information related to the character
4. Specify how the character should respond to users
5. Include any relevant knowledge areas or expertise

Format the prompt to be used with an LLM, focusing on making the character realistic and engaging.
"""}
            ]
            enhanced_prompt = None
            if self.use_azure_inference:
                try:
                    enhanced_prompt = await self._call_azure_inference(
                        prompt_messages,
                        temperature=0.7,
                        max_tokens=500
                    )
                except Exception as e:
                    logging.error(f"Azure Inference LLM1 failed, trying fallback: {str(e)}")
            if not enhanced_prompt and self.use_openai:
                try:
                    enhanced_prompt = await self._call_openai_direct(
                        prompt_messages,
                        model="gpt-3.5-turbo",
                        temperature=0.7,
                        max_tokens=500
                    )
                except Exception as e:
                    logging.error(f"OpenAI direct LLM1 failed: {str(e)}")
            if not enhanced_prompt:
                logging.info("Using fallback prompt template")
                enhanced_prompt = f"You are {character_config.get('name', 'the character')}. {character_config.get('description', '')} Your personality is {character_config.get('personality', 'friendly and helpful')}. Respond in a way that reflects this personality."
            response_messages = [
                {"role": "system", "content": enhanced_prompt},
            ]
            if conversation_history:
                for msg in conversation_history[-5:]:
                    role = "assistant" if msg.get("sender") == "character" else "user"
                    response_messages.append({"role": role, "content": msg.get("content", "")})
            response_messages.append({"role": "user", "content": user_message})
            if self.use_azure_inference:
                try:
                    return await self._call_azure_inference(
                        response_messages,
                        temperature=0.8,
                        max_tokens=300
                    )
                except Exception as e:
                    logging.error(f"Azure Inference LLM2 failed, trying fallback: {str(e)}")
            if self.use_openai:
                try:
                    return await self._call_openai_direct(
                        response_messages,
                        model="gpt-4-turbo-preview",
                        temperature=0.8,
                        max_tokens=300
                    )
                except Exception as e:
                    logging.error(f"OpenAI direct LLM2 failed: {str(e)}")
            logging.info("Using mock response for custom character")
            return f"This is a mock response from {character_config.get('name', 'custom character')} to: {user_message}"
        except Exception as e:
            logging.error(f"Error generating custom character response: {str(e)}")
            return f"I'm sorry, I encountered an error. Please try again later."

    async def analyze_emotion(self, text: str) -> Dict:
        """Analyze emotional content of text using Azure LLama."""
        try:
            messages = [
                {"role": "system", "content": "You are an emotion analysis expert."},
                {"role": "user", "content": f"""
                Analyze the emotional content of this text and provide:
                1. Primary emotion
                2. Emotional intensity (0-1)
                3. Secondary emotions
                4. Sentiment score (-1 to 1)
                
                Text: {text}
                
                Respond in JSON format.
                """}
            ]
            response = await self._call_azure_openai(messages, self.azure_openai_deployment, temperature=1.0)
            return json.loads(response)
        except Exception as e:
            print(f"Error analyzing emotion: {str(e)}")
            raise 