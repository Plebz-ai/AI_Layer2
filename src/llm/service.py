from typing import List, Dict, Optional
import os
from openai import AsyncOpenAI
from dotenv import load_dotenv
import json
import asyncio
from datetime import datetime
import aiohttp

load_dotenv()

class LLMService:
    def __init__(self):
        # Azure LLama configuration
        self.azure_endpoint = os.getenv("AZURE_LLAMA_ENDPOINT")
        self.azure_api_key = os.getenv("AZURE_API_KEY")
        self.azure_model_name = os.getenv("AZURE_MODEL_NAME", "llama-2-13b-chat")
        
        # OpenAI configuration
        self.openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        # Load predefined characters
        self.predefined_characters = self._load_predefined_characters()

    def _load_predefined_characters(self) -> Dict:
        """Load predefined character configurations."""
        try:
            with open("src/llm/data/predefined_characters.json", "r") as f:
                return json.load(f)
        except FileNotFoundError:
            return {}

    async def _call_azure_llama(self, messages: List[Dict]) -> str:
        """Call Azure LLama model."""
        try:
            async with aiohttp.ClientSession() as session:
                headers = {
                    "Content-Type": "application/json",
                    "api-key": self.azure_api_key
                }
                
                payload = {
                    "messages": messages,
                    "temperature": 0.7,
                    "max_tokens": 150,
                    "top_p": 0.95,
                    "frequency_penalty": 0.3,
                    "presence_penalty": 0.6
                }
                
                url = f"{self.azure_endpoint}/openai/deployments/{self.azure_model_name}/chat/completions?api-version=2023-05-15"
                
                async with session.post(url, headers=headers, json=payload) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise Exception(f"Azure LLama API error: {error_text}")
                    
                    result = await response.json()
                    return result["choices"][0]["message"]["content"]
                    
        except Exception as e:
            print(f"Error calling Azure LLama: {str(e)}")
            raise

    async def generate_predefined_response(
        self,
        character_id: str,
        user_message: str,
        conversation_history: List[Dict]
    ) -> str:
        """Generate response for a predefined character using Azure LLama."""
        try:
            character = self.predefined_characters.get(character_id)
            if not character:
                raise ValueError(f"Character {character_id} not found")

            # Prepare conversation context
            messages = [
                {"role": "system", "content": character["system_prompt"]},
                *conversation_history[-5:],  # Keep last 5 messages for context
                {"role": "user", "content": user_message}
            ]

            # Generate response using Azure LLama
            return await self._call_azure_llama(messages)

        except Exception as e:
            print(f"Error generating predefined response: {str(e)}")
            raise

    async def generate_custom_character_response(
        self,
        character_config: Dict,
        user_message: str,
        conversation_history: List[Dict]
    ) -> str:
        """Generate response for a custom character using two-step LLM process."""
        try:
            # Step 1: Generate enhanced prompt using Azure LLama
            prompt_messages = [
                {"role": "system", "content": "You are a prompt engineering expert."},
                {"role": "user", "content": f"""
                Create a detailed prompt for an AI character with the following traits:
                Name: {character_config['name']}
                Description: {character_config['description']}
                Personality: {character_config['personality']}
                Voice Type: {character_config['voice_type']}
                
                The prompt should include:
                1. Core personality traits
                2. Speaking style and mannerisms
                3. Knowledge areas and expertise
                4. Emotional tendencies
                5. Response patterns
                
                Format the prompt to be used with an LLM.
                """}
            ]

            enhanced_prompt = await self._call_azure_llama(prompt_messages)

            # Step 2: Generate character response using OpenAI
            response_messages = [
                {"role": "system", "content": enhanced_prompt},
                *conversation_history[-5:],
                {"role": "user", "content": user_message}
            ]

            response = await self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=response_messages,
                temperature=0.7,
                max_tokens=150,
                presence_penalty=0.6,
                frequency_penalty=0.3
            )

            return response.choices[0].message.content

        except Exception as e:
            print(f"Error generating custom character response: {str(e)}")
            raise

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

            response = await self._call_azure_llama(messages)
            return json.loads(response)

        except Exception as e:
            print(f"Error analyzing emotion: {str(e)}")
            raise 