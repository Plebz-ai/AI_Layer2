from typing import List, Dict, Optional
import os
from openai import AsyncOpenAI
from dotenv import load_dotenv
import json
import asyncio
from datetime import datetime

load_dotenv()

class LLMService:
    def __init__(self):
        self.client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.default_model = os.getenv("DEFAULT_MODEL", "gpt-4")
        self.custom_character_model = os.getenv("CUSTOM_CHARACTER_MODEL", "gpt-4")
        self.prompt_generation_model = os.getenv("PROMPT_GENERATION_MODEL", "gpt-4")
        
        # Load predefined characters
        self.predefined_characters = self._load_predefined_characters()

    def _load_predefined_characters(self) -> Dict:
        """Load predefined character configurations."""
        try:
            with open("src/llm/data/predefined_characters.json", "r") as f:
                return json.load(f)
        except FileNotFoundError:
            return {}

    async def generate_predefined_response(
        self,
        character_id: str,
        user_message: str,
        conversation_history: List[Dict]
    ) -> str:
        """Generate response for a predefined character."""
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

            # Generate response
            response = await self.client.chat.completions.create(
                model=self.default_model,
                messages=messages,
                temperature=0.7,
                max_tokens=150,
                presence_penalty=0.6,
                frequency_penalty=0.3
            )

            return response.choices[0].message.content

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
            # Step 1: Generate enhanced prompt
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

            prompt_response = await self.client.chat.completions.create(
                model=self.prompt_generation_model,
                messages=prompt_messages,
                temperature=0.7,
                max_tokens=500
            )

            enhanced_prompt = prompt_response.choices[0].message.content

            # Step 2: Generate character response
            response_messages = [
                {"role": "system", "content": enhanced_prompt},
                *conversation_history[-5:],
                {"role": "user", "content": user_message}
            ]

            character_response = await self.client.chat.completions.create(
                model=self.custom_character_model,
                messages=response_messages,
                temperature=0.7,
                max_tokens=150,
                presence_penalty=0.6,
                frequency_penalty=0.3
            )

            return character_response.choices[0].message.content

        except Exception as e:
            print(f"Error generating custom character response: {str(e)}")
            raise

    async def analyze_emotion(self, text: str) -> Dict:
        """Analyze emotional content of text."""
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

            response = await self.client.chat.completions.create(
                model=self.default_model,
                messages=messages,
                temperature=0.3,
                max_tokens=150
            )

            return json.loads(response.choices[0].message.content)

        except Exception as e:
            print(f"Error analyzing emotion: {str(e)}")
            raise 