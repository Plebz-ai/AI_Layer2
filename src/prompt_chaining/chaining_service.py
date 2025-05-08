class PromptChainingService:
    def __init__(self, llama_client):
        self.llama_client = llama_client

    def generate_enriched_prompt(self, character_data):
        prompt = f"You are an expert AI character designer. Create a detailed system prompt for the following character:\n\n"
        prompt += f"Name: {character_data['name']}\n"
        prompt += f"Description: {character_data['description']}\n"
        prompt += f"Personality: {character_data['personality']}\n"
        messages = [{"role": "user", "content": prompt}]
        return self.llama_client.generate_response(messages)

    def generate_response(self, enriched_prompt, user_input):
        messages = [
            {"role": "system", "content": enriched_prompt},
            {"role": "user", "content": user_input}
        ]
        return self.llama_client.generate_response(messages)