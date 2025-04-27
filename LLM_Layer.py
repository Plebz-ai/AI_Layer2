import os
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential

def get_llm_response(transcript):
    """
    Sends the transcript to the Azure AI Foundry Llama model and retrieves the response.
    """
    endpoint = "https://finetuned-model-wmoqzjog.eastus2.models.ai.azure.com"
    model_name = "Llama-4-Maverick-17B-128E-Instruct-FP8"
    api_key = os.getenv("AZURE_API_KEY")

    client = ChatCompletionsClient(
        endpoint=endpoint,
        credential=AzureKeyCredential(api_key),
    )

    response = client.complete(
        messages=[
            SystemMessage(content="You are a helpful assistant."),
            UserMessage(content=transcript)
        ],
        max_tokens=2048,
        temperature=0.8,
        top_p=0.1,
        presence_penalty=0.0,
        frequency_penalty=0.0,
        model=model_name
    )

    return response.choices[0].message.content