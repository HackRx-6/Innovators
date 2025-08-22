# config.py
import os
import warnings
from dotenv import load_dotenv
import aiohttp

# CONFIGURATION 
load_dotenv()
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")

# QDRANT CREDENTIALS
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

# EMBEDDINGS
from langchain_openai import AzureOpenAIEmbeddings

AZURE_OPENAI_EMB_ENDPOINT = os.getenv("AZURE_OPENAI_EMB_ENDPOINT")
AZURE_OPENAI_EMB_API_KEY = os.getenv("AZURE_OPENAI_EMB_API_KEY")

EMBEDDER = AzureOpenAIEmbeddings(
    azure_deployment="text-embedding-ada-002",
    openai_api_version="2023-05-15",
    azure_endpoint=AZURE_OPENAI_EMB_ENDPOINT,
    api_key=AZURE_OPENAI_EMB_API_KEY,
)

class CustomLLMResponse:
    def _init_(self, content: str):
        self.content = content

async def call_custom_llm(messages: list, model: str = "gpt-5-nano") -> CustomLLMResponse:
    url = "https://register.hackrx.in/llm/openai"
    headers = {
        'Content-Type': 'application/json',
        'x-subscription-key': 'sk-spgw-api01-b75c065b5d321f07d09e035766fb9330'
    }
    payload = {
        "messages": messages,
        "model": model
    }
    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(url, headers=headers, json=payload) as response:
                response.raise_for_status()
                result = await response.json()
                if 'choices' in result and len(result['choices']) > 0:
                    content = result['choices'][0]['message']['content']
                elif 'content' in result:
                    content = result['content']
                else:
                    content = str(result)
                return CustomLLMResponse(content)
        except Exception as e:
            print(f"Error calling custom LLM API: {e}")
            return CustomLLMResponse(f"Error: Failed to get response from LLM API - {e}")

def format_messages_for_api(prompt: str, system_message: str = None) -> list:
    messages = []
    if system_message:
        messages.append({"role": "system", "content": system_message})
    messages.append({"role": "user", "content": prompt})
    return messages