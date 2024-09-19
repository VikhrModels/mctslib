import requests
from typing import Dict

class VLLMClient:
    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url
        self.api_key = api_key

    def generate(self, prompt: str, model: str, temperature: float = 0.8, max_tokens: int = 100) -> str:
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
        payload = {
            'model': model,
            'prompt': prompt,
            'temperature': temperature,
            'max_tokens': max_tokens
        }
        response = requests.post(f'{self.base_url}/generate', headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
        return data['choices'][0]['text']
