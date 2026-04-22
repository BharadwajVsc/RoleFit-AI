import requests


class LLMClient:
    def __init__(self):
        self.model = "deepseek-r1:latest"
        # self.model = "gemma:latest "  # or "mistral", "phi3"
        self.base_url = "http://localhost:11434/api/generate"

    def generate(self, prompt: str):
        payload = {"model": self.model, "prompt": prompt, "stream": False}

        response = requests.post(self.base_url, json=payload)

        if response.status_code != 200:
            raise Exception(f"Ollama error: {response.text}")

        data = response.json()
        return data["response"]
