import os
from dotenv import load_dotenv
from google import genai

load_dotenv()


class LLMClient:
    def __init__(self):
        # Uses GOOGLE_API_KEY from environment automatically
        self.client = genai.Client()
        self.model = "gemini-2.5-flash"

    def generate(self, prompt: str):
        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
        )

        return response.text
