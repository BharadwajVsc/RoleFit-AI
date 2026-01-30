import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()


class LLMClient:
    def __init__(self):# Initialize api keys, endpoints, etc.
        api_key= os.getenv('GEMINI_API_KEY')
        
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables.")
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.5-flash')
        pass

    def generate(self, prompt):
        response = self.model.generate_content(prompt)
        """Replace with actual LLM call logic"""
        return response.text # example static response
