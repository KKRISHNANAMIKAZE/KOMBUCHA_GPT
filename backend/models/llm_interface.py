from groq import Groq
import os

class LLMInterface:

    def __init__(self):
        self.client = Groq(
            api_key = os.getenv("GROQ_API_KEY")
        )

    def generate(self, prompt, temperature=0.3):

        response = self.client.chat.completions.create(
            model="llama-3.1-8b-instant",   # 🔥 fast + free
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=temperature
        )

        return response.choices[0].message.content