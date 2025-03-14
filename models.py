# models.py

import os
from dotenv import load_dotenv
import google.generativeai as genai
from mistralai import Mistral
import ollama

# Load environment variables from your .env file
load_dotenv()

########################################
# GEMINI SETUP
########################################

# Configure the Gemini API using your GOOGLE_API_KEY
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Load the Gemini model (adjust if you have a different variant)
gemini_model = genai.GenerativeModel("gemini-2.0-flash")
gemini_chat = gemini_model.start_chat(history=[])

def get_gemini_response(prompt: str) -> str:
    """
    Sends the prompt to Gemini and returns the full response
    by concatenating streaming chunks.
    """
    response = gemini_chat.send_message(prompt, stream=True)
    full_response = ""
    for chunk in response:
        full_response += chunk.text
    return full_response

########################################
# MISTRAL SETUP
########################################

mistral_api_key = os.getenv("MISTRAL_API_KEY")
mistral_model_name = "mistral-large-latest"

# Initialize Mistral client
mistral_client = Mistral(api_key=mistral_api_key)

def get_mistral_response(prompt: str) -> str:
    """
    Sends the prompt to Mistral and returns the response.
    In this simple example, only the current prompt is sent.
    """
    response = mistral_client.chat.complete(
        model=mistral_model_name,
        messages=[{"role": "user", "content": prompt}],
        stream=False  # Set to True to handle streaming responses
    )
    return response.choices[0].message.content

########################################
# DEEPSEEK SETUP (via Ollama)
########################################

def get_deepseek_response(prompt: str) -> str:
    """
    Sends the prompt to Deepseek (accessed locally via Ollama)
    and returns the response.
    """
    response = ollama.generate(
        model='deepseek-r1:7b',
        prompt=prompt,
        stream=False
    )
    return response['response']
