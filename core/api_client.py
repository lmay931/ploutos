# Finnhub and Gemini clients, shared across the scripts
import finnhub
from google import genai

from core.secrets import FINNHUB_API_KEY, GEMINI_API_KEY

finnhub_client = finnhub.Client(api_key=FINNHUB_API_KEY)

gemini_client = genai.Client(api_key=GEMINI_API_KEY)

print("API Clients Initialized.")