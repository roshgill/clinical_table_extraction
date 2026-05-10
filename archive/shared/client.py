"""Gemini client setup — shared across all agents."""

import os

from dotenv import load_dotenv
from google import genai

load_dotenv(override=True)

api_key = os.environ.get("GEMINI_API_KEY", "")
if not api_key or api_key == "your-api-key-here":
    raise RuntimeError("GEMINI_API_KEY not set — update your .env file")

client = genai.Client()

DEFAULT_MODEL = "gemini-3-flash-preview"
PRO_MODEL = "gemini-3.1-pro-preview"
