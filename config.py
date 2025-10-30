# config.py

import os
import streamlit
from dotenv import load_dotenv

load_dotenv(override=True)

# --- General Configuration ---
# Fallback chunking settings (used instead of LLM-guided analysis for speed)
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200

# Cleanup time
HALF_DAY_IN_SECONDS = (
    0.5 * 24 * 60 * 60
)  # Retained UPPER_SNAKE_CASE for config constant

# Robust fallback logic:
# 1. Try to get from os.getenv (local .env file or environment variable).
# 2. If that fails, fall back to st.secrets["OPENAI_API_KEY"] (Streamlit Cloud).
# 3. If that also fails, set it to None (or raise an error)
try:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or streamlit.secrets["OPENAI_API_KEY"]
except (AttributeError, KeyError):
    # AttributeError if st is not initialized (e.g., in a non-streamlit context)
    # KeyError if 'OPENAI_API_KEY' is missing from secrets.toml
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Check if key is still missing after all attempts (Good practice)
if not OPENAI_API_KEY:
    # Set a placeholder or raise a more descriptive error in a production version
    print("Warning: OPENAI_API_KEY not found in .env or st.secrets.")

# Backend Choices: "openai" or "huggingface"
EMBEDDINGS_BACKEND: str = "openai"  # Retained UPPER_SNAKE_CASE for config constant

# Model Names
OPENAI_EMBEDDING_MODEL = (
    "text-embedding-3-small"  # Retained UPPER_SNAKE_CASE for config constant
)
HUGGINGFACE_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # Retained UPPER_SNAKE_CASE for config constant
LLM_MODEL_NAME = "gpt-4o-mini"  # Retained UPPER_SNAKE_CASE for config constant
