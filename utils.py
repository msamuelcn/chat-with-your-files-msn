# utils.py

from pathlib import Path

# Use more descriptive names for directories
BASE_DIRECTORY = Path(".")
DATA_DIRECTORY = BASE_DIRECTORY / "data"
INDEX_DIRECTORY = DATA_DIRECTORY / "indices"
UPLOAD_DIRECTORY = DATA_DIRECTORY / "uploads"

# Create directories if they don't exist
for d in (DATA_DIRECTORY, INDEX_DIRECTORY, UPLOAD_DIRECTORY):
    d.mkdir(parents=True, exist_ok=True)
