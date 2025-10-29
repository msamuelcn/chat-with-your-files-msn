from pathlib import Path

BASE_DIR = Path(".")
DATA_DIR = BASE_DIR / "data"
INDEX_DIR = DATA_DIR / "indices"
UPLOAD_DIR = DATA_DIR / "uploads"

for d in (DATA_DIR, INDEX_DIR, UPLOAD_DIR):
    d.mkdir(parents=True, exist_ok=True)
