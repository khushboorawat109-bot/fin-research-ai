import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
SEED_DIR = DATA_DIR / "seed"        # put 3-4 annual-report PDFs here
UPLOAD_DIR = DATA_DIR / "uploads"
CHROMA_DIR = DATA_DIR / "chroma"
DB_PATH = DATA_DIR / "app.db"

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
EMBED_MODEL = os.getenv("EMBED_MODEL", "gemini-embedding-001")
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-2.5-flash")

CHUNK_SIZE = 1000      # characters per chunk
CHUNK_OVERLAP = 150
EMBED_BATCH = 50       # free tier friendly

for d in (SEED_DIR, UPLOAD_DIR, CHROMA_DIR):
    d.mkdir(parents=True, exist_ok=True)

REPORT_DIR = DATA_DIR / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)
# Gemini free tier allows only a few requests/minute; this spaces LLM calls out.
LLM_MIN_INTERVAL = float(os.getenv("LLM_MIN_INTERVAL", "6.5"))
