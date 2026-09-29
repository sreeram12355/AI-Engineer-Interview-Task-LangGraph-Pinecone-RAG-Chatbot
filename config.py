"""Central configuration, driven by environment variables (.env supported)."""
import os
from dotenv import load_dotenv

load_dotenv()


def _f(name, default):
    return float(os.getenv(name, default))


def _i(name, default):
    return int(os.getenv(name, default))


PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "agentic-ai-ebook")
PINECONE_NAMESPACE = os.getenv("PINECONE_NAMESPACE", "ebook")
PINECONE_CLOUD = os.getenv("PINECONE_CLOUD", "aws")
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
EMBEDDING_DIM = _i("EMBEDDING_DIM", 1536)
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")

CHUNK_SIZE = _i("CHUNK_SIZE", 800)
CHUNK_OVERLAP = _i("CHUNK_OVERLAP", 100)

TOP_K = _i("TOP_K", 5)
MIN_RELEVANCE = _f("MIN_RELEVANCE", 0.25)          # cosine sim cut-off for retrieval
GROUNDING_THRESHOLD = _f("GROUNDING_THRESHOLD", 0.7)  # below -> regenerate
MIN_GROUNDING = _f("MIN_GROUNDING", 0.5)            # below after retries -> refuse
MAX_GENERATION_ATTEMPTS = _i("MAX_GENERATION_ATTEMPTS", 2)

PDF_URL = os.getenv("PDF_URL", "")
DEFAULT_PDF_PATH = os.path.join("data", "Ebook-Agentic-AI.pdf")
