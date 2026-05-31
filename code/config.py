"""Configuration module for the support triage agent."""

import os
from pathlib import Path

from dotenv import load_dotenv


def _normalize_anthropic_base_url(url: str) -> str:
    """Normalize Anthropic proxy base URL for the Python SDK.

    The Anthropic SDK appends `/v1/messages` internally, so a configured URL of
    `http://localhost:6655/anthropic/v1` must be normalized to
    `http://localhost:6655/anthropic`.
    """
    normalized = (url or "").rstrip("/")
    if normalized.endswith("/v1"):
        normalized = normalized[:-3]
    return normalized

# Load .env from code/ directory and repo root
load_dotenv(Path(__file__).resolve().parent / ".env")
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

# ─── Paths ───────────────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
SUPPORT_DIR = REPO_ROOT / "support_tickets"
CACHE_DIR = Path(__file__).resolve().parent / ".cache"

CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ─── API Configuration ───────────────────────────────────────────────────────

ANTHROPIC_BASE_URL = _normalize_anthropic_base_url(
    os.getenv("ANTHROPIC_BASE_URL", "http://localhost:6655/anthropic")
)
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY") or os.getenv("HAI_API_KEY") or ""

# ─── HuggingFace Fallback Configuration ──────────────────────────────────────

HF_LLM_MODEL = os.getenv("HF_LLM_MODEL", "Qwen/Qwen2.5-72B-Instruct")

# ─── Model Configuration ─────────────────────────────────────────────────────

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-large-en-v1.5")
EMBEDDING_MODEL_FALLBACKS = ["BAAI/bge-base-en-v1.5", "BAAI/bge-small-en-v1.5"]
LLM_MODEL = "claude-sonnet-4-20250514"

# ─── Chunking Parameters ─────────────────────────────────────────────────────

MAX_CHUNK_TOKENS = 512
MIN_CHUNK_TOKENS = 10
CHUNK_OVERLAP = 50

# ─── Retrieval Parameters ────────────────────────────────────────────────────

BM25_TOP_K = 20
VECTOR_TOP_K = 20
RRF_K = 60
FINAL_TOP_K = 10

# ─── Domain Configuration ────────────────────────────────────────────────────

DOMAINS = ["hackerrank", "claude", "visa"]

SKIP_FILES: set = {"consumer.md", "merchant.md", "checkout-fees-contact-form.md"}

# ─── Regex Patterns ──────────────────────────────────────────────────────────

IMAGE_PATTERN = r"!\[.*?\]\(.*?\)"
HTML_COMMENT_PATTERN = r"<!--.*?-->"  # Use with re.DOTALL

# NOTE: Product area taxonomy lives in prompts.py (PRODUCT_AREA_TAXONOMY).
# It is NOT duplicated here to avoid drift between config and prompt definitions.
