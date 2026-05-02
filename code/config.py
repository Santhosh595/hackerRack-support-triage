"""Central configuration for the triage agent."""

from __future__ import annotations

import os
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
SUPPORT_TICKETS_DIR = BASE_DIR / "support_tickets"

DOMAIN_TO_CORPUS = {
    "HackerRank": DATA_DIR / "hackerrank",
    "Claude": DATA_DIR / "claude",
    "Visa": DATA_DIR / "visa",
}

# Security-first threshold: uncertain retrieval escalates.
# Note: TF-IDF scores are naturally lower with a large corpus (800+ docs).
# Threshold of 0.07 gives good signal/noise with the full orchestrate corpus.
RETRIEVAL_CONFIDENCE_THRESHOLD = float(os.getenv("RETRIEVAL_CONFIDENCE_THRESHOLD", "0.07"))
RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "3"))

LOG_DIR = BASE_DIR / "logs"
LOG_PATH = LOG_DIR / "triage_structured.jsonl"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_CSV = SUPPORT_TICKETS_DIR / "output.csv"

# Evaluator-compatible taxonomy (must remain stable).
VALID_DOMAINS = ("HackerRank", "Claude", "Visa", "None")
VALID_REQUEST_TYPES = ("product_issue", "feature_request", "bug", "invalid")

# Optional LLM mode with multi-provider support.
# Auto-detects provider from available API keys (priority order):
# 1. OPENAI_API_KEY (OpenAI: GPT-4, GPT-4o, etc.)
# 2. ANTHROPIC_API_KEY (Anthropic: Claude)
# 3. GOOGLE_API_KEY (Google: Gemini)
# 4. AZURE_OPENAI_API_KEY (Azure OpenAI)
# 5. TRIAGE_API_KEY + TRIAGE_LLM_URL (Legacy: any OpenAI-compatible endpoint)

# For backwards compatibility, we also check TRIAGE_API_KEY
TRIAGE_API_KEY = os.environ.get("TRIAGE_API_KEY", "").strip()

# Detect if any LLM provider is configured
_PROVIDER_KEYS = {
    "openai": os.environ.get("OPENAI_API_KEY", "").strip(),
    "anthropic": os.environ.get("ANTHROPIC_API_KEY", "").strip(),
    "google": os.environ.get("GOOGLE_API_KEY", "").strip(),
    "azure": os.environ.get("AZURE_OPENAI_API_KEY", "").strip(),
    "custom": os.environ.get("TRIAGE_LLM_URL", "").strip() and TRIAGE_API_KEY,
}

USE_LLM = any(_PROVIDER_KEYS.values())
TRIAGE_LLM_URL = os.environ.get("TRIAGE_LLM_URL", "").strip()
