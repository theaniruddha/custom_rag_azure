"""
config.py — Central configuration loader for EDGAR Multi-Agent Comparator.

Design principle: every module imports from here, never from os.environ directly.
This gives us a single source of truth, early validation, and easy dev→prod swaps.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Resolve .env relative to this file so the script works from any CWD
_env_path = Path(__file__).parent / ".env"
load_dotenv(_env_path, override=True)


def _require(name: str) -> str:
    """Fail fast with a clear message if a required variable is absent."""
    val = os.getenv(name, "").strip()
    if not val:
        raise EnvironmentError(
            f"Required env var '{name}' is missing or empty.\n"
            f"  → Check your .env file at: {_env_path}"
        )
    return val


def _optional(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


# ── Azure Resource Group ──────────────────────────────────────────────────────
RESOURCE_GROUP: str = _require("RESOURCE_GROUP")
AZURE_SUBSCRIPTION_ID: str = _optional("AZURE_SUBSCRIPTION_ID")

# ── Azure AI Foundry ──────────────────────────────────────────────────────────
# Project endpoint: used for AIProjectClient (agents list, connections, deployments).
# Format: https://<hub-name>.services.ai.azure.com/api/projects/<project-name>
FOUNDRY_PROJECT_ENDPOINT: str = _require("FOUNDRY_PROJECT_ENDPOINT")

# Azure OpenAI endpoint: the hub's AI Services base URL used for chat completions
# and embeddings. Typically: https://<hub-name>.services.ai.azure.com/
# Derived automatically from FOUNDRY_PROJECT_ENDPOINT if not set explicitly.
AZURE_OPENAI_ENDPOINT: str = _optional(
    "AZURE_OPENAI_ENDPOINT",
    # Strip /api/projects/<name> to get the hub base URL
    FOUNDRY_PROJECT_ENDPOINT.split("/api/projects")[0] + "/"
    if "/api/projects" in _optional("FOUNDRY_PROJECT_ENDPOINT", "")
    else "",
)

# ── Azure AI Search ───────────────────────────────────────────────────────────
# Admin key allows index creation; use query key in production read-only paths.
SEARCH_ENDPOINT: str = _require("SEARCH_ENDPOINT")
SEARCH_KEY: str = _require("SEARCH_KEY")

# ── Model deployment names ────────────────────────────────────────────────────
# These must match the deployment names returned by client.deployments.list().
# Verify with: uv run python -c "from azure.ai.projects import AIProjectClient; ..."
CHAT_MODEL: str = _optional("CHAT_MODEL", "gpt-4.1")
EMBEDDING_MODEL: str = _optional("EMBEDDING_MODEL", "text-embedding-3-large")

# Dimensions must match the model: ada-002=1536, text-embedding-3-small=1536,
# text-embedding-3-large=3072. Reducing dimensions saves memory but lowers recall.
EMBEDDING_DIMENSIONS: int = int(_optional("EMBEDDING_DIMENSIONS", "1536"))

# ── Storage & Cosmos DB (deployed by main.bicep) ──────────────────────────────
STORAGE_CONNECTION_STRING: str = _optional("STORAGE_CONNECTION_STRING")
COSMOS_ENDPOINT: str = _optional("COSMOS_ENDPOINT")
COSMOS_DATABASE: str = _optional("COSMOS_DATABASE", "edgar_evals")
COSMOS_CONTAINER: str = _optional("COSMOS_CONTAINER", "eval_results")

# ── EDGAR filing targets ──────────────────────────────────────────────────────
# We pull the latest 10-K (annual) and one 10-Q (quarterly) for each ticker.
TICKERS: list[str] = ["MSFT", "AAPL", "GOOGL", "NVDA"]
FILING_TYPES: list[str] = ["10-K", "10-Q"]
# Fetch up to 3 recent filings per type — covers ~4-5 months of quarterly activity
FILINGS_PER_TYPE: int = 3
# Only pull filings filed on or after this date (YYYY-MM-DD); filters to recent content
FILINGS_AFTER_DATE: str = _optional("FILINGS_AFTER_DATE", "2025-12-01")

# ── Vector index names — one per search config ────────────────────────────────
#
# WHY multiple indexes?
# Azure AI Search indexes are configured at creation time.  To compare HNSW
# parameters you need separate indexes — you cannot change m/efConstruction
# after an index is built (rebuild is required).
#
# HNSW parameter cheat-sheet:
#   m             : bidirectional links per node (range 4–10 in Azure Search)
#                   Higher m → better recall, more RAM
#   efConstruction: beam width during graph build (range 100–1000)
#                   Higher → better graph quality, slower indexing
#   efSearch      : beam width during query (range 100–1000)
#                   Higher → better recall at query time, slower queries
INDEX_HNSW_FAST: str     = _optional("INDEX_HNSW_FAST",     "edgar-hnsw-fast")
INDEX_HNSW_BALANCED: str = _optional("INDEX_HNSW_BALANCED", "edgar-hnsw-balanced")
INDEX_HNSW_ACCURATE: str = _optional("INDEX_HNSW_ACCURATE", "edgar-hnsw-accurate")
INDEX_EXHAUSTIVE_KNN: str = _optional("INDEX_EXHAUSTIVE_KNN", "edgar-exhaustive-knn")
INDEX_HYBRID: str        = _optional("INDEX_HYBRID",        "edgar-hybrid")
INDEX_QUANTIZED: str     = _optional("INDEX_QUANTIZED",     "edgar-quantized")

# Ordered display map used by Streamlit and benchmark logging
ALL_INDEXES: dict[str, str] = {
    "HNSW Fast (m=4)":       INDEX_HNSW_FAST,
    "HNSW Balanced (m=6)":   INDEX_HNSW_BALANCED,
    "HNSW Accurate (m=10)":  INDEX_HNSW_ACCURATE,
    "Exhaustive KNN":         INDEX_EXHAUSTIVE_KNN,
    "Hybrid (HNSW + BM25)":  INDEX_HYBRID,
    "Quantized HNSW":         INDEX_QUANTIZED,
}

# ── Chunking parameters ───────────────────────────────────────────────────────
# Overlap ensures context isn't lost at chunk boundaries.
# Smaller chunks → more precise retrieval but higher index size.
# Larger chunks → more context per result but noisier retrieval.
CHUNK_SIZE: int    = int(_optional("CHUNK_SIZE", "800"))
CHUNK_OVERLAP: int = int(_optional("CHUNK_OVERLAP", "150"))

# ── Observability (optional) ──────────────────────────────────────────────────
APPINSIGHTS_CONN_STR: str = _optional("APPLICATIONINSIGHTS_CONNECTION_STRING")

# ── Evaluation ────────────────────────────────────────────────────────────────
EVAL_OUTPUT_PATH: str = _optional("EVAL_OUTPUT_PATH", "evaluation/results.json")


# ═════════════════════════════════════════════════════════════════════════════
# CLIENT FACTORY
# ═════════════════════════════════════════════════════════════════════════════


def get_openai_client():
    """
    Return an Azure OpenAI client authenticated via Azure CLI credential.

    WHY AzureOpenAI (not project_client.get_openai_client()):
    The Foundry project's /openai/v1/ route does not expose the embeddings
    or chat completions APIs in SDK 2.1.0.  The hub's AI Services endpoint
    (AZURE_OPENAI_ENDPOINT) does — it follows the standard Azure OpenAI
    deployment routing: /openai/deployments/<name>/...

    This function is the single place that constructs the OpenAI client,
    so changing auth or API version only requires editing here.
    """
    import openai
    from azure.identity import AzureCliCredential, get_bearer_token_provider

    token_provider = get_bearer_token_provider(
        AzureCliCredential(),
        "https://cognitiveservices.azure.com/.default",
    )

    return openai.AzureOpenAI(
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        azure_ad_token_provider=token_provider,
        api_version="2024-12-01-preview",
    )
