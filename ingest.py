# ingest.py
import logging
from pathlib import Path
from dotenv import load_dotenv
import os

from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex, SimpleField, SearchableField, SearchFieldDataType
)
from sec_edgar_downloader import Downloader

load_dotenv()

logger = logging.getLogger(__name__)

# ===================== LOAD FROM .env =====================
SEARCH_ENDPOINT = os.getenv("SEARCH_ENDPOINT")
SEARCH_KEY = os.getenv("SEARCH_KEY")           # ← This is missing in your .env
INDEX_NAME = os.getenv("INDEX_NAME", "apple-index-v54")

if not SEARCH_ENDPOINT:
    raise ValueError("SEARCH_ENDPOINT is missing in .env")

if not SEARCH_KEY:
    raise ValueError("SEARCH_KEY is missing in .env! Please add your Azure Search Admin Key.")
# ========================================================

def ingest_data():
    """Simplified Ingestion for Azure AI Foundry"""
    logger.info("🚀 Starting Ingestion...")

    data_dir = Path("data")
    data_dir.mkdir(parents=True, exist_ok=True)

    # 1. Download AAPL 10-K 2026
    logger.info("📥 Downloading AAPL 10-K (2026)...")
    dl = Downloader("FoundryAgent", "your.email@example.com", str(data_dir))
    dl.get("10-K", "AAPL", limit=1, download_details=False, after="2026-01-01")

    # 2. Read TXT file
    txt_files = list(data_dir.rglob("*.txt"))
    if not txt_files:
        logger.error("❌ No .txt file downloaded!")
        return

    raw_text = txt_files[0].read_text(encoding="utf-8", errors="ignore")
    clean_text = ' '.join(raw_text.split())

    logger.info(f"✅ Loaded {len(clean_text):,} characters")

    # 3. Index in Azure AI Search
    credential = AzureKeyCredential(SEARCH_KEY)
    index_client = SearchIndexClient(endpoint=SEARCH_ENDPOINT, credential=credential)

    # Create index if not exists
    fields = [
        SimpleField(name="id", type=SearchFieldDataType.String, key=True),
        SearchableField(name="content", type=SearchFieldDataType.String, analyzer_name="en.microsoft"),
        SimpleField(name="year", type=SearchFieldDataType.Int32, filterable=True),
        SimpleField(name="source", type=SearchFieldDataType.String),
    ]

    index = SearchIndex(name=INDEX_NAME, fields=fields)
    index_client.create_or_update_index(index)

    # Upload document
    search_client = SearchClient(endpoint=SEARCH_ENDPOINT, index_name=INDEX_NAME, credential=credential)

    doc = {
        "id": "AAPL-10K-2026",
        "content": clean_text[:15000],
        "year": 2026,
        "source": "AAPL_10K_2026"
    }

    search_client.upload_documents([doc])

    logger.info(f"🎉 Successfully indexed into index: **{INDEX_NAME}**")