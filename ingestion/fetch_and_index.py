"""
ingestion/fetch_and_index.py — EDGAR Download + Multi-Config Vector Indexing

WHAT THIS DOES (end-to-end):
  1. Downloads 10-K and 10-Q filings from SEC EDGAR for MSFT, AAPL, GOOGL, NVDA
  2. Extracts clean text from HTML/TXT filings
  3. Chunks text into overlapping windows with rich metadata
  4. Generates dense vector embeddings via Azure OpenAI
  5. Creates 6 Azure AI Search indexes with different vector configs
  6. Uploads documents to all indexes + logs per-index benchmark timings

VECTOR CONFIG LADDER (Azure AI Search limits: m∈[4,10], ef∈[100,1000]):
  ┌─────────────────────┬────┬──────────────────┬───────────┬────────────────┐
  │ Index               │ m  │ efConstruction   │ efSearch  │ Tradeoff       │
  ├─────────────────────┼────┼──────────────────┼───────────┼────────────────┤
  │ HNSW Fast           │  4 │ 400              │ 500       │ Fastest, ~90%  │
  │ HNSW Balanced       │  6 │ 600              │ 700       │ Sweet spot      │
  │ HNSW Accurate       │ 10 │ 1000             │ 1000      │ Best recall     │
  │ Exhaustive KNN      │  — │  —               │  —        │ Ground truth    │
  │ Hybrid (HNSW+BM25)  │  6 │ 600              │ 700       │ Mixed queries   │
  │ Quantized HNSW      │  6 │ 600              │ 700       │ ~75% less RAM   │
  └─────────────────────┴────┴──────────────────┴───────────┴────────────────┘

Run:  uv run python -m ingestion.fetch_and_index
"""

import csv
import json
import logging
import re
import sys
import time
from pathlib import Path
from typing import Generator

from azure.ai.projects import AIProjectClient
from azure.core.credentials import AzureKeyCredential
from azure.identity import AzureCliCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    ExhaustiveKnnAlgorithmConfiguration,
    ExhaustiveKnnParameters,
    HnswAlgorithmConfiguration,
    HnswParameters,
    ScalarQuantizationCompression,
    SearchField,
    SearchFieldDataType,
    SearchIndex,
    SearchableField,
    SemanticConfiguration,
    SemanticField,
    SemanticPrioritizedFields,
    SemanticSearch,
    SimpleField,
    VectorSearch,
    VectorSearchAlgorithmMetric,
    VectorSearchProfile,
)
from sec_edgar_downloader import Downloader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logging.getLogger("azure.core").setLevel(logging.WARNING)
logging.getLogger("azure.identity").setLevel(logging.WARNING)
log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).parent.parent / "data"
BENCHMARK_CSV = Path(__file__).parent.parent / "evaluation" / "index_benchmark.csv"
EDGAR_USER_AGENT = "edgar-comparator aglearn123@gmail.com"

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 1: EDGAR DOWNLOAD
# ═════════════════════════════════════════════════════════════════════════════


def download_filings() -> list[Path]:
    """
    Pull EDGAR filings for all tickers using sec-edgar-downloader.

    Returns a flat list of all downloaded file paths.
    The downloader saves files under: data/sec-edgar-filings/<TICKER>/<TYPE>/
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    dl = Downloader("EDGARComparator", EDGAR_USER_AGENT, str(DATA_DIR))

    all_files: list[Path] = []

    for ticker in config.TICKERS:
        for filing_type in config.FILING_TYPES:
            log.info(
                f"  ↓  {ticker} {filing_type} "
                f"(limit={config.FILINGS_PER_TYPE}, after={config.FILINGS_AFTER_DATE})"
            )
            try:
                dl.get(
                    filing_type,
                    ticker,
                    limit=config.FILINGS_PER_TYPE,
                    after=config.FILINGS_AFTER_DATE,   # only recent filings
                    download_details=False,
                )
            except Exception as exc:
                log.warning(f"    EDGAR download failed for {ticker} {filing_type}: {exc}")
                continue

            # Collect text/html files for this ticker+type.
            # One accession folder = one filing; we want at most 1 primary file
            # per accession (the largest file is the main document body).
            # Then we take up to FILINGS_PER_TYPE accessions to cover ~4-5 months.
            filing_dir = DATA_DIR / "sec-edgar-filings" / ticker / filing_type
            if filing_dir.exists():
                # Group files by their parent (accession) folder
                accession_dirs = sorted(
                    {p.parent for p in filing_dir.rglob("*.txt")} |
                    {p.parent for p in filing_dir.rglob("*.htm")},
                    reverse=True,   # most recent accession (lexicographically newest) first
                )
                chosen: list[Path] = []
                for acc_dir in accession_dirs[: config.FILINGS_PER_TYPE]:
                    candidates = list(acc_dir.glob("*.txt")) + list(acc_dir.glob("*.htm"))
                    if candidates:
                        # Primary document = largest file in the accession folder
                        candidates.sort(key=lambda p: p.stat().st_size, reverse=True)
                        chosen.append(candidates[0])

                all_files.extend(chosen)
                log.info(f"    → {len(chosen)} file(s) selected (of {len(accession_dirs)} accessions)")

    log.info(f"Total files downloaded/found: {len(all_files)}")
    return all_files


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 2: TEXT EXTRACTION & CHUNKING
# ═════════════════════════════════════════════════════════════════════════════


def extract_text(path: Path) -> str:
    """
    Extract readable text from an EDGAR filing (HTML or plain-text).

    EDGAR filings are frequently SGML-wrapped HTML. We strip tags and
    normalize whitespace to get clean prose suitable for embedding.
    """
    raw = path.read_text(encoding="utf-8", errors="ignore")

    # Strip HTML/SGML tags using a regex (faster than BeautifulSoup for this volume)
    text = re.sub(r"<[^>]+>", " ", raw)

    # Collapse whitespace and remove EDGAR boilerplate artifacts
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"&nbsp;|&amp;|&lt;|&gt;", " ", text)

    return text


# Max characters to extract per filing.
# A full 10-K is ~1M chars → ~1,250 chunks per file → too expensive to embed.
# 60K chars (~75 pages) captures risk factors, MD&A, and key financials —
# the sections most relevant for RAG Q&A.
_MAX_CHARS_PER_FILE = 60_000


def _infer_metadata(path: Path) -> tuple[str, str]:
    """
    Derive ticker and filing_type from the directory structure:
      data/sec-edgar-filings/<TICKER>/<FILING_TYPE>/<accession>/<file>
    """
    parts = path.parts
    try:
        idx = parts.index("sec-edgar-filings")
        ticker = parts[idx + 1].upper()
        filing_type = parts[idx + 2]
    except (ValueError, IndexError):
        ticker = "UNKNOWN"
        filing_type = "UNKNOWN"
    return ticker, filing_type


def chunk_text(
    text: str,
    ticker: str,
    filing_type: str,
    source: str,
) -> Generator[dict, None, None]:
    """
    Sliding-window chunker with metadata.

    Why overlap?  Sentences and key financial figures often straddle chunk
    boundaries.  A 150-character overlap ensures the embedding for each chunk
    captures the full context of boundary content.
    """
    size = config.CHUNK_SIZE
    overlap = config.CHUNK_OVERLAP
    step = size - overlap
    chunk_id = 0

    for start in range(0, len(text), step):
        chunk = text[start : start + size].strip()
        if len(chunk) < 100:  # Skip tiny trailing fragments
            continue

        yield {
            "id": f"{ticker}-{filing_type}-{source}-{chunk_id}",
            "content": chunk,
            "ticker": ticker,
            "filing_type": filing_type,
            "source": source,
            "chunk_id": chunk_id,
            # content_vector is added later after embedding
        }
        chunk_id += 1


def prepare_documents(files: list[Path]) -> list[dict]:
    """Extract, chunk, and return all documents (without embeddings yet)."""
    docs: list[dict] = []
    seen_ids: set[str] = set()  # deduplicate across repeated file entries
    for path in files:
        ticker, filing_type = _infer_metadata(path)
        text = extract_text(path)
        if not text:
            continue
        # Cap text length per file — this keeps chunk count manageable while
        # still covering the most information-dense sections of each filing.
        text = text[:_MAX_CHARS_PER_FILE]
        source = path.stem[:50]  # Use filename as source label
        for chunk in chunk_text(text, ticker, filing_type, source):
            if chunk["id"] not in seen_ids:
                seen_ids.add(chunk["id"])
                docs.append(chunk)

    log.info(f"Total chunks prepared: {len(docs):,} (capped at {_MAX_CHARS_PER_FILE:,} chars/file)")
    return docs


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 3: EMBEDDING GENERATION
# ═════════════════════════════════════════════════════════════════════════════


def get_openai_client(project_client: AIProjectClient):
    """
    Obtain an Azure OpenAI client for this project.

    Routes through config.get_openai_client() which uses the hub's AI Services
    endpoint — this is the path that correctly resolves deployment URLs for
    both embeddings and chat completions in azure-ai-projects SDK 2.1.0.
    """
    return config.get_openai_client()


def embed_batch(openai_client, texts: list[str]) -> list[list[float]]:
    """
    Generate embeddings for a batch of texts.

    Azure OpenAI embedding models accept up to 2048 tokens per string and up
    to 16 strings per batch request (safe limit; actual max varies by deployment).
    """
    response = openai_client.embeddings.create(
        model=config.EMBEDDING_MODEL,
        input=texts,
    )
    return [item.embedding for item in response.data]


def add_embeddings(docs: list[dict], openai_client) -> list[dict]:
    """
    Enrich document dicts with a 'content_vector' field.

    Batch size = 32 (safe limit for text-embedding-3-large payload size).
    We sleep 1s per batch to stay under the Standard-tier RPM limit and
    let the SDK's built-in retry handle any residual 429s automatically.
    """
    BATCH_SIZE = 32
    SLEEP_S = 1.2       # ~50 requests/min well inside Standard 120-capacity limit
    texts = [d["content"] for d in docs]
    all_embeddings: list[list[float]] = []
    total_batches = (len(texts) + BATCH_SIZE - 1) // BATCH_SIZE
    log.info(f"  Embedding {len(texts):,} chunks in {total_batches} batches "
             f"(~{total_batches * SLEEP_S / 60:.1f} min estimated)")

    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="Embedding", total=total_batches):
        batch = texts[i : i + BATCH_SIZE]
        embeddings = embed_batch(openai_client, batch)
        all_embeddings.extend(embeddings)
        time.sleep(SLEEP_S)

    for doc, vec in zip(docs, all_embeddings):
        doc["content_vector"] = vec

    return docs


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 4: INDEX SCHEMA DEFINITIONS
# ═════════════════════════════════════════════════════════════════════════════
#
# All indexes share the same field schema but differ in:
#   - algorithm (HNSW vs ExhaustiveKNN)
#   - HNSW parameters (m, efConstruction, efSearch)
#   - compressions (scalar quantization)
#   - semantic configuration (hybrid search only)
#
# This makes benchmark comparisons apples-to-apples on the same data.


def _base_fields() -> list:
    """Fields shared by all index configurations."""
    return [
        SimpleField(name="id", type=SearchFieldDataType.String, key=True),
        SearchableField(
            name="content",
            type=SearchFieldDataType.String,
            analyzer_name="en.microsoft",  # Language-aware tokenizer for English
        ),
        SimpleField(name="ticker", type=SearchFieldDataType.String, filterable=True, facetable=True),
        SimpleField(name="filing_type", type=SearchFieldDataType.String, filterable=True),
        SimpleField(name="source", type=SearchFieldDataType.String, filterable=True),
        SimpleField(name="chunk_id", type=SearchFieldDataType.Int32, filterable=True, sortable=True),
    ]


def _vector_field(profile_name: str) -> SearchField:
    """Dense vector field wired to a named VectorSearchProfile."""
    return SearchField(
        name="content_vector",
        type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
        searchable=True,
        vector_search_dimensions=config.EMBEDDING_DIMENSIONS,
        vector_search_profile_name=profile_name,
    )


def build_hnsw_index(index_name: str, m: int, ef_construction: int, ef_search: int) -> SearchIndex:
    """
    Pure HNSW vector index.

    HNSW (Hierarchical Navigable Small World) is a graph-based approximate
    nearest-neighbor algorithm. It builds a multi-layer proximity graph where
    each node has at most m bidirectional links.

    Query time: O(log n) — logarithmic in corpus size.
    Space: O(n × m × dims × 4 bytes) for float32 vectors.
    """
    algo_name = f"hnsw-algo-m{m}"
    profile_name = f"hnsw-profile-m{m}"

    vector_search = VectorSearch(
        algorithms=[
            HnswAlgorithmConfiguration(
                name=algo_name,
                parameters=HnswParameters(
                    m=m,
                    ef_construction=ef_construction,
                    ef_search=ef_search,
                    metric=VectorSearchAlgorithmMetric.COSINE,
                ),
            )
        ],
        profiles=[VectorSearchProfile(name=profile_name, algorithm_configuration_name=algo_name)],
    )

    return SearchIndex(
        name=index_name,
        fields=_base_fields() + [_vector_field(profile_name)],
        vector_search=vector_search,
    )


def build_exhaustive_knn_index(index_name: str) -> SearchIndex:
    """
    Brute-force (exact) KNN index — the ground truth baseline.

    ExhaustiveKNN computes the exact distance to every document vector.
    Time: O(n × dims).  Use only for benchmarking recall — too slow for
    production at scale but returns 100% recall by definition.
    """
    vector_search = VectorSearch(
        algorithms=[
            ExhaustiveKnnAlgorithmConfiguration(
                name="eknn-algo",
                parameters=ExhaustiveKnnParameters(
                    metric=VectorSearchAlgorithmMetric.COSINE
                ),
            )
        ],
        profiles=[VectorSearchProfile(name="eknn-profile", algorithm_configuration_name="eknn-algo")],
    )

    return SearchIndex(
        name=index_name,
        fields=_base_fields() + [_vector_field("eknn-profile")],
        vector_search=vector_search,
    )


def build_hybrid_index(index_name: str) -> SearchIndex:
    """
    Hybrid search: dense vector (HNSW) + sparse keyword (BM25), fused via RRF.

    Reciprocal Rank Fusion (RRF) merges ranked lists from both systems without
    needing score normalization.  This handles queries like:
      - "revenue in FY2023" → keyword for "revenue", vector for semantic intent
      - "earnings momentum" → purely semantic, no exact keyword match

    SemanticConfiguration adds a cross-encoder reranker as a 3rd stage.
    """
    vector_search = VectorSearch(
        algorithms=[
            HnswAlgorithmConfiguration(
                name="hybrid-algo",
                parameters=HnswParameters(
                    m=6,
                    ef_construction=600,
                    ef_search=700,
                    metric=VectorSearchAlgorithmMetric.COSINE,
                ),
            )
        ],
        profiles=[VectorSearchProfile(name="hybrid-profile", algorithm_configuration_name="hybrid-algo")],
    )

    # Semantic configuration tells the reranker which fields carry the most signal
    semantic_search = SemanticSearch(
        configurations=[
            SemanticConfiguration(
                name="edgar-semantic",
                prioritized_fields=SemanticPrioritizedFields(
                    content_fields=[SemanticField(field_name="content")],
                    keywords_fields=[SemanticField(field_name="ticker")],
                ),
            )
        ]
    )

    return SearchIndex(
        name=index_name,
        fields=_base_fields() + [_vector_field("hybrid-profile")],
        vector_search=vector_search,
        semantic_search=semantic_search,
    )


def build_quantized_index(index_name: str) -> SearchIndex:
    """
    HNSW with scalar quantization compression.

    Scalar quantization converts float32 vectors (4 bytes/dim) to int8 (1 byte/dim),
    reducing memory footprint by ~75%.  Recall degrades slightly (~1-3%) but
    query latency often improves due to smaller memory bandwidth.

    Azure AI Search supports:
      - Scalar quantization (float32 → int8)
      - Binary quantization (float32 → 1 bit, ~97% compression, more recall loss)
    """
    vector_search = VectorSearch(
        algorithms=[
            HnswAlgorithmConfiguration(
                name="quant-algo",
                parameters=HnswParameters(
                    m=6,
                    ef_construction=600,
                    ef_search=700,
                    metric=VectorSearchAlgorithmMetric.COSINE,
                ),
            )
        ],
        compressions=[
            ScalarQuantizationCompression(compression_name="scalar-sq8")
        ],
        profiles=[
            VectorSearchProfile(
                name="quant-profile",
                algorithm_configuration_name="quant-algo",
                compression_name="scalar-sq8",
            )
        ],
    )

    return SearchIndex(
        name=index_name,
        fields=_base_fields() + [_vector_field("quant-profile")],
        vector_search=vector_search,
    )


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 5: INDEX CREATION & DOCUMENT UPLOAD WITH BENCHMARKING
# ═════════════════════════════════════════════════════════════════════════════


def create_or_update_index(idx_client: SearchIndexClient, index: SearchIndex) -> None:
    """Idempotent index creation — safe to re-run if index already exists."""
    try:
        idx_client.create_or_update_index(index)
        log.info(f"  ✓  Index ready: {index.name}")
    except Exception as exc:
        log.error(f"  ✗  Index creation failed ({index.name}): {exc}")
        raise


def upload_to_index(
    index_name: str,
    docs: list[dict],
    credential: AzureKeyCredential,
) -> float:
    """
    Upload all documents to an index and return elapsed seconds.

    Returns upload time for benchmark logging.
    Documents are uploaded in batches of 1000 (Search SDK default).
    """
    client = SearchClient(
        endpoint=config.SEARCH_ENDPOINT,
        index_name=index_name,
        credential=credential,
    )

    start = time.perf_counter()
    BATCH = 500
    for i in range(0, len(docs), BATCH):
        batch = docs[i : i + BATCH]
        result = client.upload_documents(documents=batch)
        failed = [r for r in result if not r.succeeded]
        if failed:
            log.warning(f"    {len(failed)} docs failed in batch starting at {i}")
    elapsed = time.perf_counter() - start
    log.info(f"    Uploaded {len(docs):,} docs to '{index_name}' in {elapsed:.1f}s")
    return elapsed


def log_benchmark(index_name: str, label: str, upload_s: float, doc_count: int) -> None:
    """
    Append a benchmark row to CSV for later analysis in the Streamlit app.

    Columns: index_name, label, doc_count, upload_seconds, docs_per_second
    """
    BENCHMARK_CSV.parent.mkdir(parents=True, exist_ok=True)
    write_header = not BENCHMARK_CSV.exists()

    with open(BENCHMARK_CSV, "a", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["index_name", "label", "doc_count", "upload_seconds", "docs_per_second"]
        )
        if write_header:
            writer.writeheader()
        writer.writerow(
            {
                "index_name": index_name,
                "label": label,
                "doc_count": doc_count,
                "upload_seconds": round(upload_s, 2),
                "docs_per_second": round(doc_count / upload_s, 1) if upload_s > 0 else 0,
            }
        )


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 6: ORCHESTRATION
# ═════════════════════════════════════════════════════════════════════════════


def build_all_indexes(idx_client: SearchIndexClient) -> None:
    """Create (or update) all 6 index configurations."""
    log.info("Building index schemas...")

    configs = [
        (config.INDEX_HNSW_FAST,     build_hnsw_index(config.INDEX_HNSW_FAST,     m=4,  ef_construction=400,  ef_search=500)),
        (config.INDEX_HNSW_BALANCED, build_hnsw_index(config.INDEX_HNSW_BALANCED,  m=6,  ef_construction=600,  ef_search=700)),
        (config.INDEX_HNSW_ACCURATE, build_hnsw_index(config.INDEX_HNSW_ACCURATE,  m=10, ef_construction=1000, ef_search=1000)),
        (config.INDEX_EXHAUSTIVE_KNN, build_exhaustive_knn_index(config.INDEX_EXHAUSTIVE_KNN)),
        (config.INDEX_HYBRID,        build_hybrid_index(config.INDEX_HYBRID)),
        (config.INDEX_QUANTIZED,     build_quantized_index(config.INDEX_QUANTIZED)),
    ]

    for _, index in configs:
        create_or_update_index(idx_client, index)


def main() -> None:
    log.info("═" * 60)
    log.info("EDGAR Multi-Agent Comparator — Ingestion Pipeline")
    log.info("═" * 60)

    # Step 1: Auth — use Azure CLI credential (login already done)
    credential_azure = AzureCliCredential()
    search_credential = AzureKeyCredential(config.SEARCH_KEY)

    project_client = AIProjectClient(
        endpoint=config.FOUNDRY_PROJECT_ENDPOINT,
        credential=credential_azure,
    )
    idx_client = SearchIndexClient(
        endpoint=config.SEARCH_ENDPOINT,
        credential=search_credential,
    )

    # Step 2: Download filings
    log.info("\n[1/5] Downloading EDGAR filings...")
    files = download_filings()
    if not files:
        log.error("No filings found — cannot proceed.")
        sys.exit(1)

    # Step 3: Extract and chunk
    log.info("\n[2/5] Extracting text and chunking...")
    docs = prepare_documents(files)

    # Step 4: Embed (this takes a few minutes for thousands of chunks)
    log.info("\n[3/5] Generating embeddings...")
    openai_client = get_openai_client(project_client)
    docs = add_embeddings(docs, openai_client)

    # Step 5: Create all index schemas
    log.info("\n[4/5] Creating index configurations...")
    build_all_indexes(idx_client)

    # Step 6: Upload to every index + benchmark
    log.info("\n[5/5] Uploading documents to all indexes...")
    for label, index_name in config.ALL_INDEXES.items():
        elapsed = upload_to_index(index_name, docs, search_credential)
        log_benchmark(index_name, label, elapsed, len(docs))

    # Persist a sample of docs for offline inspection
    sample_path = DATA_DIR / "sample_docs.json"
    with open(sample_path, "w") as f:
        json.dump(
            [{k: v for k, v in d.items() if k != "content_vector"} for d in docs[:20]],
            f, indent=2,
        )

    log.info("\n✅ Ingestion complete!")
    log.info(f"   Chunks indexed:  {len(docs):,}")
    log.info(f"   Indexes created: {len(config.ALL_INDEXES)}")
    log.info(f"   Benchmark log:   {BENCHMARK_CSV}")
    log.info(f"   Sample docs:     {sample_path}")


if __name__ == "__main__":
    main()
