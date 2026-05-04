"""
evaluation/run_evals.py — RAG Triad Evaluation Pipeline

THE RAG TRIAD (popularized by TruLens / DeepEval):
  ┌─────────────────────┬──────────────────────────────────────────────────┐
  │ Metric              │ Question it answers                              │
  ├─────────────────────┼──────────────────────────────────────────────────┤
  │ Context Relevance   │ Are the retrieved chunks relevant to the query?  │
  │ Faithfulness        │ Is the answer grounded in the retrieved context? │
  │ Answer Relevance    │ Does the answer actually address the query?      │
  └─────────────────────┴──────────────────────────────────────────────────┘

We implement these as LLM-as-judge calls (G-Eval style) — each metric
is scored 0.0–1.0 by a GPT-4o judge on a structured rubric.

ADDITIONAL METRICS:
  - Latency          : end-to-end response time per agent run
  - Chunk Coverage   : fraction of retrieved chunks that contribute to answer
  - Cross-Index Recall : compare HNSW Fast vs Accurate vs Exhaustive KNN

Run:  uv run python -m evaluation.run_evals
      uv run python -m evaluation.run_evals --save-cosmos  (persist to Cosmos DB)
"""

import json
import logging
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

from azure.ai.projects import AIProjectClient
from azure.core.credentials import AzureKeyCredential
from azure.identity import AzureCliCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
logging.getLogger("azure.core").setLevel(logging.WARNING)

# ── Ground-truth evaluation questions ─────────────────────────────────────────
# These are hand-crafted questions with expected answer characteristics.
# A production eval set would have hundreds of these from domain experts.
EVAL_QUESTIONS: list[dict] = [
    {
        "id": "q001",
        "ticker": "MSFT",
        "query": "What was Microsoft's total revenue in the most recent fiscal year?",
        "expected_keywords": ["revenue", "billion", "fiscal year", "cloud"],
    },
    {
        "id": "q002",
        "ticker": "AAPL",
        "query": "What percentage of Apple's revenue comes from iPhone?",
        "expected_keywords": ["iPhone", "revenue", "percent", "%"],
    },
    {
        "id": "q003",
        "ticker": "NVDA",
        "query": "What is NVIDIA's Data Center revenue and its growth rate?",
        "expected_keywords": ["Data Center", "revenue", "growth", "percent"],
    },
    {
        "id": "q004",
        "ticker": "GOOGL",
        "query": "How does Alphabet's Google Cloud revenue compare to AWS and Azure?",
        "expected_keywords": ["Google Cloud", "revenue", "growth"],
    },
    {
        "id": "q005",
        "ticker": None,  # Cross-company comparison
        "query": "Which company has the highest operating margin among MSFT, AAPL, GOOGL, NVDA?",
        "expected_keywords": ["operating margin", "percent"],
    },
]


# ═════════════════════════════════════════════════════════════════════════════
# DATA CLASSES
# ═════════════════════════════════════════════════════════════════════════════


@dataclass
class EvalResult:
    question_id: str
    query: str
    ticker: Optional[str]
    index_name: str
    retrieved_chunks: list[str]
    answer: str
    latency_s: float
    context_relevance: float = 0.0
    faithfulness: float = 0.0
    answer_relevance: float = 0.0
    chunk_coverage: float = 0.0
    error: Optional[str] = None
    metadata: dict = field(default_factory=dict)


@dataclass
class BenchmarkResult:
    """Cross-index recall comparison for one query."""
    query: str
    index_results: dict[str, list[str]]  # {index_name: [chunk_ids]}
    recall_vs_knn: dict[str, float]      # {index_name: recall_score}


# ═════════════════════════════════════════════════════════════════════════════
# RETRIEVAL
# ═════════════════════════════════════════════════════════════════════════════


def vector_search(
    query_embedding: list[float],
    index_name: str,
    ticker: Optional[str],
    top_k: int = 5,
    use_hybrid: bool = False,
    query_text: Optional[str] = None,
) -> list[dict]:
    """
    Run vector (or hybrid) search against a specific index.

    For hybrid indexes, pass use_hybrid=True and provide query_text.
    The search API fuses BM25 and vector scores using Reciprocal Rank Fusion.
    """
    credential = AzureKeyCredential(config.SEARCH_KEY)
    client = SearchClient(
        endpoint=config.SEARCH_ENDPOINT,
        index_name=index_name,
        credential=credential,
    )

    vector_query = VectorizedQuery(
        vector=query_embedding,
        k_nearest_neighbors=top_k,
        fields="content_vector",
    )

    # Build OData filter for ticker scoping
    odata_filter = f"ticker eq '{ticker}'" if ticker else None

    results = client.search(
        search_text=query_text if use_hybrid else None,
        vector_queries=[vector_query],
        filter=odata_filter,
        select=["id", "content", "ticker", "filing_type", "chunk_id"],
        top=top_k,
    )

    return [
        {
            "id": r["id"],
            "content": r["content"],
            "ticker": r.get("ticker"),
            "filing_type": r.get("filing_type"),
            "score": r.get("@search.score", 0.0),
        }
        for r in results
    ]


# ═════════════════════════════════════════════════════════════════════════════
# LLM-AS-JUDGE METRICS (G-Eval style)
# ═════════════════════════════════════════════════════════════════════════════


def _llm_judge(openai_client, prompt: str) -> float:
    """
    Call the LLM judge and extract a 0.0–1.0 score.

    G-Eval uses a step-by-step chain-of-thought rubric followed by a numeric
    score.  We ask for JSON output to make parsing reliable.
    """
    response = openai_client.chat.completions.create(
        model=config.CHAT_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an objective evaluator. Score the provided text "
                    "on a 0.0 to 1.0 scale following the rubric. "
                    "Respond ONLY with valid JSON: {\"score\": <float>, \"reason\": \"<string>\"}"
                ),
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0,
        max_tokens=200,
    )
    raw = response.choices[0].message.content.strip()
    try:
        data = json.loads(raw)
        return float(data.get("score", 0.0))
    except (json.JSONDecodeError, KeyError, TypeError):
        # Fallback: attempt to extract any float in [0,1]
        import re
        matches = re.findall(r"\b0\.\d+\b|\b1\.0\b", raw)
        return float(matches[0]) if matches else 0.5


def score_context_relevance(openai_client, query: str, chunks: list[str]) -> float:
    """
    Context Relevance: Are the retrieved chunks useful for answering the query?

    Score 1.0 = all chunks are directly relevant
    Score 0.0 = chunks are completely off-topic
    """
    context_preview = "\n---\n".join(c[:300] for c in chunks[:5])
    prompt = (
        f"QUERY: {query}\n\n"
        f"RETRIEVED CONTEXT:\n{context_preview}\n\n"
        "RUBRIC:\n"
        "1.0 = All chunks are highly relevant and contain information that directly helps answer the query.\n"
        "0.7 = Most chunks are relevant; minor noise.\n"
        "0.4 = Some chunks are relevant but significant noise is present.\n"
        "0.0 = Chunks are completely irrelevant to the query.\n\n"
        "Score the context relevance."
    )
    return _llm_judge(openai_client, prompt)


def score_faithfulness(openai_client, answer: str, chunks: list[str]) -> float:
    """
    Faithfulness: Is every claim in the answer supported by the retrieved context?

    This detects hallucination — the most dangerous failure mode in RAG.
    Score 1.0 = every factual claim traces back to the context
    Score 0.0 = answer contains fabricated facts not in context
    """
    context_preview = "\n---\n".join(c[:400] for c in chunks[:5])
    prompt = (
        f"ANSWER: {answer[:800]}\n\n"
        f"CONTEXT:\n{context_preview}\n\n"
        "RUBRIC:\n"
        "1.0 = Every factual claim in the answer is directly supported by the context.\n"
        "0.7 = Most claims are supported; 1-2 minor unsupported statements.\n"
        "0.4 = Several claims are not grounded in the context.\n"
        "0.0 = Answer contains fabricated facts not present in the context (hallucination).\n\n"
        "Score faithfulness."
    )
    return _llm_judge(openai_client, prompt)


def score_answer_relevance(openai_client, query: str, answer: str) -> float:
    """
    Answer Relevance: Does the answer actually address what was asked?

    An answer can be faithful yet irrelevant (e.g., retrieved context was
    off-topic).  This metric catches that failure mode.
    """
    prompt = (
        f"QUERY: {query}\n\n"
        f"ANSWER: {answer[:800]}\n\n"
        "RUBRIC:\n"
        "1.0 = Answer directly and completely addresses the query.\n"
        "0.7 = Answer mostly addresses the query with minor gaps.\n"
        "0.4 = Answer partially addresses the query.\n"
        "0.0 = Answer is completely unrelated to the query.\n\n"
        "Score answer relevance."
    )
    return _llm_judge(openai_client, prompt)


def score_chunk_coverage(answer: str, chunks: list[str]) -> float:
    """
    Chunk Coverage: What fraction of retrieved chunks contributed to the answer?

    We use a simple keyword overlap heuristic (no LLM call needed).
    Low coverage means the retriever is returning noise — reduce top_k or
    improve the chunking strategy.
    """
    if not chunks:
        return 0.0

    answer_words = set(answer.lower().split())
    covered = 0
    for chunk in chunks:
        chunk_words = set(chunk.lower().split())
        overlap = len(answer_words & chunk_words)
        if overlap > 5:  # Heuristic threshold
            covered += 1

    return covered / len(chunks)


# ═════════════════════════════════════════════════════════════════════════════
# CROSS-INDEX RECALL BENCHMARK
# ═════════════════════════════════════════════════════════════════════════════


def benchmark_recall(
    query_embedding: list[float],
    query: str,
    ticker: Optional[str],
    top_k: int = 5,
) -> BenchmarkResult:
    """
    Compare retrieval results across all index configs.

    We use Exhaustive KNN as the ground truth (100% recall by definition)
    and compute recall@k for each HNSW variant:
      recall@k = |HNSW results ∩ KNN results| / k
    """
    index_results: dict[str, list[str]] = {}

    for label, index_name in config.ALL_INDEXES.items():
        try:
            use_hybrid = "Hybrid" in label
            results = vector_search(
                query_embedding, index_name, ticker,
                top_k=top_k, use_hybrid=use_hybrid,
                query_text=query if use_hybrid else None,
            )
            index_results[label] = [r["id"] for r in results]
        except Exception as exc:
            log.warning(f"Search failed for {label}: {exc}")
            index_results[label] = []

    # Ground truth from Exhaustive KNN
    ground_truth = set(index_results.get("Exhaustive KNN", []))

    recall_vs_knn: dict[str, float] = {}
    for label, ids in index_results.items():
        if label == "Exhaustive KNN" or not ground_truth:
            recall_vs_knn[label] = 1.0
        else:
            recall_vs_knn[label] = len(set(ids) & ground_truth) / len(ground_truth)

    return BenchmarkResult(
        query=query,
        index_results=index_results,
        recall_vs_knn=recall_vs_knn,
    )


# ═════════════════════════════════════════════════════════════════════════════
# MAIN EVAL LOOP
# ═════════════════════════════════════════════════════════════════════════════


def run_evaluations(save_cosmos: bool = False) -> list[EvalResult]:
    """Run the full RAG Triad eval suite and return all results."""

    credential = AzureCliCredential()
    project_client = AIProjectClient(
        endpoint=config.FOUNDRY_PROJECT_ENDPOINT,
        credential=credential,
    )

    # config.get_openai_client() uses the hub AI Services endpoint (correct path for SDK 2.1.0)
    openai_client = config.get_openai_client()

    results: list[EvalResult] = []

    for question in EVAL_QUESTIONS:
        log.info(f"\nEvaluating: [{question['id']}] {question['query'][:60]}...")

        # Generate query embedding
        emb_response = openai_client.embeddings.create(
            model=config.EMBEDDING_MODEL,
            input=[question["query"]],
        )
        query_embedding = emb_response.data[0].embedding

        # Retrieve from hybrid index (best retrieval quality for evals)
        start = time.perf_counter()
        chunks_raw = vector_search(
            query_embedding,
            config.INDEX_HYBRID,
            question["ticker"],
            top_k=5,
            use_hybrid=True,
            query_text=question["query"],
        )
        retrieval_latency = time.perf_counter() - start

        chunk_texts = [c["content"] for c in chunks_raw]
        context = "\n\n".join(chunk_texts)

        # Generate answer
        gen_start = time.perf_counter()
        answer_response = openai_client.chat.completions.create(
            model=config.CHAT_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a financial analyst. Answer using ONLY the provided context. "
                        "Cite exact figures. If the context doesn't contain the answer, say so."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Context:\n{context}\n\nQuestion: {question['query']}",
                },
            ],
            temperature=0,
            max_tokens=600,
        )
        answer = answer_response.choices[0].message.content
        generation_latency = time.perf_counter() - gen_start

        # Score all three RAG Triad metrics
        ctx_rel = score_context_relevance(openai_client, question["query"], chunk_texts)
        faithful = score_faithfulness(openai_client, answer, chunk_texts)
        ans_rel = score_answer_relevance(openai_client, question["query"], answer)
        coverage = score_chunk_coverage(answer, chunk_texts)

        result = EvalResult(
            question_id=question["id"],
            query=question["query"],
            ticker=question["ticker"],
            index_name=config.INDEX_HYBRID,
            retrieved_chunks=chunk_texts,
            answer=answer,
            latency_s=round(retrieval_latency + generation_latency, 2),
            context_relevance=round(ctx_rel, 3),
            faithfulness=round(faithful, 3),
            answer_relevance=round(ans_rel, 3),
            chunk_coverage=round(coverage, 3),
            metadata={
                "retrieval_latency_s": round(retrieval_latency, 3),
                "generation_latency_s": round(generation_latency, 3),
                "num_chunks": len(chunk_texts),
            },
        )
        results.append(result)

        log.info(
            f"  ctx_rel={ctx_rel:.2f}  faithful={faithful:.2f}  "
            f"ans_rel={ans_rel:.2f}  cov={coverage:.2f}  "
            f"latency={result.latency_s:.1f}s"
        )

    # Run cross-index recall benchmark on first question
    if EVAL_QUESTIONS:
        q = EVAL_QUESTIONS[0]
        emb = openai_client.embeddings.create(
            model=config.EMBEDDING_MODEL, input=[q["query"]]
        ).data[0].embedding
        bench = benchmark_recall(emb, q["query"], q["ticker"])
        log.info("\nCross-Index Recall vs Exhaustive KNN:")
        for label, recall in bench.recall_vs_knn.items():
            log.info(f"  {label:<25s}  recall={recall:.2%}")

    # Persist results
    output_path = Path(config.EVAL_OUTPUT_PATH)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)
    log.info(f"\n✅ Results saved to {output_path}")

    # Optionally persist to Cosmos DB
    if save_cosmos and config.COSMOS_ENDPOINT:
        _save_to_cosmos(results)

    return results


def _save_to_cosmos(results: list[EvalResult]) -> None:
    """Persist eval results to Cosmos DB for the Streamlit dashboard."""
    try:
        from azure.cosmos import CosmosClient, PartitionKey

        client = CosmosClient(config.COSMOS_ENDPOINT, credential=AzureCliCredential())
        db = client.create_database_if_not_exists(config.COSMOS_DATABASE)
        container = db.create_container_if_not_exists(
            id=config.COSMOS_CONTAINER,
            partition_key=PartitionKey(path="/ticker"),
        )
        for r in results:
            doc = asdict(r)
            doc["id"] = r.question_id  # Cosmos requires 'id' field
            container.upsert_item(doc)
        log.info(f"Saved {len(results)} eval results to Cosmos DB")
    except Exception as exc:
        log.error(f"Cosmos DB save failed: {exc}")


def print_summary(results: list[EvalResult]) -> None:
    """Print a formatted summary table."""
    print("\n" + "=" * 80)
    print(f"{'ID':<8} {'Ticker':<8} {'CtxRel':>7} {'Faith':>7} {'AnsRel':>7} {'Cov':>5} {'Lat(s)':>7}")
    print("-" * 80)
    for r in results:
        print(
            f"{r.question_id:<8} {r.ticker or 'ALL':<8} "
            f"{r.context_relevance:>7.2f} {r.faithfulness:>7.2f} "
            f"{r.answer_relevance:>7.2f} {r.chunk_coverage:>5.2f} {r.latency_s:>7.1f}"
        )

    avg_ctx = sum(r.context_relevance for r in results) / len(results)
    avg_faith = sum(r.faithfulness for r in results) / len(results)
    avg_ans = sum(r.answer_relevance for r in results) / len(results)
    print("-" * 80)
    print(f"{'AVG':<8} {'':8} {avg_ctx:>7.2f} {avg_faith:>7.2f} {avg_ans:>7.2f}")
    print("=" * 80)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--save-cosmos", action="store_true", help="Persist results to Cosmos DB")
    args = parser.parse_args()

    results = run_evaluations(save_cosmos=args.save_cosmos)
    print_summary(results)
