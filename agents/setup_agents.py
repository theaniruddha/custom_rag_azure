"""
agents/setup_agents.py — Multi-Agent Orchestration (Chat Completions + RAG)

ARCHITECTURE:
  This module implements multi-agent orchestration using direct chat completions
  and Azure AI Search retrieval — the most reliable and educational pattern for
  the azure-ai-projects 2.1.0 SDK.

  WHY NOT the Assistants API?
  The Foundry project endpoint exposes chat completions and embeddings but not
  the OpenAI Assistants API (returns 404). The pattern used here is equivalent
  in capability and actually more transparent: each "agent" is a function that
  (1) retrieves context from Search, (2) calls chat completions with that context.

  MULTI-AGENT PATTERN:
  ┌──────────────────────────────────────────────────────────────────────┐
  │  User Query                                                          │
  │      │                                                               │
  │      ▼                                                               │
  │  route_query()  →  LLM decides which tickers are relevant            │
  │      │                                                               │
  │      ▼                                                               │
  │  run_parallel()  →  ThreadPoolExecutor fans out to company agents    │
  │      │                 Each agent: Search(ticker filter) → LLM       │
  │      ▼                                                               │
  │  synthesize()  →  LLM combines all company responses                 │
  └──────────────────────────────────────────────────────────────────────┘

Run:
  uv run python -m agents.setup_agents --list
  uv run python -m agents.setup_agents --query "Compare NVDA and MSFT R&D spend"
"""

import argparse
import json
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

from azure.ai.projects import AIProjectClient
from azure.core.credentials import AzureKeyCredential
from azure.identity import AzureCliCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery

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

# ── Company personas ──────────────────────────────────────────────────────────
COMPANY_PERSONAS: dict[str, str] = {
    "MSFT": (
        "You are a senior equity research analyst specializing in Microsoft Corporation. "
        "Your expertise covers Azure cloud revenue, Office 365 commercial growth, "
        "LinkedIn, Xbox / Activision gaming, and AI infrastructure investments (Copilot, OpenAI partnership). "
        "Always cite specific revenue segments, YoY growth rates, and operating margin trends. "
        "Provide exact figures from the filings. If a figure is not in the context, say so."
    ),
    "AAPL": (
        "You are a senior equity research analyst specializing in Apple Inc. "
        "Your expertise covers iPhone revenue cycles, Services segment (App Store, iCloud, Apple TV+, Apple Pay), "
        "Mac, iPad, Wearables/Home/Accessories, and supply chain dynamics. "
        "Always cite segment revenue, gross margin by segment, and free cash flow data."
    ),
    "GOOGL": (
        "You are a senior equity research analyst specializing in Alphabet Inc. (Google). "
        "Your expertise covers Google Search revenue, YouTube advertising, Google Cloud, "
        "Other Bets (Waymo, DeepMind), and the advertising market cycle. "
        "Always cite revenue by segment, operating margins, and headcount trends."
    ),
    "NVDA": (
        "You are a senior equity research analyst specializing in NVIDIA Corporation. "
        "Your expertise covers Data Center (H100/A100/Blackwell GPU families), Gaming, "
        "Automotive (DRIVE platform), and Professional Visualization. "
        "Always cite Data Center revenue share, gross margins, supply constraints, and hyperscaler demand."
    ),
}

ORCHESTRATOR_SYSTEM = (
    "You are a chief investment strategist who synthesizes analysis from specialist analysts "
    "covering Microsoft, Apple, Alphabet, and NVIDIA. "
    "When given analyst responses, produce a structured comparative answer with:\n"
    "1. A concise table of key metrics (revenue, margin, growth) with exact numbers\n"
    "2. Relative strengths and risks per company\n"
    "3. An investment thesis summary\n"
    "Be precise — cite exact numbers from the analyst inputs. Do not add facts not present."
)


# ═════════════════════════════════════════════════════════════════════════════
# RETRIEVAL
# ═════════════════════════════════════════════════════════════════════════════


def get_embedding(openai_client, text: str) -> list[float]:
    """Generate a single embedding vector for the given text."""
    response = openai_client.embeddings.create(
        model=config.EMBEDDING_MODEL,
        input=[text],
    )
    return response.data[0].embedding


def retrieve_context(
    query_embedding: list[float],
    ticker: str,
    query_text: str,
    top_k: int = 5,
) -> list[dict]:
    """
    Retrieve relevant chunks from the hybrid index for a specific company.

    Uses hybrid search (vector + BM25) with an OData filter to scope
    results to the requested ticker. This is the retrieval step of RAG.
    """
    credential = AzureKeyCredential(config.SEARCH_KEY)
    client = SearchClient(
        endpoint=config.SEARCH_ENDPOINT,
        index_name=config.INDEX_HYBRID,
        credential=credential,
    )

    vector_query = VectorizedQuery(
        vector=query_embedding,
        k_nearest_neighbors=top_k,
        fields="content_vector",
    )

    results = client.search(
        search_text=query_text,       # BM25 keyword component
        vector_queries=[vector_query], # Dense vector component (fused via RRF)
        filter=f"ticker eq '{ticker}'",
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
# COMPANY AGENTS
# ═════════════════════════════════════════════════════════════════════════════


def run_company_agent(
    ticker: str,
    query: str,
    query_embedding: list[float],
    openai_client,
    top_k: int = 5,
) -> dict:
    """
    Run a single company analyst agent.

    WHAT HAPPENS:
      1. Retrieve top_k chunks from the hybrid index filtered to this ticker
      2. Format chunks as a grounded context block
      3. Call the LLM with the company persona + retrieved context
      4. Return the response with metadata

    This is stateless — each call is independent and thread-safe.
    """
    t_start = time.perf_counter()

    # Step 1: Retrieve
    try:
        chunks = retrieve_context(query_embedding, ticker, query, top_k=top_k)
    except Exception as exc:
        log.warning(f"  [{ticker}] Retrieval failed: {exc}")
        chunks = []

    retrieval_latency = time.perf_counter() - t_start

    # Format retrieved context — clearly labeled so the LLM can cite sources
    if chunks:
        context_block = "\n\n---\n\n".join(
            f"[{i+1}] {c['filing_type']} | score={c['score']:.3f}\n{c['content']}"
            for i, c in enumerate(chunks)
        )
    else:
        context_block = "No relevant filings found in the index for this query."

    # Step 2: Generate answer with company persona
    messages = [
        {
            "role": "system",
            "content": (
                f"{COMPANY_PERSONAS[ticker]}\n\n"
                "You MUST answer using ONLY information present in the CONTEXT below. "
                "Cite exact figures. If data is unavailable in context, state that explicitly."
            ),
        },
        {
            "role": "user",
            "content": (
                f"QUESTION: {query}\n\n"
                f"CONTEXT FROM {ticker} FILINGS:\n{context_block}"
            ),
        },
    ]

    try:
        gen_start = time.perf_counter()
        response = openai_client.chat.completions.create(
            model=config.CHAT_MODEL,
            messages=messages,
            temperature=0,         # Deterministic for financial analysis
            max_tokens=800,
        )
        answer = response.choices[0].message.content
        gen_latency = time.perf_counter() - gen_start
    except Exception as exc:
        log.error(f"  [{ticker}] LLM call failed: {exc}")
        answer = f"Error generating analysis for {ticker}: {exc}"
        gen_latency = 0.0

    return {
        "ticker": ticker,
        "response": answer,
        "chunks": chunks,
        "metadata": {
            "retrieval_latency_s": round(retrieval_latency, 3),
            "generation_latency_s": round(gen_latency, 3),
            "chunks_retrieved": len(chunks),
        },
    }


# ═════════════════════════════════════════════════════════════════════════════
# ORCHESTRATOR
# ═════════════════════════════════════════════════════════════════════════════


class AppOrchestrator:
    """
    Routes user queries to company agents and synthesizes comparative answers.

    Design:
    - Stateless per query (no shared mutable state between calls)
    - Thread-safe (company agents run in parallel via ThreadPoolExecutor)
    - Embedding computed once and reused across all company agents
    """

    def __init__(self, openai_client):
        self._oai = openai_client

    def route_query(self, query: str) -> list[str]:
        """
        Use the LLM to determine which companies are relevant to the query.

        Returns a subset of config.TICKERS. Falls back to all tickers on parse error.

        WHY A SEPARATE ROUTING CALL:
        Routing upfront avoids running all 4 agents when only 1-2 are relevant,
        cutting latency by 50-75% for targeted questions.
        """
        response = self._oai.chat.completions.create(
            model=config.CHAT_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a financial query router. Given a user question, "
                        "determine which of [MSFT, AAPL, GOOGL, NVDA] are relevant. "
                        "Reply ONLY with a JSON array, e.g. [\"MSFT\", \"NVDA\"]. "
                        "If the question is general or comparative across all companies, "
                        "return all four."
                    ),
                },
                {"role": "user", "content": query},
            ],
            temperature=0,
            max_tokens=50,
        )
        raw = response.choices[0].message.content.strip()
        try:
            start = raw.index("[")
            end = raw.rindex("]") + 1
            tickers = json.loads(raw[start:end])
            valid = [t for t in tickers if t in config.TICKERS]
            return valid if valid else config.TICKERS
        except (ValueError, json.JSONDecodeError):
            return config.TICKERS

    def run_parallel(
        self,
        tickers: list[str],
        query: str,
        query_embedding: list[float],
    ) -> list[dict]:
        """
        Fan out to company agents concurrently.

        The Search API and LLM calls are I/O-bound, so ThreadPoolExecutor
        gives near-linear speedup: 4 agents in ~1x latency instead of 4x.
        """
        results: list[dict] = []

        with ThreadPoolExecutor(max_workers=len(tickers)) as pool:
            futures = {
                pool.submit(run_company_agent, t, query, query_embedding, self._oai): t
                for t in tickers
            }
            for future in as_completed(futures):
                ticker = futures[future]
                try:
                    results.append(future.result())
                except Exception as exc:
                    log.error(f"Agent failed for {ticker}: {exc}")
                    results.append({
                        "ticker": ticker,
                        "response": f"Analysis unavailable: {exc}",
                        "chunks": [],
                        "metadata": {},
                    })

        # Preserve consistent ordering (MSFT, AAPL, GOOGL, NVDA)
        order = {t: i for i, t in enumerate(config.TICKERS)}
        results.sort(key=lambda r: order.get(r["ticker"], 99))
        return results

    def synthesize(self, query: str, company_responses: list[dict]) -> str:
        """
        Have the orchestrator LLM produce a final comparative answer.

        We pass all company responses as a structured prompt. The orchestrator
        has no retrieval tool — it only synthesizes what the company agents found.
        """
        sections = "\n\n".join(
            f"=== {r['ticker']} ANALYST ===\n{r['response']}"
            for r in company_responses
        )

        response = self._oai.chat.completions.create(
            model=config.CHAT_MODEL,
            messages=[
                {"role": "system", "content": ORCHESTRATOR_SYSTEM},
                {
                    "role": "user",
                    "content": (
                        f"USER QUESTION: {query}\n\n"
                        f"ANALYST RESPONSES:\n{sections}\n\n"
                        "Please synthesize these into a clear comparative analysis."
                    ),
                },
            ],
            temperature=0,
            max_tokens=1200,
        )
        return response.choices[0].message.content

    def answer(
        self,
        query: str,
        tickers: Optional[list[str]] = None,
    ) -> dict:
        """
        Full orchestration pipeline. The primary entry point for Streamlit and CLI.

        Returns:
          {
            "query": str,
            "tickers_used": list[str],
            "company_responses": list[dict],   # per-company analysis + chunks
            "synthesis": str,                  # final comparative answer
            "embedding_used": list[float],     # for Index Explorer reuse
          }
        """
        # Embed query once — reused by all company agents to avoid repeated API calls
        query_embedding = get_embedding(self._oai, query)

        # Determine relevant tickers
        active_tickers = tickers or self.route_query(query)
        log.info(f"Routing to: {active_tickers}")

        # Fan out
        company_responses = self.run_parallel(active_tickers, query, query_embedding)

        # Synthesize if multiple companies involved
        if len(company_responses) > 1:
            synthesis = self.synthesize(query, company_responses)
        else:
            synthesis = company_responses[0]["response"]

        return {
            "query": query,
            "tickers_used": active_tickers,
            "company_responses": company_responses,
            "synthesis": synthesis,
            "embedding_used": query_embedding,
        }


# ═════════════════════════════════════════════════════════════════════════════
# FACTORY
# ═════════════════════════════════════════════════════════════════════════════


def build_orchestrator(project_client: AIProjectClient) -> AppOrchestrator:
    """
    Convenience factory used by app.py.

    Returns a ready-to-use AppOrchestrator backed by the project's
    Azure OpenAI deployment via config.get_openai_client().
    """
    openai_client = config.get_openai_client()
    return AppOrchestrator(openai_client)


# ═════════════════════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════════════════════


def main() -> None:
    parser = argparse.ArgumentParser(description="EDGAR Multi-Agent CLI")
    parser.add_argument("--list", action="store_true", help="List Foundry agents in this project")
    parser.add_argument("--query", type=str, help="Run an end-to-end orchestrated query")
    parser.add_argument(
        "--tickers",
        type=str,
        help="Comma-separated tickers to restrict (e.g. MSFT,NVDA). Default: auto-route.",
    )
    parser.add_argument("--top-k", type=int, default=5, help="Chunks retrieved per company")
    args = parser.parse_args()

    credential = AzureCliCredential()
    project_client = AIProjectClient(
        endpoint=config.FOUNDRY_PROJECT_ENDPOINT,
        credential=credential,
    )

    if args.list:
        agents = list(project_client.agents.list())
        if not agents:
            print("No Foundry agents found in this project.")
        for a in agents:
            print(f"  {a.name:<30}  id={a.id}")
        return

    if args.query:
        tickers = [t.strip().upper() for t in args.tickers.split(",")] if args.tickers else None
        orchestrator = AppOrchestrator(config.get_openai_client())

        print(f"\nQuery: {args.query}")
        print("=" * 70)

        t0 = time.perf_counter()
        result = orchestrator.answer(args.query, tickers=tickers)
        elapsed = time.perf_counter() - t0

        for r in result["company_responses"]:
            m = r.get("metadata", {})
            print(f"\n── {r['ticker']} ({m.get('chunks_retrieved', 0)} chunks, "
                  f"{m.get('retrieval_latency_s', 0):.2f}s retrieval) ──")
            print(r["response"][:600])

        if len(result["company_responses"]) > 1:
            print("\n── SYNTHESIS ──")
            print(result["synthesis"])

        print(f"\nTotal elapsed: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
