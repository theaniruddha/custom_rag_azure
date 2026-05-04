"""
app.py — EDGAR Multi-Agent Comparator — Streamlit UI

Three tabs:
  1. Chat          — Multi-agent orchestrated Q&A across MSFT/AAPL/GOOGL/NVDA
  2. Index Explorer — Live comparison of vector search across 6 index configs
  3. Evaluations   — RAG Triad scores loaded from evaluation/results.json

Run:  uv run streamlit run app.py
"""

import json
import time
from pathlib import Path

import streamlit as st

# ── Page config must be first Streamlit call ──────────────────────────────────
st.set_page_config(
    page_title="EDGAR Multi-Agent Comparator",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

import sys
sys.path.insert(0, str(Path(__file__).parent))

import config

# ── Lazy imports (avoid slow SDK load on every rerun) ─────────────────────────
@st.cache_resource(show_spinner="Connecting to Azure AI Foundry...")
def get_project_client():
    from azure.ai.projects import AIProjectClient
    from azure.identity import AzureCliCredential
    return AIProjectClient(
        endpoint=config.FOUNDRY_PROJECT_ENDPOINT,
        credential=AzureCliCredential(),
    )


@st.cache_resource(show_spinner="Initializing agents...")
def get_orchestrator():
    from agents.setup_agents import build_orchestrator
    return build_orchestrator(get_project_client())


@st.cache_resource(show_spinner="Connecting to Azure AI Search...")
def get_search_clients():
    """Return {index_name: SearchClient} for all configured indexes."""
    from azure.core.credentials import AzureKeyCredential
    from azure.search.documents import SearchClient
    credential = AzureKeyCredential(config.SEARCH_KEY)
    return {
        label: SearchClient(
            endpoint=config.SEARCH_ENDPOINT,
            index_name=index_name,
            credential=credential,
        )
        for label, index_name in config.ALL_INDEXES.items()
    }


@st.cache_resource(show_spinner="Loading embedding model...")
def get_openai_client():
    # Uses the hub AI Services endpoint — the correct path for embeddings + chat in SDK 2.1.0
    return config.get_openai_client()


# ═════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.title("📊 EDGAR Comparator")
    st.caption("Multi-Agent RAG • Azure AI Foundry")
    st.divider()

    st.subheader("Project Config")
    st.code(
        f"RG: {config.RESOURCE_GROUP}\n"
        f"Model: {config.CHAT_MODEL}\n"
        f"Embed: {config.EMBEDDING_MODEL}\n"
        f"Dims: {config.EMBEDDING_DIMENSIONS}",
        language="text",
    )

    st.divider()
    st.subheader("Coverage")
    for ticker in config.TICKERS:
        st.markdown(f"- **{ticker}**: 10-K + 10-Q")

    st.divider()
    st.caption("Built with Azure AI Foundry + Azure AI Search")


# ═════════════════════════════════════════════════════════════════════════════
# MAIN TABS
# ═════════════════════════════════════════════════════════════════════════════

tab_chat, tab_explorer, tab_eval = st.tabs(
    ["💬 Multi-Agent Chat", "🔬 Vector Index Explorer", "📈 Evaluation Results"]
)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1: CHAT
# ─────────────────────────────────────────────────────────────────────────────

with tab_chat:
    st.header("Multi-Agent Financial Analyst")
    st.markdown(
        "Ask any question about MSFT, AAPL, GOOGL, or NVDA. "
        "The orchestrator routes to specialized company agents, runs them in parallel, "
        "and synthesizes a comparative answer."
    )

    # Chat history stored in session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "agent_trace" not in st.session_state:
        st.session_state.agent_trace = []

    # Render conversation history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Query input
    col1, col2 = st.columns([3, 1])
    with col2:
        selected_tickers = st.multiselect(
            "Restrict to tickers",
            config.TICKERS,
            placeholder="Auto-route",
            help="Leave empty to let the orchestrator decide which companies are relevant.",
        )

    if prompt := st.chat_input("Ask about revenues, margins, growth, risks..."):
        # Show user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Run the orchestrator
        with st.chat_message("assistant"):
            result = None
            error = None

            # st.status collapses when complete — keep only the progress log inside it
            with st.status("Running multi-agent pipeline...", expanded=True) as status:
                try:
                    orchestrator = get_orchestrator()

                    tickers_arg = selected_tickers if selected_tickers else None
                    if not tickers_arg:
                        st.write("🔀 Routing query to relevant agents...")
                        routed = orchestrator.route_query(prompt)
                        st.write(f"→ Selected: **{', '.join(routed)}**")
                        tickers_arg = routed

                    st.write(f"⚡ Running {len(tickers_arg)} agents in parallel...")
                    t_start = time.perf_counter()
                    result = orchestrator.answer(prompt, tickers=tickers_arg)
                    elapsed = time.perf_counter() - t_start
                    st.write(f"✅ Done in {elapsed:.1f}s")
                    status.update(label=f"Completed in {elapsed:.1f}s", state="complete")

                except Exception as exc:
                    error = exc
                    status.update(label="Error", state="error")

            # Results live OUTSIDE st.status so they stay visible after it collapses
            if error:
                st.error(f"Agent pipeline error: {error}")
                st.exception(error)

            elif result:
                st.session_state.agent_trace = result["company_responses"]

                # Per-company analyst responses — collapsed by default to save space
                for r in result["company_responses"]:
                    with st.expander(f"📄 {r['ticker']} Analyst Response", expanded=False):
                        st.markdown(r["response"])

                # Synthesized answer — always visible, no click required
                st.divider()
                st.markdown("**Synthesized Answer**")
                st.markdown(result["synthesis"])

                st.session_state.messages.append(
                    {"role": "assistant", "content": result["synthesis"]}
                )

    # Quick-start examples
    if not st.session_state.messages:
        st.markdown("#### Try these questions:")
        examples = [
            "Compare the gross margins of MSFT, AAPL, GOOGL, and NVDA",
            "What are NVIDIA's main revenue segments and their growth rates?",
            "Which company has the most cash and short-term investments?",
            "What are the key risk factors for Apple in the latest 10-K?",
        ]
        cols = st.columns(2)
        for i, ex in enumerate(examples):
            with cols[i % 2]:
                st.info(ex)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2: VECTOR INDEX EXPLORER
# ─────────────────────────────────────────────────────────────────────────────

with tab_explorer:
    st.header("Vector Index Explorer")
    st.markdown(
        "Compare retrieval results, latency, and relevance scores across "
        "all 6 index configurations for any query. "
        "This shows you live how HNSW parameters and quantization affect results."
    )

    # Educational sidebar
    with st.expander("📚 HNSW Parameter Guide", expanded=False):
        st.markdown("""
| Parameter | What it controls | Higher value → |
|-----------|-----------------|----------------|
| **m** | Bidirectional links per node in the HNSW graph | Better recall, more RAM |
| **efConstruction** | Beam width during index build | Better graph quality, slower build |
| **efSearch** | Beam width during query | Better recall, slower queries |

**Azure AI Search limits:** m ∈ [4, 10], ef ∈ [100, 1000]

**Scalar Quantization:** Compresses float32 → int8 (~75% memory reduction, ~1-3% recall loss)

**Hybrid Search:** Combines HNSW vector score + BM25 keyword score via Reciprocal Rank Fusion (RRF)
        """)

    col_q, col_t = st.columns([2, 1])
    with col_q:
        explorer_query = st.text_input(
            "Search query",
            placeholder="e.g. cloud revenue growth rate",
            key="explorer_query",
        )
    with col_t:
        explorer_ticker = st.selectbox(
            "Filter by ticker",
            ["All"] + config.TICKERS,
            key="explorer_ticker",
        )

    selected_configs = st.multiselect(
        "Select index configs to compare",
        list(config.ALL_INDEXES.keys()),
        default=list(config.ALL_INDEXES.keys())[:3],
        key="selected_configs",
    )

    top_k = st.slider("Results per index (top-k)", 1, 10, 3, key="top_k")

    run_btn = st.button("🔍 Compare Indexes", type="primary", disabled=not explorer_query)

    if run_btn and explorer_query:
        ticker_filter = None if explorer_ticker == "All" else explorer_ticker

        with st.spinner("Generating query embedding..."):
            try:
                openai_client = get_openai_client()
                emb_response = openai_client.embeddings.create(
                    model=config.EMBEDDING_MODEL,
                    input=[explorer_query],
                )
                query_vec = emb_response.data[0].embedding
            except Exception as exc:
                st.error(f"Embedding failed: {exc}")
                st.stop()

        from azure.search.documents.models import VectorizedQuery

        latencies: dict[str, float] = {}
        all_results: dict[str, list[dict]] = {}
        all_scores: dict[str, list[float]] = {}

        search_clients = get_search_clients()

        # Run search against each selected config
        for label in selected_configs:
            client = search_clients.get(label)
            if client is None:
                continue

            vq = VectorizedQuery(
                vector=query_vec,
                k_nearest_neighbors=top_k,
                fields="content_vector",
            )

            odata_filter = f"ticker eq '{ticker_filter}'" if ticker_filter else None
            is_hybrid = "Hybrid" in label

            t0 = time.perf_counter()
            try:
                results = list(client.search(
                    search_text=explorer_query if is_hybrid else None,
                    vector_queries=[vq],
                    filter=odata_filter,
                    select=["id", "content", "ticker", "filing_type", "chunk_id"],
                    top=top_k,
                ))
                latencies[label] = round((time.perf_counter() - t0) * 1000, 1)  # ms
                all_results[label] = results
                all_scores[label] = [r.get("@search.score", 0.0) for r in results]
            except Exception as exc:
                latencies[label] = -1
                all_results[label] = []
                st.warning(f"{label}: {exc}")

        # ── Latency comparison chart ───────────────────────────────────────────
        st.subheader("Query Latency (ms)")
        try:
            import plotly.graph_objects as go

            valid = {k: v for k, v in latencies.items() if v >= 0}
            fig = go.Figure(go.Bar(
                x=list(valid.keys()),
                y=list(valid.values()),
                marker_color=["#0078d4" if "Accurate" not in k else "#d44000" for k in valid],
                text=[f"{v}ms" for v in valid.values()],
                textposition="outside",
            ))
            fig.update_layout(
                height=300,
                margin=dict(t=20, b=20),
                xaxis_tickangle=-20,
                yaxis_title="Latency (ms)",
                showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True)
        except ImportError:
            st.json(latencies)

        # ── Side-by-side results ──────────────────────────────────────────────
        st.subheader("Retrieved Chunks")
        n_cols = min(len(selected_configs), 3)
        cols = st.columns(n_cols)

        for i, label in enumerate(selected_configs):
            with cols[i % n_cols]:
                st.markdown(f"**{label}**")
                lat = latencies.get(label, -1)
                st.caption(f"⏱ {lat}ms" if lat >= 0 else "⚠ Failed")

                for j, result in enumerate(all_results.get(label, []), 1):
                    score = result.get("@search.score", 0.0)
                    ticker = result.get("ticker", "?")
                    filing = result.get("filing_type", "?")
                    content = result.get("content", "")[:300]

                    with st.expander(f"#{j} [{ticker} {filing}] score={score:.3f}", expanded=(j == 1)):
                        st.markdown(content + "...")

        # ── Score comparison (top-1 bar) ──────────────────────────────────────
        st.subheader("Top-1 Relevance Score Comparison")
        top1_scores = {
            label: scores[0] if scores else 0.0
            for label, scores in all_scores.items()
        }
        try:
            fig2 = go.Figure(go.Bar(
                x=list(top1_scores.keys()),
                y=list(top1_scores.values()),
                marker_color="#107c10",
                text=[f"{v:.3f}" for v in top1_scores.values()],
                textposition="outside",
            ))
            fig2.update_layout(
                height=280,
                margin=dict(t=20, b=20),
                xaxis_tickangle=-20,
                yaxis_title="Cosine Similarity Score",
                yaxis_range=[0, max(top1_scores.values()) * 1.2 if top1_scores else 1],
                showlegend=False,
            )
            st.plotly_chart(fig2, use_container_width=True)
        except (ImportError, ValueError):
            st.json(top1_scores)

        # ── Latency vs Quality tradeoff scatter ───────────────────────────────
        st.subheader("Quality–Speed Tradeoff")
        st.caption(
            "Each point is one index. Ideal position = top-left (fast AND accurate). "
            "HNSW Fast trades recall for speed; KNN is the ground-truth ceiling."
        )
        try:
            scatter_labels = [l for l in selected_configs if latencies.get(l, -1) >= 0]
            scatter_x = [latencies[l] for l in scatter_labels]
            scatter_y = [top1_scores.get(l, 0.0) for l in scatter_labels]
            # Short names for marker labels
            short_names = [l.split(" (")[0] for l in scatter_labels]

            fig_scatter = go.Figure(go.Scatter(
                x=scatter_x,
                y=scatter_y,
                mode="markers+text",
                text=short_names,
                textposition="top center",
                marker=dict(
                    size=16,
                    color=scatter_y,
                    colorscale="RdYlGn",
                    showscale=True,
                    colorbar=dict(title="Score", thickness=12),
                    line=dict(width=1, color="white"),
                ),
            ))
            fig_scatter.update_layout(
                height=380,
                xaxis_title="Query Latency (ms) — lower is better →",
                yaxis_title="Top-1 Score — higher is better ↑",
                margin=dict(t=20, b=40),
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
        except (ImportError, ValueError):
            pass

        # ── Score distribution across all k results ───────────────────────────
        st.subheader("Score Distribution Across All Results")
        st.caption(
            "Wider, higher distributions = index confidently returns relevant chunks. "
            "A steep drop-off from rank 1→k means only the top result is useful."
        )
        try:
            fig_dist = go.Figure()
            palette = ["#0078d4", "#107c10", "#d44000", "#5c2d91", "#a4262c", "#004578"]
            for i, label in enumerate(selected_configs):
                scores_k = all_scores.get(label, [])
                if not scores_k:
                    continue
                ranks = list(range(1, len(scores_k) + 1))
                fig_dist.add_trace(go.Scatter(
                    x=ranks,
                    y=scores_k,
                    name=label.split(" (")[0],
                    mode="lines+markers",
                    marker=dict(size=8),
                    line=dict(color=palette[i % len(palette)], width=2),
                ))
            fig_dist.update_layout(
                height=320,
                xaxis_title="Rank (1 = best match)",
                yaxis_title="Similarity Score",
                xaxis=dict(tickmode="linear", dtick=1),
                legend=dict(orientation="h", y=1.12),
                margin=dict(t=20, b=40),
            )
            st.plotly_chart(fig_dist, use_container_width=True)
        except (ImportError, ValueError):
            pass

        # ── Result Overlap Heatmap ────────────────────────────────────────────
        st.subheader("Result Overlap Between Indexes")
        st.caption(
            "Cell value = % of retrieved document IDs shared between two indexes. "
            "100% = identical results. Low overlap = meaningfully different retrieval behavior."
        )
        try:
            # Collect result ID sets per index
            id_sets: dict[str, set] = {
                label: {r.get("id", "") for r in results_list}
                for label, results_list in all_results.items()
                if results_list
            }
            overlap_labels = list(id_sets.keys())
            n = len(overlap_labels)
            matrix = [[0.0] * n for _ in range(n)]

            for i, la in enumerate(overlap_labels):
                for j, lb in enumerate(overlap_labels):
                    a, b = id_sets[la], id_sets[lb]
                    union = a | b
                    matrix[i][j] = round(len(a & b) / len(union) * 100, 1) if union else 100.0

            short_ol = [l.split(" (")[0] for l in overlap_labels]
            fig_heat = go.Figure(go.Heatmap(
                z=matrix,
                x=short_ol,
                y=short_ol,
                colorscale="Blues",
                zmin=0,
                zmax=100,
                text=[[f"{v:.0f}%" for v in row] for row in matrix],
                texttemplate="%{text}",
                showscale=True,
                colorbar=dict(title="Overlap %", thickness=12),
            ))
            fig_heat.update_layout(
                height=350,
                margin=dict(t=20, b=20, l=120, r=20),
                xaxis_tickangle=-30,
            )
            st.plotly_chart(fig_heat, use_container_width=True)
        except (ImportError, ValueError, ZeroDivisionError):
            pass


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3: EVALUATION RESULTS
# ─────────────────────────────────────────────────────────────────────────────

with tab_eval:
    st.header("RAG Triad Evaluation Results")
    st.markdown(
        "Results from `evaluation/run_evals.py`. "
        "Run it first to populate this tab."
    )

    results_path = Path(config.EVAL_OUTPUT_PATH)
    benchmark_path = Path(__file__).parent / "evaluation" / "index_benchmark.csv"

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("🔄 Refresh Results"):
            st.cache_data.clear()

    if results_path.exists():
        with open(results_path) as f:
            results = json.load(f)

        # Summary metrics
        if results:
            avg_ctx = sum(r["context_relevance"] for r in results) / len(results)
            avg_faith = sum(r["faithfulness"] for r in results) / len(results)
            avg_ans = sum(r["answer_relevance"] for r in results) / len(results)
            avg_lat = sum(r["latency_s"] for r in results) / len(results)

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Context Relevance", f"{avg_ctx:.2f}", help="Are retrieved chunks relevant?")
            m2.metric("Faithfulness", f"{avg_faith:.2f}", help="Is answer grounded in context?")
            m3.metric("Answer Relevance", f"{avg_ans:.2f}", help="Does answer address the query?")
            m4.metric("Avg Latency", f"{avg_lat:.1f}s")

            st.divider()

        # Detailed results table
        st.subheader("Per-Question Results")
        table_data = [
            {
                "ID": r["question_id"],
                "Ticker": r.get("ticker") or "ALL",
                "Query": r["query"][:60] + "...",
                "Ctx Rel": f"{r['context_relevance']:.2f}",
                "Faithful": f"{r['faithfulness']:.2f}",
                "Ans Rel": f"{r['answer_relevance']:.2f}",
                "Coverage": f"{r['chunk_coverage']:.2f}",
                "Latency": f"{r['latency_s']:.1f}s",
            }
            for r in results
        ]
        st.dataframe(table_data, use_container_width=True)

        # RAG Triad radar / bar charts
        st.subheader("RAG Triad by Question")
        try:
            import plotly.graph_objects as go

            fig = go.Figure()
            metrics = ["context_relevance", "faithfulness", "answer_relevance"]
            labels = ["Context Relevance", "Faithfulness", "Answer Relevance"]
            colors = ["#0078d4", "#107c10", "#d44000"]

            for metric, label, color in zip(metrics, labels, colors):
                fig.add_trace(go.Bar(
                    name=label,
                    x=[r["question_id"] for r in results],
                    y=[r[metric] for r in results],
                    marker_color=color,
                ))

            fig.update_layout(
                barmode="group",
                height=380,
                yaxis_range=[0, 1.1],
                yaxis_title="Score (0–1)",
                legend=dict(orientation="h", y=1.1),
            )
            st.plotly_chart(fig, use_container_width=True)
        except ImportError:
            pass

        # Full answer drill-down
        st.subheader("Answer Drill-down")
        selected_q = st.selectbox(
            "Select question",
            [r["question_id"] for r in results],
            key="drilldown_q",
        )
        if selected_q:
            r = next(x for x in results if x["question_id"] == selected_q)
            st.markdown(f"**Query:** {r['query']}")
            st.markdown(f"**Answer:**\n\n{r['answer']}")
            with st.expander("Retrieved Chunks"):
                for i, chunk in enumerate(r["retrieved_chunks"], 1):
                    st.markdown(f"**Chunk {i}:** {chunk[:400]}...")

    else:
        st.info(
            "No evaluation results found. Run:\n\n"
            "```bash\nuv run python -m evaluation.run_evals\n```"
        )

    # Index benchmark results (from ingestion)
    if benchmark_path.exists():
        st.divider()
        st.subheader("Index Build Benchmark")
        st.caption("Upload throughput by index configuration (from ingestion run)")
        import csv as csv_module

        with open(benchmark_path) as f:
            benchmark_rows = list(csv_module.DictReader(f))

        if benchmark_rows:
            try:
                import plotly.graph_objects as go

                fig3 = go.Figure(go.Bar(
                    x=[r["label"] for r in benchmark_rows],
                    y=[float(r["docs_per_second"]) for r in benchmark_rows],
                    text=[f"{float(r['docs_per_second']):.0f} doc/s" for r in benchmark_rows],
                    textposition="outside",
                    marker_color="#5c2d91",
                ))
                fig3.update_layout(
                    height=300,
                    yaxis_title="Documents / second",
                    xaxis_tickangle=-20,
                    margin=dict(t=20),
                )
                st.plotly_chart(fig3, use_container_width=True)
            except ImportError:
                st.dataframe(benchmark_rows, use_container_width=True)
