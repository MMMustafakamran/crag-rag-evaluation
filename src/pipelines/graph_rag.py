"""
Graph RAG: build a chunk-similarity graph, retrieve seed nodes via vector search,
expand to their neighbourhood via BFS, then generate from the expanded context.
"""

from __future__ import annotations

import numpy as np
import networkx as nx
from collections import deque


# Similarity threshold above which two chunks are connected by an edge
EDGE_SIM_THRESHOLD = 0.70
# Maximum BFS depth from seed nodes
BFS_DEPTH = 2
# Maximum total nodes to collect after graph expansion
MAX_GRAPH_NODES = 15


def _build_similarity_graph(embeddings: np.ndarray, threshold: float = EDGE_SIM_THRESHOLD) -> nx.Graph:
    """
    Build an undirected graph where nodes are chunk indices and edges
    connect chunks whose cosine similarity exceeds `threshold`.

    For large corpora we only build a sparse approximate graph by computing
    similarities in blocks to stay memory-efficient.
    """
    n = len(embeddings)
    G = nx.Graph()
    G.add_nodes_from(range(n))

    # Compute similarity in blocks of 512 rows to avoid OOM on large corpora
    block_size = 512
    for i in range(0, n, block_size):
        block = embeddings[i: i + block_size]  # (B, D)
        sims = block @ embeddings.T  # (B, N) — all-pairs cosine sim for this block
        for bi, row in enumerate(sims):
            row_idx = i + bi
            # Only edges where similarity > threshold and j > row_idx (upper triangle)
            js = np.where(row > threshold)[0]
            for j in js:
                if j > row_idx:
                    G.add_edge(row_idx, int(j), weight=float(row[j]))

    return G


def _bfs_expand(G: nx.Graph, seeds: list[int], max_depth: int, max_nodes: int) -> list[int]:
    """
    BFS from seed nodes up to `max_depth` hops, collecting up to `max_nodes` total.
    Returns list of node indices in discovery order.
    """
    visited = set(seeds)
    queue = deque((s, 0) for s in seeds)
    collected = list(seeds)

    while queue and len(collected) < max_nodes:
        node, depth = queue.popleft()
        if depth >= max_depth:
            continue
        for neighbour in G.neighbors(node):
            if neighbour not in visited:
                visited.add(neighbour)
                collected.append(neighbour)
                queue.append((neighbour, depth + 1))
                if len(collected) >= max_nodes:
                    break

    return collected


def run(
    query: str,
    corpus,
    embedder,
    generator,
    top_k: int = 5,
) -> dict:
    """
    Graph RAG pipeline.

    Steps:
    1. Retrieve top-k seed chunks via vector search.
    2. Build (or reuse) a chunk-similarity graph over the corpus.
    3. BFS-expand from seed nodes to collect their graph neighbourhood.
    4. Generate answer from the expanded context.

    Returns:
        {retrieved: list[dict], answer: str, meta: {...}}
    """
    from src.retrieval import embed_text
    from src.generation import generate_answer

    # Step 1: vector retrieval to get seed nodes
    q_emb = embed_text(query, embedder)
    seeds_raw = corpus.retrieve(q_emb, top_k=top_k)
    seed_idxs = [c["chunk_idx"] for c in seeds_raw]

    # Step 2: build or use cached graph (attach to corpus object for reuse across calls)
    if not hasattr(corpus, "_graph") or corpus._graph is None:
        print("[graph_rag] Building chunk-similarity graph (first call, may take a moment)...")
        corpus._graph = _build_similarity_graph(corpus.embeddings, threshold=EDGE_SIM_THRESHOLD)
        print(f"[graph_rag] Graph built: {corpus._graph.number_of_nodes()} nodes, "
              f"{corpus._graph.number_of_edges()} edges")

    G: nx.Graph = corpus._graph

    # Step 3: BFS-expand from seeds
    expanded_idxs = _bfs_expand(G, seed_idxs, max_depth=BFS_DEPTH, max_nodes=MAX_GRAPH_NODES)

    # Build final chunk list (seeds first, then neighbours), deduplicated
    all_idxs = list(dict.fromkeys(seed_idxs + [i for i in expanded_idxs if i not in set(seed_idxs)]))
    all_idxs = all_idxs[:MAX_GRAPH_NODES]

    retrieved = []
    for idx in all_idxs:
        meta = corpus.metadata[idx] if idx < len(corpus.metadata) else {}
        is_seed = idx in set(seed_idxs)
        # Find original score if it was a seed
        orig_score = next((c["score"] for c in seeds_raw if c["chunk_idx"] == idx), 0.0)
        retrieved.append({
            "chunk_idx": idx,
            "text": corpus.chunks[idx],
            "score": orig_score,
            "page_name": meta.get("page_name", ""),
            "page_url": meta.get("page_url", ""),
            "query_id": meta.get("query_id", ""),
            "is_seed": is_seed,
        })

    # Step 4: generate from expanded neighbourhood
    answer = generate_answer(query, retrieved, generator)

    return {
        "retrieved": retrieved,
        "answer": answer,
        "meta": {
            "pipeline": "graph_rag",
            "seed_count": len(seed_idxs),
            "expanded_count": len(retrieved),
            "graph_edges": G.number_of_edges(),
            "bfs_depth": BFS_DEPTH,
        },
    }
