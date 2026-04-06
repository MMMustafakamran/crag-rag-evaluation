"""
Run all 4 pipelines on the dev set (or a subset), compute accuracy per pipeline, print results.
Usage:
    python run_evaluation.py [--limit N] [--rebuild]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def load_config(config_path: str = "config/config.yaml") -> dict:
    if not Path(config_path).exists():
        config_path = "config/config.example.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Run RAG pipeline evaluation")
    parser.add_argument("--limit", type=int, default=None,
                        help="Max examples to evaluate (default: from config)")
    parser.add_argument("--rebuild", action="store_true",
                        help="Force rebuild the corpus index even if it exists")
    parser.add_argument("--config", type=str, default="config/config.yaml",
                        help="Path to config.yaml")
    parser.add_argument("--rpm", type=int, default=None,
                        help="Max LLM requests per minute to respect during evaluation")
    args = parser.parse_args()

    config = load_config(args.config)

    dataset_path = config["dataset_path"]
    embedding_model = config.get("embedding_model", "all-MiniLM-L6-v2")
    api_key = config.get("gemini_api_key", "")
    generation_model = config.get("generation_model", "gemini-2.0-flash")
    requests_per_minute = args.rpm if args.rpm is not None else config.get("requests_per_minute", 30)
    top_k = config.get("top_k", 5)
    index_path = config.get("index_path", "index/crag_index")
    index_build_limit = config.get("index_build_limit", 500)
    eval_limit = args.limit if args.limit is not None else config.get("eval_limit", 50)

    # ── 1. Build or load corpus index ───────────────────────────────────────
    from src.corpus import build_index, load_index
    from src.retrieval import get_embedder
    from src.generation import get_generator

    index_exists = Path(index_path).exists() and not args.rebuild

    if index_exists:
        print(f"[eval] Loading existing index from {index_path} ...")
        corpus = load_index(index_path, embedding_model)
    else:
        print(f"[eval] Building index (limit={index_build_limit} rows) ...")
        corpus = build_index(dataset_path, embedding_model, index_path, limit=index_build_limit)

    embedder = get_embedder(embedding_model)
    generator = get_generator(api_key, generation_model, requests_per_minute=requests_per_minute)

    # ── 2. Load pipeline modules ─────────────────────────────────────────────
    from src.pipelines import rag_fusion, hyde, crag, graph_rag

    pipelines = {
        "RAG Fusion": rag_fusion,
        "HyDE":       hyde,
        "CRAG":       crag,
        "Graph RAG":  graph_rag,
    }

    # ── 3. Run evaluation ────────────────────────────────────────────────────
    from src.data_loader import load_examples
    from src.evaluation import is_correct, compute_accuracy

    results: dict[str, list[bool]] = {name: [] for name in pipelines}
    scores_store: dict[str, list[float]] = {name: [] for name in pipelines}
    output_rows = []

    print(f"\n[eval] Running {len(pipelines)} pipelines on up to {eval_limit} examples ...\n")
    estimated_calls_per_example = {
        "RAG Fusion": 2,
        "HyDE": 2,
        "CRAG": 1,
        "Graph RAG": 1,
    }
    calls_per_example = sum(estimated_calls_per_example.values())
    estimated_total_calls = eval_limit * calls_per_example
    estimated_minutes = estimated_total_calls / max(requests_per_minute, 1)
    print(
        f"[eval] Quota plan: about {calls_per_example} LLM calls/example, "
        f"~{estimated_total_calls} calls total, capped at {requests_per_minute} RPM "
        f"(~{estimated_minutes:.1f} min best case).\n"
    )

    for i, example in enumerate(load_examples(path=dataset_path, limit=eval_limit)):
        query = example["query"]
        answer = example["answer"]
        alt_ans = example.get("alt_ans", [])

        print(f"  [{i+1}/{eval_limit}] {query[:70]}...")
        row = {"query": query, "answer": answer, "pipelines": {}}

        for name, pipeline in pipelines.items():
            t0 = time.time()
            try:
                out = pipeline.run(query, corpus, embedder, generator, top_k=top_k)
                prediction = out.get("answer", "")
                avg_score = float(
                    sum(c.get("score", 0) for c in out.get("retrieved", [])) /
                    max(len(out.get("retrieved", [])), 1)
                )
                correct = is_correct(prediction, answer, alt_ans)
            except Exception as e:
                print(f"    [{name}] ERROR: {e}")
                prediction = ""
                avg_score = 0.0
                correct = False

            elapsed = time.time() - t0
            results[name].append(correct)
            scores_store[name].append(avg_score)
            row["pipelines"][name] = {
                "prediction": prediction,
                "correct": correct,
                "avg_score": avg_score,
                "elapsed_s": round(elapsed, 2),
            }
            status = "✓" if correct else "✗"
            print(f"    [{name}] {status}  score={avg_score:.3f}  ({elapsed:.1f}s)")

        output_rows.append(row)
        print()

    # ── 4. Save results ──────────────────────────────────────────────────────
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    results_path = results_dir / "evaluation_results.json"
    with open(results_path, "w") as f:
        json.dump(output_rows, f, indent=2, ensure_ascii=False)
    print(f"[eval] Raw results saved to {results_path}\n")

    # ── 5. Print accuracy table ──────────────────────────────────────────────
    print("=" * 60)
    print(f"{'Pipeline':<15} {'Accuracy':>10} {'Avg Score':>12} {'Count':>8}")
    print("-" * 60)
    for name in pipelines:
        acc = compute_accuracy(results[name])
        avg_sc = sum(scores_store[name]) / max(len(scores_store[name]), 1)
        count = len(results[name])
        print(f"{name:<15} {acc:>10.1%} {avg_sc:>12.4f} {count:>8}")
    print("=" * 60)
    print(f"\nEvaluation complete. Results saved to {results_path}")


if __name__ == "__main__":
    main()
