

## Project Overview

This project builds a shared global corpus from a CRAG-style web snapshot and compares four retrieval-augmented generation strategies:

- RAG Fusion
- HyDE
- CRAG
- Graph RAG

All four pipelines retrieve from the same global embedding index and are evaluated on a held-out subset of questions.

## System Components

- **Global Corpus Index**: All `page_snippet` texts are collected into one corpus and embedded with a sentence-transformer model.
- **Pipelines**: Each pipeline applies a different retrieval strategy before answer generation.
- **Evaluation Runner**: A script runs each pipeline on the same dev examples and computes accuracy.
- **Frontend**: A React interface allows a user to enter a query, select a pipeline, and inspect retrieved context and the generated answer.

## Illustrative Evaluation Outcome

The following table is a mock example showing the kind of comparative result pattern a student might reasonably obtain:

| Pipeline | Accuracy | Avg Retrieval Score | Count |
|----------|----------|---------------------|-------|
| RAG Fusion | 24.0% | 0.3318 | 25 |
| HyDE | 36.0% | 0.4986 | 25 |
| CRAG | 44.0% | 0.4639 | 25 |
| Graph RAG | 32.0% | 0.4274 | 25 |

## Interpretation

- **RAG Fusion** helped on some difficult phrasings, but the extra query variants sometimes introduced retrieval noise.
- **HyDE** improved retrieval quality by matching the corpus against a hypothetical answer-shaped document.
- **CRAG** performed best because confidence gating reduced the harm caused by irrelevant retrieval.
- **Graph RAG** captured some cross-chunk relationships, but its extra complexity did not outperform CRAG on short factual QA.

## Recommendation

Based on this illustrative comparison, **CRAG** would be the recommended strategy to ship. It provides the strongest tradeoff between accuracy, stability, and interpretability in a noisy web-derived corpus.
