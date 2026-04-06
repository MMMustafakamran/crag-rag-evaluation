# Sample Illustrative Recommendation Report

This document is a mock example for teaching. The numbers below are hypothetical and are provided only to illustrate what a reasonable student write-up and result pattern might look like for this assignment. They are not the actual measured results of this repository.

## Executive Summary

In this illustrative evaluation, **CRAG** performed best overall and is the strategy we would recommend shipping for a factual assistant built on a noisy pre-crawled web corpus. It offered the best balance between answer accuracy, trustworthiness, and robustness when retrieval quality varied from query to query.

## Sample Results

| Pipeline | Accuracy | Avg Retrieval Score | Count |
|----------|----------|---------------------|-------|
| RAG Fusion | 24.0% | 0.3318 | 25 |
| HyDE | 36.0% | 0.4986 | 25 |
| CRAG | 44.0% | 0.4639 | 25 |
| Graph RAG | 32.0% | 0.4274 | 25 |

## Pipeline Analysis

### 1. RAG Fusion

RAG Fusion generated multiple search-style rewrites of the user query, retrieved for each version, and then merged the ranked results using Reciprocal Rank Fusion. In this illustrative outcome, it helped on some multi-hop and ambiguous questions, but it also amplified noise because some rewritten queries drifted away from the original intent. As a result, it had the lowest final accuracy among the four approaches.

### 2. HyDE

HyDE improved retrieval by first generating a hypothetical answer-like document and then retrieving corpus chunks similar to that generated text. This helped more than standard query retrieval when the original question was short or underspecified. In the mock results, HyDE ranked second and showed stronger average retrieval scores than RAG Fusion.

### 3. CRAG

CRAG first retrieved context normally, then used a confidence-based decision to determine whether retrieval should be trusted for final answer generation. This made it more robust on noisy examples because weak retrieval did not automatically poison the final response. In the illustrative results, CRAG achieved the highest accuracy and was the most reliable option overall. It also aligned well with the assignment requirement to surface source-backed answers.

### 4. Graph RAG

Graph RAG expanded beyond initial seed chunks by traversing a chunk-similarity graph. This gave it an advantage on questions where related facts were spread across multiple pieces of context. However, the added complexity did not consistently outperform CRAG on short factual questions. In the mock results, Graph RAG performed better than RAG Fusion but remained behind HyDE and CRAG.

## Final Recommendation

Based on this hypothetical example, we would recommend **CRAG** for deployment.

Reasons:

- It produced the highest overall accuracy.
- It handled noisy retrieval better than the other strategies.
- It supports more trustworthy answers because it can reject weak retrieval and include citations when context is reliable.
- It is easier to justify in production than Graph RAG, which is more complex and computationally heavier.

## Teaching Note

This report is intentionally written as a model answer for students. It demonstrates the kind of evidence-based recommendation they should aim to produce: compare the pipelines fairly, identify tradeoffs, and make a final shipping decision that is supported by the evaluation table.
