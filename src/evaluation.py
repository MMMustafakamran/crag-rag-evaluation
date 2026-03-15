"""
Compare predicted answer to gold answer and alt_ans; compute accuracy.
"""

from __future__ import annotations

import re
import string


def normalize(text: str) -> str:
    """
    Normalize text for comparison:
    - lowercase
    - strip punctuation
    - collapse whitespace
    """
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def is_correct(prediction: str, answer: str, alt_ans: list[str]) -> bool:
    """
    Return True if the prediction matches (normalized) the gold answer or any alt_ans.

    Also handles partial match: if gold answer (normalized) is fully contained
    in the prediction (e.g. prediction is "The answer is Roger Federer" and
    gold is "Roger Federer").
    """
    pred_norm = normalize(prediction)
    gold_norms = [normalize(answer)] + [normalize(a) for a in (alt_ans or [])]

    for gold in gold_norms:
        if not gold:
            continue
        # Exact normalized match
        if pred_norm == gold:
            return True
        # Gold is contained in prediction (model often gives verbose answers)
        if gold in pred_norm:
            return True

    return False


def compute_accuracy(results: list[bool]) -> float:
    """Return fraction of True values in results list."""
    if not results:
        return 0.0
    return sum(results) / len(results)
