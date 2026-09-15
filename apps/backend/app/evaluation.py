from __future__ import annotations

import re
from typing import Any


def evaluate_report(report: str, evidence: list[dict[str, Any]], latency_seconds: float, usage: dict[str, int]) -> dict[str, Any]:
    """Compute deterministic production metrics; no fabricated model-judge scores."""
    cited = set(re.findall(r"\[((?:E|W)\d+)\]", report))
    available = {item["id"] for item in evidence}
    valid = cited & available
    invalid = cited - available
    return {
        "citation_accuracy": round(len(valid) / len(cited), 3) if cited else 0.0,
        "citation_coverage": round(min(len(cited) / len(evidence), 1.0), 3) if evidence else 0.0,
        "invalid_citation_count": len(invalid),
        "evidence_count": len(evidence),
        "latency_seconds": round(latency_seconds, 3),
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "completion_tokens": usage.get("completion_tokens", 0),
        "total_tokens": usage.get("total_tokens", 0),
        "faithfulness": None,
        "retrieval_recall": None,
        "evaluation_note": "Faithfulness requires an evaluator; retrieval recall requires labeled relevance judgments and is measured by the benchmark endpoint.",
    }


def retrieval_metrics(retrieved_ids: list[str], relevant_ids: list[str], k_values: tuple[int, ...] = (1, 3, 5, 10)) -> dict[str, float | int]:
    """Calculate Recall@K, Precision@K and MRR against explicit gold chunk IDs."""
    relevant = set(relevant_ids)
    if not relevant:
        return {**{f"recall_at_{k}": 0.0 for k in k_values}, **{f"precision_at_{k}": 0.0 for k in k_values}, "mrr": 0.0}
    result: dict[str, float | int] = {}
    for k in k_values:
        top = retrieved_ids[:k]
        hits = sum(1 for item in top if item in relevant)
        result[f"recall_at_{k}"] = round(hits / len(relevant), 3)
        result[f"precision_at_{k}"] = round(hits / k, 3)
    result["mrr"] = round(next((1 / (rank + 1) for rank, item in enumerate(retrieved_ids) if item in relevant), 0.0), 3)
    return result


def benchmark_retrieval(cases: list[dict[str, Any]], retrieve_fn) -> dict[str, Any]:
    """Run labeled retrieval cases and aggregate Recall@K, Precision@K and MRR."""
    totals: dict[str, float] = {}
    outputs = []
    for case in cases:
        retrieved = retrieve_fn(case["query"])
        ids = [item["id"] for item in retrieved]
        metrics = retrieval_metrics(ids, case["relevant_ids"])
        outputs.append({"id": case["id"], "query": case["query"], "retrieved_ids": ids, "metrics": metrics})
        for key, value in metrics.items():
            totals[key] = totals.get(key, 0.0) + float(value)
    count = len(cases) or 1
    return {"case_count": len(cases), "aggregate": {key: round(value / count, 3) for key, value in totals.items()}, "cases": outputs}
