from __future__ import annotations

from typing import Any


def evaluate_report(report: str, evidence: list[dict[str, Any]], latency_seconds: float, usage: dict[str, int]) -> dict[str, Any]:
    """Compute deterministic production metrics; no fabricated model-judge scores."""
    import re

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
        "evaluation_note": "Faithfulness and retrieval recall require a labeled benchmark or evaluator and are intentionally not fabricated.",
    }
