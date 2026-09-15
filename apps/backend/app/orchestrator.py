from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from .rag import Chunk, rank_chunks


@dataclass
class AgentResult:
    agent: str
    output: Any
    status: str = "completed"


def plan(question: str) -> list[str]:
    """Create explicit, inspectable research tasks from a decision question."""
    return [
        f"Define the decision criteria for: {question}",
        "Find the strongest available evidence for and against the decision",
        "Retrieve the most relevant document evidence",
        "Compare evidence, assumptions, opportunities and risks",
        "Critique unsupported claims and identify evidence gaps",
        "Synthesize a recommendation with confidence and traceable citations",
    ]


def retrieve(question: str, chunks: list[Chunk], k: int = 6) -> list[dict[str, Any]]:
    matches = rank_chunks(question, chunks, k=k)
    return [
        {"id": f"E{i}", "source": c.source, "page": c.page, "chunk_index": c.index, "text": c.text}
        for i, c in enumerate(matches, 1)
    ]


def critique(report: str, evidence: list[dict[str, Any]]) -> dict[str, Any]:
    cited = set(re.findall(r"\[(E\d+)\]", report))
    available = {e["id"] for e in evidence}
    invalid = sorted(cited - available)
    unsupported = [] if cited else ["The report contains no evidence citations."] if evidence else []
    return {
        "citation_ids": sorted(cited),
        "invalid_citations": invalid,
        "unsupported": unsupported,
        "support_score": round(len(cited & available) / max(len(cited), 1), 3),
    }


def verification(report: str, evidence: list[dict[str, Any]]) -> dict[str, Any]:
    critique_result = critique(report, evidence)
    return {
        "verified": not critique_result["invalid_citations"] and not critique_result["unsupported"],
        "citation_accuracy": critique_result["support_score"],
        "checks": [
            "citation_ids_resolve_to_retrieved_evidence",
            "no_unknown_citation_ids",
            "evidence_presence_checked",
        ],
    }


def pipeline(question: str, chunks: list[Chunk]) -> dict[str, Any]:
    tasks = plan(question)
    evidence = retrieve(question, chunks)
    return {
        "tasks": tasks,
        "agents": [
            AgentResult("Planner", tasks).__dict__,
            AgentResult("Researcher", {"evidence_count": len(evidence)}).__dict__,
            AgentResult("Retriever", evidence).__dict__,
        ],
        "evidence": evidence,
    }
