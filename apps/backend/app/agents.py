"""Agent roles used by the MVP orchestration layer."""

AGENTS = [
    {"name": "Planner", "purpose": "Decompose the decision into research tasks."},
    {"name": "Researcher", "purpose": "Gather and normalize evidence."},
    {"name": "Analyst", "purpose": "Compare evidence, assumptions and trade-offs."},
    {"name": "Critic", "purpose": "Challenge unsupported or weak claims."},
    {"name": "Synthesizer", "purpose": "Produce the final decision brief."},
    {"name": "Verifier", "purpose": "Trace claims back to evidence."},
]
