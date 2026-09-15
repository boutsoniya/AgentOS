# AgentOS Architecture

AgentOS separates orchestration, model access, retrieval and presentation so each layer can be tested independently.

```text
Next.js UI
   │
   ▼
FastAPI API ──► Planner ──► Research tasks
   │                         │
   │                         ├── Web evidence (next phase)
   │                         └── Documents (next phase)
   ▼
Synthesis / Critic / Verification
   │
   ▼
Citation-backed report + evaluation telemetry
```

The current MVP implements the API contract, structured planning, model provider boundary, upload validation, research runs and the production-facing UI. PostgreSQL/pgvector, external web search, asynchronous workers and the full benchmark harness are the next implementation layer.
