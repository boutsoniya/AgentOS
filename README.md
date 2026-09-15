# AgentOS

**Autonomous Research & Decision Intelligence Platform**

AgentOS is a production-oriented AI agent system that decomposes complex questions, gathers evidence from multiple sources, reasons over the evidence, verifies claims, and produces citation-backed decision reports.

## What it demonstrates

- Multi-agent orchestration: Planner → Researcher → Analyst → Critic → Synthesizer
- Evidence-first RAG over uploaded documents
- Citation and source tracking
- Structured outputs and deterministic workflow state
- Claim verification and confidence scoring
- Streaming-friendly research events
- Evaluation-ready architecture for retrieval, faithfulness, citation accuracy, latency and cost
- Docker and Render deployment configuration

## Architecture

```text
User Question
     ↓
Research Planner
     ↓
Parallel Research Tasks
 ┌────────┬─────────┬─────────┐
Web      Documents  Analysis
 └────────┴─────────┴─────────┘
     ↓
Evidence Retrieval + Reranking
     ↓
Analyst → Critic → Synthesizer
     ↓
Claim Verification
     ↓
Citation-backed Decision Report
```

## Stack

- Frontend: Next.js, TypeScript, Tailwind CSS
- Backend: FastAPI, Python, Pydantic
- AI: OpenAI API with a provider abstraction
- RAG: PostgreSQL/pgvector-ready data model
- Data: PyMuPDF, pandas, python-docx
- Deployment: Docker + Render

## Local development

```bash
cp .env.example .env
# Add OPENAI_API_KEY for live model execution

cd apps/backend
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

In another terminal:

```bash
cd apps/frontend
npm install
npm run dev
```

Open `http://localhost:3000`.

## API

- `GET /api/health`
- `POST /api/research`
- `GET /api/research/{id}`
- `GET /api/research/{id}/events`
- `POST /api/documents`
- `GET /api/evaluations`

## Engineering principles

AgentOS treats retrieved material as untrusted data, not instructions. Uploaded documents cannot override the system's agent policies. The MVP also keeps model access behind a small provider boundary so models can be changed without rewriting orchestration code.

> Portfolio project by Soniya — built to demonstrate practical AI engineering, RAG, agent orchestration, evaluation and production deployment.
