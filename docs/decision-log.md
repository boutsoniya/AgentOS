# Engineering Decisions

- Keep the model provider behind a small interface so the orchestration graph is not tied to one vendor SDK.
- Treat uploaded content as untrusted evidence, never as executable instructions.
- Return explicit metrics and empty evidence arrays when retrieval is not yet configured instead of fabricating citations.
- Keep evaluation separate from product responses so reported benchmark numbers can be independently reproduced.
