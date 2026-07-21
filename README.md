# Autonomous Customer Support Copilot

**Project Topic:** Autonomous Customer Support Copilot

**Submitted by:** TINU THOMAS P  
**Registered Email:** tinupadikkala1@gmail.com

---

A Retrieval-Augmented Generation (RAG) system that resolves customer support tickets by
grounding an open-source LLM in a company's own knowledge base. Built entirely with free
and open-source components.

### How it works

```
User query → Intent detection → Router → Retriever (semantic search)
                                         → LLM (cited answer)
                                         → Escalation (if needed)
                                         → Feedback loop
```

### Tech Stack

| Component | Technology |
|-----------|-----------|
| Embeddings | `nomic-embed-text` via Ollama |
| LLM | `qwen3:latest` (or `qwen-local`) via Ollama |
| Vector Store | ChromaDB (persistent, local) |
| API Server | FastAPI + uvicorn |
| Chat UI | Streamlit |
| Database | SQLite (turns, feedback, escalations) |
| Container | Docker + docker-compose |

### Quick Start

```bash
# 1. Make sure Ollama is running and models are available
ollama list  # should show nomic-embed-text and qwen3:latest

# 2. Activate the virtual environment
source .venv/bin/activate

# 3. Build the vector index from your KB documents
PYTHONPATH=src python scripts/build_index.py --kb-root data/kb_raw

# 4. Start the API server (in one terminal)
COPILOT_API_KEY=my-secret PYTHONPATH=src uvicorn copilot.serving.api:app --reload

# 5. Launch the chat UI (in another terminal)
COPILOT_API_KEY=my-secret PYTHONPATH=src streamlit run src/copilot/serving/ui/chat_app.py

# 6. (Optional) Launch the dashboard
COPILOT_API_KEY=my-secret PYTHONPATH=src streamlit run src/copilot/serving/ui/dashboard.py
```

### Docker

```bash
docker compose up -d
```

This starts Ollama, the FastAPI server (port 8000), and the Streamlit UI (port 8501).

### Project Structure

```
src/copilot/
├── config.py              # Typed settings
├── schemas.py             # Shared data models
├── branding.py            # Submission identity
├── ingestion/             # Document loaders + chunker
├── indexing/              # Embedder + vector store + index builder
├── core/                  # RAG pipeline (retriever, intent, generation, guards, escalation)
├── feedback/              # Feedback capture
├── analytics/             # Metrics + reporting
└── serving/               # FastAPI + Streamlit
    ├── api.py
    ├── security.py
    ├── deps.py
    └── ui/
        ├── chat_app.py
        └── dashboard.py

tests/                     # 50+ unit and integration tests
```

### Testing

```bash
PYTHONPATH=src pytest tests/ -v
```
