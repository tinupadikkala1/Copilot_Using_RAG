# Autonomous Customer Support Copilot

**Project Topic:** Autonomous Customer Support Copilot

**Submitted by:** TINU THOMAS P  
**Registered Email:** tinupadikkala1@gmail.com

---

> A Retrieval-Augmented Generation (RAG) system that resolves customer support tickets by
> grounding an open-source LLM in a company's own knowledge base. Built entirely with free
> and open-source components.

## Table of Contents

1. [What This Project Does](#-what-this-project-does)
2. [Tech Stack](#-tech-stack)
3. [How It Works](#-how-it-works)
4. [Prerequisites — All Operating Systems](#-prerequisites--all-operating-systems)
5. [Installation Guides by OS](#-installation-guides-by-os)
   - [Ubuntu / Debian (Linux)](#ubuntu--debian-linux)
   - [macOS](#macos)
   - [Windows](#windows)
6. [Step-by-Step Setup (All OS)](#-step-by-step-setup-all-os)
7. [Running the Application](#-running-the-application)
8. [Using the File Upload Feature](#-using-the-file-upload-feature)
9. [Using Docker](#-using-docker)
10. [Testing](#-testing)
11. [API Endpoints](#-api-endpoints)
12. [Project Structure](#-project-structure)
13. [SPARKIIT Submission Notes](#-sparkiit-submission-notes)

---

## 🤖 What This Project Does

This system is an **autonomous customer support copilot** that:

- **Ingests** your company's knowledge base (Markdown, PDF, HTML, TXT, CSV files)
- **Embeds** them into a vector database (ChromaDB) using `nomic-embed-text`
- **Answers** user questions by finding the most relevant content and generating a cited response using `qwen3:latest`
- **Detects intent** — knows if you're asking about billing, technical issues, account management, or just greeting
- **Escalates** to a human agent when it's unsure or when sensitive topics come up
- **Learns** from feedback — captures 👍/👎 ratings

**All models run locally on your machine. No internet required after setup. No paid APIs.**

---

## 🛠️ Tech Stack

| Component | Technology | Why |
|-----------|-----------|-----|
| **Embedding Model** | `nomic-embed-text` (via Ollama) | 768-dim vectors, CPU-friendly, 274 MB |
| **LLM** | `qwen3:latest` (via Ollama) | 2.5 GB local model, fast on consumer hardware |
| **Fallback LLM** | `qwen-local:latest` | 1.1 GB lighter alternative |
| **Vector Database** | ChromaDB | Persistent, local, free |
| **API Server** | FastAPI + uvicorn | Fast, modern Python web framework |
| **Chat UI** | Streamlit | Interactive web interface |
| **Dashboard** | Streamlit | Real-time metrics display |
| **Database** | SQLite | Zero-config, file-based |
| **Containerization** | Docker + docker-compose | Optional, for easy deployment |
| **CI/CD** | GitHub Actions | Automated testing on push |

---

## 🔄 How It Works

```ascii
User types a question
         │
         ▼
  ┌──────────────┐
  │ 1. Intent    │  Classifies: billing / technical / account /
  │  Detection   │  how_to / greeting / human_agent
  └──────┬───────┘
         │
         ▼
  ┌──────────────┐
  │ 2. Router    │  Routes to: RAG answer / Escalate / Smalltalk
  └──────┬───────┘
         │
         ▼
  ┌──────────────┐
  │ 3. Retriever │  Searches ChromaDB for top-5 most relevant chunks
  └──────┬───────┘
         │
         ▼
  ┌──────────────┐
  │ 4. LLM       │  Generates cited answer using qwen3:latest
  │  Generation  │  With [1][2] inline citations
  └──────┬───────┘
         │
         ▼
  ┌──────────────┐
  │ 5. Guards    │  Checks answer is grounded in retrieved context
  └──────┬───────┘
         │
    ┌────┴────┐
    │         │
    ▼         ▼
 Answer    Escalate
 (cited)   (ticket created)
    │         │
    ▼         ▼
 Feedback    Human agent
 (👍/👎)     queue
```

---

## 📋 Prerequisites — All Operating Systems

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| **Python** | 3.11 | 3.12 |
| **RAM** | 8 GB | 16 GB |
| **Disk space** | 10 GB | 20 GB |
| **OS** | Linux, macOS, or Windows | Linux (best Ollama support) |
| **Ollama** | 0.31+ | latest |

### Dependencies That Get Installed (18 packages)

| Package | Version | Purpose |
|---------|---------|---------|
| `fastapi` | 0.115+ | REST API framework |
| `uvicorn[standard]` | 0.32+ | ASGI server |
| `streamlit` | 1.39+ | Web UI framework |
| `httpx` | 0.28+ | HTTP client (calls Ollama) |
| `pydantic` | 2.9+ | Data validation |
| `pydantic-settings` | 2.6+ | Environment config |
| `chromadb` | 0.5+ | Vector database |
| `numpy` | 2.1+ | Numerical operations |
| `pypdf` | 5.0+ | PDF parsing |
| `beautifulsoup4` | 4.12+ | HTML parsing |
| `python-multipart` | 0.0.12+ | File upload support |
| `pytest` | 8.3+ | Testing (dev) |
| `black` | 24.10+ | Code formatter (dev) |
| `ruff` | 0.7+ | Linter (dev) |
| `mypy` | 1.13+ | Type checker (dev) |
| `flake8` | 7.1+ | Linter (dev) |

### Models Downloaded by Ollama (~2.8 GB total)

| Model | Size | Purpose |
|-------|------|---------|
| `nomic-embed-text:latest` | 274 MB | Text embeddings |
| `qwen3:latest` | 2.5 GB | Main LLM for answers |
| `qwen-local:latest` | 1.1 GB | Lighter fallback LLM |

---

## 🖥️ Installation Guides by OS

### Ubuntu / Debian (Linux)

#### 1. Install Python 3.12

```bash
# Update package list
sudo apt update && sudo apt upgrade -y

# Install Python 3.12 and pip
sudo apt install -y python3 python3-pip python3-venv

# Verify
python3 --version   # Should show Python 3.11 or 3.12
pip3 --version
```

#### 2. Install Ollama

```bash
# Official one-line install
curl -fsSL https://ollama.com/install.sh | sh

# Verify
ollama --version    # Should show 0.31.x or later
```

#### 3. Pull the Required Models

```bash
# Start Ollama server (background)
ollama serve &

# Pull models (this downloads ~2.8 GB total)
ollama pull nomic-embed-text   # 274 MB — embeddings
ollama pull qwen3:latest       # 2.5 GB — main LLM
ollama pull qwen-local:latest  # 1.1 GB — lighter fallback (optional)

# Verify
ollama list
# Should show:
# nomic-embed-text:latest   274 MB
# qwen3:latest              2.5 GB
# qwen-local:latest         1.1 GB
```

#### 4. Install Git

```bash
sudo apt install -y git
git --version
```

---

### macOS

#### 1. Install Python 3.12

```bash
# Option A: Using Homebrew (recommended)
brew install python@3.12

# Option B: Download from python.org
# https://www.python.org/downloads/

# Verify
python3 --version
```

#### 2. Install Ollama

```bash
# Option A: Using Homebrew
brew install ollama

# Option B: Download from https://ollama.com/download/mac
# Double-click the .dmg file and drag Ollama to Applications

# Start Ollama (Run the Ollama app from Applications folder,
# or via terminal):
ollama serve &
```

#### 3. Pull Models

```bash
# In a new terminal (keep Ollama running in the first one)
ollama pull nomic-embed-text
ollama pull qwen3:latest
ollama pull qwen-local:latest

# Verify
ollama list
```

#### 4. Install Git

```bash
# Comes pre-installed on macOS, or via Homebrew:
brew install git
git --version
```

---

### Windows

#### 1. Install Python 3.12

1. Go to https://www.python.org/downloads/
2. Download **Python 3.12.x** for Windows
3. Run the installer
4. **IMPORTANT:** Check **"Add Python to PATH"** at the bottom of the installer
5. Click **Install Now**

```powershell
# Verify in Command Prompt or PowerShell
python --version
pip --version
```

#### 2. Install Ollama

1. Go to https://ollama.com/download/windows
2. Download and run the **OllamaSetup.exe** installer
3. Ollama will start automatically as a background service
4. The Ollama icon appears in the system tray

#### 3. Pull Models

```powershell
# Open Command Prompt or PowerShell
ollama pull nomic-embed-text
ollama pull qwen3:latest
ollama pull qwen-local:latest

# Verify
ollama list
```

#### 4. Install Git

1. Go to https://git-scm.com/download/win
2. Download and run the installer (default options are fine)
3. After installation, open **Git Bash** (recommended) or **Command Prompt**

```powershell
git --version
```

#### 5. Windows-Specific Notes

- **Path issues on Windows:** The project directory contains parentheses in the path (`Copilot(By:TINU THOMAS P)`). On Windows, always use double quotes when navigating:
  ```powershell
  cd "C:\Users\...\Copilot(By:TINU THOMAS P)"
  ```
- **Virtual environment activation:**
  ```powershell
  python -m venv venv
  venv\Scripts\activate
  ```
- **PYTHONPATH on Windows:** Use `set` instead of `export`:
  ```powershell
  set PYTHONPATH=src
  ```
- Or use Git Bash for a Unix-like terminal experience.

---

## 🚀 Step-by-Step Setup (All OS)

After installing Python, Ollama, and Git, follow these steps:

### Step 1: Clone the Repository

```bash
# Linux / macOS
git clone https://github.com/tinupadikkala1/Copilot_Using_RAG.git
cd "Copilot(By:TINU THOMAS P)"

# Windows (Git Bash)
git clone https://github.com/tinupadikkala1/Copilot_Using_RAG.git
cd "Copilot(By:TINU THOMAS P)"
```

### Step 2: Create a Virtual Environment

```bash
# Linux / macOS
python3 -m venv venv
source venv/bin/activate

# Windows (Command Prompt)
python -m venv venv
venv\Scripts\activate

# Windows (PowerShell)
python -m venv venv
venv\Scripts\Activate.ps1
```

### Step 3: Upgrade pip

```bash
# All OS
python -m pip install --upgrade pip
```

### Step 4: Install All Dependencies

```bash
# All OS — this installs all 18 packages listed above
pip install -e .
```

This installs everything in one command. It may take a few minutes.

**Alternative — install without Poetry build system:**
```bash
pip install fastapi uvicorn[standard] streamlit httpx pydantic pydantic-settings chromadb numpy pypdf beautifulsoup4 python-multipart
pip install pytest pytest-cov black ruff mypy flake8  # dev tools
```

### Step 5: Verify Installation

```bash
python -c "
import fastapi, uvicorn, streamlit, httpx, pydantic, chromadb, numpy
print('✅ All libraries installed successfully!')
"
```

### Step 6: Ensure Ollama Models Are Available

```bash
# Start Ollama if not already running
ollama serve &

# Verify models
ollama list
```

Expected output:
```
NAME                     ID              SIZE      MODIFIED
nomic-embed-text:latest  xxx             274 MB    ...  
qwen3:latest             xxx             2.5 GB    ...  
qwen-local:latest        xxx             1.1 GB    ...  
```

> **If models are missing**, pull them:
> ```bash
> ollama pull nomic-embed-text
> ollama pull qwen3:latest
> ```

### Step 7: Run the Tests

```bash
# Verify everything works with the test suite
PYTHONPATH=src python -m pytest tests/ -v --tb=short
```

You should see **68 passed, 14 skipped** (the skipped tests need Ollama running).

---

## ▶️ Running the Application

You need **three terminals** to run the full stack (or use Docker for a single command).

### Terminal 1: Start the API Server

```bash
cd "/home/user/Desktop/SPARKIIT_ideas/Copilot(By:TINU THOMAS P)"  # Adjust path
source venv/bin/activate  # or venv\Scripts\activate on Windows

COPILOT_API_KEY=my-secret-key PYTHONPATH=src uvicorn copilot.serving.api:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at **http://localhost:8000**

### Terminal 2: Build the Vector Index

First, place your knowledge base documents in `data/kb_raw/`. Then:

```bash
# Using the mock KB for testing
PYTHONPATH=src python scripts/build_index.py --kb-root tests/fixtures/mock_kb

# Or using your own documents
PYTHONPATH=src python scripts/build_index.py --kb-root data/kb_raw
```

**You only need to do this once** (or whenever you add/change documents).

### Terminal 3: Launch the Chat UI

```bash
COPILOT_API_KEY=my-secret-key COPILOT_API_URL=http://localhost:8000 \
  PYTHONPATH=src streamlit run src/copilot/serving/ui/chat_app.py
```

Open **http://localhost:8501** in your browser.

### (Optional) Terminal 4: Launch the Dashboard

```bash
COPILOT_API_KEY=my-secret-key COPILOT_API_URL=http://localhost:8000 \
  PYTHONPATH=src streamlit run src/copilot/serving/ui/dashboard.py
```

Open **http://localhost:8502** in your browser.

### Quick Test (API Only)

```bash
# Test the API without the UI
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -H "x-api-key: my-secret-key" \
  -d '{"message": "How do I reset my password?"}'
```

Expected response:
```json
{
  "answer": "To reset your password, click 'Forgot password' on the login page. [1]",
  "citations": [{"marker": 1, "title": "Password Reset", ...}],
  "intent": "how_to",
  "escalated": false,
  "session_id": "...",
  ...
}
```

---

## 📤 Using the File Upload Feature

You don't need to use the terminal to build the index. The **Upload KB** page lets you upload files through the browser.

### Via the Web UI

1. Open the chat UI at **http://localhost:8501**
2. Click **"📤 Upload KB"** in the sidebar
3. Drag & drop your files (`.md`, `.html`, `.pdf`, `.txt`, `.csv`) or click to browse
4. The files list will appear — you can see file names, types, and sizes
5. Click **"🚀 Build/Refresh Vector Index"**
6. Wait for the status to show "Index built: X chunks from Y documents"
7. Switch back to **"💬 Chat"** and ask questions about your documents

### Via the API

```bash
# Upload files programmatically
curl -X POST http://localhost:8000/upload \
  -H "x-api-key: my-secret-key" \
  -F "files=@my_document.md" \
  -F "files=@manual.pdf"

# Then build the index
curl -X POST http://localhost:8000/upload/build \
  -H "x-api-key: my-secret-key"
```

### Supported File Formats

| Format | Extension | Notes |
|--------|-----------|-------|
| Markdown | `.md` | Best format — preserves structure |
| HTML | `.html`, `.htm` | Strips scripts/styles automatically |
| PDF | `.pdf` | Extracts text from all pages |
| Plain text | `.txt` | Simple, clean |
| CSV | `.csv` | Converts to pipe-delimited text |

**Max file size:** 20 MB per file. Larger files are rejected.

---

## 🐳 Using Docker

Docker bundles everything into containers — no need to install Python or dependencies manually.

### Prerequisites

- Install Docker: https://docs.docker.com/get-docker/
- Install Docker Compose: https://docs.docker.com/compose/install/

### Quick Start

```bash
# Start all services
docker compose up -d

# Check status
docker compose ps

# View logs
docker compose logs -f
```

### What Gets Started

| Service | URL | Description |
|---------|-----|-------------|
| **Ollama** | http://localhost:11434 | LLM server |
| **API** | http://localhost:8000 | FastAPI backend |
| **Streamlit** | http://localhost:8501 | Chat UI |

### Notes on Docker

- The first run will start an empty Ollama container. You need to pull models inside it:
  ```bash
  docker exec -it copilot-ollama ollama pull nomic-embed-text
  docker exec -it copilot-ollama ollama pull qwen3:latest
  ```
- Data is persisted in a Docker volume (`ollama_data`) so models survive restarts.
- The `data/` directory is mounted as a bind volume — your uploaded KB files persist.

---

## 🧪 Testing

```bash
# Run all tests (Ollama not required — Ollama tests auto-skip)
PYTHONPATH=src python -m pytest tests/ -v

# Run with coverage report
PYTHONPATH=src python -m pytest tests/ --cov=src/copilot/ --cov-report=term

# Run only non-Ollama tests (faster)
PYTHONPATH=src python -m pytest tests/ -v -k "not ollama"

# Run a specific test file
PYTHONPATH=src python -m pytest tests/test_api.py -v
```

### Test Summary

| Test File | Count | What's Covered |
|-----------|-------|----------------|
| `test_chunker.py` | 17 | Chunk size, overlap, deduplication, hash integrity |
| `test_retrieval.py` | 11 | Embedder, ChromaStore CRUD, index builder |
| `test_guards.py` | 8 | Groundedness scoring, injection defense |
| `test_escalation.py` | 7 | All escalation triggers, edge cases |
| `test_route.py` | 8 | RAG / escalate / smalltalk routing |
| `test_intent.py` | 5 | Intent classification, sensitive intents |
| `test_pipeline.py` | 3 | Full pipeline integration (greeting, RAG, escalate) |
| `test_api.py` | 18 | Auth, validation, chat, feedback, metrics, upload |
| **Total** | **82** | **68 pass, 14 skip (Ollama required)** |

---

## 📡 API Endpoints

| Method | Endpoint | Auth Required | Description |
|--------|----------|---------------|-------------|
| `GET` | `/` | No | Landing page — project identity |
| `GET` | `/healthz` | No | Health check |
| `POST` | `/chat` | Yes | Ask a support question |
| `POST` | `/feedback` | Yes | Record 👍/👎 feedback |
| `GET` | `/metrics` | Yes | Get deflection rate, CSAT, latency |
| `POST` | `/upload` | Yes | Upload KB documents |
| `POST` | `/upload/build` | Yes | Build/refresh vector index |

**Authentication:** Pass your API key in the `X-API-Key` header:
```bash
curl -H "x-api-key: your-key" http://localhost:8000/healthz
```

### Quick API Reference

```bash
# Health check (no key required)
curl http://localhost:8000/healthz

# Landing page (no key required)
curl http://localhost:8000/

# Ask a question
curl -X POST http://localhost:8000/chat \
  -H "x-api-key: my-secret-key" \
  -H "Content-Type: application/json" \
  -d '{"message": "How do I export my data?"}'

# Send feedback
curl -X POST http://localhost:8000/feedback \
  -H "x-api-key: my-secret-key" \
  -H "Content-Type: application/json" \
  -d '{"session_id": "...", "query": "...", "answer": "...", "rating": "up"}'

# Get metrics
curl http://localhost:8000/metrics \
  -H "x-api-key: my-secret-key"

# Upload a document
curl -X POST http://localhost:8000/upload \
  -H "x-api-key: my-secret-key" \
  -F "files=@document.md"

# Build the vector index
curl -X POST http://localhost:8000/upload/build \
  -H "x-api-key: my-secret-key"
```

---

## 📁 Project Structure

```
├── pyproject.toml              # Python project config + 18 dependencies
├── README.md                   # This file
├── .gitignore                  # Ignored files (.venv, .freebuff, __pycache__, *.db)
├── Dockerfile                  # Multi-stage Docker build
├── docker-compose.yml          # Ollama + API + Streamlit containers
├── configs/                    # (Optional) logging.json, settings.toml
├── data/
│   ├── kb_raw/                 # 📤 Your uploaded KB documents go here
│   └── chroma/                 # Vector index (auto-generated)
├── scripts/
│   └── build_index.py          # CLI to build/refresh vector index
├── .github/workflows/
│   └── ci.yml                  # GitHub Actions CI pipeline
├── src/copilot/
│   ├── config.py               # Typed settings (Ollama URLs, model names, paths)
│   ├── schemas.py              # Shared data models (Chunk, Citation, ChatResponse)
│   ├── branding.py             # SPARKIIT landing page with identity
│   ├── logging_setup.py        # Structured JSON logging
│   ├── ingestion/
│   │   ├── loaders.py          # Parse .md / .html / .pdf / .txt / .csv
│   │   └── chunker.py          # Split text into chunks + deduplicate
│   ├── indexing/
│   │   ├── embedder.py         # Ollama nomic-embed-text wrapper
│   │   ├── vector_store.py     # ChromaDB backend
│   │   └── index_builder.py    # ETL: load → chunk → embed → upsert
│   ├── core/
│   │   ├── retriever.py        # Semantic search (top-k)
│   │   ├── intent.py           # Intent classifier (6 categories)
│   │   ├── router.py           # Route: RAG / escalate / smalltalk
│   │   ├── prompt.py           # Citation-grounded prompt templates
│   │   ├── generation.py       # Ollama qwen3:latest chat client
│   │   ├── guards.py           # Groundedness score + injection defense
│   │   ├── escalation.py       # Auto-escalation rules + ticket creation
│   │   └── pipeline.py         # Full orchestrator
│   ├── feedback/
│   │   └── store.py            # 👍/👎 feedback persistence
│   ├── analytics/
│   │   ├── db.py               # SQLite schema
│   │   ├── metrics.py          # Per-turn metrics + latency
│   │   └── reporting.py        # Deflection rate + CSAT
│   └── serving/
│       ├── api.py              # FastAPI server (7 endpoints)
│       ├── security.py         # API key auth + rate limiting
│       ├── deps.py             # Dependency injection (singletons)
│       └── ui/
│           ├── chat_app.py     # Streamlit chat interface
│           ├── upload.py       # 📤 File upload page
│           └── dashboard.py    # Metrics dashboard
└── tests/
    ├── conftest.py             # Shared test fixtures
    ├── fixtures/mock_kb/       # 3 test documents
    ├── test_chunker.py         # 17 tests
    ├── test_retrieval.py       # 11 tests
    ├── test_intent.py          # 5 tests
    ├── test_guards.py          # 8 tests
    ├── test_escalation.py      # 7 tests
    ├── test_route.py           # 8 tests
    ├── test_pipeline.py        # 3 tests
    └── test_api.py             # 18 tests (7 endpoints)
```

---

## 📝 SPARKIIT Submission Notes

### Submission Identity

The project landing page displays the required identity information:
- **Project Topic:** Autonomous Customer Support Copilot
- **Full Name:** TINU THOMAS P
- **Registered Email:** tinupadikkala1@gmail.com

This appears on:
1. The Streamlit chat UI (top of every page)
2. The FastAPI root route (`GET /`)
3. The `README.md` header (above)

### Language

The entire project is written in **Python 3.11+** — all source code is in `.py` files with proper `snake_case` naming.

### Budget

All components are **free and open-source**. No paid APIs, no paid hosting, no paid services.
- Ollama runs locally (free)
- ChromaDB is local (free)
- FastAPI / Streamlit are open-source (free)
- All models are open-source and run on your own machine (free)
