#!/usr/bin/env bash
# =============================================================================
#  🚀 Autonomous Customer Support Copilot — One-Click Launcher
#
#  This script does EVERYTHING:
#    1. Checks Python, dependencies, and Ollama
#    2. Starts Ollama if not running
#    3. Creates a virtual environment and installs dependencies (if needed)
#    4. Builds the vector index from your documents
#    5. Starts the API server
#    6. Opens the Streamlit Chat UI + Dashboard
#
#  Usage:
#    chmod +x start.sh
#    ./start.sh
#
#  Optional flags:
#    --kb-dir PATH     Path to your knowledge base documents (default: data/kb_raw)
#    --api-key KEY     API key for authentication (default: copilot-demo-key)
#    --port PORT       Port for the API server (default: 8000)
#    --help            Show this help message
# =============================================================================

set -uo pipefail

# ─────────────────────────────────────────────────────────────────────────────
#  Config
# ─────────────────────────────────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

API_KEY="${COPILOT_API_KEY:-copilot-demo-key}"
API_PORT=8000
KB_DIR=""
STREAMLIT_PORT=8501

# Track child process PIDs for cleanup (declare early — used in Step 5+6).
_ALL_PIDS=()
_CLEANING=false

# ─────────────────────────────────────────────────────────────────────────────
#  Colors for output
# ─────────────────────────────────────────────────────────────────────────────

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

info()  { echo -e "${BLUE}[INFO]${NC}  $1"; }
ok()    { echo -e "${GREEN}[OK]${NC}    $1"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $1"; }
err()   { echo -e "${RED}[ERROR]${NC} $1"; }
step()  { echo; echo -e "${CYAN}═══════════════════════════════════════════════════${NC}"; echo -e "${CYAN}  $1${NC}"; echo -e "${CYAN}═══════════════════════════════════════════════════${NC}"; echo; }

# ─────────────────────────────────────────────────────────────────────────────
#  Help
# ─────────────────────────────────────────────────────────────────────────────

if [[ "${1:-}" == "--help" ]] || [[ "${1:-}" == "-h" ]]; then
    echo ""
    echo "  🚀 Autonomous Customer Support Copilot — One-Click Launcher"
    echo ""
    echo "  Usage:  ./start.sh [OPTIONS]"
    echo ""
    echo "  Options:"
    echo "    --kb-dir PATH   Path to knowledge base documents (default: data/kb_raw)"
    echo "    --api-key KEY   API key for authentication (default: copilot-demo-key)"
    echo "    --port PORT     Port for API server (default: 8000)"
    echo "    --help          Show this help"
    echo ""
    echo "  Environment variables:"
    echo "    COPILOT_API_KEY  API key (overrides --api-key)"
    echo ""
    exit 0
fi

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --kb-dir)     KB_DIR="$2"; shift 2 ;;
        --api-key)    API_KEY="$2"; shift 2 ;;
        --port)       API_PORT="$2"; shift 2 ;;
        *)            err "Unknown option: $1"; exit 1 ;;
    esac
done

# ─────────────────────────────────────────────────────────────────────────────
#  Step 1: Check prerequisites
# ─────────────────────────────────────────────────────────────────────────────

step "Step 1/6 — Checking prerequisites"

# Python
PYTHON=""
for cmd in python3 python; do
    if command -v "$cmd" &>/dev/null; then
        PYTHON="$cmd"
        break
    fi
done

if [[ -z "$PYTHON" ]]; then
    err "Python 3 not found. Please install Python 3.11+."
    exit 1
fi

PY_VERSION=$("$PYTHON" --version 2>&1 | grep -oP '\d+\.\d+')
info "Python: $("$PYTHON" --version 2>&1)"
info "OS:     $(uname -s) $(uname -r)"

# Check Python version (need 3.11+)
PY_MAJOR=$(echo "$PY_VERSION" | cut -d. -f1)
PY_MINOR=$(echo "$PY_VERSION" | cut -d. -f2)
if [[ "$PY_MAJOR" -lt 3 ]] || { [[ "$PY_MAJOR" -eq 3 ]] && [[ "$PY_MINOR" -lt 11 ]]; }; then
    warn "Python 3.11+ recommended. You have $("$PYTHON" --version 2>&1)"
fi

ok "Prerequisites check complete"

# ─────────────────────────────────────────────────────────────────────────────
#  Step 2: Check / start Ollama
# ─────────────────────────────────────────────────────────────────────────────

step "Step 2/6 — Setting up Ollama"

OLLAMA_RUNNING=false
if command -v ollama &>/dev/null; then
    if curl -s http://localhost:11434/api/tags &>/dev/null; then
        OLLAMA_RUNNING=true
        ok "Ollama is already running"
    else
        info "Starting Ollama in background..."
        ollama serve &
        OLLAMA_PID=$!
        sleep 3
        if curl -s http://localhost:11434/api/tags &>/dev/null; then
            ok "Ollama started (PID: $OLLAMA_PID)"
            OLLAMA_RUNNING=true
        else
            warn "Could not connect to Ollama. Make sure it's installed and run 'ollama serve' manually."
        fi
    fi
else
    warn "Ollama not found. Please install from https://ollama.com"
fi

if $OLLAMA_RUNNING; then
    # Check for required models
    MODELS=$(curl -s http://localhost:11434/api/tags 2>/dev/null | grep -oP '"name":\s*"[^"]+' | cut -d'"' -f4 || echo "")
    info "Available models:"
    echo "$MODELS" | while read -r m; do echo "    - $m"; done

    # Check for nomic-embed-text
    if ! echo "$MODELS" | grep -q "nomic-embed-text"; then
        info "Pulling embedding model (nomic-embed-text, 274 MB)..."
        ollama pull nomic-embed-text
        ok "Embedding model ready"
    else
        ok "Embedding model (nomic-embed-text) available"
    fi

    # Check for qwen-local
    if ! echo "$MODELS" | grep -q "qwen-local"; then
        info "Pulling LLM model (qwen-local, 1.1 GB)..."
        ollama pull qwen-local:latest
        ok "LLM model ready"
    else
        ok "LLM model (qwen-local) available"
    fi
fi

ok "Ollama setup complete"

# ─────────────────────────────────────────────────────────────────────────────
#  Step 3: Install Python dependencies
# ─────────────────────────────────────────────────────────────────────────────

step "Step 3/6 — Installing Python dependencies"

# Create virtual environment if it doesn't exist
VENV_DIR="/tmp/copilot-venv"

if [[ ! -f "$VENV_DIR/bin/python3" ]]; then
    info "Creating virtual environment at $VENV_DIR..."
    "$PYTHON" -m venv "$VENV_DIR"
    ok "Virtual environment created"

    info "Installing dependencies (this may take a few minutes)..."
    "$VENV_DIR/bin/pip" install --quiet --upgrade pip 2>/dev/null
    "$VENV_DIR/bin/pip" install --quiet \
        fastapi uvicorn[standard] streamlit httpx pydantic pydantic-settings \
        chromadb numpy pypdf beautifulsoup4 python-multipart \
        pytest 2>&1 | tail -1
    ok "Dependencies installed"
else
    ok "Virtual environment already exists at $VENV_DIR"
    # Check critical packages are installed
    if ! "$VENV_DIR/bin/python" -c "import fastapi" 2>/dev/null; then
        info "Reinstalling missing dependencies..."
        "$VENV_DIR/bin/pip" install --quiet \
            fastapi uvicorn streamlit httpx pydantic pydantic-settings \
            chromadb numpy pypdf beautifulsoup4 python-multipart
        ok "Dependencies installed"
    fi
fi

PYTHON="$VENV_DIR/bin/python"
PIP="$VENV_DIR/bin/pip"

ok "Python dependencies ready"

# ─────────────────────────────────────────────────────────────────────────────
#  Step 4: Build the vector index
# ─────────────────────────────────────────────────────────────────────────────

step "Step 4/6 — Building vector index"

# Use user-provided KB directory, or default
if [[ -z "$KB_DIR" ]]; then
    # Check if data/kb_raw has files
    if [[ -d "data/kb_raw" ]] && [[ "$(find data/kb_raw -type f 2>/dev/null | wc -l)" -gt 0 ]]; then
        KB_DIR="data/kb_raw"
        info "Using existing documents from: $KB_DIR"
    else
        # Check if mock KB exists for demo
        if [[ -d "tests/fixtures/mock_kb" ]]; then
            KB_DIR="tests/fixtures/mock_kb"
            info "No user documents found. Using demo mock KB: $KB_DIR"
            info "You can upload your own documents later via the Upload KB page in the UI."
        else
            mkdir -p "data/kb_raw"
            KB_DIR="data/kb_raw"
            info "No documents found. Created empty directory: $KB_DIR"
            info "Upload documents via the Upload KB page after the app starts."
        fi
    fi
fi

if [[ -d "$KB_DIR" ]] && [[ "$(find "$KB_DIR" -type f 2>/dev/null | wc -l)" -gt 0 ]]; then
    info "Building index from: $KB_DIR"
    mkdir -p data/chroma

    # Build index — use relative PYTHONPATH (avoids colon-in-pathname parsing issues).
    if PYTHONPATH="src" $PYTHON scripts/build_index.py --kb-root "$KB_DIR" --persist-dir data/chroma 2>&1; then
        ok "Vector index built successfully"
    else
        warn "Index build had issues. You can rebuild from the Upload KB page."
    fi
else
    warn "No documents to index at: $KB_DIR"
    warn "Upload documents via the Upload KB page after the app starts."
    mkdir -p "data/kb_raw"
fi

ok "Vector index ready"

# ─────────────────────────────────────────────────────────────────────────────
#  Step 5: Start the API server (background)
# ─────────────────────────────────────────────────────────────────────────────

step "Step 5/6 — Starting API server"

# Kill any existing API server on our port
_kill_port() {
    local port="$1" pid
    if command -v lsof &>/dev/null; then
        pid=$(lsof -ti:"$port" 2>/dev/null)
    elif command -v ss &>/dev/null; then
        pid=$(ss -tlnp 2>/dev/null | grep ":$port " | grep -oP 'pid=\K\d+' | head -1)
    elif command -v fuser &>/dev/null; then
        pid=$(fuser "$port/tcp" 2>/dev/null | head -1)
    fi
    if [[ -n "${pid:-}" ]]; then
        warn "Port $port is in use (PID: $pid). Stopping old process..."
        kill "$pid" 2>/dev/null || true
        sleep 1
    fi
}
_kill_port "$API_PORT"

export COPILOT_API_KEY="$API_KEY"

info "Starting FastAPI server on port $API_PORT..."

# Pass PYTHONPATH inline (relative path avoids colon-in-directory-name issues).
PYTHONPATH="src" $PYTHON -m uvicorn copilot.serving.api:app \
    --host 0.0.0.0 --port "$API_PORT" \
    --log-level warning \
    &
API_PID=$!
_ALL_PIDS+=("$API_PID")    # Wait for API to be ready
info "Waiting for API server to be ready..."
API_READY=false
for i in $(seq 1 30); do
    if curl -s "http://localhost:$API_PORT/healthz" &>/dev/null; then
        ok "API server is running (PID: $API_PID) at http://localhost:$API_PORT"
        API_READY=true
        break
    fi
    sleep 1
done

if ! $API_READY; then
    err "API server failed to start within 30 seconds."
    err "Check if port $API_PORT is available or run manually:"
    err "  COPILOT_API_KEY=$API_KEY PYTHONPATH=src $PYTHON -m uvicorn copilot.serving.api:app --host 0.0.0.0 --port $API_PORT"
    info "Continuing anyway — you can start the chat UI manually with:"
    info "  COPILOT_API_KEY=$API_KEY COPILOT_API_URL=http://localhost:$API_PORT PYTHONPATH=src $PYTHON -m streamlit run src/copilot/serving/ui/chat_app.py"
fi

# ─────────────────────────────────────────────────────────────────────────────
#  Step 6: Launch the Streamlit UI
# ─────────────────────────────────────────────────────────────────────────────

step "Step 6/6 — Launching Chat UI"

export COPILOT_API_URL="http://localhost:$API_PORT"
export COPILOT_API_KEY="$API_KEY"

info "Starting all Streamlit UIs..."
info "  💬 Chat:      http://localhost:$STREAMLIT_PORT"
info "  📤 Upload KB: http://localhost:$((STREAMLIT_PORT + 2))"
info "  📊 Dashboard: http://localhost:$((STREAMLIT_PORT + 1))"
info "  📡 API:       http://localhost:$API_PORT"
info "  📖 API Docs:  http://localhost:$API_PORT/docs"
echo ""

# Function to cleanup on exit (guard prevents double-run on SIGINT + EXIT)
cleanup() {
    $_CLEANING && return 0
    _CLEANING=true
    local exit_code=$?
    echo ""
    warn "Shutting down..."
    for pid in "${_ALL_PIDS[@]}"; do
        if kill "$pid" 2>/dev/null; then
            ok "Stopped process (PID: $pid)"
        fi
    done
    ok "All services stopped. Goodbye!"
    exit "$exit_code"
}

trap cleanup EXIT INT TERM

# Launch all UIs — pass PYTHONPATH inline (relative path avoids colon issues).
info "Starting Streamlit Chat UI (PID tracking)..."
PYTHONPATH="src" $PYTHON -m streamlit run src/copilot/serving/ui/chat_app.py \
    --server.port "$STREAMLIT_PORT" \
    --server.headless true \
    --browser.gatherUsageStats false \
    2>&1 &
CHAT_PID=$!
_ALL_PIDS+=("$CHAT_PID")

sleep 3

info "Starting Upload KB page..."
PYTHONPATH="src" $PYTHON -m streamlit run src/copilot/serving/ui/upload.py \
    --server.port "$((STREAMLIT_PORT + 2))" \
    --server.headless true \
    --browser.gatherUsageStats false \
    2>&1 &
UPLOAD_PID=$!
_ALL_PIDS+=("$UPLOAD_PID")

sleep 2

info "Starting Dashboard..."
PYTHONPATH="src" $PYTHON -m streamlit run src/copilot/serving/ui/dashboard.py \
    --server.port "$((STREAMLIT_PORT + 1))" \
    --server.headless true \
    --browser.gatherUsageStats false \
    2>&1 &
DASH_PID=$!
_ALL_PIDS+=("$DASH_PID")

sleep 2

# ─────────────────────────────────────────────────────────────────────────────
#  All done — show summary
# ─────────────────────────────────────────────────────────────────────────────

echo ""
echo -e "${GREEN}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║                                                              ║${NC}"
echo -e "${GREEN}║   🚀  Copilot is running!                                    ║${NC}"
echo -e "${GREEN}║                                                              ║${NC}"
echo -e "${GREEN}║   💬 Chat:          http://localhost:${STREAMLIT_PORT}              ║${NC}"
echo -e "${GREEN}║   📤 Upload KB:     http://localhost:$((STREAMLIT_PORT + 2))              ║${NC}"
echo -e "${GREEN}║   📊 Dashboard:     http://localhost:$((STREAMLIT_PORT + 1))              ║${NC}"
echo -e "${GREEN}║   📡 API:           http://localhost:${API_PORT}                ║${NC}"
echo -e "${GREEN}║   📖 API Docs:      http://localhost:${API_PORT}/docs           ║${NC}"
echo -e "${GREEN}║                                                              ║${NC}"
echo -e "${GREEN}║   🔑 API Key:       ${API_KEY}                    ║${NC}"
echo -e "${GREEN}║                                                              ║${NC}"
echo -e "${GREEN}║   📤 Upload your own documents in the Chat UI sidebar         ║${NC}"
echo -e "${GREEN}║                                                              ║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo "  Press Ctrl+C to stop all services."
echo ""

# Wait so the script stays alive (cleanup runs on Ctrl+C)
wait
