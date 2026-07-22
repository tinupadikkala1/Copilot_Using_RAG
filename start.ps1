<#
.SYNOPSIS
    🚀 Autonomous Customer Support Copilot — One-Click Launcher for Windows

.DESCRIPTION
    This PowerShell script does EVERYTHING needed to run the project on Windows:
      1. Checks Python, dependencies, and Ollama
      2. Starts Ollama if not running
      3. Creates a virtual environment and installs dependencies (if needed)
      4. Builds the vector index from your documents
      5. Starts the API server
      6. Opens the Streamlit Chat UI + Dashboard

    Usage:
        .\start.ps1
        .\start.ps1 -KbDir "C:\path\to\docs" -ApiKey "my-secret" -Port 8000

    Prerequisites (install before running):
        - Python 3.11+:  https://www.python.org/downloads/
        - Ollama:        https://ollama.com/download/windows
        - Git (optional): https://git-scm.com/download/win

.PARAMETER KbDir
    Path to your knowledge base documents (default: data/kb_raw)

.PARAMETER ApiKey
    API key for authentication (default: copilot-demo-key)

.PARAMETER Port
    Port for the API server (default: 8000)
#>

param(
    [string]$KbDir = "",
    [string]$ApiKey = "copilot-demo-key",
    [int]$Port = 8000
)

# ─────────────────────────────────────────────────────────────────────────────
#  Config
# ─────────────────────────────────────────────────────────────────────────────

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

$ApiKey = if ($env:COPILOT_API_KEY) { $env:COPILOT_API_KEY } else { $ApiKey }
$ApiPort = $Port
$StreamlitPort = 8501
$KB_DIR = $KbDir

# ─────────────────────────────────────────────────────────────────────────────
#  Helper functions
# ─────────────────────────────────────────────────────────────────────────────

$Host.UI.RawUI.ForegroundColor = "White"

function Write-Info($msg) { Write-Host "[INFO]  $msg" -ForegroundColor Blue }
function Write-Ok($msg)   { Write-Host "[OK]    $msg" -ForegroundColor Green }
function Write-Warn($msg) { Write-Host "[WARN]  $msg" -ForegroundColor Yellow }
function Write-Err($msg)  { Write-Host "[ERROR] $msg" -ForegroundColor Red }

function Write-Step($num, $title) {
    Write-Host ""
    Write-Host "═══════════════════════════════════════════════════" -ForegroundColor Cyan
    Write-Host "  Step $num/6 — $title" -ForegroundColor Cyan
    Write-Host "═══════════════════════════════════════════════════" -ForegroundColor Cyan
    Write-Host ""
}

# ─────────────────────────────────────────────────────────────────────────────
#  Help
# ─────────────────────────────────────────────────────────────────────────────

if ($args -contains "--help" -or $args -contains "-h" -or $args -contains "/?") {
    Write-Host ""
    Write-Host "  🚀 Autonomous Customer Support Copilot — Windows Launcher" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "  Usage: .\start.ps1 [OPTIONS]"
    Write-Host ""
    Write-Host "  Options:"
    Write-Host "    -KbDir PATH    Path to knowledge base documents (default: data/kb_raw)"
    Write-Host "    -ApiKey KEY    API key for authentication (default: copilot-demo-key)"
    Write-Host "    -Port PORT     Port for API server (default: 8000)"
    Write-Host "    -Help          Show this help"
    Write-Host ""
    Write-Host "  Environment variables:"
    Write-Host "    COPILOT_API_KEY  API key (overrides -ApiKey)"
    Write-Host ""
    exit 0
}

# ─────────────────────────────────────────────────────────────────────────────
#  Step 1: Check prerequisites
# ─────────────────────────────────────────────────────────────────────────────

Write-Step 1 "Checking prerequisites"

# Find Python
$PYTHON = ""
$pythonCandidates = @("python3", "python")
foreach ($cmd in $pythonCandidates) {
    try {
        $version = & $cmd --version 2>&1
        if ($LASTEXITCODE -eq 0) {
            $PYTHON = $cmd
            break
        }
    } catch { continue }
}

if (-not $PYTHON) {
    Write-Err "Python 3 not found."
    Write-Err "Please install Python 3.11+ from: https://www.python.org/downloads/"
    Write-Err "Make sure to check 'Add Python to PATH' during installation."
    exit 1
}

$pyVersion = & $PYTHON -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>&1
Write-Info "Python: $(& $PYTHON --version 2>&1)"
Write-Info "OS:     Windows $([Environment]::OSVersion.Version)"

# Check Python version (need 3.11+)
$versionParts = $pyVersion -split '\.'
if (([int]$versionParts[0] -lt 3) -or (([int]$versionParts[0] -eq 3) -and ([int]$versionParts[1] -lt 11))) {
    Write-Warn "Python 3.11+ recommended. You have $(& $PYTHON --version 2>&1)"
}

Write-Ok "Prerequisites check complete"

# ─────────────────────────────────────────────────────────────────────────────
#  Step 2: Check / start Ollama
# ─────────────────────────────────────────────────────────────────────────────

Write-Step 2 "Setting up Ollama"

$OLLAMA_RUNNING = $false
try {
    $ollamaCheck = & ollama --version 2>&1
    if ($LASTEXITCODE -eq 0) {
        # Check if Ollama API is responding
        try {
            $response = Invoke-WebRequest -Uri "http://localhost:11434/api/tags" -TimeoutSec 5 -ErrorAction Stop
            $OLLAMA_RUNNING = $true
            Write-Ok "Ollama is already running"
        } catch {
            Write-Info "Starting Ollama in background..."
            $ollamaProcess = Start-Process -FilePath "ollama" -ArgumentList "serve" -NoNewWindow -PassThru
            Start-Sleep -Seconds 4
            try {
                $response = Invoke-WebRequest -Uri "http://localhost:11434/api/tags" -TimeoutSec 5 -ErrorAction Stop
                Write-Ok "Ollama started"
                $OLLAMA_RUNNING = $true
            } catch {
                Write-Warn "Could not connect to Ollama. Make sure it's running."
            }
        }
    }
} catch {
    Write-Warn "Ollama not found. Please install from: https://ollama.com/download/windows"
}

if ($OLLAMA_RUNNING) {
    # Check for required models
    try {
        $tags = Invoke-RestMethod -Uri "http://localhost:11434/api/tags" -TimeoutSec 10
        $models = $tags.models.name

        Write-Info "Available models:"
        foreach ($m in $models) { Write-Host "    - $m" }

        # Check for nomic-embed-text
        $hasEmbed = ($models | Select-String "nomic-embed-text").Count -gt 0
        if (-not $hasEmbed) {
            Write-Info "Pulling embedding model (nomic-embed-text, 274 MB)..."
            & ollama pull nomic-embed-text 2>&1 | Out-Host
            Write-Ok "Embedding model ready"
        } else {
            Write-Ok "Embedding model (nomic-embed-text) available"
        }

        # Check for qwen-local
        $hasQwen = ($models | Select-String "qwen-local").Count -gt 0
        if (-not $hasQwen) {
            Write-Info "Pulling LLM model (qwen-local, 1.1 GB)..."
            & ollama pull qwen-local:latest 2>&1 | Out-Host
            Write-Ok "LLM model ready"
        } else {
            Write-Ok "LLM model (qwen-local) available"
        }
    } catch {
        Write-Warn "Could not fetch model list from Ollama"
    }
}

Write-Ok "Ollama setup complete"

# ─────────────────────────────────────────────────────────────────────────────
#  Step 3: Install Python dependencies
# ─────────────────────────────────────────────────────────────────────────────

Write-Step 3 "Installing Python dependencies"

$VENV_DIR = "$env:TEMP\copilot-venv"
$VENV_PYTHON = "$VENV_DIR\Scripts\python.exe"
$VENV_PIP = "$VENV_DIR\Scripts\pip.exe"

if (-not (Test-Path $VENV_PYTHON)) {
    Write-Info "Creating virtual environment at $VENV_DIR..."
    & $PYTHON -m venv $VENV_DIR 2>&1 | Out-Null
    Write-Ok "Virtual environment created"

    Write-Info "Installing dependencies (this may take a few minutes)..."
    & $VENV_PIP install --quiet --upgrade pip 2>&1 | Out-Null
    & $VENV_PIP install --quiet `
        fastapi uvicorn[standard] streamlit httpx pydantic pydantic-settings `
        chromadb numpy pypdf beautifulsoup4 python-multipart `
        pytest 2>&1 | Out-Null
    Write-Ok "Dependencies installed"
} else {
    Write-Ok "Virtual environment already exists at $VENV_DIR"
    # Check critical packages
    try {
        & $VENV_PYTHON -c "import fastapi" 2>&1 | Out-Null
    } catch {
        Write-Info "Reinstalling missing dependencies..."
        & $VENV_PIP install --quiet `
            fastapi uvicorn streamlit httpx pydantic pydantic-settings `
            chromadb numpy pypdf beautifulsoup4 python-multipart 2>&1 | Out-Null
        Write-Ok "Dependencies installed"
    }
}

Write-Ok "Python dependencies ready"

# ─────────────────────────────────────────────────────────────────────────────
#  Step 4: Build the vector index
# ─────────────────────────────────────────────────────────────────────────────

Write-Step 4 "Building vector index"

# Determine KB directory
if (-not $KB_DIR) {
    if ((Test-Path "data/kb_raw") -and ((Get-ChildItem "data/kb_raw" -File).Count -gt 0)) {
        $KB_DIR = "data/kb_raw"
        Write-Info "Using existing documents from: $KB_DIR"
    } elseif (Test-Path "tests/fixtures/mock_kb") {
        $KB_DIR = "tests/fixtures/mock_kb"
        Write-Info "No user documents found. Using demo mock KB: $KB_DIR"
        Write-Info "You can upload your own documents later via the Upload KB page in the UI."
    } else {
        New-Item -ItemType Directory -Force -Path "data/kb_raw" | Out-Null
        $KB_DIR = "data/kb_raw"
        Write-Info "No documents found. Created empty directory: $KB_DIR"
        Write-Info "Upload documents via the Upload KB page after the app starts."
    }
}

$env:PYTHONPATH = "$ScriptDir\src"

if ((Test-Path $KB_DIR) -and ((Get-ChildItem $KB_DIR -File).Count -gt 0)) {
    Write-Info "Building index from: $KB_DIR"
    New-Item -ItemType Directory -Force -Path "data/chroma" | Out-Null

    & $VENV_PYTHON scripts/build_index.py --kb-root "$KB_DIR" --persist-dir data/chroma 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Ok "Vector index built successfully"
    } else {
        Write-Warn "Index build had issues. You can rebuild from the Upload KB page."
    }
} else {
    Write-Warn "No documents to index at: $KB_DIR"
    Write-Warn "Upload documents via the Upload KB page after the app starts."
    New-Item -ItemType Directory -Force -Path "data/kb_raw" | Out-Null
}

Write-Ok "Vector index ready"

# ─────────────────────────────────────────────────────────────────────────────
#  Step 5: Start the API server (background)
# ─────────────────────────────────────────────────────────────────────────────

Write-Step 5 "Starting API server"

# Kill any existing process on our port
try {
    $existing = Get-NetTCPConnection -LocalPort $ApiPort -ErrorAction SilentlyContinue
    if ($existing) {
        $pidToKill = $existing.OwningProcess
        Write-Warn "Port $ApiPort is in use (PID: $pidToKill). Stopping old process..."
        Stop-Process -Id $pidToKill -Force -ErrorAction SilentlyContinue
        Start-Sleep -Seconds 1
    }
} catch {
    # Get-NetTCPConnection might not be available on older Windows
    Write-Info "Could not check port (using netstat fallback)..."
    try {
        $netstatOutput = netstat -ano | Select-String ":$ApiPort "
        if ($netstatOutput) {
            Write-Warn "Port $ApiPort appears to be in use. Please close the other program and retry."
        }
    } catch { }
}

$env:COPILOT_API_KEY = $ApiKey
$env:PYTHONPATH = "$ScriptDir\src"

Write-Info "Starting FastAPI server on port $ApiPort..."

$apiProcess = Start-Process -FilePath $VENV_PYTHON -ArgumentList @(
    "-m", "uvicorn", "copilot.serving.api:app",
    "--host", "0.0.0.0",
    "--port", $ApiPort,
    "--log-level", "warning"
) -NoNewWindow -PassThru

$API_PID = $apiProcess.Id
Write-Info "API server PID: $API_PID"

# Wait for API to be ready
Write-Info "Waiting for API server to be ready..."
$ready = $false
for ($i = 1; $i -le 30; $i++) {
    try {
        $health = Invoke-WebRequest -Uri "http://localhost:$ApiPort/healthz" -TimeoutSec 2 -ErrorAction Stop
        if ($health.StatusCode -eq 200) {
            Write-Ok "API server is running (PID: $API_PID) at http://localhost:$ApiPort"
            $ready = $true
            break
        }
    } catch { }
    Start-Sleep -Seconds 1
}
if (-not $ready) {
    Write-Err "API server failed to start within 30 seconds"
    Write-Err "Check manually with: curl http://localhost:$ApiPort/healthz"
}

# ─────────────────────────────────────────────────────────────────────────────
#  Step 6: Launch the Streamlit UI
# ─────────────────────────────────────────────────────────────────────────────

Write-Step 6 "Launching Chat UI"

$env:COPILOT_API_URL = "http://localhost:$ApiPort"
$env:COPILOT_API_KEY = $ApiKey
$env:PYTHONPATH = "$ScriptDir\src"

Write-Info "Starting Streamlit Chat UI..."
Write-Info "  Chat UI:     http://localhost:$StreamlitPort"
Write-Info "  Dashboard:   http://localhost:$($StreamlitPort + 1)"
Write-Info "  API Server:  http://localhost:$ApiPort"
Write-Info "  API Docs:    http://localhost:$ApiPort/docs"
Write-Host ""

# Launch Chat UI
$chatProcess = Start-Process -FilePath $VENV_PYTHON -ArgumentList @(
    "-m", "streamlit", "run", "src/copilot/serving/ui/chat_app.py",
    "--server.port", $StreamlitPort,
    "--server.headless", "true",
    "--browser.gatherUsageStats", "false"
) -NoNewWindow -PassThru

$CHAT_PID = $chatProcess.Id
Start-Sleep -Seconds 3

# Launch Dashboard
$dashProcess = Start-Process -FilePath $VENV_PYTHON -ArgumentList @(
    "-m", "streamlit", "run", "src/copilot/serving/ui/dashboard.py",
    "--server.port", ($StreamlitPort + 1),
    "--server.headless", "true",
    "--browser.gatherUsageStats", "false"
) -NoNewWindow -PassThru

$DASH_PID = $dashProcess.Id
Start-Sleep -Seconds 2

# ─────────────────────────────────────────────────────────────────────────────
#  All done — show summary
# ─────────────────────────────────────────────────────────────────────────────

Write-Host ""
Write-Host "╔══════════════════════════════════════════════════════════════╗" -ForegroundColor Green
Write-Host "║                                                              ║" -ForegroundColor Green
Write-Host "║   🚀  Copilot is running!                                    ║" -ForegroundColor Green
Write-Host "║                                                              ║" -ForegroundColor Green
Write-Host "║   💬 Chat UI:        http://localhost:$StreamlitPort              ║" -ForegroundColor Green
Write-Host "║   📊 Dashboard:     http://localhost:$($StreamlitPort + 1)              ║" -ForegroundColor Green
Write-Host "║   📡 API Server:    http://localhost:$ApiPort                ║" -ForegroundColor Green
Write-Host "║   📖 API Docs:      http://localhost:$ApiPort/docs           ║" -ForegroundColor Green
Write-Host "║                                                              ║" -ForegroundColor Green
Write-Host "║   🔑 API Key:       $ApiKey                    ║" -ForegroundColor Green
Write-Host "║                                                              ║" -ForegroundColor Green
Write-Host "║   📤 Upload your own documents in the Chat UI sidebar         ║" -ForegroundColor Green
Write-Host "║                                                              ║" -ForegroundColor Green
Write-Host "╚══════════════════════════════════════════════════════════════╝" -ForegroundColor Green
Write-Host ""
Write-Host "  Press Ctrl+C to stop all services." -ForegroundColor Yellow
Write-Host "  Or close this PowerShell window." -ForegroundColor Yellow
Write-Host ""

# Keep the script running
try {
    while ($true) {
        Start-Sleep -Seconds 1
    }
} finally {
    # Cleanup on exit
    Write-Host ""
    Write-Warn "Shutting down..."
    $pids = @($API_PID, $CHAT_PID, $DASH_PID) | Where-Object { $_ -gt 0 }
    foreach ($pid in $pids) {
        try {
            Stop-Process -Id $pid -Force -ErrorAction SilentlyContinue
            Write-Ok "Stopped process (PID: $pid)"
        } catch { }
    }
    Write-Ok "All services stopped. Goodbye!"
}
