# ---- Builder stage ----
FROM python:3.12-slim AS builder

WORKDIR /app
COPY pyproject.toml ./
RUN pip install --no-cache-dir .

# ---- Runtime stage ----
FROM python:3.12-slim

WORKDIR /app

# Copy installed packages from builder.
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy source code.
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY configs/ ./configs/
COPY data/ ./data/

# Expose the FastAPI port.
EXPOSE 8000

# Default command — override via docker-compose or CLI.
CMD ["uvicorn", "copilot.serving.api:app", "--host", "0.0.0.0", "--port", "8000"]
