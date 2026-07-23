# ---- Builder stage ----
FROM python:3.12-slim AS builder

WORKDIR /app
COPY pyproject.toml ./
COPY src/ ./src/

# Install poetry and project dependencies
RUN pip install --no-cache-dir poetry && \
    poetry config virtualenvs.create false && \
    poetry install --only=main --no-interaction --no-ansi

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

# Create data directories.
RUN mkdir -p data/kb_raw data/chroma

ENV PYTHONPATH=/app/src

# Expose the FastAPI port.
EXPOSE 8000

# Default command.
CMD ["uvicorn", "copilot.serving.api:app", "--host", "0.0.0.0", "--port", "8000"]
