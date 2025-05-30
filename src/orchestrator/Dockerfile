FROM python:3.11.9-slim-bookworm
WORKDIR /app
# Install build tools, audio libs, and dev headers for VAD, audio, and async Redis
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    python3-dev \
    libffi-dev \
    libsndfile1 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*
# Copy only requirements first for better cache
COPY requirements-base.txt ./
COPY requirements.txt ./
# Install Python dependencies (uvicorn[standard] for WebSocket support)
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements-base.txt && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir websockets wsproto && \
    pip show uvicorn || true && pip show websockets || true && pip show wsproto || true
# Copy only the service code
COPY orchestrator/. ./
COPY speech ./speech
COPY utils ./utils
ENV PYTHONPATH=/app:/app/..
ENV CUDA_VISIBLE_DEVICES=""
EXPOSE 8010
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8010"]
HEALTHCHECK CMD curl --fail http://localhost:8010/health || exit 1 