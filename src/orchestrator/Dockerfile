FROM python:3.11.9-slim-bookworm
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends build-essential curl libsndfile1 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*
# Copy only requirements first for better cache
COPY requirements-base.txt ./
COPY requirements.txt ./
# Uninstall any pre-existing torch/torchaudio, then install CPU-only torch and torchaudio from official wheel index
RUN pip uninstall -y torch torchaudio || true && \
    pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir torch==2.1.2+cpu torchaudio==2.1.2+cpu --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements-base.txt && \
    pip install --no-cache-dir -r requirements.txt
# Debug: print torch/torchaudio versions and check libtorchaudio.so linkage
RUN python -c "import torch; print('torch:', torch.__version__, torch.version.cuda); import torchaudio; print('torchaudio:', torchaudio.__version__)"
RUN ldd /usr/local/lib/python3.11/site-packages/torchaudio/lib/libtorchaudio.so || true
# Copy only the service code
COPY orchestrator/. ./
COPY speech ./speech
COPY utils ./utils
ENV PYTHONPATH=/app:/app/..
ENV CUDA_VISIBLE_DEVICES=""
EXPOSE 8010
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8010"]
HEALTHCHECK CMD curl --fail http://localhost:8010/health || exit 1 