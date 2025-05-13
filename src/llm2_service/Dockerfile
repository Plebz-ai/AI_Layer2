FROM python:3.11.9-slim-bookworm
WORKDIR /app
RUN apt-get update && apt-get upgrade -y && apt-get clean
COPY src/llm2_service/ .
COPY requirements.txt ./
COPY .env ./
ENV PIP_NO_CACHE_DIR=1
ENV PYTHONUSERBASE=/app/.local
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip uninstall -y openai httpx || true && \
    pip install --no-cache-dir --retries 10 --timeout 120 -r requirements.txt && \
    pip install --no-cache-dir torch==2.2.2+cpu -f https://download.pytorch.org/whl/torch_stable.html && \
    pip install --force-reinstall --no-cache-dir openai==1.56.0 httpx==0.27.2 && \
    python -c "import openai, httpx; print('openai version:', openai.__version__); print('httpx version:', httpx.__version__)" && \
    pip list
EXPOSE 8002
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8002"]
HEALTHCHECK CMD curl --fail http://localhost:8002/health || exit 1 