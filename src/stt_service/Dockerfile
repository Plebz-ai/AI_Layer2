FROM python:3.11.9-slim-bookworm
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends build-essential curl && \
    apt-get clean && rm -rf /var/lib/apt/lists/*
# Copy only requirements first for better cache
COPY ../../requirements-base.txt ./
COPY ../../requirements.txt ./
# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements-base.txt && \
    pip install --no-cache-dir -r requirements.txt
# Copy only the service code
COPY . .
ENV PYTHONPATH=/app
EXPOSE 8003
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8003"]
HEALTHCHECK CMD curl --fail http://localhost:8003/health || exit 1 