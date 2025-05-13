FROM python:3.11.9-slim-bookworm
WORKDIR /app
RUN apt-get update && apt-get upgrade -y && apt-get clean
COPY src/stt_service/ .
COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir --retries 10 --timeout 120 -r requirements.txt
EXPOSE 8003
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8003"]
HEALTHCHECK CMD curl --fail http://localhost:8003/health || exit 1 