FROM python:3.11.9-slim-bookworm
WORKDIR /app
RUN apt-get update && apt-get upgrade -y && apt-get clean
COPY src/llm1_service/ .
COPY requirements.txt ./
COPY .env ./
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir --retries 10 --timeout 120 -r requirements.txt && \
    pip install --no-cache-dir torch==2.2.2+cpu -f https://download.pytorch.org/whl/torch_stable.html
EXPOSE 8001
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]
HEALTHCHECK CMD curl --fail http://localhost:8001/health || exit 1 