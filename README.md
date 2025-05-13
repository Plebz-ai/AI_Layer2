# AI_Layer2 Microservices

This directory contains the AI microservices for the Character platform. Each service is stateless, exposes a REST API, and can be run independently.

## Services

- **llm1_service**: Prompt/context generator
- **llm2_service**: Character brain (persona response)
- **stt_service**: Speech-to-text
- **tts_service**: Text-to-speech
- **orchestrator**: Orchestrates LLM/STT/TTS flows

## Running Locally

Each service can be run with:

```bash
cd src/<service_name>
pip install fastapi uvicorn httpx  # (httpx only for orchestrator)
uvicorn main:app --host 0.0.0.0 --port <port>
```

- llm1_service: 8001
- llm2_service: 8002
- stt_service: 8003
- tts_service: 8004
- orchestrator: 8010

## Running with Docker

Each service has a `Dockerfile`. Example:

```bash
docker build -t llm1_service ./src/llm1_service
docker run -p 8001:8001 llm1_service
```

## API Endpoints Overview

### LLM1 Service
- `POST /generate-context`  
  Input: `{ user_input, character_details }`  
  Output: `{ context, rules }`

### LLM2 Service
- `POST /generate-response`  
  Input: `{ user_query, persona_context, rules, model? }`  
  Output: `{ response }`

### STT Service
- `POST /speech-to-text`  
  Input: `{ audio_data }`  
  Output: `{ transcript }`
- `POST /stream-speech-to-text` (streaming)

### TTS Service
- `POST /text-to-speech`  
  Input: `{ text, voice_name? }`  
  Output: `{ audio_data }`
- `POST /stream-text-to-speech` (streaming)

### Orchestrator
- `POST /interact`  
  Input: `{ user_input, character_details, mode, audio_data? }`  
  Output: `{ response, audio_data? }`

## Health Checks

Each service can implement a `/health` endpoint for readiness/liveness probes (recommended for production).

# AI Layer 2 - Character AI Platform

This is the AI layer for the Character AI platform, providing real-time voice and text interactions with AI characters.

## Quick Start (Local Development)

**This service is designed to run locally with minimal setup. Cloud API keys and Redis are optional. If not provided, the service will use mock responses and in-memory storage.**

### Prerequisites
- Python 3.8 or later
- ffmpeg (required for audio features; [download here](https://ffmpeg.org/download.html))

### Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your API keys and configuration (or leave blank for mock mode)
```

3. Run the service:
```bash
cd src
python main.py
```

- The API will be available at `http://localhost:5000` (or the port you set in `.env`).
- API documentation is available at `/docs`.

### Notes
- **Redis is optional.** If not running, the service will use in-memory storage.
- **Cloud API keys (OpenAI, ElevenLabs, Azure, etc.) are optional.** If not set, the service will use mock responses.
- **ffmpeg is required for real audio processing.** If not installed, audio features may not work, but the API will still run.

## Development
- Use `pytest` for testing
- Follow PEP 8 style guide
- Use type hints
- Document all functions and classes

## Architecture

The system is built using microservices architecture with the following components:

1. **Speech Service**
   - Text-to-Speech (TTS) using ElevenLabs
   - Speech-to-Text (STT) using Deepgram
   - Real-time audio streaming

2. **LLM Service**
   - Predefined character responses
   - Custom character generation
   - Prompt chaining for enhanced responses

3. **Conversation Manager**
   - Real-time conversation handling
   - State management
   - WebSocket communication

4. **Persona Management**
   - Character context and personality
   - Voice configuration
   - Response style management

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your API keys and configuration
```

3. Run the services:
```bash
# Run all services
python -m src.main

# Or run individual services
python -m src.speech.main
python -m src.llm.main
python -m src.conversation.main
```

## API Documentation

The API documentation is available at `/docs` when running the services.

## Development

- Use `pytest` for testing
- Follow PEP 8 style guide
- Use type hints
- Document all functions and classes

## Best Practices & Troubleshooting

### Environment Variables
- All required Azure/OpenAI keys must be set in `.env`.
- Services will fail fast and log a clear error if a required variable is missing.
- Never commit `.env` to git.

### Docker
- Use the provided Dockerfiles and `docker-compose.yml`.
- Healthchecks are included for all services.
- Use `.dockerignore` to keep images small and secure.

### Common Startup Errors
- **TypeError: key must be a string**: Check that your Azure API key is set and not empty in `.env`.
- **OpenAIError: Missing credentials**: Ensure all required OpenAI/Azure variables are set in `.env`.
- **Service fails to start**: Check logs for `[FATAL] Missing required environment variable`.

### Production Checklist
- [x] All required env vars set in `.env`
- [x] All dependencies listed in `requirements.txt`
- [x] `.dockerignore` present
- [x] Healthcheck endpoints implemented
- [x] Logging to stdout/stderr
- [x] No secrets in Dockerfile or code
- [x] Resource limits set in Docker Compose (optional) 