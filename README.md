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