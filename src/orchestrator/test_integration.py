import pytest
import httpx
import asyncio

@pytest.mark.asyncio
async def test_orchestrator_interact_chat():
    async with httpx.AsyncClient() as client:
        payload = {
            "user_input": "Hello!",
            "character_details": {"name": "Alice", "persona": "friendly"},
            "mode": "chat"
        }
        resp = await client.post("http://localhost:8010/interact", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert "response" in data
        assert data["response"].startswith("[O4-mini]")

@pytest.mark.asyncio
async def test_orchestrator_interact_voice():
    async with httpx.AsyncClient() as client:
        payload = {
            "user_input": "ignored for voice",
            "character_details": {"name": "Bob", "persona": "serious", "voice_type": "male"},
            "mode": "voice",
            "audio_data": "U29tZUJhc2U2NEF1ZGlv"  # "SomeBase64Audio"
        }
        resp = await client.post("http://localhost:8010/interact", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert "response" in data
        assert "audio_data" in data
        assert data["response"].startswith("[O4-mini]") or data["response"].startswith("[Llama-4]") 