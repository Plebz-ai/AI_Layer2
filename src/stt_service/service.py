import asyncio
import json
import logging
import websockets
import sys
import httpx  # For error logging
from fastapi import FastAPI, Request, Response, status, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
import os
import base64

# --- CONFIG ---
DEEPGRAM_API_KEY = os.environ.get("DEEPGRAM_API_KEY", "")
DEEPGRAM_URL = "wss://api.deepgram.com/v1/listen"

app = FastAPI()
logger = logging.getLogger("stt_service")
logging.basicConfig(level=logging.INFO)

def get_deepgram_url():
    params = {
        "encoding": "linear16",
        "channels": "1",
        "sample_rate": "16000",
        "language": "en-US"
    }
    param_str = "&".join(f"{k}={v}" for k, v in params.items())
    return f"{DEEPGRAM_URL}?{param_str}"

async def deepgram_stream(audio_chunks):
    url = get_deepgram_url()
    headers = {"Authorization": f"Token {DEEPGRAM_API_KEY}"}
    try:
        async with websockets.connect(url, additional_headers=headers, max_size=2**24) as ws:
            first_chunk = True
            first_chunk_data = None
            async def sender():
                nonlocal first_chunk, first_chunk_data
                async for chunk in audio_chunks:
                    if not chunk or len(chunk) == 0:
                        continue  # Do not send empty bytes to Deepgram
                    if first_chunk:
                        first_chunk_data = chunk
                        logger.info(f"[STT] First audio chunk to Deepgram: {chunk[:32]}... (len={len(chunk)})")
                        # Audio format validation
                        if len(chunk) % 2 != 0:
                            logger.warning("[STT] First audio chunk length is not a multiple of 2 (not valid 16-bit PCM)")
                        import array
                        pcm = array.array('h')
                        try:
                            pcm.frombytes(chunk[:min(3200, len(chunk))])
                            if all(x == 0 for x in pcm):
                                logger.warning("[STT] First audio chunk is all zeros (silence or bad format)")
                            if all(x == 32767 or x == -32768 for x in pcm):
                                logger.warning("[STT] First audio chunk is all max/min values (bad format)")
                        except Exception as e:
                            logger.warning(f"[STT] Could not parse first audio chunk as 16-bit PCM: {e}")
                        first_chunk = False
                    await ws.send(chunk)
                # Do not send b'' at the end, just close the connection
                # await ws.send(b"")  # Removed per Deepgram docs

            async def receiver():
                async for msg in ws:
                    data = json.loads(msg)
                    if "channel" in data and "alternatives" in data["channel"]:
                        alt = data["channel"]["alternatives"][0]
                        transcript = alt.get("transcript", "")
                        if transcript:
                            yield transcript

            sender_task = asyncio.create_task(sender())
            async for transcript in receiver():
                yield transcript
            await sender_task
    except websockets.exceptions.InvalidStatusCode as e:
        logger.error(f"Deepgram websocket connection failed: HTTP {e.status_code}")
        # Print Deepgram error headers if available
        dg_error = e.headers.get("dg-error") if hasattr(e, 'headers') else None
        dg_request_id = e.headers.get("dg-request-id") if hasattr(e, 'headers') else None
        if dg_error:
            logger.error(f"Deepgram dg-error: {dg_error}")
        if dg_request_id:
            logger.error(f"Deepgram dg-request-id: {dg_request_id}")
        # Try to fetch the error body with httpx for more details
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(url, headers=headers)
                logger.error(f"Deepgram error body: {resp.text}")
        except Exception as fetch_e:
            logger.error(f"Could not fetch Deepgram error body: {fetch_e}")
        logger.error("Check your API key, audio format (16kHz 16-bit mono PCM), and Deepgram parameters.")
        yield "[ERROR] Deepgram connection failed: HTTP 400. Check API key, audio format, and parameters."
    except Exception as e:
        logger.error(f"Deepgram websocket connection failed: {e}")
        yield "[ERROR] Deepgram connection failed."

@app.post("/stream-speech-to-text")
async def stream_speech_to_text(request: Request):
    body = await request.body()
    content_type = request.headers.get("content-type", "")
    logger.info(f"[STT] Incoming /stream-speech-to-text content-type: {content_type}")
    # If base64, decode
    try:
        data = json.loads(body)
        logger.info("[STT] Request is JSON, decoding base64 audio_data.")
        audio_data = base64.b64decode(data["audio_data"])
    except Exception:
        if isinstance(body, (bytes, bytearray)):
            logger.info("[STT] Request is raw PCM bytes.")
            audio_data = body
        else:
            logger.warning(f"[STT] Request body is not JSON and not bytes: type={type(body)}")
            audio_data = b""

    # Audio format validation and logging
    logger.info(f"[STT] First 32 bytes of audio: {audio_data[:32]}")
    if len(audio_data) % 2 != 0:
        logger.warning("[STT] Audio length is not a multiple of 2 (not valid 16-bit PCM)")
    import array
    pcm = array.array('h')
    try:
        pcm.frombytes(audio_data[:min(3200, len(audio_data))])
        if all(x == 0 for x in pcm):
            logger.warning("[STT] Audio chunk is all zeros (silence or bad format)")
        if all(x == 32767 or x == -32768 for x in pcm):
            logger.warning("[STT] Audio chunk is all max/min values (bad format)")
    except Exception as e:
        logger.warning(f"[STT] Could not parse audio chunk as 16-bit PCM: {e}")

    # Call Deepgram HTTP API
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "https://api.deepgram.com/v1/listen",
            headers={
                "Authorization": f"Token {DEEPGRAM_API_KEY}",
                "Content-Type": "audio/pcm"  # 16-bit linear PCM
            },
            content=audio_data,
            params={
                "model": "general",
                "language": "en-US",
                "punctuate": "true"
            }
        )
        if response.status_code != 200:
            logger.error(f"Deepgram HTTP error {response.status_code}: {response.text}")
            return Response(content=response.text, status_code=response.status_code, media_type="application/json")
        return Response(content=response.content, media_type="application/json")

@app.websocket("/ws/stream-speech-to-text")
async def websocket_speech_to_text(ws: WebSocket):
    await ws.accept()
    logger.info("[STT WS] Client connected")
    try:

        audio_queue = asyncio.Queue()
        stop_event = asyncio.Event()

        async def audio_receiver():
            try:
                while not stop_event.is_set():
                    data = await ws.receive_bytes()
                    await audio_queue.put(data)
            except WebSocketDisconnect:
                logger.info("[STT WS] Client disconnected")
                stop_event.set()
            except Exception as e:
                logger.error(f"[STT WS] Receiver error: {e}")
                stop_event.set()

        async def deepgram_sender():
            url = get_deepgram_url()
            headers = {"Authorization": f"Token {DEEPGRAM_API_KEY}"}
            try:
                async with websockets.connect(url, additional_headers=headers, max_size=2**24) as dg_ws:
                    async def sender():
                        while not stop_event.is_set():
                            chunk = await audio_queue.get()
                            await dg_ws.send(chunk)
                        await dg_ws.send(b"")
                    async def receiver():
                        async for msg in dg_ws:
                            data = json.loads(msg)
                            if "channel" in data and "alternatives" in data["channel"]:
                                alt = data["channel"]["alternatives"][0]
                                transcript = alt.get("transcript", "")
                                if transcript:
                                    await ws.send_text(transcript)
                    sender_task = asyncio.create_task(sender())
                    await receiver()
                    await sender_task
            except Exception as e:
                logger.error(f"[STT WS] Deepgram error: {e}")
                await ws.send_text("[ERROR] Deepgram connection failed.")
                stop_event.set()

        await asyncio.gather(audio_receiver(), deepgram_sender())
    except Exception as e:
        logger.error(f"[STT WS] Unexpected error: {e}")
    finally:
        await ws.close()
        logger.info("[STT WS] Connection closed")

@app.get("/health")
async def health():
    return {"status": "ok"}
