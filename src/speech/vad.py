import torch
import numpy as np
import soundfile as sf
import io

# Initialize Silero VAD model and utils at module load
vad_model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=False)
(get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils

def is_speech(audio_chunk: bytes) -> bool:
    """
    Returns True if speech is detected in the given audio chunk (PCM 16kHz mono bytes), else False.
    """
    try:
        audio_np, sr = sf.read(io.BytesIO(audio_chunk), dtype='int16')
        speech_timestamps = get_speech_timestamps(audio_np, vad_model, sampling_rate=sr)
        return bool(speech_timestamps)
    except Exception as e:
        import logging
        logging.error(f"VAD error: {e}")
        return False 