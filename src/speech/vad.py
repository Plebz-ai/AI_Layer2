import webrtcvad
import numpy as np

class StreamingVAD:
    def __init__(self, sample_rate=16000, frame_ms=30, aggressiveness=2):
        self.vad = webrtcvad.Vad(aggressiveness)
        self.sample_rate = sample_rate
        self.frame_ms = frame_ms
        self.frame_bytes = int(sample_rate * frame_ms / 1000) * 2  # 16-bit PCM

    def is_speech(self, pcm_bytes):
        """Returns True if the frame contains speech."""
        if len(pcm_bytes) != self.frame_bytes:
            raise ValueError(f"Frame must be {self.frame_bytes} bytes")
        return self.vad.is_speech(pcm_bytes, self.sample_rate)

    def stream_vad(self, pcm_stream):
        """
        Generator: yields (is_speech, frame_bytes) for each frame in the stream.
        pcm_stream: bytes or file-like object yielding raw PCM 16kHz mono.
        """
        buf = b""
        while True:
            chunk = pcm_stream.read(self.frame_bytes - len(buf))
            if not chunk:
                break
            buf += chunk
            if len(buf) < self.frame_bytes:
                continue
            frame = buf[:self.frame_bytes]
            buf = buf[self.frame_bytes:]
            yield self.is_speech(frame), frame

# Simple helper for one-off frame checks
vad_instance = StreamingVAD()
def is_speech(audio_chunk: bytes) -> bool:
    return vad_instance.is_speech(audio_chunk) 