from .base_speaker import BaseSpeaker
from typing import Optional
from ..constants import VOICE_NAME, SAMPLE_RATE
import sys
import io

class EdgeTTSSpeaker(BaseSpeaker):
    """Edge TTS speaker; per-sentence synthesis & playback.

    Uses edge-tts to synthesize MP3 -> pydub to decode -> simpleaudio to play.
    """

    def __init__(self, voice: Optional[str] = VOICE_NAME):
        self.voice = voice or 'en-US-JennyNeural'
        try:
            import edge_tts  # noqa: F401
            import pydub  # noqa: F401
            import simpleaudio  # noqa: F401
        except ImportError as e:
            print("[EdgeTTSSpeaker] Missing packages. Install: pip install edge-tts pydub simpleaudio", file=sys.stderr)
            raise

    async def speak(self, sentence: str):
        import edge_tts
        from pydub import AudioSegment
        import simpleaudio as sa
        # Synthesize
        communicate = edge_tts.Communicate(sentence, voice=self.voice)
        audio_bytes = bytearray()
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_bytes.extend(chunk["data"])
        data = bytes(audio_bytes)
        if not data:
            return
        # Decode / resample
        audio_seg = AudioSegment.from_file(io.BytesIO(data), format="mp3")
        audio_seg = audio_seg.set_frame_rate(SAMPLE_RATE).set_channels(1).set_sample_width(2)
        play_obj = sa.play_buffer(audio_seg.raw_data, num_channels=1, bytes_per_sample=2, sample_rate=audio_seg.frame_rate)
        play_obj.wait_done()
