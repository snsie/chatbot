from .base_speaker import BaseSpeaker
from .edge_tts_speaker import EdgeTTSSpeaker
from .coqui_tts_speaker import CoquiTTSSpeaker
from .pyttsx3_speaker import Pyttsx3Speaker
from .piper_speaker import PiperSpeaker
from typing import Optional
from ..constants import TTS_BACKEND, VOICE_NAME

# =============================
# Speaker Factory
# =============================
async def create_speaker() -> BaseSpeaker:
    if TTS_BACKEND.lower() == 'edge-tts':
        return EdgeTTSSpeaker(VOICE_NAME)
    elif TTS_BACKEND.lower() == 'coqui':
        return CoquiTTSSpeaker(VOICE_NAME)
    elif TTS_BACKEND.lower() == 'piper':
        # VOICE_NAME should be a .onnx path for Piper; use native model rate and natural speaking speed
        # Note: Setting a very small length_scale (< 1.0) makes speech faster and can raise perceived pitch.
        # Use length_scale=1.0 for natural prosody. Adjust slightly (e.g., 0.9–1.1) if needed.
        return PiperSpeaker(model_path=VOICE_NAME, use_native_sample_rate=False, length_scale=1)
    return Pyttsx3Speaker(VOICE_NAME)