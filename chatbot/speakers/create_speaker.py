from .base_speaker import BaseSpeaker
from .edge_tts_speaker import EdgeTTSSpeaker
from .coqui_tts_speaker import CoquiTTSSpeaker
from .pyttsx3_speaker import Pyttsx3Speaker
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
    return Pyttsx3Speaker(VOICE_NAME)