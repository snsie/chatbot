
from .voice_identifier import _VoiceIdentifier

_VI_SINGLETON = None

def get_voice_identifier(enroll_path: str = "enrollments.npz") -> _VoiceIdentifier:
    global _VI_SINGLETON
    if _VI_SINGLETON is None:
        _VI_SINGLETON = _VoiceIdentifier(enroll_path=enroll_path)
    return _VI_SINGLETON
    