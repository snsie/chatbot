from faster_whisper import WhisperModel
import numpy as np

class WhisperSTT:
    def __init__(self, model_name: str, compute: str = 'auto'):
        device, compute_type = self._select_device(compute)
        self.model = WhisperModel(model_name, device=device, compute_type=compute_type)

    def _select_device(self, compute: str):
        if compute == 'auto':
            # Try GPU (cuda) first
            try:
                import torch  # noqa: F401
                return 'cuda', 'float16'
            except Exception:
                return 'cpu', 'int8'
        if compute == 'cuda':
            return 'cuda', 'float16'
        return 'cpu', 'int8'

    def transcribe(self, audio: np.ndarray) -> str:
        # segments, info = self.model.transcribe(audio, beam_size=1, vad_filter=False)
        segments, info = self.model.transcribe(audio,vad_filter=True)

        text_parts = [seg.text.strip() for seg in segments]
        return ' '.join(part for part in text_parts if part)