# identify.py
import warnings, pathlib
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import torch
#from torch.amp import custom_fwd   # with device_type="cuda"
from speechbrain.pretrained import EncoderClassifier

class _VoiceIdentifier:
    def __init__(self, enroll_path: str = "enrollments.npz"):
        self.model = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb", savedir="pretrained_ecapa"
        )
        self.enroll = self._load_enroll(enroll_path)

    def _load_enroll(self, path: str):
        try:
            data = np.load(path, allow_pickle=True)
            names = data["names"]
            vecs  = data["vecs"]
            return {str(n): np.asarray(v, dtype=np.float32) for n, v in zip(names, vecs)}
        except Exception:
            return {}

    @torch.no_grad()
    def _embed_f32(self, mono_f32: np.ndarray, sr: int) -> np.ndarray:
        if sr != 16000:
            raise ValueError(f"Expected 16k audio, got {sr}")
        if mono_f32.ndim != 1:
            mono_f32 = mono_f32.reshape(-1)
        wav = torch.from_numpy(mono_f32).unsqueeze(0)
        emb = self.model.encode_batch(wav).squeeze(0).squeeze(0).cpu().numpy()
        n = np.linalg.norm(emb) + 1e-12
        return (emb / n).astype(np.float32)

    def identify_from_array(self, mono_f32: np.ndarray, sr: int):
        """Return (best_name, best_similarity) where similarity is cosine in [−1,1].
           If no enrollments, returns (None, 0.0)."""
        if not self.enroll:
            return None, 0.0
        q = self._embed_f32(mono_f32, sr)
        scores = {n: float(np.dot(q, v)) for n, v in self.enroll.items()}
        best_name, best_sim = max(scores.items(), key=lambda kv: kv[1])
        return best_name, best_sim

# Singleton accessor so the model loads once per process
_VI_SINGLETON = None

def get_voice_identifier(enroll_path: str = "enrollments.npz") -> _VoiceIdentifier:
    global _VI_SINGLETON
    if _VI_SINGLETON is None:
        _VI_SINGLETON = _VoiceIdentifier(enroll_path=enroll_path)
    return _VI_SINGLETON