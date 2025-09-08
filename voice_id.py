# identify.py
import warnings, pathlib
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import torch
#from torch.amp import custom_fwd   # with device_type="cuda"
from speechbrain.pretrained import EncoderClassifier


def embedding(wav_path: str) -> np.ndarray:
    """Return L2-normalized speaker embedding from a WAV file."""
    signal = classifier.load_audio(wav_path).unsqueeze(0)  # [1, T]
    with torch.no_grad():
        emb = classifier.encode_batch(signal).squeeze(0).squeeze(0)  # [d]
    v = emb.cpu().numpy()
    v = v / (np.linalg.norm(v) + 1e-12)
    return v

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))

# 1) Build an enrollment gallery
# Dict of {name: [wav_paths...]}
gallery = {
    "Dummy 2": ["data/spk2_snt1.wav", "data/spk2_snt2.wav"],
    "Dummy 1"  : ["data/spk1_snt2.wav",   "data/spk1_snt3.wav"],
    "Charlie": ["data/charlie_2.wav", "data/charlie_1.wav"],
}

enroll = {}
for name, paths in gallery.items():
    vecs = [embedding(p) for p in paths]
    enroll[name] = np.mean(vecs, axis=0)  # centroid per speaker

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