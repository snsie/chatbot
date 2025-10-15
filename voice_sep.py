# voice_sep.py
import os, tempfile
import numpy as np
import torch, torchaudio, soundfile as sf
from typing import Tuple, Optional
from speechbrain.inference.separation import SepformerSeparation as Separator

_SEPFORMER = None


def get_sepformer() -> Separator:
    """Lazy-load and cache the SpeechBrain SepFormer model."""
    global _SEPFORMER
    if _SEPFORMER is None:
        _SEPFORMER = Separator.from_hparams(
            source="speechbrain/sepformer-wsj02mix",
            savedir="pretrained_sepformer"
        )
        _SEPFORMER.separation_model.to(_device()).eval()
    return _SEPFORMER
def extract_target_voice(audio, sr, spkrec, target_emb, sim_threshold=0.65, device="cuda"):
    from voice_sep import separate
    stems = separate(audio, sr)
    best_sim, best_idx, best_stem = 0.0, None, None

    for i, stem in enumerate(stems):
        emb = spkrec.encode_batch(torch.tensor(stem).unsqueeze(0).to(device))
        sim = cosine_similarity(emb.cpu().numpy(), target_emb)
        if sim > best_sim:
            best_sim, best_idx, best_stem = sim, i, stem

    if best_sim < sim_threshold:
        return audio, best_sim, best_idx, False  # no good match
    else:
        return best_stem, best_sim, best_idx, True

def separate(
    audio: np.ndarray,
    sr: int,
    desired_sr: int = 16000,
    normalize: bool = True
) -> np.ndarray:
    """
    Returns np.ndarray shape (2, T_out) at desired_sr, dtype float32.
    """
    wav = _ensure_mono_float(audio)
    sr_in = int(sr)

    # Simple API path: write a temp wav, call .separate_file()
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        sf.write(tmp.name, wav, sr_in)
        tmp_path = tmp.name

    sep = get_sepformer()

    try:
        with torch.inference_mode():
            if _device() == "cuda":
                with torch.autocast("cuda", dtype=torch.float16):
                    est = sep.separate_file(path=tmp_path)  # torch [2, T]
            else:
                est = sep.separate_file(path=tmp_path)
    finally:
        try: os.remove(tmp_path)
        except OSError: pass

    est = est.detach().cpu()  # [2, T]

    # Resample each stem if needed
    if desired_sr != sr_in:
        est = torch.stack([_resample(s.view(1,-1), sr_in, desired_sr).squeeze(0) for s in est], dim=0)
    est_np = est.numpy().astype(np.float32, copy=False)

    if normalize:
        for i in range(est_np.shape[0]):
            peak = float(np.max(np.abs(est_np[i])) + 1e-8)
            if peak > 1.0:
                est_np[i] /= peak
    return est_np  # (2, T_out)
