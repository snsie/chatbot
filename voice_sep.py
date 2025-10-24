# voice_sep.py
import os
import tempfile
from typing import Optional, Tuple

import numpy as np
import soundfile as sf
import torch
import torchaudio
from speechbrain.inference.separation import SepformerSeparation as Separator

# -----------------------------
# Global cache for the SepFormer
# -----------------------------
_SEPFORMER: Optional[Separator] = None


def _device() -> str:
    """Return the best available device string."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def _ensure_mono_float(x: np.ndarray) -> np.ndarray:
    """
    Ensure waveform is mono float32 in [-1, 1].
    Accepts (T,), (T, C), or (C, T). Downmixes via mean if needed.
    """
    x = np.asarray(x)
    if x.ndim == 2:
        # Heuristic: treat the smaller dimension (<=8) as channels and average across it.
        if x.shape[0] <= 8 and x.shape[1] > x.shape[0]:
            # (C, T) -> (T,)
            x = x.mean(axis=0)
        elif x.shape[1] <= 8:
            # (T, C) -> (T,)
            x = x.mean(axis=1)
        else:
            # Fallback: average across the smaller axis
            axis = 0 if x.shape[0] < x.shape[1] else 1
            x = x.mean(axis=axis)
    # Ensure float32
    return x.astype(np.float32, copy=False)


def _resample(wav: torch.Tensor, sr_in: int, sr_out: int) -> torch.Tensor:
    """
    Resample [1, T] tensor from sr_in -> sr_out using torchaudio.
    Returns [1, T_out].
    """
    if sr_in == sr_out:
        return wav
    resampler = torchaudio.transforms.Resample(sr_in, sr_out)
    return resampler(wav)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two 1-D vectors."""
    a = a.reshape(-1)
    b = b.reshape(-1)
    denom = (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)
    return float(np.dot(a, b) / denom)


def get_sepformer() -> Separator:
    """Lazy-load and cache the SpeechBrain SepFormer model."""
    global _SEPFORMER
    if _SEPFORMER is None:
        _SEPFORMER = Separator.from_hparams(
            source="speechbrain/sepformer-wsj02mix",
            savedir="pretrained_sepformer",
        )
        # Move underlying torch model to device and set eval
        _SEPFORMER.separation_model.to(_device()).eval()
    return _SEPFORMER


def separate(
    audio: np.ndarray,
    sr: int,
    desired_sr: int = 16000,
    normalize: bool = True,
) -> np.ndarray:
    """
    Separate a mixture into 2 stems using SepFormer.

    Args:
        audio: np.ndarray audio (mono or stereo), typically float in [-1, 1].
        sr: input sample rate.
        desired_sr: output sample rate (default 16 kHz).
        normalize: if True, peak-normalize only if |peak| > 1.0 (prevents clipping).

    Returns:
        np.ndarray with shape (2, T_out), dtype float32, at desired_sr.
    """
    wav = _ensure_mono_float(audio)
    sr_in = int(sr)

    # Use the file-based API for simplicity
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        sf.write(tmp.name, wav, sr_in)
        tmp_path = tmp.name

    sep = get_sepformer()

    try:
        with torch.inference_mode():
            if _device() == "cuda":
                # Use autocast for faster GPU inference
                with torch.autocast("cuda", dtype=torch.float16):
                    est = sep.separate_file(path=tmp_path)  # torch [2, T]
            else:
                est = sep.separate_file(path=tmp_path)  # torch [2, T]
    finally:
        # Clean up the temp file
        try:
            os.remove(tmp_path)
        except OSError:
            pass

    # Move to CPU for post-processing
    est = est.detach().cpu()  # [2, T]

    # Optional resample per stem
    if desired_sr != sr_in:
        est = torch.stack(
            [
                _resample(s.view(1, -1), sr_in, desired_sr).squeeze(0)
                for s in est
            ],
            dim=0,
        )  # [2, T_out]

    # To numpy float32
    est_np = est.numpy().astype(np.float32, copy=False)  # (2, T_out)

    # Clip-prevention normalization only (doesn't force fixed loudness)
    if normalize:
        for i in range(est_np.shape[0]):
            peak = float(np.max(np.abs(est_np[i])) + 1e-8)
            if peak > 1.0:
                est_np[i] /= peak

    return est_np  # (2, T_out)


def extract_target_voice(
    audio: np.ndarray,
    sr: int,
    spkrec,
    target_emb: np.ndarray,
    sim_threshold: float = 0.65,
    device: Optional[str] = None,
) -> Tuple[np.ndarray, float, Optional[int], bool]:
    """
    Separate audio into 2 stems, embed each stem with `spkrec`, compare to `target_emb`,
    and return the best-matching stem if its cosine similarity >= sim_threshold.

    Args:
        audio: mixture audio array.
        sr: sample rate of `audio`.
        spkrec: speaker embedding model with an `encode_batch(tensor[1, T]) -> [1, D]`.
        target_emb: target speaker embedding as a 1-D np.ndarray of dim D.
        sim_threshold: minimum cosine similarity to accept a match.
        device: 'cuda' or 'cpu'. If None, auto-detect.

    Returns:
        (best_audio, best_sim, best_idx, matched)
          best_audio: the best stem if matched, else the original mixture.
          best_sim: highest cosine similarity score (float).
          best_idx: index of chosen stem (0 or 1), or None if not matched.
          matched: True if best_sim >= sim_threshold, else False.
    """
    device = device or _device()

    # 1) Separate into two stems at 16k (default)
    stems = separate(audio, sr)  # np.float32, (2, T_out)

    best_sim: float = 0.0
    best_idx: Optional[int] = None
    best_stem: Optional[np.ndarray] = None

    # 2) For each stem, compute an embedding and compare to target_emb
    for i, stem in enumerate(stems):
        # to torch [1, T]
        t = torch.tensor(stem, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.inference_mode():
            emb = spkrec.encode_batch(t)  # expected shape [1, D]
        emb_np = emb.squeeze(0).detach().cpu().numpy()  # [D]

        sim = cosine_similarity(emb_np, target_emb)
        if sim > best_sim:
            best_sim, best_idx, best_stem = sim, i, stem

    # 3) Decide if we have a good match
    matched = (best_stem is not None) and (best_sim >= sim_threshold)
    if matched:
        return best_stem, best_sim, best_idx, True
    else:
        return audio, best_sim, best_idx, False
