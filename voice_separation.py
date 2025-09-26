# voice_sep.py
# Minimal SpeechBrain SepFormer wrapper for 2-speaker separation.
# ---------------------------------------------------------------
# Requires: pip install speechbrain torch torchaudio soundfile numpy
#
# Usage (in your chatbot):
#   from voice_sep import separate
#   stems = separate(audio_np, sr=16000)        # -> np.ndarray shape [2, T] at 16 kHz by default
#   for stem in stems:
#       name, sim = voice_identifier.identify_from_array(stem, 16000)
#   ...pick best stem, send to ASR...

from __future__ import annotations
import os
import tempfile
from typing import Tuple
import numpy as np
import torch
import soundfile as sf
import torchaudio
from speechbrain.inference.separation import SepformerSeparation as Separator

# Singleton cache for the heavy model
_SEPFORMER = None

def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"

def get_sepformer() -> Separator:
    """
    Lazy-loads the pretrained SepFormer (WSJ0-2mix) checkpoint once and reuses it.
    Moves model to GPU if available and sets eval mode.
    """
    global _SEPFORMER
    if _SEPFORMER is None:
        _SEPFORMER = Separator.from_hparams(
            source="speechbrain/sepformer-wsj02mix",
            savedir="pretrained_sepformer"
        )
        _SEPFORMER.separation_model.to(_device()).eval()
    return _SEPFORMER

def _ensure_mono_float(audio: np.ndarray) -> np.ndarray:
    """
    Accepts (T,) or (C,T) float/int arrays. Returns (T,) float32 mono.
    """
    a = np.asarray(audio)
    if a.ndim == 2:   # (C, T)
        a = a.mean(axis=0)
    a = a.astype(np.float32, copy=False)
    # Simple peak normalization safeguard (no loudness change if already safe)
    peak = float(np.max(np.abs(a)) + 1e-8)
    if peak > 1.0:
        a = a / peak
    return a

def _resample_if_needed(wav: torch.Tensor, sr_in: int, sr_out: int) -> torch.Tensor:
    """
    wav: (1, T) float tensor
    """
    if sr_in == sr_out:
        return wav
    return torchaudio.functional.resample(wav, sr_in, sr_out)

def separate(
    audio: np.ndarray,
    sr: int,
    desired_sr: int = 16000,
    normalize: bool = True
) -> np.ndarray:
    """
    Separates a 2-speaker mixture into two stems using SepFormer.

    Parameters
    ----------
    audio : np.ndarray
        Mono waveform as (T,) or (C,T). Any dtype; will be cast to float32.
    sr : int
        Sample rate of `audio`.
    desired_sr : int, optional
        Resample separated stems to this rate for downstream (default: 16000).
    normalize : bool, optional
        If True, peak-normalize each stem to avoid clipping.

    Returns
    -------
    np.ndarray
        Array of shape (2, T_out) at `desired_sr`, dtype float32.
        Each row is one separated speaker.
    """
    # Prep input
    wav = _ensure_mono_float(audio)
    sr_in = int(sr)

    # Write a temp WAV because SepformerSeparation exposes .separate_file(...)
    # (keeps the wrapper simple & stable)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        sf.write(tmp.name, wav, sr_in)
        tmp_path = tmp.name

    sep = get_sepformer()

    try:
        with torch.inference_mode():
            if _device() == "cuda":
                # Use FP16 on GPU for speed/memory
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    est_sources = sep.separate_file(path=tmp_path)  # torch.Tensor [2, T]
            else:
                est_sources = sep.separate_file(path=tmp_path)
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass

    # est_sources is [2, T] on CPU or GPU; bring to CPU
    est = est_sources.detach().cpu()

    # Resample if needed (each stem is (T,))
    if desired_sr and desired_sr != sr_in:
        est = torch.stack([
            _resample_if_needed(s.view(1, -1), sr_in, desired_sr).squeeze(0)
            for s in est
        ], dim=0)
        sr_out = desired_sr
    else:
        sr_out = sr_in

    # Convert to numpy float32
    est_np = est.numpy().astype(np.float32, copy=False)

    # Optional per-stem peak normalize (keeps dynamics; prevents clipping)
    if normalize:
        for i in range(est_np.shape[0]):
            peak = float(np.max(np.abs(est_np[i])) + 1e-8)
            if peak > 1.0:
                est_np[i] /= peak

    # Ensure shape (2, T_out)
    return est_np

# -------------------------------
# Optional: simple CLI test hook
# -------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Separate a 2-speaker mixture with SepFormer.")
    parser.add_argument("mixture", type=str, help="Path to input WAV (mono).")
    parser.add_argument("--sr", type=int, default=16000, help="Assumed sample rate of input file.")
    parser.add_argument("--out_a", type=str, default="speaker1.wav", help="Output WAV for stem A.")
    parser.add_argument("--out_b", type=str, default="speaker2.wav", help="Output WAV for stem B.")
    parser.add_argument("--desired_sr", type=int, default=16000, help="Output sample rate.")
    args = parser.parse_args()

    # Load file
    wav, sr_in = torchaudio.load(args.mixture)
    if sr_in != args.sr:
        wav = torchaudio.functional.resample(wav, sr_in, args.sr)
        sr_in = args.sr
    # (C,T) -> (T,)
    wav_np = wav.mean(dim=0).numpy()

    stems = separate(wav_np, sr=sr_in, desired_sr=args.desired_sr)
    sf.write(args.out_a, stems[0], args.desired_sr)
    sf.write(args.out_b, stems[1], args.desired_sr)
    print(f"Saved:\n - {args.out_a}\n - {args.out_b}")
