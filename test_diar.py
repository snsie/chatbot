# test_diar.py (hardened)
import torch
import torchaudio
from pathlib import Path
from datetime import datetime

from speechbrain.inference.separation import SepformerSeparation as Separator

MIXTURE_PATH = "enroll_target.wav"  # change if needed
TARGET_SR = 8000  # WSJ0-2mix
RUN_DIR = Path("separated") / datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_DIR.mkdir(parents=True, exist_ok=True)

# 1) Load audio
wav, sr = torchaudio.load(MIXTURE_PATH)  # [C, T]
if wav.shape[0] > 1:
    wav = wav.mean(dim=0, keepdim=True)  # -> [1, T]

# 2) Resample if needed
if sr != TARGET_SR:
    print(f"Resampling from {sr} Hz to {TARGET_SR} Hz")
    wav = torchaudio.functional.resample(wav, orig_freq=sr, new_freq=TARGET_SR)
    sr = TARGET_SR

# 3) Model
separator = Separator.from_hparams(
    source="speechbrain/sepformer-wsj02mix",
    savedir="pretrained_sepformer"
)
device = next(separator.parameters()).device
wav = wav.to(device)  # [1, T]

with torch.inference_mode():
    est_sources = separator.separate_batch(wav)  # [1, nsrc, T] (expected)

print(f"[debug] raw est_sources shape: {tuple(est_sources.shape)}")

# 4) Normalize shape to [nsrc, T]
if est_sources.ndim == 3:
    # Expected [B, S, T]; drop batch if B==1
    if est_sources.shape[0] != 1:
        raise RuntimeError(f"Unexpected batch size {est_sources.shape[0]}; expected 1.")
    est_sources = est_sources.squeeze(0)  # -> [S, T]
elif est_sources.ndim != 2:
    raise RuntimeError(f"Unexpected est_sources dims: {est_sources.ndim}; expected 2.")

S, T = est_sources.shape[0], est_sources.shape[1]

# Detect transposed case [T, S] and fix it
if S > 16 and T <= 8:  # absurd #speakers, tiny time
    print("[warn] est_sources seems transposed ([time, nsrc]); fixing...")
    est_sources = est_sources.T  # -> [S, T]
    S, T = est_sources.shape

# Another heuristic: if "speakers" dimension is huge (e.g., >64),
# it’s almost certainly the time dimension; transpose.
if S > 64 and T < S:
    print("[warn] est_sources likely [time, nsrc]; transposing to [nsrc, time]...")
    est_sources = est_sources.T
    S, T = est_sources.shape

print(f"[debug] normalized est_sources shape: [S={S}, T={T}]")

# Safety: limit to at most 4 streams
if S > 4:
    print(f"[warn] S={S} is unusually large; capping saves to first 2 streams.")
    S = 2

# 5) Save exactly S streams (for WSJ0-2mix, S should be 2)
for i in range(S):
    spk = est_sources[i].unsqueeze(0).detach().to("cpu")  # [1, T]
    out = RUN_DIR / f"speaker{i+1}.wav"
    torchaudio.save(str(out), spk, sr)
    dur = spk.shape[-1] / sr
    amp = float(spk.abs().mean())
    print(f"✅ {out}  duration={dur:.2f}s  mean|amp|={amp:.6f}")

print("Done. Output folder:", RUN_DIR)

