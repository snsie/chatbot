# build_enrollments.py
"""
Builds speaker centroids from audio files and saves them to enrollments.npz.

Folder layout (example):
data/
  Alice/
    a1.wav
    a2.wav
  Bob/
    b1.wav
    b2.wav

Run:
  python build_enrollments.py --root data --out enrollments.npz

Options:
  --min-clips 2        # require at least this many clips per speaker (default 2)
  --max-clips 5        # limit clips per speaker to speed up (default 5; 0 = no limit)
  --ext wav flac       # file extensions to include (default: wav)
  --savedir pretrained_ecapa  # cache dir for SpeechBrain model
"""

from __future__ import annotations
import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly
from speechbrain.pretrained import EncoderClassifier


TARGET_SR = 16000


def l2norm(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v) + 1e-12
    return (v / n).astype(np.float32)


def load_audio_16k_mono(path: Path) -> np.ndarray:
    """Load audio with soundfile, convert to float32 mono @ 16 kHz."""
    audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if audio.ndim == 2:  # stereo -> mono
        audio = audio.mean(axis=1)
    if sr != TARGET_SR:
        # high-quality resample: resample_poly
        # Up/Down ratios chosen to avoid large numbers
        from math import gcd
        g = gcd(sr, TARGET_SR)
        up, down = TARGET_SR // g, sr // g
        audio = resample_poly(audio, up, down).astype(np.float32, copy=False)
    # Ensure contiguous 1-D float32 in [-1, 1]
    return np.ascontiguousarray(audio, dtype=np.float32)


@torch.no_grad()
def embed_waveform(model: EncoderClassifier, wav_f32: np.ndarray) -> np.ndarray:
    """wav_f32: float32 mono @16k, shape [T]"""
    wav = torch.from_numpy(wav_f32).unsqueeze(0)  # [1, T]
    emb = model.encode_batch(wav).squeeze(0).squeeze(0).cpu().numpy()
    return l2norm(emb)


def find_gallery(root: Path, exts: List[str], max_clips: int) -> Dict[str, List[Path]]:
    """Return {speaker_name: [file paths...]} scanning subdirs of root."""
    exts = [e.lower().lstrip(".") for e in exts]
    gallery: Dict[str, List[Path]] = {}
    for spk_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        files = []
        for ext in exts:
            files.extend(sorted(spk_dir.glob(f"*.{ext}")))
        if max_clips > 0:
            files = files[:max_clips]
        if files:
            gallery[spk_dir.name] = files
    return gallery


def main():
    p = argparse.ArgumentParser(description="Build enrollments.npz from audio folders.")
    p.add_argument("--root", type=Path, required=True, help="Root folder with subfolders per speaker")
    p.add_argument("--out", type=Path, default=Path("enrollments.npz"), help="Output .npz path")
    p.add_argument("--savedir", type=Path, default=Path("pretrained_ecapa"), help="Cache dir for SpeechBrain model")
    p.add_argument("--min-clips", type=int, default=2, help="Minimum clips per speaker to include")
    p.add_argument("--max-clips", type=int, default=5, help="Limit clips per speaker (0 = no limit)")
    p.add_argument("--ext", nargs="+", default=["wav"], help="File extensions to include (e.g., wav flac)")
    args = p.parse_args()

    root: Path = args.root
    if not root.exists() or not root.is_dir():
        raise SystemExit(f"[!] Root folder not found: {root}")

    # 1) Discover files
    gallery = find_gallery(root, args.ext, args.max_clips)
    if not gallery:
        raise SystemExit(f"[!] No speaker folders with audio found under: {root}")

    # 2) Load model once
    print("↓ Loading SpeechBrain ECAPA model (first run may take a bit)…")
    model = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir=str(args.savedir)
    )

    # 3) Build centroids
    vectors: Dict[str, np.ndarray] = {}
    total_files = 0
    for speaker, files in gallery.items():
        if len(files) < args.min_clips:
            print(f" - Skipping {speaker}: only {len(files)} clip(s) (< min {args.min_clips})")
            continue

        embs = []
        for f in files:
            try:
                wav = load_audio_16k_mono(f)
                if wav.size < TARGET_SR:  # <1s of audio → warn
                    print(f"   ! {speaker}: {f.name} is very short (<1s); consider longer clips")
                embs.append(embed_waveform(model, wav))
            except Exception as e:
                print(f"   ! Skipping {f} ({e})")
                continue

        if len(embs) == 0:
            print(f" - Skipping {speaker}: no valid clips after reading")
            continue

        centroid = l2norm(np.mean(np.stack(embs, axis=0), axis=0))
        vectors[speaker] = centroid
        total_files += len(embs)
        print(f" ✓ Enrolled {speaker} from {len(embs)} clip(s)")

    if not vectors:
        raise SystemExit("[!] No speakers enrolled; nothing to save.")

    # 4) Save .npz
    names = np.array(list(vectors.keys()), dtype=object)
    vecs = np.array(list(vectors.values()), dtype=object)
    np.savez(args.out, names=names, vecs=vecs)
    print(f"\n💾 Saved {len(names)} speakers ({total_files} clips) → {args.out}")

    # 5) Quick sanity print
    print("Speakers:", ", ".join(vectors.keys()))


if __name__ == "__main__":
    main()
