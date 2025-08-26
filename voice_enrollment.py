"""
Multi‑User Voice Enrollment Script
---------------------------------
Records a short sample for a person and saves a voiceprint embedding to
voiceprints/<name>.npy using Resemblyzer. Run once per person (or multiple
times to refine their print; embeddings will be averaged).

Usage:
    python enroll_user.py --name "charlie" --seconds 8

Install deps:
    pip install resemblyzer sounddevice numpy
"""

import argparse
from pathlib import Path
import numpy as np
import sounddevice as sd
from resemblyzer import VoiceEncoder, preprocess_wav

SAMPLE_RATE = 16000
CHANNELS = 1
VOICEPRINT_DIR = Path("voiceprints")
VOICEPRINT_DIR.mkdir(exist_ok=True)


def record_seconds(seconds: int) -> np.ndarray:
    print(f"[Enroll] Recording {seconds}s… Speak naturally.")
    audio = sd.rec(frames=seconds * SAMPLE_RATE, samplerate=SAMPLE_RATE,
                   channels=CHANNELS, dtype="float32")
    sd.wait()
    return audio.squeeze()


def save_or_average(name: str, emb: np.ndarray):
    fp = VOICEPRINT_DIR / f"{name}.npy"
    if fp.exists():
        old = np.load(fp)
        # Simple running average of two embeddings (good enough to smooth noise)
        new = (old + emb) / 2.0
        np.save(fp, new)
        print(f"[Enroll] Updated existing voiceprint: {fp}")
    else:
        np.save(fp, emb)
        print(f"[Enroll] Saved new voiceprint: {fp}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="Person's unique name (filename friendly)")
    ap.add_argument("--seconds", type=int, default=8, help="Recording length in seconds (6–10 is typical)")
    args = ap.parse_args()

    wav = record_seconds(args.seconds)
    print("[Enroll] Computing embedding…")
    encoder = VoiceEncoder()
    emb = encoder.embed_utterance(preprocess_wav(wav, SAMPLE_RATE))
    save_or_average(args.name.strip().lower(), emb)
    print("✅ Done.")

if __name__ == "__main__":
    main()
