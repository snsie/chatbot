#!/usr/bin/env python3
"""PersonaPlex local audio-to-audio chat runner.

This script records a user utterance, sends it to PersonaPlex (moshi.offline),
then plays back the generated response audio. PersonaPlex handles ASR + TTS
internally, so no external Whisper/Ollama/TTS stack is needed.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import tempfile
from typing import List, Optional, Tuple

import numpy as np
import sounddevice as sd
import soundfile as sf

from .profiles import ProfileError, get_profile, resolve_prompt_path


SAMPLE_RATE = 24_000
FRAME_MS = 20
FRAME_SAMPLES = int(SAMPLE_RATE * FRAME_MS / 1000)
MIN_UTTERANCE_SEC = 0.6
SILENCE_TIMEOUT_SEC = 0.7
SILENCE_RMS_THRESHOLD = 0.012


def _rms(frame: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(frame))))


def record_until_silence(
    sample_rate: int = SAMPLE_RATE,
    silence_threshold: float = SILENCE_RMS_THRESHOLD,
    silence_timeout_sec: float = SILENCE_TIMEOUT_SEC,
    min_utterance_sec: float = MIN_UTTERANCE_SEC,
) -> np.ndarray:
    """Record microphone audio until silence is detected."""
    q: "queue.Queue[np.ndarray]" = queue.Queue()

    def callback(indata, frames, time_info, status):  # noqa: ANN001
        if status:
            print(f"[Audio] {status}", file=sys.stderr)
        q.put(indata.copy())

    silence_frames_required = int(silence_timeout_sec * sample_rate / FRAME_SAMPLES)
    min_frames_required = int(min_utterance_sec * sample_rate / FRAME_SAMPLES)

    collected: List[np.ndarray] = []
    silence_frames = 0
    voiced_frames = 0

    print("🎤 Listening...", flush=True)
    with sd.InputStream(
        samplerate=sample_rate,
        channels=1,
        dtype="float32",
        blocksize=FRAME_SAMPLES,
        callback=callback,
    ):
        while True:
            frame = q.get()
            rms = _rms(frame)
            if rms >= silence_threshold:
                silence_frames = 0
                voiced_frames += 1
                collected.append(frame)
            else:
                silence_frames += 1
                if collected:
                    collected.append(frame)
                if silence_frames >= silence_frames_required and voiced_frames >= min_frames_required:
                    break

    if not collected:
        raise RuntimeError("No audio captured. Try speaking a little louder.")

    audio = np.concatenate(collected, axis=0).flatten()
    return audio


def run_personaplex_offline(
    voice_prompt: Path,
    text_prompt: Path,
    input_wav: Path,
    output_wav: Path,
    output_text: Path,
    seed: int,
    cpu_offload: bool,
) -> None:
    cmd = [
        sys.executable,
        "-m",
        "moshi.offline",
        "--voice-prompt",
        str(voice_prompt),
        "--text-prompt",
        text_prompt.read_text(),
        "--input-wav",
        str(input_wav),
        "--seed",
        str(seed),
        "--output-wav",
        str(output_wav),
        "--output-text",
        str(output_text),
    ]
    if cpu_offload:
        cmd.append("--cpu-offload")

    print("🧠 Running PersonaPlex inference...", flush=True)
    subprocess.run(cmd, check=True)


def play_audio(audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> None:
    print("🔊 Speaking...", flush=True)
    sd.play(audio, samplerate=sample_rate)
    sd.wait()


def load_profile(
    profile_name: Optional[str],
    profiles_path: Path,
    voice_prompt: Optional[str],
    text_prompt: Optional[str],
    base_dir: Path,
) -> Tuple[Path, Path, str]:
    if profile_name:
        profile = get_profile(profile_name, profiles_path)
        voice_path = resolve_prompt_path(profile.voice_prompt, base_dir)
        text_path = resolve_prompt_path(profile.text_prompt, base_dir)
        prompt_name = profile.name
    else:
        if not (voice_prompt and text_prompt):
            raise ProfileError("Provide --profile or both --voice-prompt and --text-prompt.")
        voice_path = resolve_prompt_path(voice_prompt, base_dir)
        text_path = resolve_prompt_path(text_prompt, base_dir)
        prompt_name = "custom"

    if not voice_path.exists():
        raise ProfileError(f"Voice prompt not found: {voice_path}")
    if not text_path.exists():
        raise ProfileError(f"Text prompt not found: {text_path}")

    return voice_path, text_path, prompt_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PersonaPlex offline voice chat")
    parser.add_argument(
        "--profile",
        help="Profile name from personaplex_app/profiles.json",
        default="assistant",
    )
    parser.add_argument(
        "--profiles-path",
        default="personaplex_app/profiles.json",
    )
    parser.add_argument("--voice-prompt", help="Path to voice prompt (.pt)")
    parser.add_argument("--text-prompt", help="Path to text prompt (.txt)")
    parser.add_argument("--seed", type=int, default=42424242)
    parser.add_argument("--cpu-offload", action="store_true")
    return parser.parse_args()


def main() -> None:
    if not os.getenv("HF_TOKEN"):
        print(
            "HF_TOKEN is not set. Set it to your Hugging Face token after accepting the PersonaPlex license.",
            file=sys.stderr,
        )
        sys.exit(1)

    args = parse_args()
    base_dir = Path(__file__).resolve().parents[1]
    profiles_path = (base_dir / args.profiles_path).resolve()

    try:
        voice_prompt, text_prompt, prompt_name = load_profile(
            args.profile,
            profiles_path,
            args.voice_prompt,
            args.text_prompt,
            base_dir,
        )
    except ProfileError as exc:
        print(f"Profile error: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"🎭 Using profile: {prompt_name}")

    audio = record_until_silence()

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        input_wav = tmp_path / "input.wav"
        output_wav = tmp_path / "output.wav"
        output_text = tmp_path / "output.json"

        sf.write(input_wav, audio, SAMPLE_RATE)

        run_personaplex_offline(
            voice_prompt=voice_prompt,
            text_prompt=text_prompt,
            input_wav=input_wav,
            output_wav=output_wav,
            output_text=output_text,
            seed=args.seed,
            cpu_offload=args.cpu_offload,
        )

        if output_text.exists():
            try:
                data = json.loads(output_text.read_text())
                transcript = data.get("text") or data
                print(f"📝 Model text: {transcript}")
            except json.JSONDecodeError:
                print("📝 Model text saved.")

        if output_wav.exists():
            audio_out, _ = sf.read(output_wav, dtype="float32")
            play_audio(audio_out)
        else:
            print("No output audio produced.", file=sys.stderr)


if __name__ == "__main__":
    main()
