"""
Fully streaming voice chatbot (STT -> GPT-OSS -> TTS) in Python

Features
- Half‑duplex streaming loop: listen for speech, transcribe, stream LLM tokens,
  speak sentences as soon as they complete (low latency), then return to listening.
- Local/offline by default (Whisper via faster-whisper + pyttsx3).  
  Swap TTS to Edge TTS (internet) or Piper (offline) easily.
- Works on CPU; uses CUDA automatically if available.

Requirements (install):
    pip install faster-whisper sounddevice webrtcvad numpy pyttsx3 ollama tiktoken

Optional (better voices / alternatives):
    pip install edge-tts pydub simpleaudio  # for Edge TTS streaming playback

System deps (Linux):
    sudo apt-get install -y portaudio19-dev espeak-ng  # for mic + pyttsx3 voices

Run Ollama (separate terminal):
    curl -fsSL https://ollama.com/install.sh | sh
    ollama serve
    ollama pull llama3.1:8b-instruct   # or mistral, phi, qwen2.5, etc.

Usage:
    python streaming_voice_chatbot.py

Press Ctrl+C to quit.
"""
from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import queue
import re
import sys
import threading
import time
from typing import AsyncGenerator, Generator, Iterable, Optional

import numpy as np
import sounddevice as sd
import webrtcvad
from faster_whisper import WhisperModel

# --- Configuration -----------------------------------------------------------

SAMPLE_RATE = 16000  # webrtcvad requires 8000/16000/32000/48000; Whisper expects 16k
FRAME_MS = 30        # valid: 10, 20, 30
FRAME_SAMPLES = SAMPLE_RATE * FRAME_MS // 1000
VAD_AGGRESSIVENESS = 2  # 0..3; higher = more aggressive (more filtering)

# Silence detection (utterance segmentation)
MIN_UTTERANCE_MS = 400        # require at least this much voiced audio to trigger
TRAILING_SILENCE_MS = 800     # end utterance after this much silence

# Whisper model & device
WHISPER_MODEL = "base.en"     # good baseline; options: tiny/tiny.en/base/base.en/small/small.en/medium/large-v3
WHISPER_COMPUTE = "auto"      # "auto" | "cuda" | "cpu"

# LLM via Ollama
OLLAMA_MODEL = "llama3.1:8b-instruct"  # make sure it's pulled
SYSTEM_PROMPT = (
    "You are a concise, helpful voice assistant. "
    "Keep answers short unless asked for detail."
)
MAX_TOKENS = 512

# TTS backend: "pyttsx3" (offline & simple), or "edge-tts" (internet, higher quality).
TTS_BACKEND = "pyttsx3"
VOICE_NAME = None  # pyttsx3 voice name (or None for default). For edge-tts use e.g. "en-US-JennyNeural".

# Sentence segmentation while streaming tokens from LLM
SENTENCE_END_RE = re.compile(r"([.!?…]+)(\s+|$)")

# --- Utilities ---------------------------------------------------------------

@dataclasses.dataclass
class Frame:
    bytes: bytes
    timestamp: float
    duration: float


def _now() -> float:
    return time.perf_counter()


def frame_generator(stream: sd.RawInputStream) -> Generator[Frame, None, None]:
    """Yield 30ms PCM16 frames from sounddevice RawInputStream (blocking)."""
    base_ts = _now()
    t = base_ts
    while True:
        data, overflowed = stream.read(FRAME_SAMPLES)
        if overflowed:
            # We proceed anyway; VAD is robust to occasional loss.
            pass
        yield Frame(bytes=data, timestamp=t, duration=FRAME_MS / 1000.0)
        t += FRAME_MS / 1000.0


def pcm16_bytes_to_float32(frames: Iterable[Frame]) -> np.ndarray:
    raw = b"".join(f.bytes for f in frames)
    arr = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    return arr


# --- STT: VAD-based utterance detector + Whisper transcription ---------------

class UtteranceDetector:
    def __init__(self, aggressiveness: int = VAD_AGGRESSIVENESS):
        self.vad = webrtcvad.Vad(aggressiveness)

    def collect(self, frames: Iterable[Frame]) -> Generator[list[Frame], None, None]:
        """Accumulate frames into utterances based on VAD + trailing silence.
        Yields a list[Frame] per utterance.
        """
        voiced = []
        silence_ms = 0
        voiced_ms = 0
        in_utt = False
        for f in frames:
            is_speech = False
            with contextlib.suppress(Exception):
                is_speech = self.vad.is_speech(f.bytes, SAMPLE_RATE)
            if is_speech:
                voiced.append(f)
                voiced_ms += FRAME_MS
                silence_ms = 0
                if not in_utt and voiced_ms >= MIN_UTTERANCE_MS:
                    in_utt = True
            else:
                if in_utt:
                    silence_ms += FRAME_MS
                    if silence_ms >= TRAILING_SILENCE_MS:
                        # end utterance
                        yield voiced
                        voiced = []
                        silence_ms = 0
                        voiced_ms = 0
                        in_utt = False
                else:
                    # not yet in an utterance; reset voiced buffer if too long silence
                    voiced = []
                    voiced_ms = 0


class WhisperSTT:
    def __init__(self, model_name: str = WHISPER_MODEL, compute_type: str = WHISPER_COMPUTE):
        device = None
        if compute_type == "auto":
            # Try CUDA first
            try:
                import torch  # type: ignore
                device = "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:
                device = "cpu"
        elif compute_type in ("cuda", "cpu"):
            device = compute_type
        else:
            device = "cpu"

        print(f"[Whisper] Loading model '{model_name}' on {device}…")
        self.model = WhisperModel(model_name, device=device)

    def transcribe(self, audio_f32: np.ndarray) -> str:
        # Faster-Whisper expects 16k float32 mono
        segments, _info = self.model.transcribe(
            audio_f32,
            vad_filter=False,
            beam_size=1,
            language="en",
        )
        return " ".join(seg.text.strip() for seg in segments).strip()


# --- LLM streaming via Ollama ------------------------------------------------

class StreamLLM:
    def __init__(self, model: str = OLLAMA_MODEL, system_prompt: str = SYSTEM_PROMPT):
        self.model = model
        self.system_prompt = system_prompt
        # Lazy import to keep dependencies flexible
        import ollama  # type: ignore
        self.ollama = ollama
        self.history: list[dict] = [
            {"role": "system", "content": self.system_prompt},
        ]

    async def stream_chat(self, user_text: str) -> AsyncGenerator[str, None]:
        """Stream raw token chunks from Ollama as they arrive."""
        self.history.append({"role": "user", "content": user_text})

        loop = asyncio.get_event_loop()

        def _generator() -> Iterable[str]:
            for part in self.ollama.chat(
                model=self.model,
                messages=self.history,
                stream=True,
                options={"num_predict": MAX_TOKENS},
            ):
                msg = part.get("message", {})
                yield msg.get("content", "")

        # Run the blocking generator in a thread and yield into async world
        def _producer(q: queue.Queue[str]):
            try:
                for chunk in _generator():
                    if chunk:
                        q.put(chunk)
                q.put("__END__")
            except Exception as e:
                q.put(f"__ERROR__:{e}")

        q: queue.Queue[str] = queue.Queue()
        t = threading.Thread(target=_producer, args=(q,), daemon=True)
        t.start()

        while True:
            chunk = await loop.run_in_executor(None, q.get)
            if chunk == "__END__":
                break
            if chunk.startswith("__ERROR__:"):
                raise RuntimeError(chunk)
            yield chunk

        # Append assistant final message to history (collected by caller)


# --- TTS backends ------------------------------------------------------------

class Pyttsx3Speaker:
    def __init__(self, voice_name: Optional[str] = VOICE_NAME):
        import pyttsx3  # type: ignore
        self.engine = pyttsx3.init()
        if voice_name:
            for v in self.engine.getProperty("voices"):
                if voice_name.lower() in (v.name or "").lower():
                    self.engine.setProperty("voice", v.id)
                    break
        # Slightly faster speaking rate for snappier feel
        rate = self.engine.getProperty("rate")
        self.engine.setProperty("rate", int(rate * 1.05))
        self._lock = threading.Lock()

    async def speak_sentences(self, sentences: AsyncGenerator[str, None]) -> None:
        loop = asyncio.get_event_loop()
        for sentence in sentences:
            # In case caller passed a sync generator
            if asyncio.iscoroutine(sentence):  # pragma: no cover (safety)
                sentence = await sentence
            text = sentence.strip()
            if not text:
                continue
            await loop.run_in_executor(None, self._speak_blocking, text)

    def _speak_blocking(self, text: str) -> None:
        with self._lock:
            self.engine.say(text)
            self.engine.runAndWait()


class EdgeTTSSpeaker:
    """Optional higher-quality, internet TTS. Requires: pip install edge-tts simpleaudio pydub
    Note: Playback uses pydub+simpleaudio. Ensure ffmpeg is installed.
    """
    def __init__(self, voice: str = "en-US-JennyNeural"):
        import edge_tts  # type: ignore
        from pydub import AudioSegment  # type: ignore
        import simpleaudio  # type: ignore

        self.edge_tts = edge_tts
        self.AudioSegment = AudioSegment
        self.simpleaudio = simpleaudio
        self.voice = voice

    async def speak_sentences(self, sentences: AsyncGenerator[str, None]) -> None:
        async for sentence in sentences:
            text = sentence.strip()
            if not text:
                continue
            communicate = self.edge_tts.Communicate(text=text, voice=self.voice)
            wav_bytes = bytearray()
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    wav_bytes.extend(chunk["data"])  # default is mp3 frames
            # Decode to PCM and play
            seg = self.AudioSegment.from_file_using_temporary_files(
                bytes(wav_bytes), format="mp3"
            )
            play_obj = self.simpleaudio.play_buffer(
                seg.raw_data,
                num_channels=seg.channels,
                bytes_per_sample=seg.sample_width,
                sample_rate=seg.frame_rate,
            )
            play_obj.wait_done()


# --- Sentence chunker for streaming tokens to TTS ----------------------------

async def sentence_stream(token_stream: AsyncGenerator[str, None]) -> AsyncGenerator[str, None]:
    """Accumulate incoming token chunks and yield sentences as soon as they complete.
    Uses punctuation heuristics to keep latency low.
    """
    buf = ""
    async for chunk in token_stream:
        buf += chunk
        while True:
            m = SENTENCE_END_RE.search(buf)
            if not m:
                break
            idx = m.end()
            sentence = buf[:idx]
            buf = buf[idx:]
            yield sentence.strip()
    # Flush remainder
    if buf.strip():
        yield buf.strip()


# --- Orchestration -----------------------------------------------------------

async def listen_once_and_respond(
    stt: WhisperSTT,
    llm: StreamLLM,
    speaker_backend: str = TTS_BACKEND,
) -> None:
    # 1) Listen and segment a single utterance
    print("\n🎤 Listening… (speak now)")
    vad = UtteranceDetector(VAD_AGGRESSIVENESS)
    with sd.RawInputStream(
        samplerate=SAMPLE_RATE,
        blocksize=FRAME_SAMPLES,
        dtype="int16",
        channels=1,
        latency="low",
    ) as stream:
        frames_iter = frame_generator(stream)
        for utter_frames in vad.collect(frames_iter):
            audio_f32 = pcm16_bytes_to_float32(utter_frames)
            break  # take the first utterance only

    # 2) Transcribe
    print("📝 Transcribing…")
    user_text = stt.transcribe(audio_f32)
    print(f"You: {user_text}")
    if not user_text:
        print("(heard nothing)")
        return

    # 3) LLM stream -> sentence chunker -> TTS
    print("🤖 Assistant (streaming)…")
    token_stream = llm.stream_chat(user_text)
    sentence_gen = sentence_stream(token_stream)

    if speaker_backend == "pyttsx3":
        speaker = Pyttsx3Speaker(voice_name=VOICE_NAME)
        await speaker.speak_sentences(sentence_gen)
    elif speaker_backend == "edge-tts":
        speaker = EdgeTTSSpeaker(voice=VOICE_NAME or "en-US-JennyNeural")
        await speaker.speak_sentences(sentence_gen)
    else:
        raise ValueError(f"Unknown TTS backend: {speaker_backend}")


async def main() -> None:
    print("Booting streaming voice chatbot…")
    stt = WhisperSTT(WHISPER_MODEL, WHISPER_COMPUTE)
    llm = StreamLLM(OLLAMA_MODEL, SYSTEM_PROMPT)

    try:
        while True:
            await listen_once_and_respond(stt, llm, TTS_BACKEND)
            # Small pause to avoid capturing our own final audio tail
            await asyncio.sleep(0.15)
    except KeyboardInterrupt:
        print("\nBye! 👋")


if __name__ == "__main__":
    if sys.platform == "win32":
        # Windows event loop policy fix for asyncio
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
