#added a new import, added 4 variables, and modified the process_turn function


#!/usr/bin/env python3
"""
Streaming Voice Chatbot
=======================

A half‑duplex (listen -> transcribe -> stream LLM -> speak sentences) voice assistant
with low latency sentence‑by‑sentence TTS while tokens stream from an Ollama model.

Features
--------
1. Microphone capture @ 16 kHz mono (30 ms frames) using sounddevice RawInputStream.
2. Voice Activity Detection (webrtcvad) to segment utterances.
3. Speech‑to‑Text via faster-whisper (GPU auto, fallback CPU) on captured utterance.
4. Streaming LLM responses token-by-token from Ollama (llama3.1:8b-instruct by default).
5. Sentence segmentation of streaming tokens; each completed sentence immediately sent to TTS.
6. Two TTS backends:
   - pyttsx3 (offline, default)
   - edge-tts (optional, higher quality, requires internet + ffmpeg)
7. Clean shutdown on Ctrl+C.

Configuration (edit constants below) controls sample rate, model names, thresholds, etc.

Dependencies (pip install ...)
------------------------------
Core:
  faster-whisper
  sounddevice
  webrtcvad
  numpy
  pyttsx3
  ollama
  tiktoken  (optional, not strictly needed but listed per spec)

Optional for enhanced TTS:
  edge-tts
  pydub
  simpleaudio

Optional for neural TTS (Coqui):
  TTS
  torch
  torchaudio

System packages (Ubuntu examples):
  sudo apt-get update && sudo apt-get install -y \
       portaudio19-dev ffmpeg espeak-ng

Quick Start
-----------
1. Start Ollama server (separate terminal):
     ollama serve
2. Pull desired model (first time):
     ollama pull llama3.1:8b-instruct
3. Run this script:
     python streaming_voice_chatbot.py

Runtime Flow
------------
Loop:
  🎤 Listening… -> capture utterance
  📝 Transcribing… (print transcript as You: <text>)
  🤖 Assistant (streaming)… -> sentences spoken as generated
  (short tail delay) -> back to listening

Press Ctrl+C to exit cleanly.
"""

# =============================
# Configuration Block
# =============================
SAMPLE_RATE = 16000
FRAME_MS = 30  # ms per frame for capture + VAD
VAD_AGGRESSIVENESS = 2  # 0-3 (higher = more aggressive speech detection)
MIN_UTTERANCE_MS = 400  # minimum voiced audio required to accept an utterance
TRAILING_SILENCE_MS = 800  # silence to mark end of utterance

WHISPER_MODEL = "base.en"
WHISPER_COMPUTE = "cuda"  # 'auto' | 'cpu' | 'cuda'

OLLAMA_MODEL = "gpt-oss:20b"
# OLLAMA_MODEL="llama2:latest"
# - Use a friendly, conversational tone

SYSTEM_PROMPT = """
You are a helpful voice assistant named Cora.
Response Rules:
- Only respond to user queries that include the word "Cora" or phonetically similar words like "Kora", "Quora", "Core", "Coral", "Corona", etc.
- Speech-to-text may mishear "Cora" as similar sounding words - be flexible with variations and expect it to be the first word in the query
- When responding, consider the sentence that follows the wake word
- Keep responses concise (1-2 sentences typically)
- Speak as if having a natural conversation
- IMPORTANT: Do not use any tools or function calls. Only provide direct text responses.
- If you're unsure whether the user said "Cora", err on the side of responding rather than staying silent
- Before answering, double-check that your reply follows all these rules.
"""

MAX_TOKENS = 512

TTS_BACKEND = "edge-tts"  # 'pyttsx3' | 'edge-tts' | 'coqui'
# VOICE_NAME = "tts_models/en/vctk/vits"  # substring filter (pyttsx3) or exact edge-tts voice like 'en-US-JennyNeural' or coqui model name
VOICE_NAME = "en-US-JennyNeural"  # High-quality neural female voice

TAIL_DELAY_SEC = 0.5  # Delay after TTS before returning to mic (reduce capturing own voice)
PRINT_PARTIAL_SENTENCES = True  # Print sentences as they are spoken

# =============================
# Imports
# =============================
import asyncio
import sys
import threading
import queue
import time
import math
import re
import io
import json
from dataclasses import dataclass
from typing import AsyncGenerator, List, Optional, Iterable

import numpy as np
import sounddevice as sd
import webrtcvad

# STT / LLM
from faster_whisper import WhisperModel
import ollama  # Official Ollama Python client

# TTS (pyttsx3 always imported; optional edge-tts, pydub, simpleaudio imported lazily)
import pyttsx3

# Voice ID 
from voice_id import get_voice_identifier
ENABLE_SPEAKER_GATE = True
ENROLL_PATH = "enrollments.npz"
SIM_THRESHOLD = 0.6

voice_identifier = get_voice_identifier(ENROLL_PATH) if ENABLE_SPEAKER_GATE else None


# =============================
# Utility
# =============================
FRAME_SAMPLES = int(SAMPLE_RATE * FRAME_MS / 1000)  # samples per frame (e.g. 480)
MIN_VOICED_FRAMES = math.ceil(MIN_UTTERANCE_MS / FRAME_MS)
TRAILING_SILENCE_FRAMES = math.ceil(TRAILING_SILENCE_MS / FRAME_MS)

SENTENCE_END_CHARS = "\.\!\?…"  # regex set
SENTENCE_END_REGEX = re.compile(rf"(.+?[{SENTENCE_END_CHARS}](?:[\"'\)\]]*)\s+)", re.DOTALL)

# =============================
# Utterance Detection
# =============================
class UtteranceDetector:
    """Segments microphone audio into utterances using WebRTC VAD.

    Logic:
      - Collect 30 ms frames.
      - Accumulate frames until at least MIN_VOICED_FRAMES voiced frames observed.
      - After start, keep frames until TRAILING_SILENCE_FRAMES consecutive non-voiced frames.
      - Return utterance as float32 numpy array normalized to [-1,1].
    """

    def __init__(self, aggressiveness: int = VAD_AGGRESSIVENESS):
        self.vad = webrtcvad.Vad(aggressiveness)
        self._stream = None
        self._is_muted = False

    def mute_microphone(self):
        """Temporarily mute the microphone."""
        self._is_muted = True
        if self._stream:
            try:
                self._stream.close()
                self._stream = None
            except:
                pass

    def unmute_microphone(self):
        """Unmute the microphone."""
        self._is_muted = False

    def record_once(self) -> Optional[np.ndarray]:
        """Blocking capture of a single utterance. Returns float32 waveform or None."""
        if self._is_muted:
            return None
            
        q: 'queue.Queue[bytes]' = queue.Queue()
        started = False
        voiced_count = 0
        silence_count = 0
        collected: List[bytes] = []
        overflow_counter = 0

        def callback(indata, frames, time_info, status):
            nonlocal overflow_counter
            if status.input_overflow:
                overflow_counter += 1
            if not self._is_muted:  # Only collect if not muted
                q.put(bytes(indata))

        self._stream = sd.RawInputStream(
            samplerate=SAMPLE_RATE,
            blocksize=FRAME_SAMPLES,
            channels=1,
            dtype='int16',
            callback=callback,
        )

        with self._stream:
            while True:
                if self._is_muted:
                    return None
                    
                try:
                    frame = q.get(timeout=0.1)
                except queue.Empty:
                    continue
                except KeyboardInterrupt:
                    raise
                    
                is_speech = False
                try:
                    is_speech = self.vad.is_speech(frame, SAMPLE_RATE)
                except Exception:
                    # If VAD fails (rare), treat as silence
                    is_speech = False
                if not started:
                    if is_speech:
                        voiced_count += 1
                        collected.append(frame)
                        if voiced_count >= MIN_VOICED_FRAMES:
                            started = True
                    else:
                        # Reset (noise or short blips)
                        voiced_count = 0
                        collected.clear()
                    continue
                # After started
                collected.append(frame)
                if is_speech:
                    silence_count = 0
                else:
                    silence_count += 1
                    if silence_count >= TRAILING_SILENCE_FRAMES:
                        break  # end of utterance
        if not collected:
            return None
        # Remove trailing silence frames for cleaner STT input
        if silence_count:
            collected = collected[:-silence_count] or collected
        if len(collected) < MIN_VOICED_FRAMES:
            return None  # Too short / discard
        pcm = b''.join(collected)
        audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        return audio

# =============================
# Whisper STT Wrapper
# =============================
class WhisperSTT:
    def __init__(self, model_name: str, compute: str = 'auto'):
        device, compute_type = self._select_device(compute)
        self.model = WhisperModel(model_name, device=device, compute_type=compute_type)

    def _select_device(self, compute: str):
        if compute == 'auto':
            # Try GPU (cuda) first
            try:
                import torch  # noqa: F401
                return 'cuda', 'float16'
            except Exception:
                return 'cpu', 'int8'
        if compute == 'cuda':
            return 'cuda', 'float16'
        return 'cpu', 'int8'

    def transcribe(self, audio: np.ndarray) -> str:
        segments, info = self.model.transcribe(audio, beam_size=1, vad_filter=False)
        text_parts = [seg.text.strip() for seg in segments]
        return ' '.join(part for part in text_parts if part)

# =============================
# Ollama Streaming
# =============================
async def ollama_stream_chat(conversation: List[dict], model: str, max_tokens: int) -> AsyncGenerator[str, None]:
    """Async generator yielding text chunks from Ollama chat streaming.

    conversation: list of {'role': 'system'|'user'|'assistant', 'content': str}
    """
    loop = asyncio.get_running_loop()
    q: asyncio.Queue = asyncio.Queue()

    def worker():
        try:
            # streaming=True returns incremental responses
            for part in ollama.chat(model=model, messages=conversation, stream=True, options={"num_predict": max_tokens}):
                try:
                    msg = part.get('message', {})
                    content = msg.get('content')
                    if content:
                        asyncio.run_coroutine_threadsafe(q.put(content), loop)
                except Exception:
                    continue
        except Exception as e:
            asyncio.run_coroutine_threadsafe(q.put(f"[Error: {e} ]"), loop)
        finally:
            asyncio.run_coroutine_threadsafe(q.put(None), loop)

    threading.Thread(target=worker, daemon=True).start()
    while True:
        chunk = await q.get()
        if chunk is None:
            break
        yield chunk

# =============================
# Sentence Segmentation of Token Stream
# =============================
async def sentence_stream(token_stream: AsyncGenerator[str, None]) -> AsyncGenerator[str, None]:
    """Yield completed sentences as tokens stream in.

    Sentence ends when we see end punctuation followed by whitespace OR we flush at end.
    """
    buffer = ''
    async for chunk in token_stream:
        buffer += chunk
        # Find full sentence(s)
        while True:
            match = SENTENCE_END_REGEX.search(buffer)
            if not match:
                break
            sentence = match.group(1)
            # Remove from buffer
            buffer = buffer[len(sentence):]
            yield sentence.strip()
    # Flush leftover
    tail = buffer.strip()
    if tail:
        yield tail

# =============================
# TTS Speakers
# =============================
class BaseSpeaker:
    async def speak(self, sentence: str):  # pragma: no cover - interface
        raise NotImplementedError

    async def close(self):  # optional cleanup
        pass

class Pyttsx3Speaker(BaseSpeaker):
    """Threaded pyttsx3 speaker; async 'speak' returns when sentence finished."""

    def __init__(self, voice_filter: Optional[str] = VOICE_NAME):
        self.queue: 'queue.Queue[Optional[tuple[str, asyncio.Future]]]' = queue.Queue()
        self.loop = asyncio.get_event_loop()
        self.thread = threading.Thread(target=self._worker, args=(voice_filter,), daemon=True)
        self.thread.start()

    def _worker(self, voice_filter: Optional[str]):
        engine = pyttsx3.init()
        if voice_filter:
            for v in engine.getProperty('voices'):
                name = getattr(v, 'name', '') or ''
                if voice_filter.lower() in name.lower():
                    engine.setProperty('voice', v.id)
                    break
        while True:
            item = self.queue.get()
            if item is None:
                break
            text, fut = item
            try:
                engine.say(text)
                engine.runAndWait()
                self.loop.call_soon_threadsafe(fut.set_result, True)
            except Exception as e:
                self.loop.call_soon_threadsafe(fut.set_exception, e)

    async def speak(self, sentence: str):
        fut = self.loop.create_future()
        self.queue.put((sentence, fut))
        await fut

    async def close(self):
        self.queue.put(None)
        self.thread.join(timeout=1)

class EdgeTTSSpeaker(BaseSpeaker):
    """Edge TTS speaker; per-sentence synthesis & playback.

    Uses edge-tts to synthesize MP3 -> pydub to decode -> simpleaudio to play.
    """

    def __init__(self, voice: Optional[str] = VOICE_NAME):
        self.voice = voice or 'en-US-JennyNeural'
        try:
            import edge_tts  # noqa: F401
            import pydub  # noqa: F401
            import simpleaudio  # noqa: F401
        except ImportError as e:
            print("[EdgeTTSSpeaker] Missing packages. Install: pip install edge-tts pydub simpleaudio", file=sys.stderr)
            raise

    async def speak(self, sentence: str):
        import edge_tts
        from pydub import AudioSegment
        import simpleaudio as sa
        # Synthesize
        communicate = edge_tts.Communicate(sentence, voice=self.voice)
        audio_bytes = bytearray()
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_bytes.extend(chunk["data"])
        data = bytes(audio_bytes)
        if not data:
            return
        # Decode / resample
        audio_seg = AudioSegment.from_file(io.BytesIO(data), format="mp3")
        audio_seg = audio_seg.set_frame_rate(SAMPLE_RATE).set_channels(1).set_sample_width(2)
        play_obj = sa.play_buffer(audio_seg.raw_data, num_channels=1, bytes_per_sample=2, sample_rate=audio_seg.frame_rate)
        play_obj.wait_done()

class CoquiTTSSpeaker(BaseSpeaker):
    """Coqui TTS speaker using neural voice synthesis.
    
    Uses TTS library for high-quality neural text-to-speech synthesis.
    Enhanced with better audio processing and prosody control.
    """

    def __init__(self, model_name: Optional[str] = VOICE_NAME):
        self.model_name = model_name or "tts_models/en/vctk/vits"  # Better multi-speaker model
        try:
            from TTS.api import TTS
            import torch
            import sounddevice as sd
            import tempfile
            import os
            import wave
        except ImportError as e:
            print("[CoquiTTSSpeaker] Missing packages. Install: pip install TTS torch torchaudio", file=sys.stderr)
            raise
        
        # Initialize TTS model with GPU acceleration
        import torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[CoquiTTS] Loading model {self.model_name} on {self.device}...")
        
        try:
            self.tts = TTS(model_name=self.model_name).to(self.device)
            print(f"[CoquiTTS] Successfully loaded on {self.device}")
        except Exception as e:
            print(f"[CoquiTTS] Failed to load {self.model_name} on {self.device}, trying fallback...")
            # Try CPU fallback if GPU fails
            if self.device == "cuda":
                self.device = "cpu"
                print(f"[CoquiTTS] Falling back to CPU...")
                try:
                    self.tts = TTS(model_name=self.model_name).to(self.device)
                except Exception as e2:
                    print(f"[CoquiTTS] CPU fallback failed, trying simpler model...")
                    self.model_name = "tts_models/en/ljspeech/tacotron2-DDC"
                    self.tts = TTS(model_name=self.model_name).to(self.device)
            else:
                print(f"[CoquiTTS] Trying simpler model on CPU...")
                self.model_name = "tts_models/en/ljspeech/tacotron2-DDC"
                self.tts = TTS(model_name=self.model_name).to(self.device)
        
        # Check if model supports speaker selection
        if hasattr(self.tts, 'speakers') and self.tts.speakers:
            print(f"[CoquiTTS] Available speakers: {self.tts.speakers[:3]}...")  # Show first 3
            self.speaker_name = self.tts.speakers[0]
        else:
            self.speaker_name = None
        
        print(f"[CoquiTTS] Model loaded successfully. Speaker: {self.speaker_name or 'default'}")

    def _enhance_text(self, sentence: str) -> str:
        """Add prosody and cleanup to make speech more natural."""
        # Clean up text
        sentence = sentence.strip()
        if not sentence:
            return sentence
            
        # Add slight pauses for better pacing
        sentence = sentence.replace(',', ', ')  # Pause after commas
        sentence = sentence.replace(';', '; ')  # Pause after semicolons
        
        # Emphasize questions and exclamations
        if sentence.endswith('?'):
            sentence = sentence[:-1] + '?'  # Ensure proper question intonation
        elif sentence.endswith('!'):
            sentence = sentence[:-1] + '!'  # Ensure proper exclamation intonation
            
        return sentence

    async def speak(self, sentence: str):
        import sounddevice as sd
        import numpy as np
        import torch
        
        # Enhance text for better prosody
        enhanced_sentence = self._enhance_text(sentence)
        if not enhanced_sentence.strip():
            return
        
        try:
            # Use GPU acceleration for synthesis
            with torch.no_grad():  # Disable gradients for inference
                if self.device == "cuda":
                    # GPU-optimized synthesis
                    torch.cuda.empty_cache()  # Clear cache before synthesis
                
                # Generate audio with proper parameters
                if self.speaker_name:
                    wav = self.tts.tts(text=enhanced_sentence, speaker=self.speaker_name)
                else:
                    wav = self.tts.tts(text=enhanced_sentence)
                
                # Move to CPU for audio processing if on GPU
                if isinstance(wav, torch.Tensor):
                    wav = wav.cpu().numpy()
                elif not isinstance(wav, np.ndarray):
                    wav = np.array(wav, dtype=np.float32)
            
            # Normalize audio to prevent clipping
            if len(wav) > 0:
                max_val = np.max(np.abs(wav))
                if max_val > 0:
                    wav = wav / max_val * 0.85  # Leave some headroom
            
            # Get sample rate from TTS model
            if hasattr(self.tts, 'synthesizer') and hasattr(self.tts.synthesizer, 'output_sample_rate'):
                sample_rate = getattr(self.tts.synthesizer.output_sample_rate, 'value', 22050)
            else:
                sample_rate = 22050
            
            # Play using sounddevice with optimized settings
            sd.play(wav, samplerate=sample_rate, blocking=True)
            
            # Clear GPU memory after synthesis
            if self.device == "cuda":
                torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"[CoquiTTS] Synthesis error: {e}")
            # Clear GPU memory on error too
            if self.device == "cuda":
                try:
                    torch.cuda.empty_cache()
                except:
                    pass
            # Fallback: skip this sentence
            pass

    async def close(self):
        """Clean up GPU resources when done."""
        if hasattr(self, 'device') and self.device == "cuda":
            try:
                import torch
                torch.cuda.empty_cache()
                print("[CoquiTTS] GPU memory cleared")
            except:
                pass

# =============================
# Speaker Factory
# =============================
async def create_speaker() -> BaseSpeaker:
    if TTS_BACKEND.lower() == 'edge-tts':
        return EdgeTTSSpeaker(VOICE_NAME)
    elif TTS_BACKEND.lower() == 'coqui':
        return CoquiTTSSpeaker(VOICE_NAME)
    return Pyttsx3Speaker(VOICE_NAME)

# =============================
# Conversation Memory
# =============================
class Conversation:
    def __init__(self, system_prompt: str):
        self.base_system_prompt = system_prompt
        self.current_style_modifier = ""
        self.messages: List[dict] = [{"role": "system", "content": self._get_full_system_prompt()}]

    def _get_full_system_prompt(self) -> str:
        """Combine base prompt with current style modifier."""
        if self.current_style_modifier:
            return f"{self.base_system_prompt}\n\nCURRENT STYLE OVERRIDE: {self.current_style_modifier}"
        return self.base_system_prompt

    def _update_system_prompt(self):
        """Update the system message with current style."""
        self.messages[0] = {"role": "system", "content": self._get_full_system_prompt()}

    def _detect_style_commands(self, user_text: str) -> bool:
        """Detect and apply style change commands. Returns True if command was processed."""
        text_lower = user_text.lower().strip()
        
        style_commands = {
            "speak more formally": "Use formal, professional language with detailed explanations and proper grammar.",
            "be more formal": "Use formal, professional language with detailed explanations and proper grammar.",
            "be more casual": "Use casual, informal language with contractions and conversational style.",
            "speak more casually": "Use casual, informal language with contractions and conversational style.",
            "be more technical": "Include technical details, terminology, and in-depth explanations.",
            "explain like i'm 5": "Use very simple language, basic analogies, and concepts a child would understand.",
            "explain like im 5": "Use very simple language, basic analogies, and concepts a child would understand.",
            "eli5": "Use very simple language, basic analogies, and concepts a child would understand.",
            "be more concise": "Give very brief, to-the-point responses with minimal elaboration.",
            "be more detailed": "Provide comprehensive explanations with examples and context.",
            "be more creative": "Use creative language, metaphors, and imaginative explanations.",
            "be more professional": "Use business-appropriate language and maintain professional demeanor.",
            "reset style": "",  # Empty string resets to default
            "default style": "",
            "normal style": ""
        }
        
        for command, modifier in style_commands.items():
            if command in text_lower:
                self.current_style_modifier = modifier
                self._update_system_prompt()
                return True
        
        return False

    def add_user(self, text: str):
        # Check for style commands before adding to conversation
        is_style_command = self._detect_style_commands(text)
        
        if is_style_command:
            # Add a confirmation message for the style change
            style_name = "default" if not self.current_style_modifier else "updated"
            self.messages.append({"role": "user", "content": text})
            self.messages.append({"role": "assistant", "content": f"Got it! I've switched to {style_name} response style."})
        else:
            self.messages.append({"role": "user", "content": text})

    def add_assistant(self, text: str):
        self.messages.append({"role": "assistant", "content": text})

    def history(self) -> List[dict]:
        return list(self.messages)

    def get_current_style(self) -> str:
        """Get current style description for debugging."""
        return self.current_style_modifier or "Default conversational style"

# =============================
# Main Loop Logic
# =============================
async def process_turn(detector: UtteranceDetector, stt: WhisperSTT, convo: Conversation, speaker: BaseSpeaker):
    print("🎤 Listening…", flush=True)

    audio = await asyncio.to_thread(detector.record_once)  # Add back the asyncio.to_thread()
    if audio is None or not len(audio):
        print("🛑 No audio captured.")
        return  # Nothing captured; loop again
    

    if ENABLE_SPEAKER_GATE and voice_identifier is not None:
        try:
            best_name, best_sim = voice_identifier.identify_from_array(audio, 16000)
            if best_name is None:
                await speaker.speak("I don't have any enrolled voices yet.")
                return
            if best_sim < SIM_THRESHOLD:
                print(f"[Gate] ❌ Deny: best={best_name}, sim={best_sim:.3f}")
                await speaker.speak("Sorry, I didn’t recognize your voice. Please try again.")
                return
            print(f"[Gate] ✅ Allow: {best_name} (sim={best_sim:.3f})")
        except Exception as e:
            print(f"[Gate Error] {e}")
            await speaker.speak("Voice check failed. Please try again.")
            return
        
    try:
        print("📝 Transcribing…", flush=True)
        transcript = await asyncio.to_thread(stt.transcribe, audio)
    except Exception as e:
        print(f"[STT Error] {e}")
        return
    if not transcript.strip():
        return
    print(f"You: {transcript}")
    
    # Handle style commands differently
    is_style_command = convo._detect_style_commands(transcript)
    convo.add_user(transcript)
    
    if is_style_command:
        # For style commands, give immediate feedback instead of calling LLM
        current_style = convo.get_current_style()
        response = f"I've updated my response style. Current style: {current_style}"
        print(f"Assistant ↳ {response}")
        await speaker.speak(response)
        convo.add_assistant(response)
        await asyncio.sleep(TAIL_DELAY_SEC)
        return

    print("🤖 Assistant (streaming)…", flush=True)

    # Streaming generation
    assistant_buffer = []
    sentences_queue: asyncio.Queue = asyncio.Queue()
    speak_consumer_task = asyncio.create_task(_speak_consumer(sentences_queue, speaker, detector))  # Pass detector

    async for sentence in sentence_stream(ollama_stream_chat(convo.history(), OLLAMA_MODEL, MAX_TOKENS)):
        assistant_buffer.append(sentence)
        await sentences_queue.put(sentence)
        if PRINT_PARTIAL_SENTENCES:
            print(f"Assistant ↳ {sentence}")

    # Signal completion
    await sentences_queue.put(None)
    await speak_consumer_task

    full_assistant_text = ' '.join(assistant_buffer)
    convo.add_assistant(full_assistant_text)
    # Note: TAIL_DELAY_SEC is now handled in _speak_consumer

async def _speak_consumer(q: 'asyncio.Queue[Optional[str]]', speaker: BaseSpeaker, detector: UtteranceDetector):
    sentences_spoken = 0
    
    # Mute microphone when starting to speak
    detector.mute_microphone()
    
    while True:
        sentence = await q.get()
        if sentence is None:
            break
        try:
            await speaker.speak(sentence)
            sentences_spoken += 1
        except Exception as e:
            print(f"[TTS Error] {e}")
    
    # Dynamic delay based on how much was spoken
    dynamic_delay = 0.5
    await asyncio.sleep(dynamic_delay)
    
    # Unmute microphone after speaking is completely done
    detector.unmute_microphone()
    print("🔊 Microphone reactivated")

# =============================
# Entry Point
# =============================
async def main():
    if sys.platform.startswith('win'):
        try:
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())  # type: ignore
        except Exception:
            pass
    print("Booting streaming voice chatbot…")

    detector = UtteranceDetector()
    stt = WhisperSTT(WHISPER_MODEL, WHISPER_COMPUTE)
    speaker = await create_speaker()
    convo = Conversation(SYSTEM_PROMPT)

    try:
        while True:
            await process_turn(detector, stt, convo, speaker)
    except KeyboardInterrupt:
        print("\nExiting…")
    finally:
        await speaker.close()

if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
