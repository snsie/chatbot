#!/usr/bin/env python3
"""
Text-to-Speech speakers for the Streaming Voice Chatbot.
"""

import asyncio
import sys
import threading
import queue
import io
from typing import Optional
import pyttsx3

from .config import default_config


class BaseSpeaker:
    """Base class for TTS speakers."""
    
    async def speak(self, sentence: str):  # pragma: no cover - interface
        raise NotImplementedError

    async def close(self):  # optional cleanup
        pass


class Pyttsx3Speaker(BaseSpeaker):
    """Threaded pyttsx3 speaker; async 'speak' returns when sentence finished."""

    def __init__(self, config=None, voice_filter=None):
        self.config = config or default_config
        voice_filter = voice_filter or self.config.VOICE_NAME
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

    def __init__(self, config=None, voice=None):
        self.config = config or default_config
        self.voice = voice or self.config.VOICE_NAME or 'en-US-JennyNeural'
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
        audio_seg = audio_seg.set_frame_rate(self.config.SAMPLE_RATE).set_channels(1).set_sample_width(2)
        play_obj = sa.play_buffer(audio_seg.raw_data, num_channels=1, bytes_per_sample=2, sample_rate=audio_seg.frame_rate)
        play_obj.wait_done()


class CoquiTTSSpeaker(BaseSpeaker):
    """Coqui TTS speaker using neural voice synthesis.
    
    Uses TTS library for high-quality neural text-to-speech synthesis.
    Enhanced with better audio processing and prosody control.
    """

    def __init__(self, config=None, model_name=None):
        self.config = config or default_config
        self.model_name = model_name or self.config.VOICE_NAME or "tts_models/en/vctk/vits"
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


async def create_speaker(config=None) -> BaseSpeaker:
    """Factory function to create the appropriate speaker based on configuration."""
    config = config or default_config
    
    if config.TTS_BACKEND.lower() == 'edge-tts':
        return EdgeTTSSpeaker(config)
    elif config.TTS_BACKEND.lower() == 'coqui':
        return CoquiTTSSpeaker(config)
    return Pyttsx3Speaker(config)
