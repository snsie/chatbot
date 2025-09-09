#!/usr/bin/env python3
"""
Audio detection and processing utilities for the Streaming Voice Chatbot.
"""

import queue
from typing import Optional, List
import numpy as np
import sounddevice as sd
import webrtcvad
from faster_whisper import WhisperModel

from .config import default_config


class UtteranceDetector:
    """Segments microphone audio into utterances using WebRTC VAD.

    Logic:
      - Collect frames based on configuration.
      - Accumulate frames until at least MIN_VOICED_FRAMES voiced frames observed.
      - After start, keep frames until TRAILING_SILENCE_FRAMES consecutive non-voiced frames.
      - Return utterance as float32 numpy array normalized to [-1,1].
    """

    def __init__(self, config=None, aggressiveness=None):
        self.config = config or default_config
        aggressiveness = aggressiveness or self.config.VAD_AGGRESSIVENESS
        self.vad = webrtcvad.Vad(aggressiveness)

    def record_once(self, stt=None) -> Optional[np.ndarray]:
        """Blocking capture of a single utterance. Returns float32 waveform or None.
        May block indefinitely until speech occurs or user interrupts.
        """
        q: 'queue.Queue[bytes]' = queue.Queue()
        started = False
        voiced_count = 0
        silence_count = 0
        collected: List[bytes] = []
        overflow_counter = 0

        def callback(indata, frames, time_info, status):  # sounddevice RawInputStream callback
            nonlocal overflow_counter
            if status.input_overflow:
                overflow_counter += 1  # We tolerate overflow; frames still usable.
            q.put(bytes(indata))

        with sd.RawInputStream(
            samplerate=self.config.SAMPLE_RATE,
            blocksize=self.config.FRAME_SAMPLES,
            channels=1,
            dtype='int16',
            callback=callback,
        ):
            while True:
                try:
                    frame = q.get()
                except KeyboardInterrupt:
                    raise
                if frame is None:
                    continue
                is_speech = False
                try:
                    is_speech = self.vad.is_speech(frame, self.config.SAMPLE_RATE)
                except Exception:
                    # If VAD fails (rare), treat as silence
                    is_speech = False
                print('1' if is_speech else '0', end='', flush=True)  # Debug: show VAD decisions
                if not started:
                    if is_speech:
                        voiced_count += 1
                        collected.append(frame)
                        if voiced_count >= self.config.MIN_VOICED_FRAMES:
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
                    if silence_count >= self.config.TRAILING_SILENCE_FRAMES:
                        break  # end of utterance
                
        if not collected:
            return None
        # Remove trailing silence frames for cleaner STT input
        if silence_count:
            collected = collected[:-silence_count] or collected
        if len(collected) < self.config.MIN_VOICED_FRAMES:
            return None  # Too short / discard
        pcm = b''.join(collected)
        audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        return audio


class WhisperSTT:
    """Whisper Speech-to-Text wrapper with GPU acceleration support."""
    
    def __init__(self, config=None, model_name=None, compute=None):
        self.config = config or default_config
        model_name = model_name or self.config.WHISPER_MODEL
        compute = compute or self.config.WHISPER_COMPUTE
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
