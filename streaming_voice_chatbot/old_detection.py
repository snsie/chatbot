#!/usr/bin/env python3
"""
Audio detection and processing utilities for the Streaming Voice Chatbot.
"""

import asyncio
import queue
import time
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


class StreamingUtteranceDetector:
    """Async streaming utterance detector using asyncio.Queue for non-blocking operation.
    
    Continuously captures audio in the background and provides utterances via async methods.
    """
    
    def __init__(self, config=None, aggressiveness=None):
        self.config = config or default_config
        aggressiveness = aggressiveness or self.config.VAD_AGGRESSIVENESS
        self.vad = webrtcvad.Vad(aggressiveness)
        self.audio_queue: asyncio.Queue = asyncio.Queue(maxsize=100)  # Limit queue size
        self.stream = None
        self._recording_task = None
        
    async def start_recording(self):
        """Start continuous audio recording in background."""
        if self.stream is not None:
            return  # Already recording
            
        def audio_callback(indata, frames, time_info, status):
            """Audio callback that feeds data to the asyncio queue."""
            if status.input_overflow:
                # Handle overflow but continue
                pass
            try:
                # Convert the input data to the right format
                # indata is already int16 from sounddevice
                audio_bytes = bytes(indata)
                # Try to put in queue, drop if full (non-blocking)
                try:
                    self.audio_queue.put_nowait(audio_bytes)
                except asyncio.QueueFull:
                    # Drop old frames to prevent memory buildup
                    try:
                        self.audio_queue.get_nowait()  # Remove oldest
                        self.audio_queue.put_nowait(audio_bytes)  # Add new
                    except asyncio.QueueEmpty:
                        pass
            except Exception as e:
                print(f"Audio callback error: {e}")
        
        self.stream = sd.RawInputStream(
            samplerate=self.config.SAMPLE_RATE,
            blocksize=self.config.FRAME_SAMPLES,
            channels=1,
            dtype='int16',
            callback=audio_callback,
        )
        self.stream.start()
        
    async def stop_recording(self):
        """Stop the background recording."""
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.stream = None
            
    async def get_utterance(self,stt, timeout: float = 30.0) -> Optional[np.ndarray]:
        """Get next complete utterance (non-blocking with timeout).
        
        Args:
            timeout: Maximum time to wait for an utterance in seconds
            
        Returns:
            Audio data as float32 numpy array or None if timeout/no utterance
        """
        if self.stream is None:
            await self.start_recording()
            
        started = False
        voiced_count = 0
        silence_count = 0
        collected: List[bytes] = []
        start_time = asyncio.get_event_loop().time()
        transcript_counter = 0
        while True:
            transcript_counter += 1
            try:
                # Check timeout
                if asyncio.get_event_loop().time() - start_time > timeout:
                    return None
                    
                # Wait for audio chunk with short timeout to allow responsiveness
                try:
                    frame = await asyncio.wait_for(
                        self.audio_queue.get(), 
                        timeout=0.1
                    )
                except asyncio.TimeoutError:
                    # No audio data, check if we have enough for an utterance
                    if started and collected and silence_count >= self.config.TRAILING_SILENCE_FRAMES:
                        break
                    continue
                
                # Process the frame
                is_speech = False
                try:
                    is_speech = self.vad.is_speech(frame, self.config.SAMPLE_RATE)
                except Exception:
                    # If VAD fails, treat as silence
                    is_speech = False
                    
                # print('1' if is_speech else '0', end='', flush=True)  # Debug output
                
                if not started:
                    if is_speech:
                        voiced_count += 1
                        collected.append(frame)
                        if voiced_count >= self.config.MIN_VOICED_FRAMES:
                            started = True
                            silence_count = 0
                    else:
                        # Reset on silence before utterance starts
                        voiced_count = 0
                        collected.clear()
                else:
                    if transcript_counter % 25 == 0 and stt is not None:
                        time.sleep(0.1)  # Yield to event loop

                        pcm = b''.join(collected)
                        audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
                        transcript = await asyncio.to_thread(stt.transcribe, audio)
                        print(f"\nInterim transcript: {transcript}\n", flush=True)
                    # After started, collect all frames
                    collected.append(frame)
                    if is_speech:
                        silence_count = 0
                    else:
                        silence_count += 1
                        if silence_count >= self.config.TRAILING_SILENCE_FRAMES:
                            break  # End of utterance
                            
            except Exception as e:
                print(f"Error processing audio frame: {e}")
                continue
        
        if not collected:
            return None
            
        # Remove trailing silence frames for cleaner STT input
        if silence_count > 0:
            collected = collected[:-silence_count] or collected
            
        if len(collected) < self.config.MIN_VOICED_FRAMES:
            return None  # Too short, discard
            
        # Convert to audio
        pcm = b''.join(collected)
        audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        return audio
        
    def _is_silence(self, audio_chunk: np.ndarray, threshold: float = 0.01) -> bool:
        """Check if audio chunk is mostly silence."""
        return np.mean(np.abs(audio_chunk)) < threshold
        
    async def cleanup(self):
        """Clean up resources."""
        await self.stop_recording()


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
