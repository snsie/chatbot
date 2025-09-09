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
import whisper

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
                  
    async def get_utterance(self, stt, timeout: float = 30.0, transcript_check_interval: int = 100) -> Optional[str]:
        """Get next complete utterance (non-blocking with timeout).
        
        Args:
            timeout: Maximum time to wait for an utterance in seconds
            transcript_check_interval: How often (in frames) to check interim transcription if stt is provided
            
        Returns:
            Transcribed text or None if timeout/no utterance
        """
        if self.stream is None:
            await self.start_recording()
            
        started = False
        voiced_count = 0
        silence_count = 0
        collected: List[bytes] = []
        start_time = asyncio.get_event_loop().time()
        transcript_counter = 0
        prev_transcript_length = 0
        transcript = ""
        
        while True:
            transcript_counter += 1
            try:
                # Check timeout
                if asyncio.get_event_loop().time() - start_time > timeout:
                    return None
                    
                # Wait for audio chunk with short timeout
                try:
                    frame = await asyncio.wait_for(
                        self.audio_queue.get(), 
                        timeout=0.1
                    )
                except asyncio.TimeoutError:
                    if started and collected and silence_count >= self.config.TRAILING_SILENCE_FRAMES:
                        break
                    continue
                
                # Process the frame
                is_speech = False
                try:
                    is_speech = self.vad.is_speech(frame, self.config.SAMPLE_RATE)
                except Exception:
                    is_speech = False
                    
                if not started:
                    if is_speech:
                        voiced_count += 1
                        collected.append(frame)
                        if voiced_count >= self.config.MIN_VOICED_FRAMES:
                            prev_transcript_length = 0
                            started = True
                            silence_count = 0
                            transcript_counter = 0  # Reset counter when starting
                    else:
                        voiced_count = 0
                        collected.clear()
                else:
                    collected.append(frame)
                    
                    # Only transcribe periodically and when we have enough audio
                    if (transcript_counter % transcript_check_interval == 0 and 
                        stt is not None and len(collected) >= self.config.MIN_VOICED_FRAMES):
                        
                        # Yield control to event loop without blocking sleep
                        
                        pcm = b''.join(collected)
                        audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
                        
                        
                        try:
                            # Use asyncio.to_thread for better async handling
                            transcript = await asyncio.to_thread(stt.transcribe, audio)
                            
                            if len(transcript) > prev_transcript_length:
                                print(f'New transcript: {transcript}', flush=True)
                                silence_count = 0  # Reset silence counter on new content
                                prev_transcript_length = len(transcript)
                            elif transcript.strip():  # Only count if we have actual content
                                # Check if transcript stopped growing
                                silence_frames_equivalent = transcript_counter * transcript_check_interval
                                if silence_frames_equivalent >= self.config.TRAILING_SILENCE_FRAMES:
                                    print("Transcript stopped growing, ending utterance.", flush=True)
                                    print('', flush=True)
                                    break
                                    
                        except Exception as e:
                            print(f"[STT Error] {e}")
                            
                    # if is_speech:
                    #     silence_count = 0
                    # else:
                    #     silence_count += 1
                    #     if silence_count >= self.config.TRAILING_SILENCE_FRAMES:
                    #         break
                            
            except Exception as e:
                print(f"Error processing audio frame: {e}")
                continue
        
        if not transcript or not transcript.strip():
            print('No valid transcript found', flush=True)
            return None
            
        print('Returning transcript:', transcript, flush=True)
        return transcript.strip()
        
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
        self.device = self._select_device(compute)
        self.model = whisper.load_model(model_name, device=self.device)

    def _select_device(self, compute: str):
        if compute == 'auto':
            # Try GPU (cuda) first
            try:
                import torch
                if torch.cuda.is_available():
                    return 'cuda'
                else:
                    return 'cpu'
            except Exception:
                return 'cpu'
        if compute == 'cuda':
            try:
                import torch
                if torch.cuda.is_available():
                    return 'cuda'
                else:
                    print("Warning: CUDA requested but not available, falling back to CPU")
                    return 'cpu'
            except Exception:
                print("Warning: CUDA requested but PyTorch not available, falling back to CPU")
                return 'cpu'
        return 'cpu'

    def transcribe(self, audio: np.ndarray) -> str:
        # Ensure audio is the right format for openai-whisper
        # openai-whisper expects float32 audio in range [-1, 1]
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        
        # Ensure audio is normalized to [-1, 1]
        if audio.max() > 1.0 or audio.min() < -1.0:
            audio = audio / max(abs(audio.max()), abs(audio.min()))
        
        try:
            result = self.model.transcribe(audio, verbose=False)
            return result["text"].strip()
        except Exception as e:
            print(f"Transcription error: {e}")
            return ""
