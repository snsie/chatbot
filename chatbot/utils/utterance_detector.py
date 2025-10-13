from typing import List, Optional
from ..constants import SAMPLE_RATE, VAD_AGGRESSIVENESS, FRAME_MS, MIN_VOICED_FRAMES, TRAILING_SILENCE_FRAMES, FRAME_SAMPLES
import webrtcvad
import numpy as np
import sounddevice as sd
import queue


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