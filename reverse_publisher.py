# reverse_publisher.py
# Purpose: take TTS PCM, normalize to mono/16k/int16, chunk into 30 ms (480-sample)
# frames, and publish them to a bounded, thread-safe queue for AEC use.

from collections import deque
import threading
import numpy as np

class ReverseAudioPublisher:
    """
    Publishes render (speaker) audio as fixed 30 ms frames (480 samples @ 16kHz) into a
    bounded queue for AEC. Use from your TTS playback path *before* writing to the device.
    """
    def __init__(self, sample_rate: int = 16000, frame_ms: int = 30, max_frames: int = 50):
        self.sample_rate = sample_rate
        self.frame_ms = frame_ms
        self.frame_samples = int(sample_rate * frame_ms / 1000)  # 480 at 16k/30ms
        self._carry = np.zeros(0, dtype=np.int16)  # holds remainders between calls
        self.queue = deque(maxlen=max_frames)      # ~1.5 s of history
        self._last_frame = np.zeros(self.frame_samples, dtype=np.int16)
        self._lock = threading.Lock()

    # ---------- public API ----------
    def publish_pcm(self, pcm, src_sample_rate: int, channels: int = 1, dtype=np.int16) -> int:
        """
        Feed a chunk of PCM that you're about to play to speakers.
        - pcm: either a numpy array or a bytes-like buffer
        - src_sample_rate: the PCM's current sample rate
        - channels: 1 (mono) or 2 (stereo)
        - dtype: dtype of the input buffer (e.g., np.int16, np.float32)

        Returns: number of 30 ms frames published to the queue.
        """
        x = self._to_mono_int16_16k(pcm, src_sample_rate, channels, dtype)
        return self._chunk_and_enqueue(x)

    def flush_tail(self, pad: bool = True) -> int:
        """
        Call once at the END of an utterance.
        If there's a remainder < 480 samples in the carry buffer:
          - pad=True: zero-pad and publish one last 480-sample frame
          - pad=False: drop it
        Returns: number of frames published (0 or 1).
        """
        with self._lock:
            published = 0
            r = len(self._carry)
            if r > 0:
                if pad:
                    tail = np.zeros(self.frame_samples, dtype=np.int16)
                    tail[:r] = self._carry
                    self.queue.append(tail)
                    self._last_frame = tail
                    published = 1
                # clear remainder either way
                self._carry = np.zeros(0, dtype=np.int16)
            return published

    def get_latest_frame(self) -> np.ndarray:
        """
        For the mic side (later): get the most recent reverse frame.
        If the queue is empty, returns the last published frame (or silence initially).
        """
        with self._lock:
            if self.queue:
                self._last_frame = self.queue.pop()
                # keep latest as last_frame, and discard older ones to avoid latency
                self.queue.clear()
            return self._last_frame

    # ---------- internals ----------
    def _to_mono_int16_16k(self, pcm, src_sr: int, channels: int, dtype) -> np.ndarray:
        """Normalize incoming PCM to mono / 16 kHz / int16 with proper clipping."""
        # bytes -> ndarray
        if isinstance(pcm, (bytes, bytearray, memoryview)):
            x = np.frombuffer(pcm, dtype=dtype)
        else:
            x = np.asarray(pcm)

        # ensure 1D
        x = x.reshape(-1, channels) if channels > 1 else x.reshape(-1)

        # downmix stereo to mono if needed
        if channels > 1:
            # If integer type, upcast to float during downmix to avoid overflow
            if np.issubdtype(x.dtype, np.integer):
                x = x.astype(np.float32)
            x = x.mean(axis=1)

        # convert to float for resampling/math
        if np.issubdtype(x.dtype, np.integer):
            # assume full-scale int ranges; map to [-1,1)
            max_abs = np.iinfo(x.dtype).max
            x = x.astype(np.float32) / max_abs
        else:
            x = x.astype(np.float32)

        # resample to 16k if needed
        if src_sr != self.sample_rate:
            x = self._resample_float(x, src_sr, self.sample_rate)

        # scale back to int16 with clipping
        x = np.clip(x, -1.0, 0.9999695)  # just under 1.0 to prevent wrap
        x = (x * 32768.0).round().astype(np.int16)
        return x

    def _resample_float(self, x: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
        """Resample float32 mono using high-quality polyphase if available, else linear."""
        # Prefer scipy.signal.resample_poly if present
        try:
            from scipy.signal import resample_poly
            # Use gcd-like factorization to keep kernels small
            import math
            g = math.gcd(src_sr, dst_sr)
            up, down = dst_sr // g, src_sr // g
            return resample_poly(x, up, down).astype(np.float32, copy=False)
        except Exception:
            # Fallback: linear interpolation (good enough for AEC reverse reference)
            ratio = dst_sr / float(src_sr)
            n_out = int(round(len(x) * ratio))
            if n_out <= 1 or len(x) <= 1:
                return np.zeros(n_out, dtype=np.float32)
            xp = np.linspace(0.0, len(x) - 1, num=n_out, dtype=np.float32)
            idx = np.arange(len(x), dtype=np.float32)
            return np.interp(xp, idx, x).astype(np.float32, copy=False)

    def _chunk_and_enqueue(self, x: np.ndarray) -> int:
        """Append to carry, emit fixed-size frames into bounded queue. Returns frames published."""
        with self._lock:
            if len(self._carry) == 0:
                buf = x
            else:
                buf = np.concatenate([self._carry, x])

            published = 0
            start = 0
            end = len(buf)
            step = self.frame_samples

            while start + self.frame_samples <= end:
                frame = buf[start:start+step]
                # Enqueue newest; if full, deque drops oldest automatically
                self.queue.append(frame.astype(np.int16, copy=False))
                self._last_frame = frame
                start += step
                published += 1

            # Keep remainder for next call
            self._carry = buf[start:] if start < end else np.zeros(0, dtype=np.int16)
            return published
