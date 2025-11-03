from typing import List, Optional
from ..constants import SAMPLE_RATE, VAD_AGGRESSIVENESS, FRAME_MS, MIN_VOICED_FRAMES, TRAILING_SILENCE_FRAMES, FRAME_SAMPLES
import webrtcvad
import numpy as np
import sounddevice as sd
import queue
# AEC
from chatbot.utils.AEC_publisher import ReverseAudioPublisher
reverse_pub = ReverseAudioPublisher(sample_rate=16000, frame_ms=30, max_frames=50)
from webrtc_audio_processing import AudioProcessingModule
# Create the processor
apm = AudioProcessingModule()

apm.set_stream_format(16000, 1)           # mic stream format
apm.set_reverse_stream_format(16000, 1)   # playback/ref stream format

# Tune the processing modules.
# Typical ranges are small ints, e.g. 0=off/low ... 2 or 3=stronger.
apm.set_aec_level(2)        # echo cancellation aggressiveness
apm.set_ns_level(3)         # noise suppression strength
apm.set_agc_level(2)        # automatic gain control mode/strength
apm.set_agc_target(12)    # target loudness-ish; tweak later
apm.set_vad_level(2)        # VAD sensitivity (lower = stricter voice detection)

# How much audio output latency (ms) to expect between far-end and mic.
# Start with 0; you can increase this if you get weird residual echo.
apm.set_system_delay(0)


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

                # 1) Feed latest speaker frame to AEC
                # frame: 30 ms mic bytes (960 bytes @ 16k, int16 mono)
                # reverse_pub returns a 30 ms np.int16[480]; convert to bytes
                reverse30 = reverse_pub.get_latest_frame().tobytes()  # 480 * 2 = 960 bytes

                # def chunks_10ms(buf: bytes):
                #     # 10 ms @ 16 kHz int16 mono = 160 samples = 320 bytes
                #     for i in range(3):
                #         start = i * 320
                #         yield buf[start:start+320]
                # Helper: slice a 30 ms block into 3 × 10 ms subframes (320 bytes each)
                def chunks_10ms(buf: bytes):
                    # Split a FRAME_MS block into 10 ms chunks at SAMPLE_RATE (int16 mono)
                    chunk_ms = 10
                    samples_per_chunk = (SAMPLE_RATE * chunk_ms) // 1000
                    bytes_per_chunk = samples_per_chunk * 2  # 2 bytes per int16 sample
                    # print('samples_per_chunk:', samples_per_chunk, 'bytes_per_chunk:', bytes_per_chunk)
                    # Yield only full 10 ms chunks (APM expects exact 10 ms)
                    for start in range(0, len(buf) - (len(buf) % bytes_per_chunk), bytes_per_chunk):
                        yield buf[start:start + bytes_per_chunk]

                # 960 bytes comes from: 16_000 Hz * 30 ms = 480 samples; int16 = 2 bytes/sample; mono = 1 channel
                CHANNELS = 1
                BYTES_PER_SAMPLE = np.dtype(np.int16).itemsize  # 2 bytes for int16

                expected_len = int(SAMPLE_RATE * (FRAME_MS / 1000)) * BYTES_PER_SAMPLE * CHANNELS
                # print('Expected length:', expected_len, 'Actual length:', len(reverse30))
                # If reverse publisher is empty for some reason, use silence
                if len(reverse30) != expected_len:
                    reverse30 = b"\x00" * expected_len

                # 1) AEC: feed reverse 10 ms chunk, then process mic 10 ms chunk — do this 3 times
                out_parts = []
                for rev10, mic10 in zip(chunks_10ms(reverse30), chunks_10ms(frame)):
                    apm.process_reverse_stream(rev10)         # expects bytes (10 ms)
                    out10 = apm.process_stream(mic10)         # expects bytes (10 ms), returns bytes
                    out_parts.append(out10)

                # Reassemble processed 30 ms block for VAD + collection
                proc_bytes = b"".join(out_parts)              # 960 bytes (480 samples)
                frame=proc_bytes
                # print(f"Processed frame length: {len(proc_bytes)} bytes")
                # print(f"Original frame length: {len(frame)} bytes")
                # 3) VAD decision on processed frame
                is_speech = False
                try:
                    is_speech = self.vad.is_speech(frame, SAMPLE_RATE)
                except Exception:
                    # If VAD fails (rare), treat as silence
                    is_speech = False
                print(f"VAD decision: {'speech' if is_speech else 'silence'}")
                # 4) Collect frames based on VAD
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