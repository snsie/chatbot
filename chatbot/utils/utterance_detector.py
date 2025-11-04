from typing import List, Optional
from ..constants import SAMPLE_RATE, VAD_AGGRESSIVENESS, FRAME_MS, MIN_VOICED_FRAMES, TRAILING_SILENCE_FRAMES, FRAME_SAMPLES
import webrtcvad
import numpy as np
import sounddevice as sd
import queue
# AEC
from chatbot.utils.AEC_publisher import ReverseAudioPublisher
reverse_pub = ReverseAudioPublisher(sample_rate=16000, frame_ms=30, max_frames=50)
# Prefer Rust APM (PyO3) if available for low-latency, fallback to Python APM
try:
    import apm_rs  # built via maturin from rust/apm_rs
    _USE_RUST_APM = True
except Exception:
    _USE_RUST_APM = False
    from webrtc_audio_processing import AudioProcessingModule
    # Create the processor

import math

# Additional gating to reduce false positives from steady noises (e.g., rain)
# Frames with RMS below this dBFS are treated as silence regardless of VAD
MIN_SPEECH_DBFS = -40.0  # tweak between -45 .. -35 to taste

def rms_dbfs(pcm_bytes: bytes) -> float:
    """Return RMS level in dBFS for 16-bit mono PCM."""
    if not pcm_bytes:
        return -120.0
    pcm = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32)
    rms = np.sqrt(np.mean(pcm ** 2) + 1e-12)
    # 16-bit full scale is 32768
    dbfs = 20 * math.log10(rms / 32768.0 + 1e-12)
    return dbfs

if _USE_RUST_APM:
    print(_USE_RUST_APM)
    # Initialize Rust APM: 16kHz mono, levels roughly matching prior Python config
    apm = apm_rs.ApmProcessor(
        sample_rate=16000,
        capture_channels=1,
        render_channels=1,
        aec_level=2,
        ns_level=3,
        agc_level=2,
        vad_level=2,
    )
else:
    apm = AudioProcessingModule()
    apm.set_stream_format(16000, 1)           # mic stream format
    apm.set_reverse_stream_format(16000, 1)   # playback/ref stream format
    # Tune the processing modules.
    apm.set_aec_level(2)        # echo cancellation aggressiveness
    apm.set_ns_level(2)         # noise suppression strength
    apm.set_agc_level(2)        # automatic gain control mode/strength
    apm.set_agc_target(0)       # target loudness-ish; tweak later
    # apm.enable_vad(True)      # optional
    apm.set_vad_level(2)        # VAD sensitivity (lower = stricter)
    # System delay hint
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

                in_level = rms_dbfs(frame)

                if _USE_RUST_APM:
                    # Rust extension expects raw 30 ms bytes for mic and reverse
                    proc_bytes = apm.process_stream_30ms(frame, reverse30)
                else:
                    # 1) AEC: feed reverse 10 ms chunk, then process mic 10 ms chunk — do this 3 times
                    out_parts = []
                    for rev10, mic10 in zip(chunks_10ms(reverse30), chunks_10ms(frame)):
                        apm.process_reverse_stream(rev10)         # expects bytes (10 ms)
                        out10 = apm.process_stream(mic10)         # expects bytes (10 ms), returns bytes
                        out_parts.append(out10)
                    # Reassemble processed 30 ms block for VAD + collection
                    proc_bytes = b"".join(out_parts)              # 960 bytes (480 samples)
                out_level = rms_dbfs(proc_bytes)
                # print(f"APM levels: in={in_level:.1f} dBFS, out={out_level:.1f} dBFS")
                # Treat very-quiet output as silence to avoid AGC/NS artifacts
                if out_level < MIN_SPEECH_DBFS:
                    is_speech = False
                else:
                    # Vote VAD over 3×10ms subframes to reduce false positives
                    votes = 0
                    for sub10 in chunks_10ms(proc_bytes):
                        try:
                            if self.vad.is_speech(sub10, SAMPLE_RATE):
                                votes += 1
                        except Exception:
                            # ignore malformed subframes
                            pass
                    # print('votes',votes)
                    is_speech = votes >= 2  # require at least 2/3 subframes to be speech
                # print(f"VAD decision: {'speech' if is_speech else 'silence'} (level={out_level:.1f} dBFS)")

                frame = proc_bytes
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