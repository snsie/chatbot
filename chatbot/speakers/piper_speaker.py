from .base_speaker import BaseSpeaker
from typing import Optional
from ..constants import VOICE_NAME, SAMPLE_RATE
import os
import sys
import io


class PiperSpeaker(BaseSpeaker):
    """Piper TTS speaker; per-sentence synthesis & playback.

    Expects a Piper voice model (.onnx) path. Synthesizes to WAV bytes via the
    piper-tts Python API, decodes with pydub, and plays with simpleaudio.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        *,
        length_scale: float = 1.0,
        noise_scale: float = 0.0,
        noise_w: float = 0,
        espeak_data_dir: Optional[str] = None,
        use_native_sample_rate: bool = False,
        use_cuda: Optional[bool] = None,
    ):
        # Resolve model path preference order: explicit arg -> VOICE_NAME (if looks like a path) -> env var
        resolved_model = model_path or (
            VOICE_NAME if isinstance(VOICE_NAME, str) and ("/" in VOICE_NAME or VOICE_NAME.endswith(".onnx")) else None
        ) or os.environ.get("PIPER_MODEL")

        if not resolved_model:
            print(
                "[PiperSpeaker] No model_path provided. Set VOICE_NAME to a .onnx path, pass model_path, or set PIPER_MODEL.",
                file=sys.stderr,
            )
            raise ImportError("Piper model path is required")

        self.model_path = resolved_model
        # self.length_scale = length_scale
        # self.noise_scale = noise_scale
        # self.noise_w = noise_w
        self.use_cuda = use_cuda
        # Try to resolve espeak-ng data directory if not provided
        if espeak_data_dir:
            self.espeak_data_dir = espeak_data_dir
        else:
            default_espeak = os.environ.get("ESPEAK_DATA_PATH")
            if not default_espeak:
                # common Linux path
                candidate = "/usr/share/espeak-ng-data"
                default_espeak = candidate if os.path.isdir(candidate) else None
            self.espeak_data_dir = default_espeak
        self.use_native_sample_rate = use_native_sample_rate

        try:
            import pydub  # noqa: F401
            import simpleaudio  # noqa: F401
            import piper  # noqa: F401
        except ImportError:
            print(
                "[PiperSpeaker] Missing packages. Install: pip install piper-tts pydub simpleaudio",
                file=sys.stderr,
            )
            raise

        # Load the Piper voice model once
        import piper

        # piper.PiperVoice.load accepts file path or file-like; prefer path for simplicity
        try:
            # Let caller control CUDA usage; default to library's auto if None
            kwargs = {}
            if self.espeak_data_dir:
                kwargs["data_dir"] = self.espeak_data_dir
            if self.use_cuda is not None:
                kwargs["use_cuda"] = self.use_cuda
            self._voice = piper.PiperVoice.load(self.model_path, **kwargs)
        except TypeError:
            # Fallback for older piper versions without data_dir kwarg
            self._voice = piper.PiperVoice.load(self.model_path,use_cuda=True)

    async def speak(self, sentence: str):
        from pydub import AudioSegment
        import simpleaudio as sa

        if not sentence:
            return

        # Synthesize to in-memory WAV using Piper's synthesize_wav API
        import wave
        from piper.config import SynthesisConfig

        # syn_cfg = SynthesisConfig(
        #     length_scale=self.length_scale,
        #     noise_scale=self.noise_scale,
        #     noise_w_scale=self.noise_w,
        # )

        buf = io.BytesIO()
        with wave.open(buf, 'wb') as wav_file:
            # set_wav_format=True will set correct WAV header/format
            self._voice.synthesize_wav(sentence, wav_file,  set_wav_format=True)

        data = buf.getvalue()
        if not data:
            # Likely espeak-ng data missing. Provide helpful guidance and fail fast.
            hint = "Install espeak-ng (Linux: sudo apt-get install espeak-ng espeak-ng-data)"
            if not self.espeak_data_dir:
                hint += " or set ESPEAK_DATA_PATH to your espeak-ng data directory."
            raise RuntimeError(f"Piper synthesis produced no audio. {hint}")

        # Decode / resample to project sample rate and play
        buf.seek(0)
        audio_seg = AudioSegment.from_file(buf, format="wav")
        if self.use_native_sample_rate:
            # Keep model's native sample rate; ensure mono 16-bit for simpleaudio
            audio_seg = audio_seg.set_channels(1).set_sample_width(2)
        else:
            # Resample explicitly to the project sample rate if requested
            audio_seg = (
                audio_seg.set_frame_rate(SAMPLE_RATE).set_channels(1).set_sample_width(2)
            )
        # Optional: lightweight debug to verify playback parameters
        if os.environ.get("DEBUG_AUDIO"):
            print(
                f"[PiperSpeaker] frame_rate={audio_seg.frame_rate}Hz, channels=1, bytes_per_sample=2, duration={round(audio_seg.duration_seconds, 3)}s",
                file=sys.stderr,
            )
        play_obj = sa.play_buffer(
            audio_seg.raw_data,
            num_channels=1,
            bytes_per_sample=2,
            sample_rate=audio_seg.frame_rate,
        )
        play_obj.wait_done()

    async def close(self):
        # Piper doesn't require explicit cleanup, but drop reference just in case
        self._voice = None
