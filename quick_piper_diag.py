import io
from pydub import AudioSegment
import piper
from piper.config import SynthesisConfig
import wave
from chatbot.constants import VOICE_NAME
import os

out_path = os.path.abspath("piper_diag_out.wav")
voice = piper.PiperVoice.load(VOICE_NAME)
syn_cfg = SynthesisConfig(length_scale=1.1, noise_scale=0.667, noise_w_scale=0.8)
with wave.open(out_path, "wb") as wav_file:
    voice.synthesize_wav(
        "This is a longer Piper diagnostic test. If you hear this clearly, audio playback is working.",
        wav_file,
        syn_config=syn_cfg,
        set_wav_format=True,
    )

print("wrote:", out_path, "exists:", os.path.exists(out_path), "size:", os.path.getsize(out_path) if os.path.exists(out_path) else 0)

if os.path.exists(out_path) and os.path.getsize(out_path) > 44:
    a = AudioSegment.from_file(out_path, format="wav")
    print("frame_rate:", a.frame_rate, "channels:", a.channels, "sample_width:", a.sample_width, "duration_s:", round(a.duration_seconds, 2))
    try:
        import simpleaudio as sa
        play_obj = sa.play_buffer(a.set_channels(1).set_sample_width(2).raw_data, 1, 2, a.frame_rate)
        play_obj.wait_done()
        print("played")
    except Exception as e:
        print("playback error:", e)
else:
    print("No output audio produced by Piper (file missing or tiny).")
