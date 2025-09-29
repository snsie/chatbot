# --- imports ---
from pathlib import Path
import numpy as np
import torch
import torchaudio
import os, warnings
from speechbrain.inference.separation import SepformerSeparation as Separator
# Hide most warnings (keep this first)
warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio")
warnings.filterwarnings("ignore", category=UserWarning, module="speechbrain")
warnings.filterwarnings("ignore", category=FutureWarning, module="speechbrain")
# (Optional) nuke all warnings:
# warnings.simplefilter("ignore")
# or via env: os.environ["PYTHONWARNINGS"] = "ignore"
# Use the new SpeechBrain import path (pretrained -> inference)
from speechbrain.inference import EncoderClassifier

# --- utils ---
print("WAVs here:", [p.name for p in Path.cwd().glob("*.wav")])

PROJECT_DIR = Path(__file__).parent.resolve()
SR_PIPELINE = 16000  # keep the rest of your stack at 16k

def wav_to_tensor(path_like) -> torch.Tensor:
    """Load audio, resample to 16k mono float32, return shape [1, T]."""
    p = Path(path_like)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")
    wav, sr = torchaudio.load(str(p))  # [C, T]
    if wav.shape[0] > 1:
        wav = torch.mean(wav, dim=0, keepdim=True)
    if sr != SR_PIPELINE:
        wav = torchaudio.functional.resample(wav, sr, SR_PIPELINE)
    return wav.contiguous().float()

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = a / (np.linalg.norm(a) + 1e-8)
    b = b / (np.linalg.norm(b) + 1e-8)
    return float(np.dot(a, b))

# --- enroll target speaker ---
print("WAVs here:", [p.name for p in Path.cwd().glob("*.wav")])

ENROLL_WAV = PROJECT_DIR / "enroll_target.wav"
enroll_wav = wav_to_tensor(ENROLL_WAV)

MODEL_ID = "speechbrain/sepformer-wsj02mix"   # <= 2-speaker separation @ 8 kHz
sep = Separator.from_hparams(
    source=MODEL_ID,
    savedir=str(PROJECT_DIR / "pretrained" / MODEL_ID.replace("/", "_")),
    run_opts={"device": "cuda"}  # or "cpu"
)

with torch.inference_mode():
    # returns shape typically [n_src, T, 1]
    est = sep.separate_file(path=str(ENROLL_WAV))

# Save as s1.wav / s2.wav at model sample rate (8 kHz)
torchaudio.save(str(PROJECT_DIR / "s1.wav"), est[0, :, 0].cpu().unsqueeze(0), 8000)
torchaudio.save(str(PROJECT_DIR / "s2.wav"), est[1, :, 0].cpu().unsqueeze(0), 8000)

# --- your separation outputs ---
OUT_A = PROJECT_DIR / "stemA.wav"
OUT_B = PROJECT_DIR / "stemB.wav"

stemA = wav_to_tensor(OUT_A)
stemB = wav_to_tensor(OUT_B)

with torch.inference_mode():
    embA = spk_model.encode_batch(stemA).detach().cpu().numpy()[0]
    embB = spk_model.encode_batch(stemB).detach().cpu().numpy()[0]

scoreA = cosine(enroll_emb, embA)
scoreB = cosine(enroll_emb, embB)

target_path = OUT_A if scoreA >= scoreB else OUT_B
print(f"Target selected: {target_path}  (cosA={scoreA:.3f}, cosB={scoreB:.3f})")
print("WAVs here:", [p.name for p in Path.cwd().glob("*.wav")])
