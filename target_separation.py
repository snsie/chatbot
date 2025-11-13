# target_separation.py
from typing import Tuple
import numpy as np, torch, torch.nn.functional as F

@torch.inference_mode()
def extract_target_voice(
    mix_audio: np.ndarray, sr: int,
    target_emb: torch.Tensor, *,
    sep_model, spkrec_model,
    max_speakers: int = 2,
    sim_floor: float = 0.65,
    enable: bool = True,
    device: str = "cuda",
    hard_cap_sec: float = 6.0
) -> Tuple[np.ndarray, float, int, bool]:
    if not enable or mix_audio is None or len(mix_audio) == 0:
        return mix_audio, -1.0, -1, False
    if len(mix_audio) > int(hard_cap_sec * sr):
        mix_audio = mix_audio[: int(hard_cap_sec * sr)]

    mix_t = torch.from_numpy(mix_audio).to(device=device, dtype=torch.float32).unsqueeze(0)  # [1,T]
    try:
        sep = sep_model.separate_batch(mix_t)  # [1,S,T]
    except Exception:
        return mix_audio, -1.0, -1, False
    if sep.ndim != 3:
        return mix_audio, -1.0, -1, False

    S = min(sep.shape[1], max_speakers)
    best_idx, best_sim = -1, -1.0
    for i in range(S):
        voice_i = sep[0, i, :].unsqueeze(0)             # [1,T]
        emb_i = spkrec_model.encode_batch(voice_i)      # [1,D]
        sim_i = float(F.cosine_similarity(target_emb, emb_i).item())
        if sim_i > best_sim: best_sim, best_idx = sim_i, i

    if best_sim < sim_floor or best_idx < 0:
        return mix_audio, best_sim, -1, True  # tried but keep original
    best = sep[0, best_idx, :].detach().to("cpu").float().numpy()
    return np.clip(best, -1.0, 1.0).astype(np.float32), best_sim, best_idx, True
