import torch, torchaudio, soundfile as sf
from pathlib import Path
from speechbrain.inference.separation import SepformerSeparation as Separator

# Choose ONE of these:
# MODEL = "speechbrain/sepformer-wsj02mix"            # clean speech
MODEL = "speechbrain/sepformer-wham16k-separation"    # noisy/real-world

INPUT = "enroll_target.wav"
OUT_SR = 44100  # for easy playback; keep 16000 if you prefer

def get_model_sr(separator, default=16000):
    # Try to read SR from model hparams
    for k in ("sample_rate", "sr", "fs"):
        if k in separator.hparams:
            return int(separator.hparams[k])
    return default

def to_sources_time(out):
    # Normalize to [S, T] from either [1,S,T] or [1,T,S]
    if out.dim() != 3 or out.shape[0] != 1:
        raise RuntimeError(f"Unexpected output shape: {tuple(out.shape)} (expected [1,*,*])")
    B, A, C = out.shape
    # Heuristic: time dimension >> num sources (<=8 typically)
    if A <= 8 and C > 100:         # [1, S, T]
        return out[0]              # [S, T]
    elif C <= 8 and A > 100:       # [1, T, S]
        return out.transpose(1, 2)[0]  # [S, T]
    else:
        # Fallback: choose the variant with longer time axis
        a = out[0]                 # assume [S,T]
        b = out.transpose(1,2)[0]  # swapped
        return a if a.shape[1] >= b.shape[1] else b

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    separator = Separator.from_hparams(
        source=MODEL,
        savedir=f"./pretrained_models/{MODEL.split('/')[-1]}",
        run_opts={"device": device},
    )
    SB_SR = get_model_sr(separator, default=16000)

    wav = Path(INPUT)
    mix, sr = torchaudio.load(str(wav))   # [C, T]
    if mix.shape[0] > 1:
        mix = mix.mean(dim=0, keepdim=True)
    if sr != SB_SR:
        mix = torchaudio.functional.resample(mix, sr, SB_SR)
        sr = SB_SR

    mono = mix.squeeze(0).to(device)      # [T]
    with torch.no_grad():
        out = separator.separate_batch(mono.unsqueeze(0))  # [1,*,*] (layout varies)

    est = to_sources_time(out).cpu()      # [S, T]
    # Keep exactly 2 stems (top-2 energy if >2)
    if est.shape[0] > 2:
        energies = (est**2).mean(dim=1)
        idx = torch.topk(energies, k=2).indices
        est = est[idx]
    elif est.shape[0] < 2:
        import torch as t
        pad = t.zeros(2 - est.shape[0], est.shape[1])
        est = t.cat([est, pad], dim=0)

    # Peak-normalize to -1 dBFS and save as PCM16 @ OUT_SR
    out_dir = wav.parent / f"{wav.stem}_separated"
    out_dir.mkdir(exist_ok=True)
    for i in range(2):
        s = est[i]
        peak = float(s.abs().max())
        if peak > 0:
            s = s * (10**(-1/20) / peak)
        if sr != OUT_SR:
            s = torchaudio.functional.resample(s.unsqueeze(0), sr, OUT_SR).squeeze(0)
            sr_out = OUT_SR
        else:
            sr_out = sr
        sf.write(str(out_dir / f"{wav.stem}_source_{i+1}.wav"),
                 (s.clamp(-1,1).numpy() * 32767).astype("int16"),
                 sr_out, subtype="PCM_16")
        print(f"✅ Wrote {out_dir / f'{wav.stem}_source_{i+1}.wav'} ({sr_out} Hz, {s.numel()/sr_out:.2f}s)")

if __name__ == "__main__":
    main()

