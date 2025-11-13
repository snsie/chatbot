# apm_rs

PyO3-powered Rust extension that wraps WebRTC AudioProcessing (AEC/NS/AGC/VAD) for low-latency real-time use.

- Backend: `webrtc-audio-processing` crate (wrapper over Google WebRTC APM)
- Frames: Internally uses 30 ms frames at 16 kHz (480 samples), but also supports 10 ms streaming API.
- Python: Built with `maturin` and importable as `apm_rs`.

## Build and install (Linux)

Prereqs:
- Rust toolchain (rustup)
- Python 3.8+ (ABI3)
- `pip install maturin`
- Build tools for bundled APM (recommended to avoid pkg-config mismatches):
  - Debian/Ubuntu: `sudo apt-get install -y build-essential pkg-config automake autoconf libtool`
  - Fedora: `sudo dnf install -y gcc gcc-c++ pkgconf automake autoconf libtool`

Build wheel and install into current env:

```
# from repo root (contains rust/apm_rs)
cd rust/apm_rs
maturin develop --release
```

This builds a native extension (cdylib) and installs it in your active Python environment. The `webrtc-audio-processing` crate
is configured with the `bundled` feature, so it compiles the underlying C++ library automatically (no system `.pc` file needed).

## Python usage

```python
import apm_rs

# Initialize for 16kHz mono, typical WebRTC settings
proc = apm_rs.ApmProcessor(
    sample_rate=16000,
    capture_channels=1,
    render_channels=1,
    aec_level=2,      # 0..3 typical
    ns_level=3,       # 0..3 typical
    agc_level=2,      # 0..3 typical (mode internally mapped)
    vad_level=2       # 0..3 typical
)

# 30 ms processing (preferred for simplicity)
# mic_30 and rev_30 are bytes/array of int16, length 480 samples (mono)
out_30 = proc.process_stream_30ms(mic_30, rev_30)  # returns bytes for 480 samples

# 10 ms streaming variant
# Provide 160-sample frames; returns b"" for the first two calls,
# then returns 160-sample processed bytes each call thereafter.
out_10 = proc.process_stream_10ms(mic_10, rev_10)  # len(out_10) in {0, 320}

# Zero-copy into a preallocated output buffer (30ms)
# `out` must be a writable buffer of 960 bytes (480 int16)
proc.process_stream_into_30ms(out, mic_30, rev_30)
```

## Latency and performance
- Target per-frame processing time: < 5 ms on typical Linux x86_64.
- The 10 ms API buffers three 10 ms chunks and processes them in one 30 ms call (WebRTC APM requires 30 ms frames). This adds up to ~20 ms algorithmic delay. Use the 30 ms API in Python to keep integration simple and predictable.
- Hot path avoids reallocations by reusing internal buffers.

## Test against Python APM (golden output)

1) Generate golden output with the existing Python `webrtc_audio_processing` module:

```
python tools/gen_golden_apm.py \
  --mic tests/data/mic_30ms.pcm \
  --rev tests/data/rev_30ms.pcm \
  --out tests/data/expected_out_30ms.pcm
```

2) Run Rust tests:

```
cargo test -- --nocapture
```

If the `expected_out_30ms.pcm` file is present, the test compares exact bytes; otherwise it skips.

## Notes
- This wrapper uses f32 internally because APM operates on floats; int16 I/O is converted with saturation.
- VAD state is enabled through APM. For additional VAD (webrtcvad) in Python you can keep your existing logic.

## Troubleshooting

- If you prefer dynamic linking to the system package instead of `bundled`, edit `Cargo.toml` to remove the `bundled` feature.
  On Ubuntu, the pkg-config file is often named `webrtc-audio-processing-1.pc` at `/usr/lib/x86_64-linux-gnu/pkgconfig/`.
  If your build tool searches for `webrtc-audio-processing.pc`, either set `PKG_CONFIG_PATH` appropriately and create an alias symlink,
  or use the `bundled` feature as provided by default in this crate.
