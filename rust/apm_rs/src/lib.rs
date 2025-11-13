use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyByteArray};
use webrtc_audio_processing as wap;

const SR_16K: i32 = 16_000;
const CH_MONO: usize = 1;
const SAMPLES_PER_30MS: usize = wap::NUM_SAMPLES_PER_FRAME as usize; // 480 at 16k mono
const SAMPLES_PER_10MS: usize = SAMPLES_PER_30MS / 3; // 160 at 16k mono

fn i16_to_f32_clip(v: i16) -> f32 {
    // Scale to [-1,1] similar to Python side
    (v as f32) / 32768.0
}

fn f32_to_i16_clip(v: f32) -> i16 {
    // Saturating conversion
    let s = (v * 32768.0).round();
    if s > i16::MAX as f32 {
        i16::MAX
    } else if s < i16::MIN as f32 {
        i16::MIN
    } else {
        s as i16
    }
}

#[pyclass]
pub struct ApmProcessor {
    proc: wap::Processor,
    sample_rate: i32,
    cap_ch: usize,
    ren_ch: usize,
    // Rolling buffers for 10ms API
    mic_accum: Vec<i16>,
    rev_accum: Vec<i16>,
    // Output staging to avoid reallocation
    cap_frame_f32: Vec<f32>,         // length = cap_ch * 480
    ren_frame_f32: Vec<f32>,         // length = ren_ch * 480
    out_frame_i16: Vec<i16>,         // length = cap_ch * 480 (interleaved mono)
}

#[pymethods]
impl ApmProcessor {
    #[new]
    #[pyo3(signature = (sample_rate=16000, capture_channels=1, render_channels=1, aec_level=2, ns_level=3, agc_level=2, vad_level=2))]
    pub fn new(
        sample_rate: i32,
        capture_channels: usize,
        render_channels: usize,
        aec_level: u8,
        ns_level: u8,
        agc_level: u8,
        vad_level: u8,
    ) -> PyResult<Self> {
        if sample_rate != SR_16K {
            return Err(PyValueError::new_err("Only 16kHz is supported in this build"));
        }
        if capture_channels != CH_MONO || render_channels != CH_MONO {
            return Err(PyValueError::new_err("Only mono is supported in this build"));
        }

        let init_cfg = wap::InitializationConfig {
            num_capture_channels: capture_channels as i32,
            num_render_channels: render_channels as i32,
            enable_experimental_agc: false,
            enable_intelligibility_enhancer: false,
        };

        let mut proc = wap::Processor::new(&init_cfg)
            .map_err(|e| PyValueError::new_err(format!("Processor::new failed: {e}")))?;

        // Configure modules
        let aec = wap::EchoCancellation {
            suppression_level: match aec_level {
                0 => wap::EchoCancellationSuppressionLevel::Low,
                1 => wap::EchoCancellationSuppressionLevel::Moderate,
                _ => wap::EchoCancellationSuppressionLevel::High,
            },
            enable_extended_filter: true,
            enable_delay_agnostic: true,
            stream_delay_ms: None,
        };
        let ns = wap::NoiseSuppression {
            suppression_level: match ns_level {
                0 => wap::NoiseSuppressionLevel::Low,
                1 => wap::NoiseSuppressionLevel::Moderate,
                2 => wap::NoiseSuppressionLevel::High,
                _ => wap::NoiseSuppressionLevel::VeryHigh,
            },
        };
        let gc = wap::GainControl {
            mode: match agc_level {
                0 => wap::GainControlMode::AdaptiveDigital,
                _ => wap::GainControlMode::AdaptiveDigital,
            },
            target_level_dbfs: 0,
            compression_gain_db: 9,
            enable_limiter: true,
        };
        let vd = wap::VoiceDetection {
            detection_likelihood: match vad_level {
                0 => wap::VoiceDetectionLikelihood::VeryLow,
                1 => wap::VoiceDetectionLikelihood::Low,
                2 => wap::VoiceDetectionLikelihood::Moderate,
                _ => wap::VoiceDetectionLikelihood::High,
            },
        };

        let cfg = wap::Config {
            echo_cancellation: Some(aec),
            noise_suppression: Some(ns),
            gain_control: Some(gc),
            voice_detection: Some(vd),
            ..Default::default()
        };
        proc.set_config(cfg);

        Ok(Self {
            proc,
            sample_rate,
            cap_ch: capture_channels,
            ren_ch: render_channels,
            mic_accum: Vec::with_capacity(SAMPLES_PER_30MS),
            rev_accum: Vec::with_capacity(SAMPLES_PER_30MS),
            cap_frame_f32: vec![0.0; capture_channels * SAMPLES_PER_30MS],
            ren_frame_f32: vec![0.0; render_channels * SAMPLES_PER_30MS],
            out_frame_i16: vec![0; capture_channels * SAMPLES_PER_30MS],
        })
    }

    pub fn reset(&mut self) {
        self.mic_accum.clear();
        self.rev_accum.clear();
    }

    /// Process one 30 ms mono frame (480 samples). Returns bytes for 480 int16 samples.
    #[pyo3(text_signature = "($self, mic_30ms_bytes, rev_30ms_bytes)")]
    pub fn process_stream_30ms<'py>(&mut self, py: Python<'py>, mic_30ms_bytes: &Bound<'py, PyBytes>, rev_30ms_bytes: &Bound<'py, PyBytes>) -> PyResult<Bound<'py, PyBytes>> {
        let mic_bytes = mic_30ms_bytes.as_bytes();
        let rev_bytes = rev_30ms_bytes.as_bytes();
        if mic_bytes.len() != SAMPLES_PER_30MS * std::mem::size_of::<i16>() {
            return Err(PyValueError::new_err(format!(
                "mic_30ms must be {} int16 samples", SAMPLES_PER_30MS
            )));
        }
        if rev_bytes.len() != SAMPLES_PER_30MS * std::mem::size_of::<i16>() {
            return Err(PyValueError::new_err(format!(
                "rev_30ms must be {} int16 samples", SAMPLES_PER_30MS
            )));
        }
        self.process_internal_30ms_from_bytes(mic_bytes, rev_bytes)?;
        // Return as bytes to avoid Python-side per-sample overhead
        let bytes = unsafe {
            std::slice::from_raw_parts(
                self.out_frame_i16.as_ptr() as *const u8,
                self.out_frame_i16.len() * std::mem::size_of::<i16>(),
            )
        };
        Ok(PyBytes::new_bound(py, bytes))
    }

    /// Zero-copy into an output buffer (expects 480 int16 samples = 960 bytes)
    #[pyo3(text_signature = "($self, out_bytearray, mic_30ms_bytes, rev_30ms_bytes)")]
    pub fn process_stream_into_30ms<'py>(&mut self, _py: Python<'py>, out_bytearray: &Bound<'py, PyByteArray>, mic_30ms_bytes: &Bound<'py, PyBytes>, rev_30ms_bytes: &Bound<'py, PyBytes>) -> PyResult<()> {
        let out_bytes = unsafe { out_bytearray.as_bytes_mut() };
        if out_bytes.len() != SAMPLES_PER_30MS * std::mem::size_of::<i16>() {
            return Err(PyValueError::new_err(format!(
                "out must be {} int16 samples", SAMPLES_PER_30MS
            )));
        }
    let mic_bytes = mic_30ms_bytes.as_bytes();
    let rev_bytes = rev_30ms_bytes.as_bytes();
        if mic_bytes.len() != SAMPLES_PER_30MS * std::mem::size_of::<i16>() {
            return Err(PyValueError::new_err("mic_30ms length incorrect"));
        }
        if rev_bytes.len() != SAMPLES_PER_30MS * std::mem::size_of::<i16>() {
            return Err(PyValueError::new_err("rev_30ms length incorrect"));
        }
        self.process_internal_30ms_from_bytes(mic_bytes, rev_bytes)?;
        // copy i16 bytes into provided out buffer
        let src_bytes = unsafe {
            std::slice::from_raw_parts(
                self.out_frame_i16.as_ptr() as *const u8,
                self.out_frame_i16.len() * std::mem::size_of::<i16>(),
            )
        };
        out_bytes.copy_from_slice(src_bytes);
        Ok(())
    }

    /// 10 ms streaming API: buffers 3x10ms, processes, then returns 10 ms bytes (empty until first process completes)
    #[pyo3(text_signature = "($self, mic_10ms_bytes, rev_10ms_bytes)")]
    pub fn process_stream_10ms<'py>(&mut self, py: Python<'py>, mic_10ms_bytes: &Bound<'py, PyBytes>, rev_10ms_bytes: &Bound<'py, PyBytes>) -> PyResult<Bound<'py, PyBytes>> {
        let mic_bytes = mic_10ms_bytes.as_bytes();
        let rev_bytes = rev_10ms_bytes.as_bytes();
        if mic_bytes.len() != SAMPLES_PER_10MS * std::mem::size_of::<i16>() {
            return Err(PyValueError::new_err(format!(
                "mic_10ms must be {} int16 samples", SAMPLES_PER_10MS
            )));
        }
        if rev_bytes.len() != SAMPLES_PER_10MS * std::mem::size_of::<i16>() {
            return Err(PyValueError::new_err(format!(
                "rev_10ms must be {} int16 samples", SAMPLES_PER_10MS
            )));
        }
        // Append samples by decoding little-endian i16s from bytes
        append_i16_le_bytes(mic_bytes, &mut self.mic_accum);
        append_i16_le_bytes(rev_bytes, &mut self.rev_accum);

        if self.mic_accum.len() < SAMPLES_PER_30MS {
            // Not enough yet; return empty
            return Ok(PyBytes::new_bound(py, &[]));
        }

        // Process 30ms
        let mic_window_bytes = unsafe {
            std::slice::from_raw_parts(
                self.mic_accum.as_ptr() as *const u8,
                SAMPLES_PER_30MS * std::mem::size_of::<i16>(),
            )
        };
        let rev_window_bytes = unsafe {
            std::slice::from_raw_parts(
                self.rev_accum.as_ptr() as *const u8,
                SAMPLES_PER_30MS * std::mem::size_of::<i16>(),
            )
        };
        self.process_internal_30ms_from_bytes(mic_window_bytes, rev_window_bytes)?;

        // Shift accumulators
        self.mic_accum.drain(..SAMPLES_PER_30MS);
        self.rev_accum.drain(..SAMPLES_PER_30MS);

        // Return the first 10ms slice of the processed block (keep latency bounded)
        let bytes = unsafe {
            let out_10 = &self.out_frame_i16[..SAMPLES_PER_10MS];
            std::slice::from_raw_parts(
                out_10.as_ptr() as *const u8,
                out_10.len() * std::mem::size_of::<i16>(),
            )
        };
        Ok(PyBytes::new_bound(py, bytes))
    }
}

impl ApmProcessor {
    fn process_internal_30ms_from_bytes(&mut self, mic_bytes: &[u8], rev_bytes: &[u8]) -> PyResult<()> {
        debug_assert_eq!(mic_bytes.len(), SAMPLES_PER_30MS * 2);
        debug_assert_eq!(rev_bytes.len(), SAMPLES_PER_30MS * 2);

        // Convert little-endian i16 to f32 in-place buffers
        for i in 0..SAMPLES_PER_30MS {
            let mi = i * 2;
            let ri = i * 2;
            let m = i16::from_le_bytes([mic_bytes[mi], mic_bytes[mi + 1]]);
            let r = i16::from_le_bytes([rev_bytes[ri], rev_bytes[ri + 1]]);
            self.cap_frame_f32[i] = i16_to_f32_clip(m);
            self.ren_frame_f32[i] = i16_to_f32_clip(r);
        }
        // Render then capture per APM expectations
        self.proc
            .process_render_frame(&mut self.ren_frame_f32[..])
            .map_err(|e| PyValueError::new_err(format!("process_render_frame failed: {e}")))?;
        self.proc
            .process_capture_frame(&mut self.cap_frame_f32[..])
            .map_err(|e| PyValueError::new_err(format!("process_capture_frame failed: {e}")))?;

        for i in 0..SAMPLES_PER_30MS {
            self.out_frame_i16[i] = f32_to_i16_clip(self.cap_frame_f32[i]);
        }
        Ok(())
    }
}

// removed generic bytes helpers to simplify lifetimes; APIs accept PyBytes/PyByteArray directly

fn append_i16_le_bytes(src: &[u8], dst: &mut Vec<i16>) {
    debug_assert!(src.len() % 2 == 0);
    let mut i = 0;
    while i < src.len() {
        let v = i16::from_le_bytes([src[i], src[i + 1]]);
        dst.push(v);
        i += 2;
    }
}

#[pymodule]
fn apm_rs(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<ApmProcessor>()?;
    m.add("SAMPLES_PER_30MS", SAMPLES_PER_30MS as u32)?;
    m.add("SAMPLES_PER_10MS", SAMPLES_PER_10MS as u32)?;
    Ok(())
}
