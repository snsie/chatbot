use std::fs;
use std::path::PathBuf;
use webrtc_audio_processing as wap;

fn to_i16(bytes: &[u8]) -> Vec<i16> {
    let mut v = Vec::with_capacity(bytes.len() / 2);
    for chunk in bytes.chunks_exact(2) {
        let val = i16::from_le_bytes([chunk[0], chunk[1]]);
        v.push(val);
    }
    v
}

fn to_bytes_i16(samples: &[i16]) -> Vec<u8> {
    let mut out = Vec::with_capacity(samples.len() * 2);
    for s in samples {
        out.extend_from_slice(&s.to_le_bytes());
    }
    out
}

#[test]
fn compare_with_golden_if_present() {
    // This test compares against golden expected output if present.
    // Otherwise, it skips silently.
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let data = root.join("tests").join("data");
    let mic_path = data.join("mic_30ms.pcm");
    let rev_path = data.join("rev_30ms.pcm");
    let exp_path = data.join("expected_out_30ms.pcm");

    if !(mic_path.exists() && rev_path.exists() && exp_path.exists()) {
        eprintln!("[golden] skipping: data files not found");
        return;
    }

    let mic_bytes = fs::read(mic_path).unwrap();
    let rev_bytes = fs::read(rev_path).unwrap();
    let exp_bytes = fs::read(exp_path).unwrap();

    assert_eq!(mic_bytes.len(), rev_bytes.len());
    assert_eq!(mic_bytes.len(), exp_bytes.len());

    let mic_i16 = to_i16(&mic_bytes);
    let rev_i16 = to_i16(&rev_bytes);

    let init_cfg = wap::InitializationConfig {
        sample_rate_hz: 16_000,
        num_capture_channels: 1,
        num_render_channels: 1,
    };
    let mut proc = wap::Processor::new(&init_cfg.into()).unwrap();
    proc.set_config(wap::Config::default());

    let n = wap::NUM_SAMPLES_PER_FRAME as usize; // 480
    let mut cap = vec![0.0f32; n];
    let mut ren = vec![0.0f32; n];
    for i in 0..n { ren[i] = (rev_i16[i] as f32) / 32768.0; }
    for i in 0..n { cap[i] = (mic_i16[i] as f32) / 32768.0; }

    proc.process_render_frame(std::slice::from_mut(&mut &mut ren[..])).unwrap();
    proc.process_capture_frame(std::slice::from_mut(&mut &mut cap[..])).unwrap();

    let out_i16: Vec<i16> = cap.iter().map(|&x| {
        let s = (x * 32768.0).round();
        if s > i16::MAX as f32 { i16::MAX } else if s < i16::MIN as f32 { i16::MIN } else { s as i16 }
    }).collect();

    let out_bytes = to_bytes_i16(&out_i16);
    assert_eq!(out_bytes, exp_bytes, "Rust APM output differs from golden");
}
