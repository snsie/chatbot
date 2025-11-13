#!/usr/bin/env python3
import argparse
import struct

try:
    from webrtc_audio_processing import AudioProcessingModule
except Exception as e:
    raise SystemExit(f"webrtc_audio_processing not available: {e}")

SAMPLES_PER_30MS = 480


def read_pcm_i16(path):
    with open(path, 'rb') as f:
        data = f.read()
    if len(data) != SAMPLES_PER_30MS * 2:
        raise ValueError(f"{path}: expected 960 bytes, got {len(data)}")
    return data


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mic', required=True, help='30ms mono int16 PCM (480 samples)')
    p.add_argument('--rev', required=True, help='30ms mono int16 PCM (480 samples)')
    p.add_argument('--out', required=True, help='write expected processed output')
    args = p.parse_args()

    apm = AudioProcessingModule()
    apm.set_stream_format(16000, 1)
    apm.set_reverse_stream_format(16000, 1)
    apm.set_aec_level(2)
    apm.set_ns_level(3)
    apm.set_agc_level(2)
    apm.set_agc_target(0)
    apm.set_vad_level(2)
    apm.set_system_delay(0)

    mic = read_pcm_i16(args.mic)
    rev = read_pcm_i16(args.rev)

    # split to 10ms chunks
    def chunks_10ms(buf):
        for i in range(3):
            yield buf[i*320:(i+1)*320]

    out_parts = []
    for r10, m10 in zip(chunks_10ms(rev), chunks_10ms(mic)):
        apm.process_reverse_stream(r10)
        out10 = apm.process_stream(m10)
        out_parts.append(out10)

    out = b''.join(out_parts)
    with open(args.out, 'wb') as f:
        f.write(out)
    print(f"Wrote golden: {args.out}")


if __name__ == '__main__':
    main()
