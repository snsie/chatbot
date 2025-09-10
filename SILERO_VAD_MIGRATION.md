# Migration from WebRTC VAD to Silero VAD

## Summary of Changes

This document summarizes the changes made to replace WebRTC VAD with Silero VAD in the streaming voice chatbot.

### Key Changes

1. **Updated Dependencies**
   - Removed: `webrtcvad==2.0.10`
   - Added: `silero-vad` and `torch` (already present)

2. **Import Changes**
   ```python
   # Old
   import webrtcvad
   
   # New
   import torch
   from silero_vad import load_silero_vad, get_speech_timestamps
   ```

3. **Configuration Updates**
   - `FRAME_MS`: Changed from 10ms to 32ms (required for Silero VAD)
   - `VAD_AGGRESSIVENESS`: Replaced with `VAD_THRESHOLD` (0.0-1.0 range)
   - `TRAILING_SILENCE_MS`: Increased from 200ms to 500ms for better performance

4. **UtteranceDetector Class**
   - Complete rewrite to use Silero VAD
   - Automatic GPU/CPU detection and usage
   - Proper chunk size handling (512 samples = 32ms at 16kHz)
   - Enhanced error handling

### Technical Details

- **Chunk Size**: Silero VAD requires exactly 512 samples (32ms) at 16kHz
- **Threshold**: VAD threshold of 0.5 (adjustable from 0.0 to 1.0)
- **Performance**: GPU acceleration when available, CPU fallback
- **Accuracy**: Silero VAD typically provides better accuracy than WebRTC VAD

### Benefits of Silero VAD

1. **Higher Accuracy**: Better speech detection, especially in noisy environments
2. **Deep Learning Based**: Uses neural networks for more sophisticated detection
3. **GPU Acceleration**: Faster inference when GPU is available
4. **Active Development**: Regularly updated and maintained

### Testing

- Created `test_silero_vad.py` to verify functionality
- Confirmed GPU acceleration works (when available)
- Verified proper probability outputs for silence vs speech-like signals

### Backward Compatibility

The interface remains the same - the UtteranceDetector class still provides the same `record_once()` method, so no changes are needed in the main loop logic.
