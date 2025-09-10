#!/usr/bin/env python3
"""
Test script for Silero VAD implementation
"""

import numpy as np
import torch
from silero_vad import load_silero_vad

def test_silero_vad():
    """Test basic Silero VAD functionality"""
    print("Testing Silero VAD...")
    
    # Load model
    vad_model = load_silero_vad()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vad_model.to(device)
    print(f"Model loaded on device: {device}")
    
    # Create test audio (silence) - exactly 512 samples for 16kHz
    sample_rate = 16000
    chunk_samples = 512  # Required for Silero VAD at 16kHz
    silence = np.zeros(chunk_samples, dtype=np.float32)
    
    # Create test audio (synthetic speech-like signal)
    t = np.linspace(0, chunk_samples/sample_rate, chunk_samples)
    speech_like = (np.sin(2 * np.pi * 440 * t) * 0.1 * np.random.random(len(t))).astype(np.float32)
    
    # Test VAD on silence
    silence_tensor = torch.from_numpy(silence).unsqueeze(0).to(device)
    with torch.no_grad():
        silence_prob = vad_model(silence_tensor, sample_rate).item()
    print(f"Silence probability: {silence_prob:.3f}")
    
    # Test VAD on speech-like signal
    speech_tensor = torch.from_numpy(speech_like).unsqueeze(0).to(device)
    with torch.no_grad():
        speech_prob = vad_model(speech_tensor, sample_rate).item()
    print(f"Speech-like probability: {speech_prob:.3f}")
    
    print("✓ Silero VAD test completed successfully!")

if __name__ == "__main__":
    test_silero_vad()
