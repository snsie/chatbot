#!/usr/bin/env python3
"""
Test script to compare voice quality improvements in Coqui TTS
"""

import asyncio
import sys
import os

# Add the current directory to Python path to import our chatbot modules
sys.path.insert(0, os.path.dirname(__file__))

from streaming_voice_chatbot import CoquiTTSSpeaker, EdgeTTSSpeaker, Pyttsx3Speaker

async def test_voices():
    """Test different TTS backends with the same text"""
    
    test_sentences = [
        "Hello! I'm your AI assistant. How can I help you today?",
        "The weather looks beautiful today, doesn't it?", 
        "I can answer questions, help with tasks, or just have a conversation!",
        "What would you like to talk about?"
    ]
    
    print("🎤 Voice Quality Comparison Test")
    print("=" * 50)
    
    # Test Coqui TTS
    print("\n🧠 Testing Coqui TTS (Neural)...")
    try:
        coqui_speaker = CoquiTTSSpeaker("tts_models/en/vctk/vits")
        for i, sentence in enumerate(test_sentences, 1):
            print(f"  {i}. Speaking: {sentence}")
            await coqui_speaker.speak(sentence)
            await asyncio.sleep(0.5)  # Brief pause between sentences
        await coqui_speaker.close()
        print("✅ Coqui TTS test completed")
    except Exception as e:
        print(f"❌ Coqui TTS failed: {e}")
    
    print("\n🔊 Testing Edge TTS...")
    try:
        edge_speaker = EdgeTTSSpeaker("en-US-JennyNeural")
        for i, sentence in enumerate(test_sentences, 1):
            print(f"  {i}. Speaking: {sentence}")
            await edge_speaker.speak(sentence)
            await asyncio.sleep(0.5)
        await edge_speaker.close()
        print("✅ Edge TTS test completed")
    except Exception as e:
        print(f"❌ Edge TTS failed: {e}")
    
    print("\n🗣️  Testing pyttsx3 (Baseline)...")
    try:
        pyttsx3_speaker = Pyttsx3Speaker()
        for i, sentence in enumerate(test_sentences, 1):
            print(f"  {i}. Speaking: {sentence}")
            await pyttsx3_speaker.speak(sentence)
            await asyncio.sleep(0.5)
        await pyttsx3_speaker.close()
        print("✅ pyttsx3 test completed")
    except Exception as e:
        print(f"❌ pyttsx3 failed: {e}")
    
    print("\n🎉 Voice comparison test finished!")
    print("Compare the naturalness, clarity, and emotion in each voice.")

if __name__ == "__main__":
    asyncio.run(test_voices())
