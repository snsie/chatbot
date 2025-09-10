#!/usr/bin/env python3
"""
Test Edge-TTS functionality
"""

import asyncio
import tempfile
import os

async def test_edge_tts():
    """Test edge-TTS audio generation and playback."""
    print("🧪 Testing Edge-TTS...")
    
    try:
        import edge_tts
        from pydub import AudioSegment
        from pydub.playback import play
        
        # Test text
        text = "Hello! This is a test of the edge TTS system."
        voice = "en-US-AriaNeural"
        
        print(f"Generating speech: '{text}'")
        
        # Generate speech
        communicate = edge_tts.Communicate(text, voice)
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp_file:
            temp_path = tmp_file.name
        
        await communicate.save(temp_path)
        print(f"Audio saved to: {temp_path}")
        
        # Check file size
        file_size = os.path.getsize(temp_path)
        print(f"File size: {file_size} bytes")
        
        if file_size > 0:
            # Play the audio
            print("Playing audio...")
            audio = AudioSegment.from_file(temp_path)
            play(audio)
            print("✅ Audio playback completed!")
        else:
            print("❌ Audio file is empty")
        
        # Clean up
        os.unlink(temp_path)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_edge_tts())
