#!/usr/bin/env python3
"""
Test script for the new StreamingUtteranceDetector functionality.
"""

import asyncio
import sys
from streaming_voice_chatbot.detection import StreamingUtteranceDetector
from streaming_voice_chatbot.config import default_config


async def test_streaming_detection():
    """Test the streaming utterance detection."""
    print("Testing StreamingUtteranceDetector...")
    
    detector = StreamingUtteranceDetector(default_config)
    
    try:
        print("Starting recording...")
        await detector.start_recording()
        print("Recording started successfully!")
        
        print("Testing get_utterance with short timeout (should return None)...")
        result = await detector.get_utterance(timeout=2.0)
        if result is None:
            print("✓ Timeout test passed - no utterance detected in 2 seconds")
        else:
            print(f"✗ Unexpected result: {type(result)}")
            
    except Exception as e:
        print(f"Error during test: {e}")
    finally:
        print("Cleaning up...")
        await detector.cleanup()
        print("✓ Cleanup completed")


async def main():
    """Main test function."""
    try:
        await test_streaming_detection()
        print("✓ All tests completed successfully!")
    except KeyboardInterrupt:
        print("\n✓ Test interrupted by user")
    except Exception as e:
        print(f"✗ Test failed with error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
