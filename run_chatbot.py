#!/usr/bin/env python3
"""
Runner script for the Streaming Voice Chatbot.

This script provides a simple way to run the chatbot with default settings.
For more advanced usage, import the module and create a custom configuration.

Usage:
    python run_chatbot.py
"""

import asyncio
from streaming_voice_chatbot import StreamingVoiceChatbot


async def main():
    """Run the chatbot with default configuration."""
    chatbot = StreamingVoiceChatbot()
    await chatbot.run()


if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
