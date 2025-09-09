#!/usr/bin/env python3
"""
Example usage of the Streaming Voice Chatbot module with custom configuration.

This demonstrates how to customize the chatbot behavior by modifying the configuration.
"""

import asyncio
from streaming_voice_chatbot import StreamingVoiceChatbot, Config


class CustomConfig(Config):
    """Custom configuration example."""
    
    # Use a different model
    OLLAMA_MODEL = "llama2:latest"
    
    # Use different TTS backend
    TTS_BACKEND = "pyttsx3"
    
    # Customize system prompt
    SYSTEM_PROMPT = """
    You are Cora, a helpful and friendly AI assistant. You speak in a warm, conversational tone 
    and provide helpful, concise responses. Keep your answers brief but informative.
    """
    
    # More aggressive voice detection
    VAD_AGGRESSIVENESS = 3
    
    # Print sentences as they're generated
    PRINT_PARTIAL_SENTENCES = True


async def main():
    """Run the chatbot with custom configuration."""
    # Create custom config
    config = CustomConfig()
    
    # Initialize chatbot with custom config
    chatbot = StreamingVoiceChatbot(config=config)
    
    # Run the chatbot
    await chatbot.run()


if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nGoodbye!")
        pass
