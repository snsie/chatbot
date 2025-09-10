#!/usr/bin/env python3
"""
LangChain Voice Chatbot - Usage Example
======================================

This script demonstrates how to use the LangChain voice chatbot.
Run this to see a demo of the text-based version.
"""

import asyncio
import sys
from langchain_voice_chatbot import LangChainChatbot

async def demo_conversation():
    """Demonstrate the chatbot with a few example exchanges."""
    print("🎯 LangChain Voice Chatbot Demo")
    print("=" * 50)
    
    # Initialize chatbot
    chatbot = LangChainChatbot()
    
    # Example conversations
    test_inputs = [
        "Hello! What's your name?",
        "Can you explain what LangChain is in simple terms?",
        "What's the weather like today?",
        "Tell me a short programming joke."
    ]
    
    for i, user_input in enumerate(test_inputs, 1):
        print(f"\n--- Conversation {i} ---")
        print(f"You: {user_input}")
        print("🤖 Assistant: ", end="", flush=True)
        
        try:
            async for chunk in chatbot.get_response_stream(user_input):
                print(chunk, end="", flush=True)
            print()  # New line
            
        except KeyboardInterrupt:
            print("\n\n👋 Demo interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
    
    print("\n✅ Demo completed!")
    print("\nTo use the full voice chatbot, run:")
    print("  ./start_voice_chatbot.sh")
    print("or:")
    print("  source .venv/bin/activate && python langchain_voice_chatbot.py")

if __name__ == "__main__":
    try:
        asyncio.run(demo_conversation())
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
