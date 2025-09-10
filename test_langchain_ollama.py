#!/usr/bin/env python3
"""
Simple LangChain Ollama Test
===========================

Test script to verify LangChain integration with Ollama's gpt-oss model.
"""

import asyncio
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

# Configuration
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "gpt-oss:20b"
MAX_TOKENS = 512
TEMPERATURE = 0.7

async def test_langchain_ollama():
    """Test LangChain with Ollama."""
    print(f"Testing LangChain with Ollama model: {OLLAMA_MODEL}")
    
    try:
        # Initialize ChatOpenAI with Ollama configuration
        llm = ChatOpenAI(
            model=OLLAMA_MODEL,
            base_url=OLLAMA_BASE_URL + "/v1",
            api_key="ollama",  # Dummy key for Ollama
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
        )
        
        # Test messages
        messages = [
            SystemMessage(content="You are a helpful AI assistant. Keep responses concise."),
            HumanMessage(content="Hello! How are you today?")
        ]
        
        print("Sending test message...")
        response = llm.invoke(messages)
        
        print(f"Response: {response.content}")
        print("✅ LangChain + Ollama integration working!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False
    
    return True

if __name__ == "__main__":
    asyncio.run(test_langchain_ollama())
