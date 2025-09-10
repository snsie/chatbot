#!/usr/bin/env python3
"""
Quick LangChain + Ollama Integration Test
========================================

Test the basic LangChain streaming functionality with gpt-oss model.
"""

import asyncio
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

# Configuration
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "gpt-oss:20b"

async def test_streaming():
    """Test streaming response from LangChain + Ollama."""
    print(f"🧪 Testing LangChain streaming with {OLLAMA_MODEL}")
    
    llm = ChatOpenAI(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL + "/v1",
        api_key="ollama",
        max_tokens=100,
        temperature=0.7,
        streaming=True
    )
    
    messages = [
        SystemMessage(content="You are a helpful assistant. Keep responses brief."),
        HumanMessage(content="Tell me a short joke about programming.")
    ]
    
    print("🤖 Assistant: ", end="", flush=True)
    
    try:
        async for chunk in llm.astream(messages):
            if hasattr(chunk, 'content') and chunk.content:
                print(chunk.content, end="", flush=True)
        
        print("\n✅ Streaming test successful!")
        return True
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return False

if __name__ == "__main__":
    asyncio.run(test_streaming())
