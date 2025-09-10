#!/usr/bin/env python3
"""
LangChain Text Chatbot Test
===========================

Text-only version to test the LangChain chatbot functionality before adding voice.
"""

import asyncio
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, AIMessage, SystemMessage

# Configuration
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "gpt-oss:20b"
MAX_TOKENS = 512
TEMPERATURE = 0.7
SYSTEM_PROMPT = """You are a helpful AI assistant. Keep your responses conversational and concise, 
typically 1-2 sentences. Speak naturally as if having a friendly conversation."""

class SimpleLangChainChatbot:
    """Simple LangChain chatbot for testing."""
    
    def __init__(self):
        self.llm = ChatOpenAI(
            model=OLLAMA_MODEL,
            base_url=OLLAMA_BASE_URL + "/v1",
            api_key="ollama",
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
            streaming=True
        )
        
        self.conversation_history = [
            SystemMessage(content=SYSTEM_PROMPT)
        ]
        
        print(f"✅ Chatbot initialized with model: {OLLAMA_MODEL}")
    
    async def get_response_stream(self, user_input: str):
        """Get streaming response from the chatbot."""
        try:
            self.conversation_history.append(HumanMessage(content=user_input))
            
            # Keep history manageable
            if len(self.conversation_history) > 21:
                self.conversation_history = [self.conversation_history[0]] + self.conversation_history[-20:]
            
            full_response = ""
            print("🤖 Assistant: ", end="", flush=True)
            
            async for chunk in self.llm.astream(self.conversation_history):
                if hasattr(chunk, 'content') and chunk.content:
                    print(chunk.content, end="", flush=True)
                    full_response += chunk.content
            
            print()  # New line
            
            if full_response:
                self.conversation_history.append(AIMessage(content=full_response))
            
            return full_response
            
        except Exception as e:
            print(f"❌ Error: {e}")
            return "Sorry, I encountered an error."

async def main():
    """Main test loop."""
    chatbot = SimpleLangChainChatbot()
    print("💬 LangChain Text Chatbot Test")
    print("Type 'quit' to exit")
    print("-" * 40)
    
    while True:
        try:
            user_input = input("\nYou: ").strip()
            if user_input.lower() in ['quit', 'exit', 'bye']:
                break
            
            if user_input:
                await chatbot.get_response_stream(user_input)
                
        except KeyboardInterrupt:
            break
    
    print("\n👋 Goodbye!")

if __name__ == "__main__":
    asyncio.run(main())
