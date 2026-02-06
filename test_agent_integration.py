#!/usr/bin/env python3
"""
Quick test to verify agent integration without running full chatbot
"""
import asyncio
from agent_tools.patient_tools import get_patient_data
from chatbot.memory.conversation_memory import Conversation
from chatbot.constants import SYSTEM_PROMPT

async def test_tool():
    """Test the patient tool directly"""
    print("=" * 60)
    print("TEST 1: Direct Tool Call")
    print("=" * 60)
    
    result = await asyncio.to_thread(get_patient_data, "101")
    print(result)
    print()

async def test_conversation_agent():
    """Test conversation with agent integration"""
    print("=" * 60)
    print("TEST 2: Conversation Agent Detection")
    print("=" * 60)
    
    convo = Conversation(SYSTEM_PROMPT)
    
    # Test needs_tool_use detection
    test_queries = [
        "Cora, tell me about patient 101",
        "Cora, what's the weather?",
        "Show me patient 205",
        "How are you doing?",
        "Get patient information for patient 101"
    ]
    
    for query in test_queries:
        needs_tool = convo.needs_tool_use(query)
        print(f"Query: {query}")
        print(f"  → Needs tool: {needs_tool}")
        print()

if __name__ == "__main__":
    print("Testing Agent Integration\n")
    asyncio.run(test_tool())
    asyncio.run(test_conversation_agent())
    print("\n✅ All tests completed!")
