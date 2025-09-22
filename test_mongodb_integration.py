#!/usr/bin/env python3
"""
Test MongoDB Integration for Voice Chatbot
==========================================

This script tests the MongoDB conversation storage functionality.
"""

import sys
import os
from datetime import datetime

# Add the current directory to sys.path to import from the main script
sys.path.append(os.path.dirname(__file__))

# Import the MongoDB store class
from streaming_voice_chatbot_connect_mongo import MongoDBConversationStore, Conversation

def test_mongodb_connection():
    """Test basic MongoDB connection."""
    print("Testing MongoDB connection...")
    try:
        store = MongoDBConversationStore()
        if store.client is None:
            print("❌ MongoDB connection failed - is MongoDB running?")
            return False
        print("✅ MongoDB connection successful")
        store.close()
        return True
    except Exception as e:
        print(f"❌ MongoDB connection error: {e}")
        return False

def test_conversation_storage():
    """Test conversation storage and retrieval."""
    print("\nTesting conversation storage...")
    
    try:
        # Initialize MongoDB store
        store = MongoDBConversationStore()
        if store.client is None:
            print("❌ Cannot test storage - MongoDB not connected")
            return False
        
        # Create a test conversation
        system_prompt = "You are a test assistant for testing MongoDB integration."
        conversation = Conversation(system_prompt, store)
        
        print(f"✅ Started new conversation: {conversation.conversation_id}")
        
        # Add some test messages
        conversation.add_user("Hello, this is a test message.")
        conversation.add_assistant("Hello! I received your test message and stored it in MongoDB.")
        conversation.add_user("Can you confirm the conversation is being saved?")
        conversation.add_assistant("Yes, all our messages are being saved to MongoDB!")
        
        print("✅ Added test messages to conversation")
        
        # Retrieve conversation history
        history = store.get_conversation_history()
        print(f"✅ Retrieved conversation history: {len(history)} messages")
        
        # Print the stored messages
        print("\nStored conversation:")
        for msg in history:
            role = msg.get('role', 'unknown')
            content = msg.get('content', '')
            timestamp = msg.get('timestamp', 'no timestamp')
            print(f"  {role}: {content[:50]}{'...' if len(content) > 50 else ''}")
        
        # Get all conversations
        all_conversations = store.get_all_conversations()
        print(f"✅ Total conversations in database: {len(all_conversations)}")
        
        store.close()
        return True
        
    except Exception as e:
        print(f"❌ Conversation storage test failed: {e}")
        return False

def test_multiple_conversations():
    """Test handling of multiple conversations."""
    print("\nTesting multiple conversations...")
    
    try:
        store = MongoDBConversationStore()
        if store.client is None:
            print("❌ Cannot test multiple conversations - MongoDB not connected")
            return False
        
        # Create first conversation
        conv1 = Conversation("You are assistant 1.", store)
        conv1.add_user("Message for conversation 1")
        conv1.add_assistant("Response from assistant 1")
        
        # Create second conversation
        store2 = MongoDBConversationStore()
        conv2 = Conversation("You are assistant 2.", store2)
        conv2.add_user("Message for conversation 2")
        conv2.add_assistant("Response from assistant 2")
        
        # Verify both conversations exist
        all_conversations = store.get_all_conversations()
        print(f"✅ Total conversations after creating 2: {len(all_conversations)}")
        
        store.close()
        store2.close()
        return True
        
    except Exception as e:
        print(f"❌ Multiple conversations test failed: {e}")
        return False

if __name__ == "__main__":
    print("MongoDB Integration Test for Voice Chatbot")
    print("=" * 50)
    
    # Run tests
    tests_passed = 0
    total_tests = 3
    
    if test_mongodb_connection():
        tests_passed += 1
    
    if test_conversation_storage():
        tests_passed += 1
    
    if test_multiple_conversations():
        tests_passed += 1
    
    print(f"\nTest Results: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        print("🎉 All tests passed! MongoDB integration is working correctly.")
    else:
        print("⚠️  Some tests failed. Check MongoDB setup and connection.")
        
    print("\nTo view stored conversations in MongoDB:")
    print("  mongo")
    print("  use chatbot_conversations")
    print("  db.conversations.find().pretty()")