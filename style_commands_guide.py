#!/usr/bin/env python3
"""
Style Command Test - Test different response styles for the voice chatbot
"""

def test_style_commands():
    """Test script to show available style commands"""
    
    print("🎭 Voice Assistant Style Commands")
    print("=" * 50)
    print("\nAvailable style commands you can say:")
    print()
    
    commands = [
        ("speak more formally", "→ Professional language with detailed explanations"),
        ("be more casual", "→ Informal, conversational tone"),
        ("be more technical", "→ Technical details and terminology"),
        ("explain like I'm 5", "→ Simple language for easy understanding"),
        ("be more concise", "→ Brief, to-the-point responses"),
        ("be more detailed", "→ Comprehensive explanations with examples"),
        ("be more creative", "→ Creative language with metaphors"),
        ("be more professional", "→ Business-appropriate language"),
        ("reset style", "→ Return to default conversational tone")
    ]
    
    for command, description in commands:
        print(f"  '{command}' {description}")
    
    print("\n📝 Example conversation flow:")
    print("  You: 'What is machine learning?'")
    print("  Assistant: [Gives normal explanation]")
    print()
    print("  You: 'explain like I'm 5'") 
    print("  Assistant: 'I've updated my response style.'")
    print()
    print("  You: 'What is machine learning?'")
    print("  Assistant: [Gives child-friendly explanation]")
    print()
    print("  You: 'be more technical'")
    print("  Assistant: 'I've updated my response style.'") 
    print()
    print("  You: 'What is machine learning?'")
    print("  Assistant: [Gives technical explanation with algorithms, etc.]")
    
    print("\n🚀 Enhanced System Prompt Features:")
    print("  • Dynamic style switching during conversation")
    print("  • Immediate confirmation of style changes")
    print("  • Persistent style until changed or reset")
    print("  • Voice-optimized responses (no formatting lists)")
    print("  • Special commands are processed locally for speed")
    
    print("\n💡 Tips:")
    print("  • Style changes persist throughout the conversation")
    print("  • You can chain multiple questions in the same style")
    print("  • Say 'reset style' to return to normal")
    print("  • The AI will adapt its vocabulary and explanation depth")

if __name__ == "__main__":
    test_style_commands()
