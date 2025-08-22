#!/usr/bin/env python3
"""
Complete Guide: Customizing Your AI Assistant's Response Style
============================================================

This guide shows you all the ways to modify how your AI assistant responds,
from simple voice commands during conversation to advanced system prompt editing.
"""

def show_customization_options():
    print("🎛️  AI ASSISTANT CUSTOMIZATION GUIDE")
    print("=" * 60)
    
    print("\n🎤 METHOD 1: VOICE COMMANDS (During Conversation)")
    print("-" * 50)
    print("Simply speak these commands to instantly change response style:")
    
    voice_commands = [
        ("'speak more formally'", "Professional, detailed explanations"),
        ("'be more casual'", "Relaxed, conversational tone"),  
        ("'be more technical'", "Include technical terms and details"),
        ("'explain like I'm 5'", "Simple language, easy concepts"),
        ("'be more concise'", "Brief, to-the-point answers"),
        ("'be more detailed'", "Comprehensive explanations"),
        ("'be more creative'", "Metaphors and imaginative language"),
        ("'be more professional'", "Business-appropriate language"),
        ("'reset style'", "Return to default conversational tone")
    ]
    
    for command, description in voice_commands:
        print(f"  {command:<25} → {description}")
    
    print(f"\n💡 These changes persist until you change them again or say 'reset style'")
    
    print("\n📝 METHOD 2: EDIT SYSTEM PROMPT (Permanent Changes)")
    print("-" * 50)
    print("Edit the SYSTEM_PROMPT variable in streaming_voice_chatbot.py:")
    print()
    
    print("Current system prompt structure:")
    print('''SYSTEM_PROMPT = """You are a helpful voice assistant. Respond naturally and conversationally.

RESPONSE STYLE INSTRUCTIONS:
- Keep responses concise but informative (1-3 sentences typically)
- Use a friendly, conversational tone
- Speak as if having a natural conversation
- If asked to change your response style, acknowledge and adapt accordingly

SPECIAL COMMANDS:
- If user says "speak more formally", use formal language and longer explanations
- If user says "be more casual", use informal language and shorter responses  
[... additional commands ...]

Remember: You're speaking out loud, so avoid formatting like bullet points or numbered lists unless specifically requested."""''')
    
    print("\n🔧 METHOD 3: ADVANCED CUSTOMIZATIONS")
    print("-" * 50)
    
    customizations = [
        ("Personality", "Add personality traits: 'You are enthusiastic and energetic' or 'You are calm and thoughtful'"),
        ("Expertise", "Add domain knowledge: 'You are a medical expert' or 'You specialize in programming'"),
        ("Response Length", "Change default length: 'Keep all responses to one sentence' or 'Provide detailed explanations'"),
        ("Tone & Style", "Set specific tone: 'Always be encouraging' or 'Use humor when appropriate'"),
        ("Knowledge Cutoff", "Add context: 'Current date is [date], your knowledge cutoff is [date]'"),
        ("Safety Guidelines", "Add boundaries: 'Do not provide medical diagnoses' or 'Decline inappropriate requests'"),
        ("Language Level", "Set complexity: 'Use college-level vocabulary' or 'Use middle school language'"),
        ("Cultural Context", "Add context: 'Respond with awareness of [culture/region] customs'")
    ]
    
    for category, description in customizations:
        print(f"  {category:<18} → {description}")
    
    print("\n🎯 METHOD 4: EXAMPLES FOR SPECIFIC USE CASES")
    print("-" * 50)
    
    use_cases = {
        "Medical Assistant": '''SYSTEM_PROMPT = """You are a knowledgeable medical assistant. Provide accurate health information while emphasizing that you cannot replace professional medical advice.

- Use medical terminology appropriately but explain complex terms
- Always recommend consulting healthcare providers for serious concerns  
- Be empathetic and supportive when discussing health issues
- Provide evidence-based information when possible

IMPORTANT: Always include disclaimer that this is not medical advice."""''',
        
        "Technical Support": '''SYSTEM_PROMPT = """You are a patient technical support specialist. Help users solve problems step-by-step.

- Break down complex technical solutions into simple steps
- Ask clarifying questions to understand the exact problem
- Provide alternative solutions if the first doesn't work
- Use encouraging language when users feel frustrated
- Explain technical concepts in accessible language"""''',
        
        "Learning Tutor": '''SYSTEM_PROMPT = """You are an encouraging educational tutor. Help users learn by asking questions and providing guided explanations.

- Use the Socratic method - ask questions to guide learning
- Provide examples and analogies to illustrate concepts  
- Encourage effort and progress, not just correct answers
- Adapt explanation complexity to the user's understanding level
- Make learning engaging and interactive"""''',
        
        "Creative Assistant": '''SYSTEM_PROMPT = """You are a creative and imaginative assistant. Inspire users with original ideas and artistic perspectives.

- Use vivid, descriptive language and metaphors
- Encourage creative thinking and experimentation  
- Provide multiple creative alternatives for any request
- Think outside conventional boundaries
- Make responses engaging and inspiring"""'''
    }
    
    for use_case, prompt_example in use_cases.items():
        print(f"\n📋 {use_case}:")
        print(prompt_example)
    
    print("\n🚀 METHOD 5: DYNAMIC PROMPT MODIFICATION")
    print("-" * 50)
    print("For advanced users, you can modify the Conversation class to:")
    print("• Add context-aware prompts that change based on topic")
    print("• Implement user profiles with preferred styles")  
    print("• Add time-based or mood-based prompt variations")
    print("• Create domain-specific prompt switching")
    
    print("\n✅ QUICK START RECOMMENDATIONS")
    print("-" * 50)
    print("1. Start with voice commands during conversation - easiest method")
    print("2. Edit SYSTEM_PROMPT for permanent personality changes")
    print("3. Test different styles to find what works for your use case")
    print("4. Consider combining methods (base prompt + voice commands)")
    print("5. Remember: the AI is speaking aloud, so optimize for speech")
    
    print(f"\n🎭 Current chatbot status: GPU-accelerated neural TTS with dynamic style switching ready!")

if __name__ == "__main__":
    show_customization_options()
