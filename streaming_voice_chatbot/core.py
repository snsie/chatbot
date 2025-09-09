#!/usr/bin/env python3
"""
Core streaming voice chatbot implementation.
"""

import asyncio
import sys
import string
from typing import Optional

from .config import default_config, Config
from .detection import UtteranceDetector, WhisperSTT
from .llm import Conversation, ollama_stream_chat, sentence_stream
from .tts import create_speaker, BaseSpeaker


class StreamingVoiceChatbot:
    """
    Main streaming voice chatbot class.
    
    A half‑duplex (listen -> transcribe -> stream LLM -> speak sentences) voice assistant
    with low latency sentence‑by‑sentence TTS while tokens stream from an Ollama model.
    """
    
    def __init__(self, config: Optional[Config] = None):
        """Initialize the chatbot with optional custom configuration."""
        self.config = config or default_config
        self.detector = None
        self.stt = None
        self.speaker = None
        self.conversation = None
        
    async def initialize(self):
        """Initialize all components asynchronously."""
        print("Booting streaming voice chatbot…")
        
        # Set Windows event loop policy if needed
        if sys.platform.startswith('win'):
            try:
                asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())  # type: ignore
            except Exception:
                pass
        
        self.detector = UtteranceDetector(self.config)
        self.stt = WhisperSTT(self.config)
        self.speaker = await create_speaker(self.config)
        self.conversation = Conversation(self.config)
        
    async def process_turn(self):
        """Process a single conversation turn."""
        print("🎤 Listening…", flush=True)
        audio = await asyncio.to_thread(self.detector.record_once, self.stt)
        print("🛑 Stopped listening.", flush=True)
        
        if audio is None or not len(audio):
            print("🛑 No audio captured.")
            return  # Nothing captured; loop again
            
        try:
            transcript = await asyncio.to_thread(self.stt.transcribe, audio)
        except Exception as e:
            print(f"[STT Error] {e}")
            return
            
        if not transcript.strip():
            return
            
        print("📝 Transcribing…", flush=True)
        print(f"You: {transcript}")
        
        # Check for wake word
        if not self._check_wake_word(transcript):
            print("No wake word detected; ignoring input.")
            return
            
        # Add user message to conversation
        self.conversation.add_user(transcript)
        
        print("🤖 Assistant (streaming)…", flush=True)

        # Streaming generation
        assistant_buffer = []
        sentences_queue: asyncio.Queue = asyncio.Queue()
        speak_consumer_task = asyncio.create_task(
            self._speak_consumer(sentences_queue, self.speaker)
        )
        
        async for sentence in sentence_stream(
            ollama_stream_chat(
                self.conversation.history(), 
                self.config.OLLAMA_MODEL, 
                self.config.MAX_TOKENS
            ),
            self.config
        ):
            assistant_buffer.append(sentence)
            await sentences_queue.put(sentence)
            if self.config.PRINT_PARTIAL_SENTENCES:
                print(f"Assistant ↳ {sentence}")

        # Signal completion
        await sentences_queue.put(None)
        await speak_consumer_task

        full_assistant_text = ' '.join(assistant_buffer)
        self.conversation.add_assistant(full_assistant_text)
        await asyncio.sleep(self.config.TAIL_DELAY_SEC)
        
    def _check_wake_word(self, transcript: str) -> bool:
        """Check if transcript contains a wake word."""
        # Remove punctuation and create word list
        transcript_no_punct = transcript.translate(str.maketrans('', '', string.punctuation))
        words_list = transcript_no_punct.split()
        print(f"Words: {words_list}")
        
        for word in words_list:
            if word in self.config.SIMILAR_NAMES:
                print(f"Found similar name: {word}")
                return True
        return False
        
    async def _speak_consumer(self, q: 'asyncio.Queue[Optional[str]]', speaker: BaseSpeaker):
        """Consumer task for speaking sentences as they arrive."""
        while True:
            sentence = await q.get()
            if sentence is None:
                break
            try:
                await speaker.speak(sentence)
            except Exception as e:
                print(f"[TTS Error] {e}")
                
    async def run(self):
        """Run the main chatbot loop."""
        await self.initialize()
        
        try:
            while True:
                await self.process_turn()
        except KeyboardInterrupt:
            print("\nExiting…")
        finally:
            await self.cleanup()
            
    async def cleanup(self):
        """Clean up resources."""
        if self.speaker:
            await self.speaker.close()


# Standalone function for backward compatibility
async def main():
    """Main entry point for running the chatbot."""
    chatbot = StreamingVoiceChatbot()
    await chatbot.run()
