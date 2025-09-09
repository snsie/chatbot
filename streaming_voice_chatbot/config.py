#!/usr/bin/env python3
"""
Configuration settings for the Streaming Voice Chatbot.
"""

import math

class Config:
    """Configuration class for the Streaming Voice Chatbot."""
    
    # Audio Configuration
    SAMPLE_RATE = 16000
    FRAME_MS = 10  # ms per frame for capture + VAD
    VAD_AGGRESSIVENESS = 2  # 0-3 (higher = more aggressive speech detection)
    MIN_UTTERANCE_MS = 300  # minimum voiced audio required to accept an utterance
    TRAILING_SILENCE_MS = 200  # silence to mark end of utterance

    # Speech-to-Text Configuration
    WHISPER_MODEL = "small.en"
    WHISPER_COMPUTE = "cuda"  # 'auto' | 'cpu' | 'cuda'

    # LLM Configuration
    OLLAMA_MODEL = "gpt-oss:20b"
    # OLLAMA_MODEL="llama2:latest"

    # Wake words configuration
    SIMILAR_NAMES = [
        "Cora", "Kora", "Korra", "Quora", "Core", "Cori", "Corey", "Coral",
        "Corrie", "Cory", "Corin", "Corie", "Corry", "Kory", "Korey", "Kori",
        "Korrie", "Corah", "Corra", "Corca", "Korla", "Korrah",
        "Cour", "Cor", "Coor", "Koor", "Korr", "Corr", "Quora", "Quorra"
    ]

    # System prompt
    SYSTEM_PROMPT = """
You are a helpful voice assistant named Cora. Keep responses concise (1-2 sentences typically). Speak as if having a natural conversation.
"""

    # LLM Response Configuration
    MAX_TOKENS = 512

    # Text-to-Speech Configuration
    TTS_BACKEND = "edge-tts"  # 'pyttsx3' | 'edge-tts' | 'coqui'
    VOICE_NAME = "en-GB-SoniaNeural"  # Alternative: en-US-AriaNeural, en-US-GuyNeural, en-AU-NatashaNeural
    # VOICE_NAME='en-AU-NatashaNeural'
    # Runtime Configuration
    TAIL_DELAY_SEC = 0.15  # Delay after TTS before returning to mic (reduce capturing own voice)
    PRINT_PARTIAL_SENTENCES = True  # Print sentences as they are spoken

    # Sentence segmentation
    SENTENCE_END_CHARS = "\.\!\?…"  # regex set

    # Derived constants (computed properties)
    @property
    def FRAME_SAMPLES(self):
        return int(self.SAMPLE_RATE * self.FRAME_MS / 1000)  # samples per frame (e.g. 480)
    
    @property
    def MIN_VOICED_FRAMES(self):
        return math.ceil(self.MIN_UTTERANCE_MS / self.FRAME_MS)
    
    @property
    def TRAILING_SILENCE_FRAMES(self):
        return math.ceil(self.TRAILING_SILENCE_MS / self.FRAME_MS)


# Default configuration instance
default_config = Config()
