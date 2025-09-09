#!/usr/bin/env python3
"""
Streaming Voice Chatbot Module
=============================

A half‑duplex (listen -> transcribe -> stream LLM -> speak sentences) voice assistant
with low latency sentence‑by‑sentence TTS while tokens stream from an Ollama model.

Features
--------
1. Microphone capture @ 16 kHz mono (30 ms frames) using sounddevice RawInputStream.
2. Voice Activity Detection (webrtcvad) to segment utterances.
3. Speech‑to‑Text via faster-whisper (GPU auto, fallback CPU) on captured utterance.
4. Streaming LLM responses token-by-token from Ollama (llama3.1:8b-instruct by default).
5. Sentence segmentation of streaming tokens; each completed sentence immediately sent to TTS.
6. Two TTS backends:
   - pyttsx3 (offline, default)
   - edge-tts (optional, higher quality, requires internet + ffmpeg)
7. Clean shutdown on Ctrl+C.

Quick Start
-----------
```python
from streaming_voice_chatbot import StreamingVoiceChatbot
import asyncio

async def main():
    chatbot = StreamingVoiceChatbot()
    await chatbot.run()

if __name__ == '__main__':
    asyncio.run(main())
```
"""

from .core import StreamingVoiceChatbot
from .config import Config
from .detection import UtteranceDetector, WhisperSTT
from .llm import Conversation, ollama_stream_chat, sentence_stream
from .tts import BaseSpeaker, Pyttsx3Speaker, EdgeTTSSpeaker, CoquiTTSSpeaker, create_speaker

__version__ = "1.0.0"
__all__ = [
    'StreamingVoiceChatbot',
    'Config',
    'UtteranceDetector',
    'WhisperSTT',
    'Conversation',
    'ollama_stream_chat',
    'sentence_stream',
    'BaseSpeaker',
    'Pyttsx3Speaker',
    'EdgeTTSSpeaker',
    'CoquiTTSSpeaker',
    'create_speaker'
]
