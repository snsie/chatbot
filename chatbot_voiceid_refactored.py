#added a new import, added 4 variables, and modified the process_turn function


#!/usr/bin/env python3
"""
Streaming Voice Chatbot
=======================

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

Configuration (edit constants below) controls sample rate, model names, thresholds, etc.

Dependencies (pip install ...)
------------------------------
Core:
  faster-whisper
  sounddevice
  webrtcvad
  numpy
  pyttsx3
  ollama
  tiktoken  (optional, not strictly needed but listed per spec)

Optional for enhanced TTS:
  edge-tts
  pydub
  simpleaudio

Optional for neural TTS (Coqui):
  TTS
  torch
  torchaudio

System packages (Ubuntu examples):
  sudo apt-get update && sudo apt-get install -y \
       portaudio19-dev ffmpeg espeak-ng

Quick Start
-----------
1. Start Ollama server (separate terminal):
     ollama serve
2. Pull desired model (first time):
     ollama pull llama3.1:8b-instruct
3. Run this script:
     python streaming_voice_chatbot.py

Runtime Flow
------------
Loop:
  🎤 Listening… -> capture utterance
  📝 Transcribing… (print transcript as You: <text>)
  🤖 Assistant (streaming)… -> sentences spoken as generated
  (short tail delay) -> back to listening

Press Ctrl+C to exit cleanly.
"""

# =============================
# Imports
# =============================
import asyncio

import sys

# Voice ID
from chatbot.voice_id.get_voice_identifier import get_voice_identifier

from chatbot.constants import (SAMPLE_RATE, FRAME_MS, VAD_AGGRESSIVENESS, MIN_UTTERANCE_MS, TRAILING_SILENCE_MS,
                              WHISPER_MODEL, WHISPER_COMPUTE,
                              OLLAMA_MODEL, MAX_TOKENS,
                              TTS_BACKEND, VOICE_NAME,
                              PRINT_PARTIAL_SENTENCES, SYSTEM_PROMPT, ENROLL_PATH, ENABLE_SPEAKER_GATE, 
                              SIMILAR_NAMES, SIM_THRESHOLD
)


from chatbot.speakers import EdgeTTSSpeaker, create_speaker

from chatbot.utils import UtteranceDetector, setup_windows_event_loop

# Voice Separation
import asyncio, numpy as np

#from voice_id import identify_from_array  # or your class instance method
#SIM_THRESHOLD = 0.65
USE_SEPARATION = True  # flip on/off easily


# Mongo DB for logging
from scipy.signal import resample_poly
import re
from chatbot.speakers import BaseSpeaker
from chatbot.listeners import WhisperSTT

# from chatbot.main_loop import main_loop
from chatbot.main_functionality import MainFunctionality
from chatbot.memory import Conversation


voice_identifier = get_voice_identifier(ENROLL_PATH) if ENABLE_SPEAKER_GATE else None


async def main():
    setup_windows_event_loop()
    print("Booting streaming voice chatbot…")

    bot = MainFunctionality()

    try:
        while True:
            await bot.main_loop()
    except KeyboardInterrupt:
        print("\nExiting…")
    finally:
        if hasattr(bot.speaker, "close"):
            await bot.speaker.close()

if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass