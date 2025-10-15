from .speakers import _speak_consumer, BaseSpeaker, create_speaker
from .voice_id.voice_identifier import _VoiceIdentifier
from .listeners.whisper_stt import WhisperSTT
from .memory.conversation_memory import Conversation
from .utils.utterance_detector import UtteranceDetector
from .streaming.sentence_stream import sentence_stream
from .streaming.ollama_stream_chat import ollama_stream_chat
from .constants import (
    OLLAMA_MODEL, MAX_TOKENS, PRINT_PARTIAL_SENTENCES,
    ENABLE_SPEAKER_GATE, SIMILAR_NAMES, ENROLL_PATH, SIM_THRESHOLD  
)
import asyncio
import string
import re
from pathlib import Path
import numpy as np
import soundfile as sf
import subprocess
from pymongo import MongoClient
from pydantic import ValidationError
from .voice_id.person import Person
from dotenv import load_dotenv
import os
from .voice_id.get_voice_identifier import get_voice_identifier
from typing import List
from .utils import validate_audio_capture, transcribing_audio


class MainFunctionality:
  def __init__(self):
    self.detector = UtteranceDetector
    self.stt = WhisperSTT
    self.convo = Conversation
    self.speaker = BaseSpeaker
    self.voice_identifier = _VoiceIdentifier
    self.people = List[Person]

  async def main_loop(self):
    print("🎤 Listening…", flush=True)

    self.audio = await asyncio.to_thread(self.detector.record_once)  # Add back the asyncio.to_thread()

    if not validate_audio_capture(self.audio):
      return
    
    if not transcribing_audio(self.audio):
      return

  async def validate_audio_capture(self):
    pass

  async def transcribing_audio(self):
    pass