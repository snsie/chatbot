#!/usr/bin/env python3
"""
LangChain Streaming Voice Chatbot
================================

A streaming voice chatbot using LangChain with ChatOpenAI interface configured
for Ollama's chatgpt-oss model. Features voice input via Whisper STT and 
voice output via edge-TTS.

Features
--------
1. LangChain integration with ChatOpenAI interface for Ollama
2. Streaming responses with real-time audio output
3. Voice Activity Detection (webrtcvad) for utterance segmentation
4. Speech-to-Text via faster-whisper
5. Text-to-Speech via edge-tts
6. Simple conversation chain with memory
7. Clean shutdown on Ctrl+C

Dependencies
-----------
pip install langchain langchain-openai

Quick Start
-----------
1. Start Ollama server:
   ollama serve
2. Pull the chatgpt-oss model:
   ollama pull chatgpt-oss
3. Run this script:
   python langchain_voice_chatbot.py
"""

# =============================
# Configuration
# =============================

# Audio Configuration
SAMPLE_RATE = 16000
FRAME_MS = 10  # ms per frame for capture + VAD
VAD_AGGRESSIVENESS = 2  # 0-3 (higher = more aggressive speech detection)
MIN_UTTERANCE_MS = 300  # minimum voiced audio required
TRAILING_SILENCE_MS = 200  # silence to mark end of utterance

# STT Configuration
WHISPER_MODEL = "small.en"
WHISPER_COMPUTE = "auto"  # 'auto' | 'cpu' | 'cuda'

# LLM Configuration
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "gpt-oss:20b"  # The model name in Ollama
MAX_TOKENS = 512
TEMPERATURE = 0.7

# TTS Configuration
TTS_VOICE = "en-US-AriaNeural"  # edge-tts voice
TAIL_DELAY_SEC = 0.5  # Delay after TTS before returning to mic

# System Configuration
SYSTEM_PROMPT = """You are a helpful AI assistant. Keep your responses conversational and concise, 
typically 1-2 sentences. Speak naturally as if having a friendly conversation."""

# =============================
# Imports
# =============================
import asyncio
import sys
import threading
import queue
import time
import math
import re
import io
import json
import tempfile
import os
from typing import AsyncGenerator, List, Optional

import numpy as np
import sounddevice as sd
import webrtcvad

# STT
from faster_whisper import WhisperModel

# LangChain imports
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationChain
from langchain.schema import BaseMessage

# TTS
import edge_tts
import simpleaudio as sa

# =============================
# Utility Functions
# =============================
FRAME_SAMPLES = int(SAMPLE_RATE * FRAME_MS / 1000)
MIN_VOICED_FRAMES = math.ceil(MIN_UTTERANCE_MS / FRAME_MS)
TRAILING_SILENCE_FRAMES = math.ceil(TRAILING_SILENCE_MS / FRAME_MS)

SENTENCE_END_CHARS = r"\.\!\?…"
SENTENCE_END_REGEX = re.compile(rf"(.+?[{SENTENCE_END_CHARS}](?:[\"'\)\]]*)\s*)", re.DOTALL)

def extract_sentences(text: str) -> List[str]:
    """Extract complete sentences from streaming text."""
    sentences = []
    matches = SENTENCE_END_REGEX.findall(text)
    for match in matches:
        sentence = match.strip()
        if sentence:
            sentences.append(sentence)
    return sentences

# =============================
# Voice Activity Detection
# =============================
class UtteranceDetector:
    """Segments microphone audio into utterances using WebRTC VAD."""
    
    def __init__(self):
        self.vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)
        self.reset()
    
    def reset(self):
        """Reset internal state for new utterance detection."""
        self.frames = []
        self.voiced_frames = 0
        self.silence_frames = 0
        self.in_utterance = False
    
    def process_frame(self, frame_bytes: bytes) -> Optional[bytes]:
        """Process audio frame and return complete utterance if detected."""
        is_speech = self.vad.is_speech(frame_bytes, SAMPLE_RATE)
        self.frames.append(frame_bytes)
        
        if is_speech:
            self.voiced_frames += 1
            self.silence_frames = 0
            if not self.in_utterance and self.voiced_frames >= MIN_VOICED_FRAMES:
                self.in_utterance = True
                print("🎤 Listening...")
        else:
            if self.in_utterance:
                self.silence_frames += 1
                if self.silence_frames >= TRAILING_SILENCE_FRAMES:
                    # End of utterance
                    utterance = b''.join(self.frames)
                    self.reset()
                    return utterance
        
        return None

# =============================
# Speech-to-Text
# =============================
class WhisperSTT:
    """Speech-to-Text using faster-whisper."""
    
    def __init__(self):
        print(f"Loading Whisper model: {WHISPER_MODEL} on {WHISPER_COMPUTE}")
        self.model = WhisperModel(WHISPER_MODEL, device=WHISPER_COMPUTE)
    
    def transcribe(self, audio_data: bytes) -> str:
        """Transcribe audio bytes to text."""
        # Convert bytes to numpy array
        audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
        
        # Transcribe
        segments, _ = self.model.transcribe(audio_np, language="en")
        
        # Combine segments
        text = " ".join(segment.text.strip() for segment in segments)
        return text.strip()

# =============================
# LangChain LLM Interface
# =============================
class LangChainChatbot:
    """LangChain-based chatbot using ChatOpenAI interface for Ollama."""
    
    def __init__(self):
        # Initialize ChatOpenAI with Ollama configuration
        self.llm = ChatOpenAI(
            model=OLLAMA_MODEL,
            base_url=OLLAMA_BASE_URL + "/v1",  # Ollama OpenAI-compatible endpoint
            api_key="ollama",  # Dummy key for Ollama
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
            streaming=True
        )
        
        # Simple conversation history
        self.conversation_history = [
            SystemMessage(content=SYSTEM_PROMPT)
        ]
        
        print(f"✅ LangChain chatbot initialized with model: {OLLAMA_MODEL}")
    
    async def get_response_stream(self, user_input: str) -> AsyncGenerator[str, None]:
        """Get streaming response from the chatbot."""
        try:
            # Add user message to history
            self.conversation_history.append(HumanMessage(content=user_input))
            
            # Keep conversation history manageable (last 10 messages + system)
            if len(self.conversation_history) > 21:  # 10 pairs + system message
                self.conversation_history = [self.conversation_history[0]] + self.conversation_history[-20:]
            
            # Get streaming response
            full_response = ""
            async for chunk in self.llm.astream(self.conversation_history):
                if hasattr(chunk, 'content') and chunk.content:
                    full_response += chunk.content
                    yield chunk.content
            
            # Add assistant response to history
            if full_response:
                self.conversation_history.append(AIMessage(content=full_response))
            
        except Exception as e:
            print(f"❌ Error getting LLM response: {e}")
            yield "I'm sorry, I encountered an error processing your request."

# =============================
# Text-to-Speech
# =============================
class EdgeTTS:
    """Text-to-Speech using edge-tts."""
    
    def __init__(self):
        self.voice = TTS_VOICE
        print(f"✅ Edge-TTS initialized with voice: {self.voice}")
    
    async def speak_text(self, text: str):
        """Convert text to speech and play it."""
        if not text.strip():
            return
        
        try:
            # Generate speech
            communicate = edge_tts.Communicate(text, self.voice)
            
            # Save to temporary file (edge-tts will determine format)
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp_file:
                temp_path = tmp_file.name
            
            # Save the audio
            await communicate.save(temp_path)
            
            # Use pydub to handle the audio format conversion and playback
            try:
                from pydub import AudioSegment
                from pydub.playback import play
                
                # Load the audio file (pydub can handle various formats)
                audio = AudioSegment.from_file(temp_path)
                
                # Play the audio
                play(audio)
                
            except ImportError:
                # Fallback: convert to WAV and use simpleaudio
                from pydub import AudioSegment
                
                # Load and convert to WAV
                audio = AudioSegment.from_file(temp_path)
                wav_path = temp_path.replace('.mp3', '.wav')
                audio.export(wav_path, format="wav")
                
                # Play with simpleaudio
                wave_obj = sa.WaveObject.from_wave_file(wav_path)
                play_obj = wave_obj.play()
                play_obj.wait_done()
                
                # Clean up WAV file
                os.unlink(wav_path)
            
            # Clean up original file
            os.unlink(temp_path)
            
        except Exception as e:
            print(f"❌ TTS Error: {e}")
            # Fallback to espeak if available
            try:
                import subprocess
                result = subprocess.run(['espeak', text], check=False, capture_output=True)
                if result.returncode != 0:
                    print("❌ Espeak fallback failed")
            except FileNotFoundError:
                print("❌ No TTS fallback available")

# =============================
# Main Voice Chatbot
# =============================
class LangChainVoiceChatbot:
    """Main voice chatbot class coordinating all components."""
    
    def __init__(self):
        self.detector = UtteranceDetector()
        self.stt = WhisperSTT()
        self.chatbot = LangChainChatbot()
        self.tts = EdgeTTS()
        
        self.audio_queue = queue.Queue()
        self.running = True
        
        print("🚀 LangChain Voice Chatbot initialized!")
        print("Press Ctrl+C to exit")
    
    def audio_callback(self, indata, frames, time, status):
        """Callback for audio input stream."""
        if status:
            print(f"Audio status: {status}")
        
        # Convert buffer to numpy array
        audio_array = np.frombuffer(indata, dtype=np.int16)
        
        # Convert to bytes and queue
        self.audio_queue.put(audio_array.tobytes())

    async def process_audio(self):
        """Process audio frames for utterance detection."""
        while self.running:
            try:
                # Get audio frame with timeout
                frame_bytes = self.audio_queue.get(timeout=0.1)
                
                # Process frame for utterance detection
                utterance = self.detector.process_frame(frame_bytes)
                
                if utterance:
                    await self.handle_utterance(utterance)
                    
            except queue.Empty:
                continue
            except Exception as e:
                print(f"❌ Error processing audio: {e}")
    
    async def handle_utterance(self, utterance: bytes):
        """Handle a complete utterance: transcribe, get response, speak."""
        try:
            # Transcribe
            print("📝 Transcribing...")
            transcript = self.stt.transcribe(utterance)
            
            if not transcript.strip():
                print("❌ No transcript detected")
                return
            
            print(f"You: {transcript}")
            
            # Get chatbot response with streaming
            print("🤖 Assistant: ", end="", flush=True)
            
            # Collect response in chunks and speak complete sentences
            response_buffer = ""
            
            async for chunk in self.chatbot.get_response_stream(transcript):
                if chunk:
                    print(chunk, end="", flush=True)
                    response_buffer += chunk
                    
                    # Check for complete sentences to speak immediately
                    sentences = extract_sentences(response_buffer)
                    for sentence in sentences:
                        # Speak the sentence in background while continuing to stream
                        asyncio.create_task(self.tts.speak_text(sentence))
                        # Remove spoken sentence from buffer
                        response_buffer = response_buffer.replace(sentence, "", 1).strip()
            
            print()  # New line after response
            
            # Speak any remaining text that didn't form a complete sentence
            if response_buffer.strip():
                await self.tts.speak_text(response_buffer)
            
            # Small delay before returning to listening
            await asyncio.sleep(TAIL_DELAY_SEC)
            
        except Exception as e:
            print(f"❌ Error handling utterance: {e}")
    
    async def run(self):
        """Main run loop."""
        # Start audio stream
        stream = sd.RawInputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype=np.int16,
            blocksize=FRAME_SAMPLES,
            callback=self.audio_callback
        )
        
        try:
            with stream:
                print("🎤 Voice chatbot is listening...")
                await self.process_audio()
        except KeyboardInterrupt:
            print("\n👋 Shutting down...")
        finally:
            self.running = False

# =============================
# Entry Point
# =============================
async def main():
    """Main entry point."""
    chatbot = LangChainVoiceChatbot()
    await chatbot.run()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        sys.exit(0)
