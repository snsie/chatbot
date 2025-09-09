# Streaming Voice Chatbot Module

A modular, half‑duplex voice assistant with low latency sentence‑by‑sentence TTS while tokens stream from an Ollama model.

## Features

1. **Microphone capture** @ 16 kHz mono (30 ms frames) using sounddevice RawInputStream
2. **Voice Activity Detection** (webrtcvad) to segment utterances
3. **Speech‑to‑Text** via faster-whisper (GPU auto, fallback CPU) on captured utterance
4. **Streaming LLM responses** token-by-token from Ollama (llama3.1:8b-instruct by default)
5. **Sentence segmentation** of streaming tokens; each completed sentence immediately sent to TTS
6. **Multiple TTS backends**:
   - pyttsx3 (offline, default)
   - edge-tts (optional, higher quality, requires internet + ffmpeg)
   - coqui-tts (neural TTS, requires GPU for best performance)
7. **Clean shutdown** on Ctrl+C
8. **Wake word detection** - responds only when wake words are detected
9. **Style commands** - change response style on-the-fly

## Installation

```bash
# Install required packages
pip install -r requirements.txt

# For edge-tts support (optional)
pip install edge-tts pydub simpleaudio

# For coqui-tts support (optional)
pip install TTS torch torchaudio
```

## Quick Start

### Basic Usage

```python
from streaming_voice_chatbot import StreamingVoiceChatbot
import asyncio

async def main():
    chatbot = StreamingVoiceChatbot()
    await chatbot.run()

if __name__ == '__main__':
    asyncio.run(main())
```

### Using the Runner Script

```bash
python run_chatbot.py
```

### Custom Configuration

```python
from streaming_voice_chatbot import StreamingVoiceChatbot, Config

class MyConfig(Config):
    OLLAMA_MODEL = "llama2:latest"
    TTS_BACKEND = "edge-tts"
    VOICE_NAME = "en-US-AriaNeural"
    SYSTEM_PROMPT = "You are a helpful assistant named Cora."

async def main():
    config = MyConfig()
    chatbot = StreamingVoiceChatbot(config=config)
    await chatbot.run()
```

## Module Structure

```
streaming_voice_chatbot/
├── __init__.py          # Main module exports
├── config.py            # Configuration settings
├── core.py              # Main chatbot class
├── detection.py         # Audio detection and STT
├── llm.py              # LLM streaming and conversation
└── tts.py              # Text-to-speech speakers
```

## Configuration Options

### Audio Settings
- `SAMPLE_RATE`: Audio sample rate (default: 16000 Hz)
- `VAD_AGGRESSIVENESS`: Voice activity detection sensitivity (0-3)
- `MIN_UTTERANCE_MS`: Minimum utterance length in milliseconds
- `TRAILING_SILENCE_MS`: Silence duration to end utterance

### LLM Settings
- `OLLAMA_MODEL`: Ollama model to use (default: "gpt-oss:20b")
- `SYSTEM_PROMPT`: System prompt for the assistant
- `MAX_TOKENS`: Maximum tokens per response
- `SIMILAR_NAMES`: Wake words list

### TTS Settings
- `TTS_BACKEND`: TTS engine ("pyttsx3", "edge-tts", or "coqui")
- `VOICE_NAME`: Voice to use (backend-specific)

## Runtime Flow

1. 🎤 **Listening** → capture utterance using VAD
2. 📝 **Transcribing** → convert speech to text with Whisper
3. 🔍 **Wake word check** → only respond if wake word detected
4. 🤖 **Assistant streaming** → generate and speak response sentence-by-sentence
5. ⏸️ **Brief pause** → prevent self-capture
6. **Repeat**

## Wake Words

The chatbot responds only when it detects wake words in the transcript:
- Cora, Kora, Korra, Quora, Core, Cori, Corey, Coral
- Plus variations and similar-sounding names

## Style Commands

You can change the assistant's response style with voice commands:
- "Be more formal" / "Speak more formally"
- "Be more casual" / "Speak more casually"
- "Be more technical"
- "Explain like I'm 5" / "ELI5"
- "Be more concise"
- "Be more detailed"
- "Be more creative"
- "Be more professional"

## Prerequisites

1. **Ollama server running**:
   ```bash
   ollama serve
   ```

2. **Pull desired model** (first time):
   ```bash
   ollama pull gpt-oss:20b
   # or
   ollama pull llama2:latest
   ```

3. **Required Python packages** (see requirements.txt)

## Error Handling

The module includes robust error handling for:
- Audio device issues
- Network connectivity problems
- Model loading failures
- TTS synthesis errors

Errors are logged but don't crash the application.

## Development

To extend the module:

1. **Custom TTS Backend**: Inherit from `BaseSpeaker` in `tts.py`
2. **Custom STT**: Modify `WhisperSTT` in `detection.py`
3. **Custom LLM**: Modify `ollama_stream_chat` in `llm.py`
4. **Custom Configuration**: Inherit from `Config` in `config.py`

## License

This module is based on the original streaming voice chatbot script and maintains the same functionality while providing a clean, modular interface.
