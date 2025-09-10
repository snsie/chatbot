# LangChain Streaming Voice Chatbot

A streaming voice chatbot built with LangChain, using the ChatOpenAI interface configured to work with Ollama's `gpt-oss:20b` model. Features real-time voice input via Whisper STT and voice output via edge-TTS.

## Features

- **LangChain Integration**: Uses ChatOpenAI interface with Ollama's OpenAI-compatible endpoint
- **Streaming Responses**: Real-time token streaming from the LLM
- **Voice Input**: Speech-to-text using faster-whisper with voice activity detection
- **Voice Output**: Text-to-speech using edge-TTS with high-quality voices
- **Conversation Memory**: Maintains conversation history (last 10 exchanges)
- **Sentence-by-Sentence TTS**: Speaks complete sentences as they're generated
- **Clean Shutdown**: Graceful handling of Ctrl+C interruption

## Requirements

### System Dependencies
```bash
sudo apt-get update && sudo apt-get install -y \
    portaudio19-dev ffmpeg espeak-ng
```

### Python Dependencies
All required packages are listed in `requirements.txt`. Key dependencies:
- `langchain` and `langchain-openai` for LLM integration
- `faster-whisper` for speech-to-text
- `edge-tts` for text-to-speech
- `sounddevice` and `webrtcvad` for audio processing
- `ollama` for model communication

### Ollama Setup
1. Install and start Ollama:
   ```bash
   # Install Ollama (if not already installed)
   curl -fsSL https://ollama.ai/install.sh | sh
   
   # Start Ollama server
   ollama serve
   ```

2. Pull the required model:
   ```bash
   ollama pull gpt-oss:20b
   ```

## Quick Start

1. **Start Ollama server** (in a separate terminal):
   ```bash
   ollama serve
   ```

2. **Run the voice chatbot**:
   ```bash
   ./start_voice_chatbot.sh
   ```
   
   Or manually:
   ```bash
   source .venv/bin/activate
   python langchain_voice_chatbot.py
   ```

## Usage

1. The chatbot will display "🎤 Voice chatbot is listening..." when ready
2. Speak into your microphone - the system uses voice activity detection
3. Your speech will be transcribed and displayed as "You: <transcript>"
4. The assistant will respond with streaming text and speech
5. Press Ctrl+C to exit cleanly

## Configuration

Edit the configuration section in `langchain_voice_chatbot.py`:

```python
# Audio Configuration
SAMPLE_RATE = 16000
VAD_AGGRESSIVENESS = 2  # 0-3 (higher = more aggressive)
MIN_UTTERANCE_MS = 300
TRAILING_SILENCE_MS = 200

# STT Configuration
WHISPER_MODEL = "small.en"
WHISPER_COMPUTE = "auto"  # 'auto' | 'cpu' | 'cuda'

# LLM Configuration
OLLAMA_MODEL = "gpt-oss:20b"
MAX_TOKENS = 512
TEMPERATURE = 0.7

# TTS Configuration
TTS_VOICE = "en-US-AriaNeural"  # edge-tts voice
```

## Available TTS Voices

Some popular edge-TTS voices:
- `en-US-AriaNeural` (default, female, US)
- `en-US-GuyNeural` (male, US)
- `en-GB-SoniaNeural` (female, UK)
- `en-AU-NatashaNeural` (female, Australia)

## Testing

### Test LangChain Integration Only
```bash
source .venv/bin/activate
python test_langchain_ollama.py
```

### Test Text-Only Chatbot
```bash
source .venv/bin/activate
python test_langchain_text.py
```

## Architecture

```
Microphone → VAD → Whisper STT → LangChain → Ollama (gpt-oss:20b)
                                      ↓
Speaker ← edge-TTS ← Sentence Segmentation ← Streaming Response
```

## Components

- **UtteranceDetector**: WebRTC VAD for speech segmentation
- **WhisperSTT**: faster-whisper for speech recognition
- **LangChainChatbot**: ChatOpenAI + conversation memory
- **EdgeTTS**: High-quality text-to-speech
- **LangChainVoiceChatbot**: Main coordinator class

## Troubleshooting

### Audio Issues
- Check microphone permissions and levels
- Verify portaudio installation: `python -c "import sounddevice; print(sounddevice.query_devices())"`

### Ollama Issues
- Ensure Ollama is running: `curl http://localhost:11434/api/tags`
- Check model availability: `ollama list`
- Verify model works: `ollama run gpt-oss:20b "Hello"`

### LangChain Issues
- Verify installation: `python -c "from langchain_openai import ChatOpenAI; print('OK')"`
- Check OpenAI endpoint: `curl http://localhost:11434/v1/models`

## Performance Notes

- **GPU Acceleration**: Whisper will use CUDA if available
- **Memory Usage**: The 20B model requires significant RAM
- **Latency**: Voice activity detection + transcription + LLM response + TTS
- **Streaming**: Sentences are spoken as soon as they're complete

## License

This project uses the same license as the existing chatbot codebase.
