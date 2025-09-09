# Migration Guide: Script to Module

This guide helps you migrate from using the original `streaming_voice_chatbot.py` script to the new modular version.

## What Changed

The original monolithic script has been refactored into a proper Python module with the following structure:

```
streaming_voice_chatbot/
├── __init__.py          # Main module exports
├── config.py            # Configuration settings
├── core.py              # Main chatbot class
├── detection.py         # Audio detection and STT
├── llm.py              # LLM streaming and conversation
└── tts.py              # Text-to-speech speakers
```

## Quick Migration

### Before (Original Script)
```bash
python streaming_voice_chatbot.py
```

### After (Module)
```bash
python run_chatbot.py
```

Or install and use as a package:
```bash
pip install -e .
streaming-voice-chatbot
```

## Code Migration

### Simple Usage Migration

**Before:**
```python
# All code was in one file
# Had to modify constants at the top of the file
```

**After:**
```python
from streaming_voice_chatbot import StreamingVoiceChatbot
import asyncio

async def main():
    chatbot = StreamingVoiceChatbot()
    await chatbot.run()

if __name__ == '__main__':
    asyncio.run(main())
```

### Configuration Migration

**Before:**
```python
# Constants at the top of streaming_voice_chatbot.py
SAMPLE_RATE = 16000
OLLAMA_MODEL = "gpt-oss:20b"
TTS_BACKEND = "edge-tts"
VOICE_NAME = "en-GB-SoniaNeural"
# ... etc
```

**After:**
```python
from streaming_voice_chatbot import StreamingVoiceChatbot, Config

class MyConfig(Config):
    SAMPLE_RATE = 16000
    OLLAMA_MODEL = "gpt-oss:20b"
    TTS_BACKEND = "edge-tts"
    VOICE_NAME = "en-GB-SoniaNeural"

async def main():
    config = MyConfig()
    chatbot = StreamingVoiceChatbot(config=config)
    await chatbot.run()
```

### Custom Component Migration

**Before:** (modifying functions directly in the script)
```python
# Had to modify classes like UtteranceDetector, WhisperSTT directly
```

**After:** (inherit and extend)
```python
from streaming_voice_chatbot import UtteranceDetector, StreamingVoiceChatbot

class MyUtteranceDetector(UtteranceDetector):
    def __init__(self, config=None):
        super().__init__(config, aggressiveness=3)  # Custom aggressiveness
        
class MyStreamingVoiceChatbot(StreamingVoiceChatbot):
    async def initialize(self):
        await super().initialize()
        self.detector = MyUtteranceDetector(self.config)
```

## Benefits of the Module

1. **Reusable**: Import and use in other projects
2. **Configurable**: Easy configuration without modifying code
3. **Extensible**: Inherit and extend components
4. **Maintainable**: Separated concerns in different modules
5. **Installable**: Can be installed as a proper Python package
6. **Testable**: Each component can be tested independently

## Compatibility

The module maintains 100% functional compatibility with the original script. All features work exactly the same:

- ✅ Voice activity detection
- ✅ Speech-to-text with Whisper
- ✅ Ollama LLM streaming
- ✅ Multiple TTS backends (pyttsx3, edge-tts, coqui-tts)
- ✅ Wake word detection
- ✅ Style commands
- ✅ Sentence-by-sentence TTS
- ✅ Clean shutdown

## Installation Options

### Development Installation
```bash
pip install -e .
```

### With Optional TTS Backends
```bash
# For edge-tts
pip install -e .[edge-tts]

# For coqui-tts  
pip install -e .[coqui-tts]

# For all features
pip install -e .[all]
```

## File Organization

You can keep both versions during migration:

```
chatbot/
├── streaming_voice_chatbot.py       # Original script (keep for reference)
├── streaming_voice_chatbot/         # New module
│   ├── __init__.py
│   ├── config.py
│   ├── core.py
│   ├── detection.py
│   ├── llm.py
│   └── tts.py
├── run_chatbot.py                   # Simple runner
├── example_custom_config.py         # Example with custom config
└── setup.py                        # Package setup
```

## Getting Help

If you encounter issues during migration:

1. Check that all dependencies are installed
2. Verify that Ollama is running (`ollama serve`)
3. Ensure your model is pulled (`ollama pull your-model`)
4. Test with the simple runner first (`python run_chatbot.py`)
5. Check the example files for reference implementations

The module is designed to be a drop-in replacement with additional flexibility and modularity.
