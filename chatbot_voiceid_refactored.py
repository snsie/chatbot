# =============================
# Imports
# =============================
import asyncio

# Voice ID helper (only used for optional global preloading; the CoraChatbot manages its own identifier)
# from chatbot.voice_id.get_voice_identifier import get_voice_identifier

# Project-wide constants for models, thresholds, and flags
from chatbot.constants import (SAMPLE_RATE, FRAME_MS, VAD_AGGRESSIVENESS, MIN_UTTERANCE_MS, TRAILING_SILENCE_MS,
                              WHISPER_MODEL, WHISPER_COMPUTE,
                              OLLAMA_MODEL, MAX_TOKENS,
                              TTS_BACKEND, VOICE_NAME,
                              PRINT_PARTIAL_SENTENCES, SYSTEM_PROMPT, ENROLL_PATH, ENABLE_SPEAKER_GATE, 
                              SIMILAR_NAMES, SIM_THRESHOLD
)

# Platform utils (Windows-specific event loop setup)
from chatbot.utils import setup_windows_event_loop

USE_SEPARATION = True  # flip on/off easily

# High-level orchestrator for a single voice turn
from chatbot.cora_chatbot import CoraChatbot

# =============================
# Main Entry Point
# =============================
async def main():
    """
    Entry point for the voice chatbot runner.

    Responsibilities:
    - Perform platform-specific event loop setup (Windows only).
    - Create a CoraChatbot instance that orchestrates one half‑duplex turn.
    - Continuously run interaction turns until interrupted.
    - Ensure TTS resources are closed cleanly on exit.

    Notes:
    - The actual capture/transcribe/gate/enroll/stream logic lives in CoraChatbot.
    - This runner keeps the loop simple and focuses on lifecycle management.
    """
    setup_windows_event_loop()
    print("Booting streaming voice chatbot…")

    bot = CoraChatbot()

    try:
        while True:
            # Run a single interaction turn (listen -> transcribe -> respond)
            await bot.run_turn()
    except KeyboardInterrupt:
        print("\nExiting…")
    finally:
        # Close TTS if it was initialized
        if hasattr(bot.speaker, "close"):
            await bot.speaker.close()

if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass