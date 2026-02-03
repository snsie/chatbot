# PersonaPlex Voice Chatbot

This workspace now uses **NVIDIA PersonaPlex** for speech-to-speech interaction. PersonaPlex handles ASR and TTS internally, so the legacy Whisper/Ollama/TTS stack is no longer required for the main flow.

## What’s inside

- `personaplex_app/personaplex_chat.py` — records a user utterance, runs PersonaPlex (via `moshi.offline`), and plays the response audio.
- `personaplex_app/profiles.json` — simple profile mapping (voice + persona prompt).
- `personaplex_app/prompts/assistant.txt` — default persona text prompt.
- `assets/voice_prompts/` — place PersonaPlex voice prompt embeddings here.

## Prerequisites

1. **Accept the PersonaPlex license** on Hugging Face: https://huggingface.co/nvidia/personaplex-7b-v1
2. **Set your token**:
   - `HF_TOKEN` must be exported in your shell environment.
3. **Install the PersonaPlex runtime** (moshi) and audio dependencies.

## Usage

Run the offline chat loop:

```
python -m personaplex_app.personaplex_chat
```

Optional flags:

- `--profile assistant` (default)
- `--cpu-offload` to reduce GPU memory use

## Profiles

Edit `personaplex_app/profiles.json` to add new voices or personas. Each profile needs:

- `voice_prompt`: path to a `.pt` embedding (see `assets/voice_prompts/README.md`)
- `text_prompt`: path to a prompt `.txt`

## Notes

- Audio runs at 24 kHz to match PersonaPlex expectations.
- If output audio is missing, check your GPU memory, voice prompt path, and `HF_TOKEN`.
