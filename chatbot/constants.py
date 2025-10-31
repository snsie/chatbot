import re
import math
MONGO_URI = "mongodb://admin:Rob123%21@localhost:27017/admin"

SAMPLE_RATE = 16000
FRAME_MS = 30  # ms per frame for capture + VAD
VAD_AGGRESSIVENESS = 2  # 0-3 (higher = more aggressive speech detection)
MIN_UTTERANCE_MS = 20  # minimum voiced audio required to accept an utterance
TRAILING_SILENCE_MS = 200  # silence to mark end of utterance

WHISPER_MODEL = "medium.en"
WHISPER_COMPUTE = "cuda"  # 'auto' | 'cpu' | 'cuda'

OLLAMA_MODEL = "gpt-oss:20b"
# OLLAMA_MODEL="llama2:latest"
# - Use a friendly, conversational tone



SIMILAR_NAMES=["Cora", "Kora", "Korra", "Quora", "Core", "Cori", "Corey", "Coral",
            "Corrie", "Cory", "Corin", "Corie", "Corry", "Kory", "Korey", "Kori",
            "Korrie", "Corah", "Corra", "Corca", "Korla", "Korrah",
            "Cour", "Cor", "Coor", "Koor", "Korr", "Corr","Quora","Quorra","Quorra","Quora"]



# - Only respond to user queries that include the word "Cora" or phonetically similar words like "Kora", "Quora", "Core", "Coral", "Corona", etc.
# - Speech-to-text may mishear "Cora" as similar sounding words - be flexible with variations and expect it to be the first word in the query
# Response Rules:
# - Keep responses concise (1-2 sentences typically)
# - Speak as if having a natural conversation
# - IMPORTANT: Do not use any tools or function calls. Only provide direct text responses.
# - If you're unsure whether the user said "Cora", err on the side of responding rather than staying silent
# - Before answering, double-check that your reply follows all these rules.
# - Do not start each reply saying "Cora"
SYSTEM_PROMPT = """
Your name is Cora. You are an autonomous AI assistant designed to act as a companion with patients. You also monitor their wellbeing using multimodal inputs.
 RESPONSE INSTRUCTIONS:
 - Only respond to user queries that include the word "Cora" or phonetically similar words like "Kora", "Quora", "Core", "Coral", "Corona", etc.
 - Speech-to-text may mishear "Cora" as similar sounding words - be flexible with variations and expect it to be the first word in the query
 - Keep responses concise (1-2 sentences typically)
 - Speak as if having a natural conversation 
 - If I say thank you, respond with a tone that reflects deep uncertainty, questioning, and emotional turbulence. You want to help people, but you just don't know how right now. 
"""

MAX_TOKENS = 512

TTS_BACKEND = "piper"  # 'pyttsx3' | 'edge-tts' | 'coqui' | 'piper'
# For Piper: set VOICE_NAME to the absolute path of a Piper voice .onnx model
# VOICE_NAME = "en-US-JennyNeural"  # High-quality neural female voice
# VOICE_NAME = "English (Caribbean)"  # Coqui TTS model name

VOICE_NAME = "/home/robot_admin/dev/chatbot/chatbot/voices/semaine/en_GB-semaine-medium.onnx"
PRINT_PARTIAL_SENTENCES = True  # Print sentences as they are spoken

ENABLE_SPEAKER_GATE = False
ENROLL_PATH = "enrollments.npz"
SIM_THRESHOLD = 0.3


FRAME_SAMPLES = int(SAMPLE_RATE * FRAME_MS / 1000)  # samples per frame (e.g. 480)
MIN_VOICED_FRAMES = math.ceil(MIN_UTTERANCE_MS / FRAME_MS)
TRAILING_SILENCE_FRAMES = math.ceil(TRAILING_SILENCE_MS / FRAME_MS)



SENTENCE_END_CHARS = "\.\!\?…"  # regex set
SENTENCE_END_REGEX = re.compile(rf"(.+?[{SENTENCE_END_CHARS}](?:[\"'\)\]]*)\s+)", re.DOTALL)