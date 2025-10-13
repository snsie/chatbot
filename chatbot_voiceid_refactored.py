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
from pymongo import MongoClient
from scipy.signal import resample_poly
import re
from chatbot.speakers import BaseSpeaker
from chatbot.listeners import WhisperSTT

from chatbot.main_loop import main_loop
from chatbot.memory import Conversation

MONGO_URI = "mongodb://admin:Rob123%21@localhost:27017/admin"
mongo = MongoClient(MONGO_URI)
people = mongo["voice_db"]["people"]  # single collection for profiles



voice_identifier = get_voice_identifier(ENROLL_PATH) if ENABLE_SPEAKER_GATE else None


async def main():

    setup_windows_event_loop()
    print("Booting streaming voice chatbot…")

    detector = UtteranceDetector()
    stt = WhisperSTT(WHISPER_MODEL, WHISPER_COMPUTE)
    speaker = await create_speaker()
    convo = Conversation(SYSTEM_PROMPT)

    try:
        while True:
            await main_loop(detector, stt, convo, speaker, voice_identifier, people)
    except KeyboardInterrupt:
        print("\nExiting…")
    finally:
        await speaker.close()


if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass




# ########## remodeled code #########
# async def active_listening(detector:UtteranceDetector, stt: WhisperSTT) -> bool:
#     print("🎤 Listening…", flush=True)

#     audio = await asyncio.to_thread(detector.record_once)  # Add back the asyncio.to_thread()
#     if audio is None or not len(audio):
#         print("🛑 No audio captured.")
#         return False # Nothing captured; loop again
#     try:
#         print("📝 Transcribing…", flush=True)
#         transcript = await asyncio.to_thread(stt.transcribe, audio)
#         cora_word_found = False
#         transcript_no_punct = transcript.translate(str.maketrans('', '', string.punctuation))
#         words_list = transcript_no_punct.split()

#         # listening for "cora"
#         for word in words_list:
#             if word in SIMILAR_NAMES:
#                 print(f"Found similar name: {word}")
#                 cora_word_found = True
#                 return transcript, audio
#         if not cora_word_found:
#             print("No wake word detected; ignoring input.")
#             return False
        
#     except Exception as e:
#         print(f"[STT Error] {e}")
#         return False

# async def voice_registration(detector: UtteranceDetector, stt: WhisperSTT, convo: Conversation, speaker: BaseSpeaker) -> bool | str:
#     # 1) Ask to enroll
#     await speaker.speak("I didn’t recognize your voice. Would you like to register it now?")
    
#     resp_audio = await asyncio.to_thread(detector.record_once)
#     resp_text  = (await asyncio.to_thread(stt.transcribe, resp_audio)).strip().lower() if resp_audio is not None else ""
#     if not any(k in resp_text for k in ["yes", "yeah", "yep", "sure", "ok", "okay"]):
#         await speaker.speak("Okay, I won't enroll right now.")
#         return False
    
#     # 2) Ask for a display name
#     await speaker.speak("What name should I save this voice under?")
#     name_audio = await asyncio.to_thread(detector.record_once)
#     user_name  = await asyncio.to_thread(stt.transcribe, name_audio)
#     user_name  = user_name.strip()
#     user_name = re.sub(r"[^\w\s-]", "", user_name) # remove special chars
#     user_name = re.sub(r"\s+", " ", user_name)
#     user_dir = Path("data") / user_name
#     user_dir.mkdir(parents=True, exist_ok=True)

#     # 3) Collect 3–5 short enrollment utterances (~2–5 s each)
#     prompts = [
#         "Please say: 'I enjoy pasta for dinner Cora.'",
#         "Please say: 'I start my morning with coffee.'",
#         "Please say: 'Hey Cora, the weather might change later today.'",
#         "Please say: 'I try to exercise regularly and eat healthy foods.'"
#     ]
#     embs = []
#     for i, p in enumerate(prompts[:4]):      # collect 4 by default
#         await speaker.speak(p)
#         clip = await asyncio.to_thread(detector.record_once)
#         if clip is None or len(clip) == 0:
#             continue

#         # Save raw WAV file under data/<user_name>/<user_name>_i.wav
#         out_path = user_dir / f"{user_name}_{i}.wav"
#         sf.write(str(out_path), clip, 16000)
#         print(f"[Enroll] Saved {out_path}")
        

#     # 4) re runs build_enrollments.py to update enrollments.npz and reload into voice_id memory
#     subprocess.run(
#         ["python", "build_enrollments.py", "--root", "data", "--out", ENROLL_PATH],
#         check=True
#     )
#     voice_identifier.reload_enrollments(ENROLL_PATH)

#     # 5) Read the centroid for this user from enrollments.npz and upsert ONE Mongo doc
#     npz = np.load(ENROLL_PATH, allow_pickle=True)  # has arrays: names, vecs :contentReference[oaicite:3]{index=3}
#     names, vecs = npz["names"], npz["vecs"]
#     centroid = None
#     for n, v in zip(names, vecs):
#         if str(n) == user_name:
#             centroid = v
#             break
#     if centroid is None:
#         await speaker.speak("I couldn’t finalize your enrollment. Please try again later.")
#         return False
    
#     try:
#         person_data = {
#             "name": user_name,
#             "current_embedding": {
#                 "vec": centroid.tolist(),
#                 "model": "speechbrain/ecapa-voxceleb"
#             },
#             "file_path": str(user_dir),
#             # created_at is auto-set by Pydantic at object creation
#         }
        
#         # Validate with Pydantic --> creating Person object that checks against types of person_data
#         person = Person(**person_data)
        
#         # insert into mongodb if successful validation of data types
#         people.update_one(
#             {"name": user_name},
#             {"$set": {
#                 "current_embedding": {
#                     "vec": person.current_embedding.vec,
#                     "model": person.current_embedding.model
#                 },
#                 "file_path": person.file_path,
#                 "created_at": person.created_at,
#             }},
#             upsert=True
#         )
        
#     except ValidationError as e:
#         await speaker.speak("Error saving your voice profile. Please try again.")
#         print(f"Pydantic validation error: {e}")
#         return False
    
    
#     await speaker.speak(f"Thanks {user_name}. Your voice has been registered.")

#     # 5) Optional immediate re-check
#     await speaker.speak("Say one more sentence to confirm.")
#     confirm = await asyncio.to_thread(detector.record_once)
#     conf_name, conf_sim = voice_identifier.identify_from_array(confirm, 16000)
#     if conf_name == user_name:
#         await speaker.speak(f"Verification passed with similarity {conf_sim:.2f}.")
#         # return user_name
#     else:
#         await speaker.speak("Verification was low; we can add more samples later.")
#         # return False
#     return user_name

# async def cora_response(detector: UtteranceDetector, stt: WhisperSTT, convo: Conversation, speaker: BaseSpeaker, best_name) -> None:
#     print("🤖 Assistant (streaming)…", flush=True)

#     # Streaming generation
#     assistant_buffer = []
#     sentences_queue: asyncio.Queue = asyncio.Queue()
#     speak_consumer_task = asyncio.create_task(_speak_consumer(sentences_queue, speaker, detector))  # Pass detector
    
#     get_last_text=convo.history()[-1] if convo.history() else None
#     last_content=get_last_text['content'] if get_last_text else "No history"
#     sentence_to_add=f"Please say 'Hey {best_name}' before speaking to me. "
#     # sentence_to_add
#     convo.messages[-1]['content']=sentence_to_add+convo.messages[-1]['content']
#     print('last_content',convo.messages[-1]['content'])
    
#     # best_name
#     async for sentence in sentence_stream(ollama_stream_chat(convo.history(), OLLAMA_MODEL, MAX_TOKENS)):
#         assistant_buffer.append(sentence)
#         await sentences_queue.put(sentence)
#         if PRINT_PARTIAL_SENTENCES:
#             print(f"Assistant ↳ {sentence}")

#     # Signal completion
#     await sentences_queue.put(None)
#     await speak_consumer_task

#     full_assistant_text = ' '.join(assistant_buffer)
#     convo.add_assistant(full_assistant_text)

# async def _speak_consumer(q: 'asyncio.Queue[Optional[str]]', speaker: BaseSpeaker, detector: UtteranceDetector) -> None:
#     sentences_spoken = 0
    
#     # Mute microphone when starting to speak
#     detector.mute_microphone()
    
#     while True:
#         sentence = await q.get()
#         if sentence is None:
#             break
#         try:
#             await speaker.speak(sentence)
#             sentences_spoken += 1
#         except Exception as e:
#             print(f"[TTS Error] {e}")
    
#     # Dynamic delay based on how much was spoken
#     dynamic_delay = 0.5
#     await asyncio.sleep(dynamic_delay)
    
#     # Unmute microphone after speaking is completely done
#     detector.unmute_microphone()
#     print("🔊 Microphone reactivated")


# async def process_turn(detector: UtteranceDetector, stt: WhisperSTT, convo: Conversation, speaker: BaseSpeaker):
#     ##### 1. active listening ######
#     transcript, audio = await active_listening(detector, stt)
#     if not transcript or audio is None or len(audio) == 0:
#         return

#     ######### 2. voice recognition #########
#     if ENABLE_SPEAKER_GATE and voice_identifier is not None:
#         try:
#             best_name, best_sim = voice_identifier.identify_from_array(audio, 16000)
#             if best_name is None or best_sim < SIM_THRESHOLD:
#                 user_name = await voice_registration(detector, stt, convo, speaker)
#                 if not user_name:
#                     return
#                 best_name = user_name
#         except Exception as e:
#             print(f"[Gate Error] {e}")
#             await speaker.speak("Voice check failed. Please try again.")
#             return

#     if not transcript.strip():
#         return
#     print(f"You: {transcript}")
    
#     # Handle style commands differently
#     is_style_command = convo._detect_style_commands(transcript)
#     convo.add_user(transcript)
    
#     if is_style_command:
#         # For style commands, give immediate feedback instead of calling LLM
#         current_style = convo.get_current_style()
#         response = f"I've updated my response style. Current style: {current_style}"
#         print(f"Assistant ↳ {response}")
#         await speaker.speak(response)
#         convo.add_assistant(response)
#         return

#     ############ 5. cora response #############
#     await cora_response(detector, stt, convo, speaker, best_name)

# # =============================
# # Entry Point
# # =============================
# async def main():
#     if sys.platform.startswith('win'):
#         try:
#             asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())  # type: ignore
#         except Exception:
#             pass
#     print("Booting streaming voice chatbot…")

#     detector = UtteranceDetector()
#     stt = WhisperSTT(WHISPER_MODEL, WHISPER_COMPUTE)
#     speaker = await create_speaker()
#     convo = Conversation(SYSTEM_PROMPT)

#     try:
#         while True:
#             await process_turn(detector, stt, convo, speaker)
#     except KeyboardInterrupt:
#         print("\nExiting…")
#     finally:
#         await speaker.close()

# if __name__ == '__main__':
#     try:
#         asyncio.run(main())
#     except KeyboardInterrupt:
#         pass