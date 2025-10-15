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

class MainFunctionality:
  def __init__(self):
    self.detector = UtteranceDetector
    self.stt = WhisperSTT
    self.convo = Conversation
    self.speaker = BaseSpeaker
    self.voice_identifier = _VoiceIdentifier
    self.people = List[Person]

  async def voice_check_and_registration(self):
     print("hi")

  async def validate_audio_capture(self) -> bool:
    # Check if audio was successfully captured
    if self.audio is None or not len(self.audio):
        print("🛑 No audio captured.")
        return False
    return True
  
  async def transcribing_audio(self) -> bool:
    try:
      print("📝 Transcribing…", flush=True)
      self.transcript = await asyncio.to_thread(self.stt.transcribe, self.audio)
      cora_word_found = False
      transcript_no_punct = self.transcript.translate(str.maketrans('', '', string.punctuation))
      words_list = transcript_no_punct.split()

      for word in words_list:
          if word in SIMILAR_NAMES:
              print(f"Found similar name: {word}")
              cora_word_found = True
              break
      if not cora_word_found:
          print("No wake word detected; ignoring input.")
          return False
      return True
      
    except Exception as e:
        print(f"[STT Error] {e}")
        return False

  async def voice_check_and_registration(self) -> bool:
    try:
        best_name, best_sim = self.voice_identifier.identify_from_array(self.audio, 16000)
        if best_name is None or best_sim < SIM_THRESHOLD:
            # 1) Ask to enroll
            await self.speaker.speak("I didn’t recognize your voice. Would you like to register it now?")
            
            resp_audio = await asyncio.to_thread(self.detector.record_once)
            resp_text  = (await asyncio.to_thread(self.stt.transcribe, resp_audio)).strip().lower() if resp_audio is not None else ""
            if not any(k in resp_text for k in ["yes", "yeah", "yep", "sure", "ok", "okay"]):
                await self.speaker.speak("Okay, I won't enroll right now.")
                return False
            
            # 2) Ask for a display name
            await self.speaker.speak("What name should I save this voice under?")
            name_audio = await asyncio.to_thread(self.detector.record_once)
            user_name  = await asyncio.to_thread(self.stt.transcribe, name_audio)
            user_name  = user_name.strip()
            user_name = re.sub(r"[^\w\s-]", "", user_name) # remove special chars
            user_name = re.sub(r"\s+", " ", user_name)
            user_dir = Path("data") / user_name
            user_dir.mkdir(parents=True, exist_ok=True)

            # 3) Collect 3–5 short enrollment utterances (~2–5 s each)
            prompts = [
                "Please say: 'I enjoy pasta for dinner Cora.'",
                "Please say: 'I start my morning with coffee.'",
                "Please say: 'Hey Cora, the weather might change later today.'",
                "Please say: 'I try to exercise regularly and eat healthy foods.'"
            ]
            embs = []
            for i, p in enumerate(prompts[:4]):      # collect 4 by default
                await self.speaker.speak(p)
                clip = await asyncio.to_thread(self.detector.record_once)
                if clip is None or len(clip) == 0:
                    continue

                # Save raw WAV file under data/<user_name>/<user_name>_i.wav
                out_path = user_dir / f"{user_name}_{i}.wav"
                sf.write(str(out_path), clip, 16000)
                print(f"[Enroll] Saved {out_path}")
                

            # 4) re runs build_enrollments.py to update enrollments.npz and reload into voice_id memory
            subprocess.run(
                ["python", "build_enrollments.py", "--root", "data", "--out", ENROLL_PATH],
                check=True
            )
            self.voice_identifier.reload_enrollments(ENROLL_PATH)

            # 5) Read the centroid for this user from enrollments.npz and upsert ONE Mongo doc
            npz = np.load(ENROLL_PATH, allow_pickle=True)  # has arrays: names, vecs :contentReference[oaicite:3]{index=3}
            names, vecs = npz["names"], npz["vecs"]
            centroid = None
            for n, v in zip(names, vecs):
                if str(n) == user_name:
                    centroid = v
                    break
            if centroid is None:
                await self.speaker.speak("I couldn’t finalize your enrollment. Please try again later.")
                return False
            
            try:
                print("insidee try")
                person_data = {
                    "name": user_name,
                    "current_embedding": {
                        "vec": centroid.tolist(),
                        "model": "speechbrain/ecapa-voxceleb"
                    },
                    "file_path": str(user_dir),
                    # created_at is auto-set by Pydantic at object creation
                }
                
                # Validate with Pydantic --> creating Person object that checks against types of person_data
                person = Person(**person_data)
                
                # insert into mongodb if successful validation of data types
                self.people.update_one(
                    {"name": user_name},
                    {"$set": {
                        "current_embedding": {
                            "vec": person.current_embedding.vec,
                            "model": person.current_embedding.model
                        },
                        "file_path": person.file_path,
                        "created_at": person.created_at,
                    }},
                    upsert=True
                )
                print("success")
                
            except ValidationError as e:
                await self.speaker.speak("Error saving your voice profile. Please try again.")
                print(f"Pydantic validation error: {e}")
                return False
            
            await self.speaker.speak(f"Thanks {user_name}. Your voice has been registered.")

            # 5) Optional immediate re-check
            await self.speaker.speak("Say one more sentence to confirm.")
            confirm = await asyncio.to_thread(self.detector.record_once)
            conf_name, conf_sim = self.voice_identifier.identify_from_array(confirm, 16000)
            if conf_name == user_name:
                await self.speaker.speak(f"Verification passed with similarity {conf_sim:.2f}.")
            else:
                await self.speaker.speak("Verification was low; we can add more samples later.")
            return True
        # else:
        #     print(f"[Gate] ✅ Allow: {best_name} (sim={best_sim:.3f})")
        #     await speaker.speak(f"Hey {best_name}.")
    except Exception as e:
        print(f"[Gate Error] {e}")
        await self.speaker.speak("Voice check failed. Please try again.")
        return False

  async def style_commands(self) -> bool:
    is_style_command = self.convo._detect_style_commands(self.transcript)
    self.convo.add_user(self.transcript)
    
    if is_style_command:
        # For style commands, give immediate feedback instead of calling LLM
        current_style = self.convo.get_current_style()
        response = f"I've updated my response style. Current style: {current_style}"
        print(f"Assistant ↳ {response}")
        await self.speaker.speak(response)
        self.convo.add_assistant(response)
        return

  async def main_loop(self):
    print("🎤 Listening…", flush=True)

    self.audio = await asyncio.to_thread(self.detector.record_once)  # Add back the asyncio.to_thread()

    if not self.validate_audio_capture():
      return # no audio caputured
    
    if not self.transcribing_audio():
      return # transcription failed or "cora" not found
    
    if ENABLE_SPEAKER_GATE and self.voice_identifier is not None:
       if not self.voice_check_and_registration():
          return
       
    if not self.transcript.strip():
        return
    print(f"You: {self.transcript}")