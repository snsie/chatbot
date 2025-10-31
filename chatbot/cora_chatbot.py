from .speakers import _speak_consumer, create_speaker
from .listeners.whisper_stt import WhisperSTT
from .memory.conversation_memory import Conversation
from .utils.utterance_detector import UtteranceDetector
from .streaming.sentence_stream import sentence_stream
from .streaming.ollama_stream_chat import ollama_stream_chat
from .constants import (
    OLLAMA_MODEL, MAX_TOKENS, PRINT_PARTIAL_SENTENCES,
    ENABLE_SPEAKER_GATE, SIMILAR_NAMES, ENROLL_PATH, SIM_THRESHOLD, WHISPER_COMPUTE, WHISPER_MODEL, SYSTEM_PROMPT
)
import asyncio
import string
import re
from pathlib import Path
import numpy as np
import soundfile as sf
import subprocess
from .utils.database import get_people_collection
from pydantic import ValidationError
from .voice_id.person import Person
from .voice_id.get_voice_identifier import get_voice_identifier
from time import perf_counter
from contextlib import contextmanager
import os


class TurnProfiler:
    """
    Simple per-turn profiler to measure durations of labeled stages.

    Usage:
        prof = TurnProfiler(enabled=True)
        with prof.measure("capture_audio"):
            ...
        prof.finish()
    """
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self._t0 = perf_counter()
        self._steps = []  # list of (name, duration)

    @contextmanager
    def measure(self, name: str):
        if not self.enabled:
            yield
            return
        t0 = perf_counter()
        try:
            yield
        finally:
            dt = perf_counter() - t0
            self._steps.append((name, dt))

    def finish(self):
        if not self.enabled:
            return
        total = perf_counter() - self._t0
        # print("[Profile] Turn summary:")
        # for n, dt in self._steps:
        #     print(f" - {n}: {dt*1000:.1f} ms")
        # print(f" - TOTAL: {total*1000:.1f} ms")

class CoraChatbot:
    """
    Orchestrates one half‑duplex voice interaction turn:
      1) Capture a single utterance via VAD.
      2) Transcribe with Whisper.
      3) Optionally gate on recognized speaker and enroll if unknown.
      4) Stream LLM response while speaking sentences via TTS.
      5) Append the response to conversation history.

    Attributes:
        detector: VAD-based utterance detector for microphone capture.
        stt: Whisper speech-to-text wrapper.
        convo: Conversation memory for system/user/assistant messages.
        speaker: Active TTS speaker instance (created lazily).
        voice_identifier: Voice-ID helper for gating and enrollment.
        people: MongoDB collection handle for voice profiles.
        audio: Last captured utterance audio (float32 numpy array) or None.
        transcript: Last transcribed text for the captured utterance.
        assistant_buffer: Accumulates streamed assistant sentences for history.
        is_style_command: Flag when the user issues a style command.
        best_name: Recognized/enrolled display name for the current speaker (safe default).
    """
    def __init__(self):        
        self.detector = UtteranceDetector()
        self.stt = WhisperSTT(WHISPER_MODEL, compute=WHISPER_COMPUTE)
        self.convo = Conversation(SYSTEM_PROMPT)
        self.speaker = None
        self.voice_identifier = get_voice_identifier(ENROLL_PATH) if ENABLE_SPEAKER_GATE else None
        self.people = get_people_collection()

        self.audio = None
        self.transcript = ""
        self.assistant_buffer = []
        self.is_style_command = False
        self.best_name = None
        # Enable/disable profiling via env var CHATBOT_PROFILE ("0"/"false" disables)
        self.profile_enabled = os.getenv("CHATBOT_PROFILE", "1").lower() not in ("0", "false")

    async def ensure_tts_ready(self):
        """
        Lazily create and cache the TTS speaker instance.

        Rationale:
        - __init__ cannot await.
        - TTS backends are heavy; initialize only when needed.
        - Idempotent; safe to call multiple times.
        """
        if self.speaker is None:
            self.speaker = await create_speaker()

    async def validate_captured_audio(self) -> bool:
        """
        Validate that an utterance was captured into self.audio.

        Returns:
            True if audio is present and non-empty; False otherwise.
        """
        if self.audio is None or not len(self.audio):
            print("🛑 No audio captured.")
            return False
        return True
  
    async def transcribe_and_check_wake_word(self) -> bool:
        """
        Transcribe self.audio and check for a wake-word token.

        Behavior:
            - Uses Whisper to transcribe the captured utterance.
            - Strips punctuation and scans tokens for any in SIMILAR_NAMES.
            - On success, sets self.transcript and returns True.
            - If wake word not found, returns False (turn is ignored).

        Returns:
            True on successful transcription and wake-word match; else False.
        """
        try:
            print("📝 Transcribing…", flush=True)
            self.transcript = await asyncio.to_thread(self.stt.transcribe, self.audio)
            cora_word_found = False
            transcript_no_punct = self.transcript.translate(str.maketrans('', '', string.punctuation))
            words_list = transcript_no_punct.split()
            print('words list',words_list)
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

    async def verify_speaker_or_enroll(self) -> bool:
        """
        Gate on recognized speaker; offer interactive enrollment if unknown.

        Flow:
            - Identify current speaker from self.audio.
            - If not recognized or below SIM_THRESHOLD:
                1) Ask if the user wants to enroll.
                2) If yes, collect a display name and several short samples.
                3) Rebuild enrollments and reload in-memory voice-ID.
                4) Persist centroid embedding and metadata to MongoDB.
                5) Thank the user and optionally confirm with a final sample.
            - On recognized speaker, allow the turn to proceed.

        Returns:
            True if the turn should proceed; False if user declined or on failure.
        """
        try:
            self.best_name, best_sim = self.voice_identifier.identify_from_array(self.audio, 16000)
            if self.best_name is None or best_sim < SIM_THRESHOLD:
                # 1) Ask to enroll
                await self.speaker.speak("I didn’t recognize your voice. Would you like to register it now?")
                print('listening now')
                resp_audio = await asyncio.to_thread(self.detector.record_once)
                resp_text  = await asyncio.to_thread(self.stt.transcribe, resp_audio)
                print('before heard this',resp_text)
                resp_text =resp_text.strip().lower() if resp_audio is not None else "" 
                # resp_text  = (await asyncio.to_thread(self.stt.transcribe, resp_audio)).strip().lower() if resp_audio is not None else ""
                print('heard this', resp_text)
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
                await asyncio.to_thread(subprocess.run, ["python", "build_enrollments.py", "--root", "data", "--out", ENROLL_PATH], check=True)

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
                self.best_name = user_name
                # 5) Optional immediate re-check
                await self.speaker.speak("Say one more sentence to confirm.")
                confirm = await asyncio.to_thread(self.detector.record_once)
                conf_name, conf_sim = self.voice_identifier.identify_from_array(confirm, 16000)
                if conf_name == user_name:
                    await self.speaker.speak(f"Verification passed with similarity {conf_sim:.2f}.")
                else:
                    await self.speaker.speak("Verification was low; we can add more samples later.")
                return True
            else:
                print(f"[Gate] ✅ Allow: {self.best_name} (sim={best_sim:.3f})")
                return True
        except Exception as e:
            print(f"[Gate Error] {e}")
            await self.speaker.speak("Voice check failed. Please try again.")
            return False

    async def handle_style_command_feedback(self) -> None:
        """
        If the latest user message was a style command, provide immediate
        confirmation feedback and persist it to the conversation.
        """
        if self.is_style_command:
            # For style commands, give immediate feedback instead of calling LLM
            current_style = self.convo.get_current_style()
            response = f"I've updated my response style. Current style: {current_style}"
            print(f"Assistant ↳ {response}")
            await self.speaker.speak(response)
            self.convo.add_assistant(response)
            return

    async def stream_llm_and_tts(self):
        """
        Stream the LLM response and speak it sentence-by-sentence.

        Behavior:
            - Mutates the last user message to prepend a wake-word reminder.
            - Streams tokens from Ollama, segments into sentences,
              and feeds each sentence to TTS via a consumer task.
            - Appends the full assistant response to conversation history.
        """
        self.assistant_buffer = []
        self.sentences_queue: asyncio.Queue = asyncio.Queue()
        speak_consumer_task = asyncio.create_task(_speak_consumer(self.sentences_queue, self.speaker, self.detector))  # Pass detector
        
        get_last_text=self.convo.history()[-1] if self.convo.history() else None
        last_content=get_last_text['content'] if get_last_text else "No history"
        if self.best_name is not None:
            sentence_to_add=f"Please say 'Hey {self.best_name}' before speaking to me. "
        else:
            sentence_to_add=""
        # sentence_to_add
        self.convo.messages[-1]['content']=sentence_to_add+self.convo.messages[-1]['content']
        print('last_content',self.convo.messages[-1]['content'])

        # best_name
        async for sentence in sentence_stream(ollama_stream_chat(self.convo.history(), OLLAMA_MODEL, MAX_TOKENS)):
            self.assistant_buffer.append(sentence)
            await self.sentences_queue.put(sentence)
            if PRINT_PARTIAL_SENTENCES:
                print(f"Assistant ↳ {sentence}")

        # Signal completion
        await self.sentences_queue.put(None)
        await speak_consumer_task

        full_assistant_text = ' '.join(self.assistant_buffer)
        self.convo.add_assistant(full_assistant_text)
        

    async def run_turn(self):
        """
        Execute one interaction turn end-to-end:
            capture -> transcribe -> speaker gate/enroll -> stream + speak.
        """
        profiler = TurnProfiler(enabled=self.profile_enabled)
        with profiler.measure("ensure_tts_ready"):
            await self.ensure_tts_ready()
        print("🎤 Listening…", flush=True)

        with profiler.measure("capture_audio"):
            self.audio = await asyncio.to_thread(self.detector.record_once)  # Add back the asyncio.to_thread()

        if not await self.validate_captured_audio():
            profiler.finish()
            return  # no audio captured

        ok_transcribe = False
        with profiler.measure("transcribe_and_wakeword"):
            ok_transcribe = await self.transcribe_and_check_wake_word()
        if not ok_transcribe:
            profiler.finish()
            return  # transcription failed

        if ENABLE_SPEAKER_GATE and self.voice_identifier is not None:
            with profiler.measure("speaker_gate_or_enroll"):
                gate_ok = await self.verify_speaker_or_enroll()
            if not gate_ok:
                profiler.finish()
                return  # speaker enrollment declined or failed

        if not self.transcript.strip():
            profiler.finish()
            return
        print(f"You: {self.transcript}")

        with profiler.measure("style_detection_and_add_user"):
            self.is_style_command = self.convo._detect_style_commands(self.transcript)
            self.convo.add_user(self.transcript)

        if self.is_style_command:
            with profiler.measure("style_feedback"):
                await self.handle_style_command_feedback()
            profiler.finish()
            return

        print("🤖 Assistant (streaming)…", flush=True)
        with profiler.measure("llm_stream_and_tts"):
            await self.stream_llm_and_tts()  # generate and stream LLM response with TTS
        profiler.finish()

