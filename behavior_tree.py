''' resources 
example code: https://medium.com/@thehummingbird/hands-on-py-trees-part-1-1df1910128e4

documentation: https://py-trees.readthedocs.io/en/devel/behaviours.html
https://docs.ros.org/en/rolling/p/py_trees/py_trees.decorators.html
https://docs.ros.org/en/ros2_packages/jazzy/api/py_trees/py_trees.composites.html
https://py-trees.readthedocs.io/en/release-2.1.x/_modules/py_trees/decorators.html
https://docs.ros.org/en/melodic/api/py_trees/html/modules.html

demos: https://py-trees.readthedocs.io/en/devel/demos.html#py-trees-demo-sequence-program

'''


from time import sleep
from typing import Optional
import py_trees
import asyncio

from pydantic_core import ValidationError
import chatbot_voiceid as cb
from py_trees.behaviour import Behaviour # used to create Action and Condition nodes (execution nodes)
from py_trees.common import Status # used to create the status of a node (success, failure, running)
from py_trees.composites import Sequence, Selector # used to create a sequence of nodes (parent nodes)
from py_trees.decorators import EternalGuard
from py_trees import logging as log_tree # used for terminal prints (visualization of ticks)

async def active_listening(detector:UtteranceDetector, stt: WhisperSTT) -> bool:
    print("🎤 Listening…", flush=True)

    audio = await asyncio.to_thread(detector.record_once)  # Add back the asyncio.to_thread()
    if audio is None or not len(audio):
        print("🛑 No audio captured.")
        return False # Nothing captured; loop again
    try:
        print("📝 Transcribing…", flush=True)
        transcript = await asyncio.to_thread(stt.transcribe, audio)
        cora_word_found = False
        transcript_no_punct = transcript.translate(str.maketrans('', '', string.punctuation))
        words_list = transcript_no_punct.split()

        # listening for "cora"
        for word in words_list:
            if word in SIMILAR_NAMES:
                print(f"Found similar name: {word}")
                cora_word_found = True
                return transcript, audio
        if not cora_word_found:
            print("No wake word detected; ignoring input.")
            return False
        
    except Exception as e:
        print(f"[STT Error] {e}")
        return False

async def voice_registration(detector: UtteranceDetector, stt: WhisperSTT, convo: Conversation, speaker: BaseSpeaker) -> bool | str:
    # 1) Ask to enroll
    await speaker.speak("I didn’t recognize your voice. Would you like to register it now?")
    
    resp_audio = await asyncio.to_thread(detector.record_once)
    resp_text  = (await asyncio.to_thread(stt.transcribe, resp_audio)).strip().lower() if resp_audio is not None else ""
    if not any(k in resp_text for k in ["yes", "yeah", "yep", "sure", "ok", "okay"]):
        await speaker.speak("Okay, I won't enroll right now.")
        return False
    
    # 2) Ask for a display name
    await speaker.speak("What name should I save this voice under?")
    name_audio = await asyncio.to_thread(detector.record_once)
    user_name  = await asyncio.to_thread(stt.transcribe, name_audio)
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
        await speaker.speak(p)
        clip = await asyncio.to_thread(detector.record_once)
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
    voice_identifier.reload_enrollments(ENROLL_PATH)

    # 5) Read the centroid for this user from enrollments.npz and upsert ONE Mongo doc
    npz = np.load(ENROLL_PATH, allow_pickle=True)  # has arrays: names, vecs :contentReference[oaicite:3]{index=3}
    names, vecs = npz["names"], npz["vecs"]
    centroid = None
    for n, v in zip(names, vecs):
        if str(n) == user_name:
            centroid = v
            break
    if centroid is None:
        await speaker.speak("I couldn’t finalize your enrollment. Please try again later.")
        return False
    
    try:
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
        people.update_one(
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
        
    except ValidationError as e:
        await speaker.speak("Error saving your voice profile. Please try again.")
        print(f"Pydantic validation error: {e}")
        return False
    
    
    await speaker.speak(f"Thanks {user_name}. Your voice has been registered.")

    # 5) Optional immediate re-check
    await speaker.speak("Say one more sentence to confirm.")
    confirm = await asyncio.to_thread(detector.record_once)
    conf_name, conf_sim = voice_identifier.identify_from_array(confirm, 16000)
    if conf_name == user_name:
        await speaker.speak(f"Verification passed with similarity {conf_sim:.2f}.")
        # return user_name
    else:
        await speaker.speak("Verification was low; we can add more samples later.")
        # return False
    return user_name

async def cora_response(detector: UtteranceDetector, stt: WhisperSTT, convo: Conversation, speaker: BaseSpeaker, best_name) -> None:
    print("🤖 Assistant (streaming)…", flush=True)

    # Streaming generation
    assistant_buffer = []
    sentences_queue: asyncio.Queue = asyncio.Queue()
    speak_consumer_task = asyncio.create_task(_speak_consumer(sentences_queue, speaker, detector))  # Pass detector
    
    get_last_text=convo.history()[-1] if convo.history() else None
    last_content=get_last_text['content'] if get_last_text else "No history"
    sentence_to_add=f"Please say 'Hey {best_name}' before speaking to me. "
    # sentence_to_add
    convo.messages[-1]['content']=sentence_to_add+convo.messages[-1]['content']
    print('last_content',convo.messages[-1]['content'])
    
    # best_name
    async for sentence in sentence_stream(ollama_stream_chat(convo.history(), OLLAMA_MODEL, MAX_TOKENS)):
        assistant_buffer.append(sentence)
        await sentences_queue.put(sentence)
        if PRINT_PARTIAL_SENTENCES:
            print(f"Assistant ↳ {sentence}")

    # Signal completion
    await sentences_queue.put(None)
    await speak_consumer_task

    full_assistant_text = ' '.join(assistant_buffer)
    convo.add_assistant(full_assistant_text)

async def _speak_consumer(q: 'asyncio.Queue[Optional[str]]', speaker: BaseSpeaker, detector: UtteranceDetector) -> None:
    sentences_spoken = 0
    
    # Mute microphone when starting to speak
    detector.mute_microphone()
    
    while True:
        sentence = await q.get()
        if sentence is None:
            break
        try:
            await speaker.speak(sentence)
            sentences_spoken += 1
        except Exception as e:
            print(f"[TTS Error] {e}")
    
    # Dynamic delay based on how much was spoken
    dynamic_delay = 0.5
    await asyncio.sleep(dynamic_delay)
    
    # Unmute microphone after speaking is completely done
    detector.unmute_microphone()
    print("🔊 Microphone reactivated")


async def process_turn(detector: UtteranceDetector, stt: WhisperSTT, convo: Conversation, speaker: BaseSpeaker):
    ##### 1. active listening ######
    transcript, audio = await active_listening(detector, stt)
    if not transcript or not audio:
        return

    ######### 2. voice recognition #########
    if ENABLE_SPEAKER_GATE and voice_identifier is not None:
        try:
            best_name, best_sim = voice_identifier.identify_from_array(audio, 16000)
            if best_name is None or best_sim < SIM_THRESHOLD:
                user_name = await voice_registration(detector, stt, convo, speaker)
                if not user_name:
                    return
                best_name = user_name
        except Exception as e:
            print(f"[Gate Error] {e}")
            await speaker.speak("Voice check failed. Please try again.")
            return

    if not transcript.strip():
        return
    print(f"You: {transcript}")
    
    # Handle style commands differently
    is_style_command = convo._detect_style_commands(transcript)
    convo.add_user(transcript)
    
    if is_style_command:
        # For style commands, give immediate feedback instead of calling LLM
        current_style = convo.get_current_style()
        response = f"I've updated my response style. Current style: {current_style}"
        print(f"Assistant ↳ {response}")
        await speaker.speak(response)
        convo.add_assistant(response)
        return

    ############ 5. cora response #############
    await cora_response(detector, stt, convo, speaker, best_name)

# =============================
# Entry Point
# =============================
async def main():
    if sys.platform.startswith('win'):
        try:
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())  # type: ignore
        except Exception:
            pass
    print("Booting streaming voice chatbot…")

    detector = UtteranceDetector()
    stt = WhisperSTT(WHISPER_MODEL, WHISPER_COMPUTE)
    speaker = await create_speaker()
    convo = Conversation(SYSTEM_PROMPT)

    try:
        while True:
            await process_turn(detector, stt, convo, speaker)
    except KeyboardInterrupt:
        print("\nExiting…")
    finally:
        await speaker.close()

if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass



####### pytrees classes ########
class CoraResponseAction(Behaviour):
    def __init__(self, name): # 1-time initialization when node object is created
        super(CoraResponseAction, self).__init__(name)
        self.logger.debug("%s.__init__()" % (self.__class__.__name__))


    def setup(self): # delayed 1-timeinitialization used for initializing members you don't want to in __init__
        self.logger.debug(f"Action::setup {self.name}")

    def initialise(self): # initialization when node is ticked for the first time and anytime the status is not RUNNING thereafter, resets behavior before running
        self.logger.debug(f"Action::initialise {self.name}")

    def update(self): # main function that is called when the node is ticked
        self.logger.debug(f"Action::update {self.name}")
        return Status.SUCCESS
    
    def terminate(self, new_status): # code called when node switches to a non-RUNNING state (SUCCESS or FAILURE)
        self.logger.debug(f"Action::terminate {self.name} to {new_status}")

class RegisterVoiceAction(Behaviour):
    def __init__(self, name): # 1-time initialization when node object is created
        super(RegisterVoiceAction, self).__init__(name)
        self.logger.debug("%s.__init__()" % (self.__class__.__name__))


    def setup(self): # delayed 1-timeinitialization used for initializing members you don't want to in __init__
        self.logger.debug(f"Action::setup {self.name}")

    def initialise(self): # initialization when node is ticked for the first time and anytime the status is not RUNNING thereafter, resets behavior before running
        self.logger.debug(f"Action::initialise {self.name}")

    def update(self): # main function that is called when the node is ticked
        self.logger.debug(f"Action::update {self.name}")
        return Status.SUCCESS
    
    def terminate(self, new_status): # code called when node switches to a non-RUNNING state (SUCCESS or FAILURE)
        self.logger.debug(f"Action::terminate {self.name} to {new_status}")


class AudioDetectedCondition(Behaviour):
    def __init__(self, name):
        super(AudioDetectedCondition, self).__init__(name)
        self.logger.debug("%s.__init__()" % (self.__class__.__name__))
        self.blackboard = py_trees.blackboard.Blackboard() # global memory for sharing data between nodes

    def setup(self):
        self.logger.debug(f"Condition::setup {self.name}")

    def initialise(self):
        self.logger.debug(f"Condition::initialise {self.name}")

    def update(self):
        self.logger.debug(f"Condition::update {self.name}")
        if getattr(self.blackboard, "audio_detected", False): # params (object, attribute, default)
            self.blackboard.audio_detected = False # reset audio detected
            return Status.SUCCESS
        return Status.FAILURE
    
    def terminate(self, new_status):
        self.logger.debug(f"Condition::terminate {self.name} to {new_status}")

async def audio_detector_logic(detector):
    blackboard = py_trees.blackboard.Blackboard()
    print("🎤 Listening…", flush=True)

    audio = await asyncio.to_thread(detector.record_once)  # Add back the asyncio.to_thread()
    if audio is not None and len(audio):
        print("Audio captured!")
        blackboard.audio_detected = True # audio successfully captured    

    # blackboard = py_trees.blackboard.Blackboard()
    # print("🎤 Listening…", flush=True)
    # blackboard.audio_detected = False
    # blackboard.audio_token = 0
    # while True:
    #     audio = await asyncio.to_thread(detector.record_once)  # Add back the asyncio.to_thread()
    #     if audio is not None and len(audio):
    #         blackboard.latest_audio = audio
    #         blackboard.audio_detected = True
    #         blackboard.audio_token = (blackboard.audio_token or 0) + 1
    #     await asyncio.sleep(0)

class KnownVoiceCondition(Behaviour):
    def __init__(self, name):
        super(KnownVoiceCondition, self).__init__(name, cb.voice_identifier, sim_threshold_key="SIM_THRESHOLD")
        self.logger.debug("%s.__init__()" % (self.__class__.__name__))
        self.blackboard = py_trees.blackboard.Blackboard()
        self.voice_identifier = cb.voice_identifier
        self.sim_threshold_key = "SIM_THRESHOLD"
        self.last_seen_token = None

    def setup(self):
        self.logger.debug(f"Condition::setup {self.name}")

    def initialise(self):
        self.logger.debug(f"Condition::initialise {self.name}")

    def update(self):
        if getattr(self.blackboard, "unknown_voice", False): # if it is an unknown_voice --> return status.FAILURE to register voice
            self.blackboard.unknown_voice = False
            return Status.FAILURE
        return Status.SUCCESS
    
        self.logger.debug(f"Condition::update {self.name}")
        token = getattr(self.blackboard, "audio_token", None)
        audio = getattr(self.blackboard, "latest_audio", None)

        if token is None or audio is None: # no audio captured (is this necessary? audiodetectedcondition should handle this)
            return Status.FAILURE

        if token == self.last_seen_token:
            ok = getattr(self.blackboard, "id_ok", False)
            return Status.SUCCESS if ok else Status.FAILURE
        
        self.last_seen_token = token
    
    def terminate(self, new_status):
        self.logger.debug(f"Condition::terminate {self.name} to {new_status}")

async def known_voice_logic(audio, speaker: cb.BaseSpeaker):
    blackboard = py_trees.blackboard.Blackboard()
    if cb.ENABLE_SPEAKER_GATE and cb.voice_identifier is not None:
        try:
            best_name, best_sim = cb.voice_identifier.identify_from_array(audio, 16000)
            if best_name is None or best_sim < cb.SIM_THRESHOLD: # if voice isn't known
                blackboard.unknown_voice = True
        except Exception as e:
                print(f"[Gate Error] {e}")
                await speaker.speak("Voice check failed. Please try again.")
                return

def make_bt():
    root_1 = Sequence(name="sequence", memory=False) # figure out which memory to use

    audio_detected_2 = AudioDetectedCondition("Audio Detected?")
    process_audio_3 = Sequence("Processing Audio", memory=False)

    root_1.add_children([audio_detected_2, process_audio_3])

    recognize_voice_4 = Selector("Recognizing Voice", memory=False)
    cora_response_5 = CoraResponseAction("Cora Response")

    process_audio_3.add_children([recognize_voice_4, cora_response_5])

    known_voice_6 = KnownVoiceCondition("Known Voice?")
    register_voice_7 = RegisterVoiceAction("Register Voice")

    recognize_voice_4.add_children([known_voice_6, register_voice_7])

    return root_1

def main():
    log_tree.level = log_tree.Level.DEBUG
    tree = make_bt()
    for i in range(3):
        print(f"\n----- Tick {i} -----")
        tree.tick_once()
        sleep(1)

if __name__ == "__main__":
    main()