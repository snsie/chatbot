
import asyncio
import re
from pathlib import Path
from ..constants import ENABLE_SPEAKER_GATE, SIM_THRESHOLD

async def voice_check_and_registration(voice_identifier, audio, speaker, detector, stt, ):
  if ENABLE_SPEAKER_GATE and voice_identifier is not None:
    try:
        best_name, best_sim = voice_identifier.identify_from_array(audio, 16000)
        if best_name is None or best_sim < SIM_THRESHOLD:
            # 1) Ask to enroll
            await speaker.speak("I didn’t recognize your voice. Would you like to register it now?")
            
            resp_audio = await asyncio.to_thread(detector.record_once)
            resp_text  = (await asyncio.to_thread(stt.transcribe, resp_audio)).strip().lower() if resp_audio is not None else ""
            if not any(k in resp_text for k in ["yes", "yeah", "yep", "sure", "ok", "okay"]):
                await speaker.speak("Okay, I won't enroll right now.")
                return
            
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
                return
            
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
                print("success")
                
            except ValidationError as e:
                await speaker.speak("Error saving your voice profile. Please try again.")
                print(f"Pydantic validation error: {e}")
                return
            
            
            await speaker.speak(f"Thanks {user_name}. Your voice has been registered.")

            # 5) Optional immediate re-check
            await speaker.speak("Say one more sentence to confirm.")
            confirm = await asyncio.to_thread(detector.record_once)
            conf_name, conf_sim = voice_identifier.identify_from_array(confirm, 16000)
            if conf_name == user_name:
                await speaker.speak(f"Verification passed with similarity {conf_sim:.2f}.")
            else:
                await speaker.speak("Verification was low; we can add more samples later.")
        # else:
        #     print(f"[Gate] ✅ Allow: {best_name} (sim={best_sim:.3f})")
        #     await speaker.speak(f"Hey {best_name}.")
    except Exception as e:
        print(f"[Gate Error] {e}")
        await speaker.speak("Voice check failed. Please try again.")
        return