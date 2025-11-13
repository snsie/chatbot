# import asyncio
# import string
# from ..constants import SIMILAR_NAMES


# async def transcribing_audio(stt, audio):
#   try:
#       print("📝 Transcribing…", flush=True)
#       transcript = await asyncio.to_thread(stt.transcribe, audio)
#       cora_word_found = False
#       transcript_no_punct = transcript.translate(str.maketrans('', '', string.punctuation))
#       words_list = transcript_no_punct.split()

#       for word in words_list:
#           if word in SIMILAR_NAMES:
#               print(f"Found similar name: {word}")
#               cora_word_found = True
#               break
#       if not cora_word_found:
#           print("No wake word detected; ignoring input.")
#           return False
#       return True
      
#   except Exception as e:
#       print(f"[STT Error] {e}")
#       return False