

import asyncio

from typing import Optional
from .base_speaker import BaseSpeaker
from ..utils.utterance_detector import UtteranceDetector

async def _speak_consumer(q: 'asyncio.Queue[Optional[str]]', speaker: BaseSpeaker, detector: UtteranceDetector):
    """
    Asynchronously consumes text sentences from a queue and speaks them using a TTS speaker.
    This function manages the speaking process by:
    1. Muting the microphone before speaking to prevent audio feedback
    2. Processing sentences from the queue until a None sentinel is received
    3. Speaking each sentence using the provided speaker
    4. Handling TTS errors gracefully with exception logging
    5. Applying a dynamic delay after all speaking is complete
    6. Unmuting the microphone to resume audio input
    Args:
        q (asyncio.Queue[Optional[str]]): Queue containing sentences to speak. 
            A None value signals the end of speaking.
        speaker (BaseSpeaker): TTS speaker instance used to vocalize the text.
        detector (UtteranceDetector): Audio detector that manages microphone 
            muting/unmuting to prevent feedback during speech.
    Returns:
        None
    Raises:
        Exception: TTS errors are caught and logged but do not stop processing.
            Other exceptions may propagate from queue operations or speaker methods.
    Note:
        The function includes a 0.5 second delay after speaking completion
        before unmuting the microphone to ensure clean audio transitions.
    """
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
