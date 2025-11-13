from typing import AsyncGenerator, List, Optional, Iterable
from ..constants import VOICE_NAME
import asyncio
import queue
import threading
import pyttsx3
from .base_speaker import BaseSpeaker


class Pyttsx3Speaker(BaseSpeaker):
    """Threaded pyttsx3 speaker; async 'speak' returns when sentence finished."""

    def __init__(self, voice_filter: Optional[str] = VOICE_NAME):
        self.queue: 'queue.Queue[Optional[tuple[str, asyncio.Future]]]' = queue.Queue()
        self.loop = asyncio.get_event_loop()
        self.thread = threading.Thread(target=self._worker, args=(voice_filter,), daemon=True)
        self.thread.start()

    def _worker(self, voice_filter: Optional[str]):
        engine = pyttsx3.init()
        if voice_filter:
            for v in engine.getProperty('voices'):
                name = getattr(v, 'name', '') or ''
                if voice_filter.lower() in name.lower():
                    engine.setProperty('voice', v.id)
                    break
        while True:
            item = self.queue.get()
            if item is None:
                break
            text, fut = item
            try:
                engine.say(text)
                engine.runAndWait()
                self.loop.call_soon_threadsafe(fut.set_result, True)
            except Exception as e:
                self.loop.call_soon_threadsafe(fut.set_exception, e)

    async def speak(self, sentence: str):
        fut = self.loop.create_future()
        self.queue.put((sentence, fut))
        await fut

    async def close(self):
        self.queue.put(None)
        self.thread.join(timeout=1)