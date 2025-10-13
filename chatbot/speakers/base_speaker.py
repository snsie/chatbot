class BaseSpeaker:
    async def speak(self, sentence: str):  # pragma: no cover - interface
        raise NotImplementedError

    async def close(self):  # optional cleanup
        pass