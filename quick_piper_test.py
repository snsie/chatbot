import asyncio
from chatbot.speakers.piper_speaker import PiperSpeaker

async def main():
    sp = PiperSpeaker()
    print("[test] PiperSpeaker instantiated with model:", sp.model_path)
    await sp.speak("Hello from Piper. This is a quick test.")
    await sp.close()
    print("[test] Piper test complete.")

if __name__ == "__main__":
    asyncio.run(main())
