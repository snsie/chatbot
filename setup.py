#!/usr/bin/env python3
"""
Setup script for the Streaming Voice Chatbot module.
"""

from setuptools import setup, find_packages

with open("streaming_voice_chatbot/README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="streaming-voice-chatbot",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A half‑duplex voice assistant with low latency sentence‑by‑sentence TTS",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/streaming-voice-chatbot",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Multimedia :: Sound/Audio :: Speech",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy",
        "sounddevice",
        "webrtcvad",
        "faster-whisper",
        "ollama",
        "pyttsx3",
        "pydantic>=2.0.0",
        "asyncio",
    ],
    extras_require={
        "edge-tts": [
            "edge-tts",
            "pydub",
            "simpleaudio",
        ],
        "coqui-tts": [
            "TTS",
            "torch",
            "torchaudio",
        ],
        "all": [
            "edge-tts",
            "pydub", 
            "simpleaudio",
            "TTS",
            "torch",
            "torchaudio",
        ],
    },
    entry_points={
        "console_scripts": [
            "streaming-voice-chatbot=streaming_voice_chatbot.core:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
