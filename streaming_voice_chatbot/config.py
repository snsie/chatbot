#!/usr/bin/env python3
"""
Configuration settings for the Streaming Voice Chatbot using Pydantic.
"""

import math
from typing import List, Literal, Union
from pydantic import BaseModel, Field, computed_field, ConfigDict, validator


class Config(BaseModel):
    """
    Configuration class for the Streaming Voice Chatbot using Pydantic for validation.
    
    This provides type checking, validation, and automatic documentation of all settings.
    """
    
    model_config = ConfigDict(
        extra='forbid',  # Don't allow extra fields
        validate_assignment=True,  # Validate on assignment
        frozen=False,  # Allow modification after creation
    )
    
    # Audio Configuration
    sample_rate: int = Field(
        default=16000,
        description="Audio sample rate in Hz",
        ge=8000,  # Greater than or equal to 8000
        le=48000  # Less than or equal to 48000
    )
    
    frame_ms: int = Field(
        default=10,
        description="Milliseconds per frame for capture + VAD",
        ge=1,
        le=100
    )
    
    vad_aggressiveness: int = Field(
        default=2,
        description="Voice Activity Detection aggressiveness (0-3, higher = more aggressive)",
        ge=0,
        le=3
    )
    
    min_utterance_ms: int = Field(
        default=300,
        description="Minimum voiced audio required to accept an utterance (ms)",
        ge=100,
        le=5000
    )
    
    trailing_silence_ms: int = Field(
        default=200,
        description="Silence duration to mark end of utterance (ms)",
        ge=50,
        le=2000
    )

    # Speech-to-Text Configuration
    whisper_model: str = Field(
        default="small.en",
        description="Whisper model to use for speech-to-text"
    )
    
    whisper_compute: Literal["auto", "cpu", "cuda"] = Field(
        default="cuda",
        description="Compute device for Whisper model"
    )

    # LLM Configuration
    ollama_model: str = Field(
        default="gpt-oss:20b",
        description="Ollama model name to use for chat completion"
    )

    # Wake words configuration
    similar_names: List[str] = Field(
        default=[
            "Cora", "Kora", "Korra", "Quora", "Core", "Cori", "Corey", "Coral",
            "Corrie", "Cory", "Corin", "Corie", "Corry", "Kory", "Korey", "Kori",
            "Korrie", "Corah", "Corra", "Corca", "Korla", "Korrah",
            "Cour", "Cor", "Coor", "Koor", "Korr", "Corr", "Quora", "Quorra"
        ],
        description="List of wake words that trigger the assistant"
    )

    # System prompt
    system_prompt: str = Field(
        default="""
You are a helpful voice assistant named Cora. Keep responses concise (1-2 sentences typically). Speak as if having a natural conversation.
""",
        description="System prompt for the LLM assistant"
    )

    # LLM Response Configuration
    max_tokens: int = Field(
        default=512,
        description="Maximum tokens per LLM response",
        ge=1,
        le=4096
    )

    # Text-to-Speech Configuration
    tts_backend: Literal["pyttsx3", "edge-tts", "coqui"] = Field(
        default="edge-tts",
        description="TTS backend to use"
    )
    
    voice_name: str = Field(
        default="en-GB-SoniaNeural",
        description="Voice name (backend-specific)"
    )

    # Runtime Configuration
    tail_delay_sec: float = Field(
        default=0.15,
        description="Delay after TTS before returning to mic (reduce capturing own voice)",
        ge=0.0,
        le=5.0
    )
    
    print_partial_sentences: bool = Field(
        default=True,
        description="Print sentences as they are spoken"
    )

    # Sentence segmentation
    sentence_end_chars: str = Field(
        default=r"\.\!\?…",
        description="Regex character set for sentence endings"
    )

    # Computed properties (derived from other fields)
    @computed_field
    @property
    def frame_samples(self) -> int:
        """Number of audio samples per frame."""
        return int(self.sample_rate * self.frame_ms / 1000)
    
    @computed_field
    @property
    def min_voiced_frames(self) -> int:
        """Minimum number of voiced frames required."""
        return math.ceil(self.min_utterance_ms / self.frame_ms)
    
    @computed_field
    @property
    def trailing_silence_frames(self) -> int:
        """Number of silence frames to mark end of utterance."""
        return math.ceil(self.trailing_silence_ms / self.frame_ms)

    @validator('similar_names')
    def validate_similar_names(cls, v):
        """Ensure wake words list is not empty."""
        if not v:
            raise ValueError("similar_names cannot be empty")
        return v

    @validator('system_prompt')
    def validate_system_prompt(cls, v):
        """Ensure system prompt is not empty."""
        if not v.strip():
            raise ValueError("system_prompt cannot be empty")
        return v

    def model_dump_config(self) -> dict:
        """Return configuration as a dictionary, including computed fields."""
        data = self.model_dump()
        # Add computed fields
        data.update({
            'frame_samples': self.frame_samples,
            'min_voiced_frames': self.min_voiced_frames,
            'trailing_silence_frames': self.trailing_silence_frames
        })
        return data


# Backward compatibility: provide uppercase property access for existing code
class CompatConfig(Config):
    """Backward-compatible configuration that provides uppercase property access."""
    
    @property
    def SAMPLE_RATE(self): return self.sample_rate
    @property 
    def FRAME_MS(self): return self.frame_ms
    @property
    def VAD_AGGRESSIVENESS(self): return self.vad_aggressiveness
    @property
    def MIN_UTTERANCE_MS(self): return self.min_utterance_ms
    @property
    def TRAILING_SILENCE_MS(self): return self.trailing_silence_ms
    @property
    def WHISPER_MODEL(self): return self.whisper_model
    @property
    def WHISPER_COMPUTE(self): return self.whisper_compute
    @property
    def OLLAMA_MODEL(self): return self.ollama_model
    @property
    def SIMILAR_NAMES(self): return self.similar_names
    @property
    def SYSTEM_PROMPT(self): return self.system_prompt
    @property
    def MAX_TOKENS(self): return self.max_tokens
    @property
    def TTS_BACKEND(self): return self.tts_backend
    @property
    def VOICE_NAME(self): return self.voice_name
    @property
    def TAIL_DELAY_SEC(self): return self.tail_delay_sec
    @property
    def PRINT_PARTIAL_SENTENCES(self): return self.print_partial_sentences
    @property
    def SENTENCE_END_CHARS(self): return self.sentence_end_chars
    @property
    def FRAME_SAMPLES(self): return self.frame_samples
    @property
    def MIN_VOICED_FRAMES(self): return self.min_voiced_frames
    @property
    def TRAILING_SILENCE_FRAMES(self): return self.trailing_silence_frames


# Default configuration instance
default_config = CompatConfig()
