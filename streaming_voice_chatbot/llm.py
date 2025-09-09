#!/usr/bin/env python3
"""
LLM streaming and conversation management for the Streaming Voice Chatbot.
"""

import asyncio
import threading
import re
from typing import AsyncGenerator, List, Dict, Any
import ollama

from .config import default_config


class Conversation:
    """Manages conversation history and style modifications."""
    
    def __init__(self, config=None, system_prompt=None):
        self.config = config or default_config
        self.base_system_prompt = system_prompt or self.config.SYSTEM_PROMPT
        self.current_style_modifier = ""
        self.messages: List[Dict[str, Any]] = [{"role": "system", "content": self._get_full_system_prompt()}]

    def _get_full_system_prompt(self) -> str:
        """Combine base prompt with current style modifier."""
        if self.current_style_modifier:
            return f"{self.base_system_prompt}\n\nCURRENT STYLE OVERRIDE: {self.current_style_modifier}"
        return self.base_system_prompt

    def _update_system_prompt(self):
        """Update the system message with current style."""
        self.messages[0] = {"role": "system", "content": self._get_full_system_prompt()}

    def _detect_style_commands(self, user_text: str) -> bool:
        """Detect and apply style change commands. Returns True if command was processed."""
        text_lower = user_text.lower().strip()
        
        style_commands = {
            "speak more formally": "Use formal, professional language with detailed explanations and proper grammar.",
            "be more formal": "Use formal, professional language with detailed explanations and proper grammar.",
            "be more casual": "Use casual, informal language with contractions and conversational style.",
            "speak more casually": "Use casual, informal language with contractions and conversational style.",
            "be more technical": "Include technical details, terminology, and in-depth explanations.",
            "explain like i'm 5": "Use very simple language, basic analogies, and concepts a child would understand.",
            "explain like im 5": "Use very simple language, basic analogies, and concepts a child would understand.",
            "eli5": "Use very simple language, basic analogies, and concepts a child would understand.",
            "be more concise": "Give very brief, to-the-point responses with minimal elaboration.",
            "be more detailed": "Provide comprehensive explanations with examples and context.",
            "be more creative": "Use creative language, metaphors, and imaginative explanations.",
            "be more professional": "Use business-appropriate language and maintain professional demeanor.",
        }
        
        for command, modifier in style_commands.items():
            if command in text_lower:
                self.current_style_modifier = modifier
                self._update_system_prompt()
                return True
        
        return False

    def add_user(self, text: str):
        """Add user message to conversation history."""
        # Check for style commands before adding to conversation
        is_style_command = self._detect_style_commands(text)
        
        if is_style_command:
            # Add a confirmation message for the style change
            style_name = "default" if not self.current_style_modifier else "updated"
            self.messages.append({"role": "user", "content": text})
            self.messages.append({"role": "assistant", "content": f"Got it! I've switched to {style_name} response style."})
        else:
            self.messages.append({"role": "user", "content": text})

    def add_assistant(self, text: str):
        """Add assistant message to conversation history."""
        self.messages.append({"role": "assistant", "content": text})

    def history(self) -> List[Dict[str, Any]]:
        """Get conversation history."""
        return list(self.messages)

    def get_current_style(self) -> str:
        """Get current style description for debugging."""
        return self.current_style_modifier or "Default conversational style"


async def ollama_stream_chat(conversation: List[dict], model: str, max_tokens: int) -> AsyncGenerator[str, None]:
    """Async generator yielding text chunks from Ollama chat streaming.

    conversation: list of {'role': 'system'|'user'|'assistant', 'content': str}
    """
    loop = asyncio.get_running_loop()
    q: asyncio.Queue = asyncio.Queue()
    
    def worker():
        try:
            # streaming=True returns incremental responses
            for part in ollama.chat(model=model, messages=conversation, stream=True, options={"num_predict": max_tokens}):
                try:
                    msg = part.get('message', {})
                    content = msg.get('content')
                    if content:
                        asyncio.run_coroutine_threadsafe(q.put(content), loop)
                except Exception:
                    continue
        except Exception as e:
            asyncio.run_coroutine_threadsafe(q.put(f"[Error: {e} ]"), loop)
        finally:
            asyncio.run_coroutine_threadsafe(q.put(None), loop)

    threading.Thread(target=worker, daemon=True).start()
    while True:
        chunk = await q.get()
        if chunk is None:
            break
        yield chunk


async def sentence_stream(token_stream: AsyncGenerator[str, None], config=None) -> AsyncGenerator[str, None]:
    """Yield completed sentences as tokens stream in.

    Sentence ends when we see end punctuation followed by whitespace OR we flush at end.
    """
    config = config or default_config
    # Sentence segmentation regex
    SENTENCE_END_REGEX = re.compile(rf"(.+?[{config.SENTENCE_END_CHARS}](?:[\"'\)\]]*)\s+)", re.DOTALL)
    
    buffer = ''
    async for chunk in token_stream:
        buffer += chunk
        # Find full sentence(s)
        while True:
            match = SENTENCE_END_REGEX.search(buffer)
            if not match:
                break
            sentence = match.group(1)
            # Remove from buffer
            buffer = buffer[len(sentence):]
            yield sentence.strip()
    # Flush leftover
    tail = buffer.strip()
    if tail:
        yield tail
