from typing import AsyncGenerator
import re
from ..constants import SAMPLE_RATE, SENTENCE_END_REGEX

# =============================
# Sentence Segmentation of Token Stream
# =============================
async def sentence_stream(token_stream: AsyncGenerator[str, None]) -> AsyncGenerator[str, None]:
    """Yield completed sentences as tokens stream in.

    Sentence ends when we see end punctuation followed by whitespace OR we flush at end.
    """
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